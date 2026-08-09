#!/usr/bin/env python3
"""
GPSite-style Geometric Featurizer
===================================

参考: Yuan et al., eLife 2024 (GPSite)
GitHub: https://github.com/biomed-AI/GPSite

从 5 参考原子坐标 (N, Ca, C, O, R_group) 动态计算丰富的几何特征。

Node geometric features (184-dim):
  - node_angles: backbone dihedral + bond angles sin/cos  (12-dim)
  - node_dist: RBF encoding of 10 intra-residue atom-pair distances (160-dim)
  - node_direction: 4 atom directions in local frame (12-dim)

Edge geometric features (450-dim):
  - pos_embeddings: sinusoidal positional encoding of sequence distance (16-dim)
  - edge_orientation: quaternion of relative frame rotation (4-dim)
  - edge_dist: RBF encoding of 25 inter-residue atom-pair distances (400-dim)
  - edge_direction: bidirectional 5-atom directions in local frame (30-dim)
"""

import numpy as np
import torch
import torch.nn.functional as F


# ================================================================
# Core geometric functions (consistent with GPSite)
# ================================================================

def _rbf(D, D_min=0., D_max=20., D_count=16):
    """Radial basis function distance encoding."""
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)


def _positional_embeddings(edge_index, num_embeddings=16):
    """Sinusoidal positional encoding of sequence distance."""
    d = edge_index[0] - edge_index[1]
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32,
                     device=edge_index.device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    PE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return PE


def _get_angle(X, eps=1e-7):
    """
    Backbone dihedral angles (psi, omega, phi) and bond angles (alpha, beta, gamma).
    X: (N, 5, 3) -> node_angles: (N, 12)
    """
    X_backbone = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
    dX = X_backbone[1:] - X_backbone[:-1]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    # Dihedral angles
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1)  # (N, 6)

    # Bond angles
    cosD = (u_2 * u_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)  # (N, 6)

    node_angles = torch.cat((dihedral, bond_angles), 1)  # (N, 12)
    return node_angles


def _get_distance(X, edge_index):
    """
    RBF distance encoding for intra-residue and inter-residue atom pairs.
    X: (N, 5, 3), edge_index: (2, E)
    Returns: node_dist (N, 160), edge_dist (E, 400)
    """
    atom_N  = X[:, 0]
    atom_Ca = X[:, 1]
    atom_C  = X[:, 2]
    atom_O  = X[:, 3]
    atom_R  = X[:, 4]

    # Node: 10 intra-residue atom pairs
    node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C',
                 'R-N', 'R-Ca', 'R-C', 'R-O']
    atoms = {'N': atom_N, 'Ca': atom_Ca, 'C': atom_C, 'O': atom_O, 'R': atom_R}
    node_dist = []
    for pair in node_list:
        a1, a2 = pair.split('-')
        E_vectors = atoms[a1] - atoms[a2]
        rbf = _rbf(E_vectors.norm(dim=-1))
        node_dist.append(rbf)
    node_dist = torch.cat(node_dist, dim=-1)  # (N, 160)

    # Edge: 25 inter-residue atom pairs (5x5)
    atom_names = ['N', 'Ca', 'C', 'O', 'R']
    edge_dist = []
    for a1 in atom_names:
        for a2 in atom_names:
            E_vectors = atoms[a1][edge_index[0]] - atoms[a2][edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1))
            edge_dist.append(rbf)
    edge_dist = torch.cat(edge_dist, dim=-1)  # (E, 400)

    return node_dist, edge_dist


def _get_direction_orientation(X, edge_index):
    """
    Local coordinate system directions and relative orientations.
    X: (N, 5, 3), edge_index: (2, E)
    Returns: node_direction (N, 12), edge_direction (E, 30), edge_orientation (E, 4)
    """
    X_N  = X[:, 0]
    X_Ca = X[:, 1]
    X_C  = X[:, 2]

    # Local coordinate system from N, Ca, C
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v), dim=-1)
    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1)  # (N, 3, 3)

    node_j, node_i = edge_index

    # Node direction: 4 atoms (N,C,O,R) relative to Ca in local frame
    t = F.normalize(X[:, [0, 2, 3, 4]] - X_Ca.unsqueeze(1), dim=-1)  # (N, 4, 3)
    node_direction = torch.matmul(t, local_frame).reshape(t.shape[0], -1)  # (N, 12)

    # Edge direction (bidirectional)
    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1)  # (E, 5, 3)
    edge_direction_ji = torch.matmul(t, local_frame[node_i]).reshape(
        t.shape[0], -1)  # (E, 15)
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1)  # (E, 5, 3)
    edge_direction_ij = torch.matmul(t, local_frame[node_j]).reshape(
        t.shape[0], -1)  # (E, 15)
    edge_direction = torch.cat([edge_direction_ji, edge_direction_ij],
                               dim=-1)  # (E, 30)

    # Edge orientation: relative rotation quaternion
    r = torch.matmul(local_frame[node_i].transpose(-1, -2),
                     local_frame[node_j])  # (E, 3, 3)
    edge_orientation = _quaternions(r)  # (E, 4)

    return node_direction, edge_direction, edge_orientation


def _quaternions(R):
    """Convert rotation matrices to quaternions. R: (E, 3, 3) -> Q: (E, 4)"""
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
        Rxx - Ryy - Rzz,
        -Rxx + Ryy - Rzz,
        -Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i, j: R[:, i, j]
    signs = torch.sign(torch.stack([
        _R(2, 1) - _R(1, 2),
        _R(0, 2) - _R(2, 0),
        _R(1, 0) - _R(0, 1)
    ], -1))
    xyz = signs * magnitudes
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)
    return Q


# ================================================================
# Main interface
# ================================================================

def get_geo_feat(X, edge_index):
    """
    Compute full GPSite geometric features from 5-reference-atom coordinates.
    
    Args:
        X: (N, 5, 3) - 5 reference atom coords [N, Ca, C, O, R_group]
        edge_index: (2, E) - edge indices
    
    Returns:
        geo_node_feat: (N, 184)
        geo_edge_feat: (E, 450)
    """
    pos_embeddings = _positional_embeddings(edge_index)         # (E, 16)
    node_angles = _get_angle(X)                                  # (N, 12)
    node_dist, edge_dist = _get_distance(X, edge_index)         # (N, 160), (E, 400)
    node_direction, edge_direction, edge_orientation = \
        _get_direction_orientation(X, edge_index)                # (N, 12), (E, 30), (E, 4)

    geo_node_feat = torch.cat([node_angles, node_dist, node_direction],
                              dim=-1)                            # (N, 184)
    geo_edge_feat = torch.cat([pos_embeddings, edge_orientation,
                               edge_dist, edge_direction],
                              dim=-1)                            # (E, 450)

    return geo_node_feat, geo_edge_feat


# Dimension constants
GEO_NODE_DIM = 184
GEO_EDGE_DIM = 450


def precompute_geo_node_feat(X):
    """
    Precompute node geometric features that don't depend on edge_index.
    
    Args:
        X: (N, 5, 3) - 5 reference atom coords
    
    Returns:
        geo_node: (N, 184) - angles(12) + dist(160) + direction(12)
    """
    node_angles = _get_angle(X)  # (N, 12)
    
    atom_N  = X[:, 0]
    atom_Ca = X[:, 1]
    atom_C  = X[:, 2]
    atom_O  = X[:, 3]
    atom_R  = X[:, 4]
    
    node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C',
                 'R-N', 'R-Ca', 'R-C', 'R-O']
    atoms = {'N': atom_N, 'Ca': atom_Ca, 'C': atom_C, 'O': atom_O, 'R': atom_R}
    node_dist = []
    for pair in node_list:
        a1, a2 = pair.split('-')
        E_vectors = atoms[a1] - atoms[a2]
        rbf = _rbf(E_vectors.norm(dim=-1))
        node_dist.append(rbf)
    node_dist = torch.cat(node_dist, dim=-1)  # (N, 160)
    
    # Local coordinate system
    u = F.normalize(atom_Ca - atom_N, dim=-1)
    v = F.normalize(atom_C - atom_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v), dim=-1)
    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1)  # (N, 3, 3)
    
    t = F.normalize(X[:, [0, 2, 3, 4]] - atom_Ca.unsqueeze(1), dim=-1)  # (N, 4, 3)
    node_direction = torch.matmul(t, local_frame).reshape(t.shape[0], -1)  # (N, 12)
    
    return torch.cat([node_angles, node_dist, node_direction], dim=-1)  # (N, 184)
