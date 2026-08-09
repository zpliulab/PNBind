#!/usr/bin/env python3
"""
PNBind-v2 Surface-Enhanced Model
=================================

基于 PNBindGVPHierESM 最优配置，添加 MetalBind 论文启发的 4 项几何增强：

  1. SubspaceModalityAttention — 子空间感知的模态注意力融合（替代 GateFusion）
  2. Quasi-geodesic distance — 法向量感知的准测地距离
  3. Learnable RBF encoding — 可学习径向基函数距离编码
  4. Heavy-tail kernel — 重尾核（rational-quadratic）边权调制

所有增强均为可选开关，不修改任何现有代码文件。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from pnbind.models.pnbind_v2_gvp_hier_esm import (
    PNBindGVPHierESM, ESM3_DIM, MSA_DIM, ESMC_DIM, ESM2_DIM,
)
from pnbind.models.pnbind_v2_gvp_ipa import build_frames
from pnbind.models.gpsite_geo_featurizer import get_geo_feat


# ============================================================================
# Enhancement 1: SubspaceModalityAttention
# ============================================================================

class SubspaceModalityAttention(nn.Module):
    """
    子空间感知模态注意力融合（inspired by MetalBind / ULSAM）。
    接口与 GateFusion 兼容: forward(esm_feat, phys_feat) -> fused
    """
    def __init__(self, hidden_dim, num_subspaces=4, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_subspaces == 0
        assert num_heads % num_subspaces == 0
        self.S = num_subspaces
        self.Es = hidden_dim // num_subspaces
        hs = num_heads // num_subspaces

        self.proj_esm = nn.Linear(hidden_dim, hidden_dim)
        self.proj_phys = nn.Linear(hidden_dim, hidden_dim)

        self.sub_attns = nn.ModuleList([
            nn.MultiheadAttention(self.Es, hs, dropout=dropout, batch_first=True)
            for _ in range(num_subspaces)
        ])
        self.out_projs = nn.ModuleList([
            nn.Linear(self.Es, self.Es) for _ in range(num_subspaces)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, esm_feat, phys_feat):
        N = esm_feat.shape[0]
        p_esm = self.proj_esm(esm_feat)
        p_phys = self.proj_phys(phys_feat)
        tokens = torch.stack([p_esm, p_phys], dim=1)  # (N, 2, E)
        tokens = tokens.view(N, 2, self.S, self.Es)

        sub_outputs = []
        for s in range(self.S):
            t_s = tokens[:, :, s, :]  # (N, 2, Es)
            out_s, _ = self.sub_attns[s](t_s, t_s, t_s)
            pooled = self.out_projs[s](out_s.mean(dim=1))  # (N, Es)
            sub_outputs.append(pooled)

        return self.norm(torch.cat(sub_outputs, dim=-1))


# ============================================================================
# Enhancement 2+3+4: EdgeGeometryEnhancer
# ============================================================================

class EdgeGeometryEnhancer(nn.Module):
    """边几何增强: 准测地距离 + RBF编码 + 重尾核调制。"""
    def __init__(self, hidden_dim,
                 use_quasi_geodesic=True, use_rbf=True, use_heavy_tail=True,
                 rbf_K=16, rbf_rmax=20.0,
                 rq_length_scale=6.0, rq_alpha=3.0, dropout=0.1):
        super().__init__()
        self.use_qg = use_quasi_geodesic
        self.use_rbf = use_rbf
        self.use_ht = use_heavy_tail

        if use_rbf:
            self.rbf_K = rbf_K
            centers = torch.linspace(0, rbf_rmax, rbf_K)
            self.register_buffer('rbf_centers', centers)
            spacing = rbf_rmax / max(rbf_K - 1, 1)
            self.log_sigma = nn.Parameter(torch.tensor(math.log(0.6 * spacing)))
            self.rbf_proj = nn.Sequential(
                nn.Linear(rbf_K, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )

        if use_heavy_tail:
            self.log_rq_l = nn.Parameter(torch.tensor(math.log(rq_length_scale)))
            self.log_rq_alpha = nn.Parameter(torch.tensor(math.log(rq_alpha)))
            self.kernel_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, pos, node_vectors, edge_index, edge_s):
        src, dst = edge_index

        if self.use_qg:
            v0 = node_vectors[:, 0, :]
            v1 = node_vectors[:, 1, :]
            normals = F.normalize(torch.cross(v0, v1, dim=-1), dim=-1, eps=1e-8)
            diff = pos[src] - pos[dst]
            euclid = torch.norm(diff, dim=-1)
            cos_nn = (normals[src] * normals[dst]).sum(dim=-1)
            distances = euclid * (2.0 - cos_nn)
        else:
            diff = pos[src] - pos[dst]
            distances = torch.norm(diff, dim=-1)

        if self.use_rbf:
            sigma = torch.exp(self.log_sigma)
            rbf = torch.exp(
                -((distances.unsqueeze(-1) - self.rbf_centers) ** 2) / (2 * sigma ** 2)
            )
            edge_s = edge_s + self.rbf_proj(rbf)

        if self.use_ht:
            l = torch.exp(self.log_rq_l)
            alpha = torch.exp(self.log_rq_alpha)
            kernel = (1.0 + distances ** 2 / (2.0 * alpha * l ** 2 + 1e-8)) ** (-alpha)
            gate = torch.sigmoid(self.kernel_gate)
            edge_s = edge_s * (1.0 + gate * (kernel.unsqueeze(-1) - 1.0))

        return edge_s


# ============================================================================
# Main Model
# ============================================================================

class PNBindSurfaceEnhanced(PNBindGVPHierESM):
    """PNBindGVPHierESM + MetalBind 启发增强。"""
    def __init__(self,
                 use_subspace_attn=False,
                 use_quasi_geodesic=False,
                 use_rbf_encoding=False,
                 use_heavy_tail_kernel=False,
                 subspace_count=4, subspace_heads=8,
                 rbf_K=16, rbf_rmax=20.0,
                 rq_length_scale=6.0, rq_alpha=3.0,
                 **kwargs):
        super().__init__(**kwargs)

        self._use_subspace_attn = use_subspace_attn
        self._use_edge_enhance = (
            use_quasi_geodesic or use_rbf_encoding or use_heavy_tail_kernel
        )
        dropout = kwargs.get('dropout', 0.3)

        if use_subspace_attn:
            self.gate_fusion = SubspaceModalityAttention(
                self.hidden_scalar_dim, subspace_count, subspace_heads, dropout)

        if self._use_edge_enhance:
            self.edge_enhancer = EdgeGeometryEnhancer(
                self.hidden_scalar_dim,
                use_quasi_geodesic=use_quasi_geodesic,
                use_rbf=use_rbf_encoding,
                use_heavy_tail=use_heavy_tail_kernel,
                rbf_K=rbf_K, rbf_rmax=rbf_rmax,
                rq_length_scale=rq_length_scale, rq_alpha=rq_alpha,
                dropout=dropout)

    def forward(self, data):
        if not self._use_edge_enhance:
            return super().forward(data)

        # ── 父类 forward + edge_enhancer 注入 ──

        # ESM3 特征
        esm3_override = None
        if self.esm3_large_only and self.esm3_large_only_encoder is not None \
                and hasattr(data, 'esm3_large_feat') and data.esm3_large_feat is not None:
            # ESM3-large only 路径（绕开 LLW + 4 层加权）
            h_esm = self.esm3_large_only_encoder(data.esm3_large_feat)
        elif hasattr(data, 'esm3_layers') and data.esm3_layers is not None:
            layers = data.esm3_layers
            if (self.esm3_large_proj is not None
                    and hasattr(data, 'esm3_large_feat')
                    and data.esm3_large_feat is not None):
                large_proj = self.esm3_large_proj(data.esm3_large_feat)
                layers = layers.clone()
                layers[-1] = large_proj
            esm3_init = self.esm3_layer_weighting(layers)
            h_esm = self.esm3_encoder(esm3_init)
        else:
            esm3_init = data.esm_feat[:, :ESM3_DIM]
            h_esm = self.esm3_encoder(esm3_init)

        # 物理特征
        phys_input = data.phys_feat
        if self.use_geo_feat and hasattr(data, 'geo_node_feat') and data.geo_node_feat is not None:
            phys_input = torch.cat([phys_input, data.geo_node_feat], dim=-1)
        h_phys = self.phys_encoder(phys_input)

        h_s = self.gate_fusion(h_esm, h_phys)

        # 多 ESM 层权重融合
        msa_fused = None
        if hasattr(data, 'msa_layers') and data.msa_layers is not None:
            msa_fused = self.msa_layer_weighting(data.msa_layers)
        esmc_fused = None
        if hasattr(data, 'esmc_layers') and data.esmc_layers is not None:
            esmc_fused = self.esmc_layer_weighting(data.esmc_layers)
        esm2_fused = None
        if hasattr(data, 'esm2_layers') and data.esm2_layers is not None:
            esm2_fused = self.esm2_layer_weighting(data.esm2_layers)
            N = esm2_fused.shape[0]
            esm2_fused = torch.cat([esm2_fused,
                torch.zeros(N, ESM2_DIM - esm2_fused.shape[1],
                            device=esm2_fused.device)], dim=-1)

        # Contact map
        contact_edge = None
        if hasattr(data, 'contact_map') and data.contact_map is not None:
            src, dst = data.edge_index[0], data.edge_index[1]
            contact_vals = data.contact_map[src, dst]
            if contact_vals.dim() == 1:
                contact_vals = contact_vals.unsqueeze(-1).expand(-1, 2)
            contact_edge = self.contact_encoder(contact_vals).squeeze(-1)

        h_v = self.node_proj_v(data.node_vectors.transpose(1, 2)).transpose(1, 2)
        edge_s = self.edge_proj_s(data.edge_attr)
        edge_v = data.edge_vec.unsqueeze(1)

        if (self.enable_geo_edge and self.use_geo_feat
                and hasattr(data, 'coords_5ref') and data.coords_5ref is not None):
            _, geo_edge = get_geo_feat(data.coords_5ref, data.edge_index)
            geo_edge_s = self.geo_edge_encoder(geo_edge)
            edge_s = edge_s + geo_edge_s

        if contact_edge is not None:
            edge_s = edge_s + contact_edge.unsqueeze(-1)

        # ★ Edge Geometry Enhancement ★
        edge_s = self.edge_enhancer(data.pos, data.node_vectors, data.edge_index, edge_s)

        for layer in self.gvp_layers:
            h_s, h_v = layer(h_s=h_s, h_v=h_v, edge_index=data.edge_index,
                             edge_s=edge_s, edge_v=edge_v)

        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else None
        h_s = self.hier_esm(h_s, data.esm_feat, batch=batch,
                            msa_fused=msa_fused, esmc_fused=esmc_fused,
                            esm2_fused=esm2_fused, esm3_override=esm3_override)

        R = build_frames(data.node_vectors)
        t = data.pos
        for ipa in self.ipa_layers:
            h_s = ipa(h_s, R, t, data.edge_index)

        if hasattr(data, 'head_logits') and data.head_logits is not None:
            head_enc = self.head_logits_encoder(data.head_logits)
            h_s = h_s + torch.sigmoid(self.head_gate) * head_enc

        # 级联辅助（最优配置未启用，保留兼容性）
        ratio_pred = None
        if self.use_length_ratio_cascade:
            from torch_geometric.nn import global_mean_pool
            if batch is not None:
                num_graphs = batch.max().item() + 1
                num_nodes = torch.zeros(num_graphs, device=h_s.device)
                num_nodes.scatter_add_(0, batch,
                                       torch.ones_like(batch, dtype=torch.float))
            else:
                num_nodes = torch.tensor([h_s.shape[0]], device=h_s.device,
                                         dtype=torch.float)
            len_feat = (torch.log(num_nodes + 1.0) / math.log(2000.0)).unsqueeze(-1)
            len_scale = self.len_film_scale(len_feat)
            len_shift = self.len_film_shift(len_feat)
            if batch is not None:
                len_scale, len_shift = len_scale[batch], len_shift[batch]
            h_s = h_s * len_scale + len_shift
            if batch is not None:
                graph_feat = global_mean_pool(h_s, batch)
            else:
                graph_feat = h_s.mean(dim=0, keepdim=True)
            ratio_input = torch.cat([graph_feat, len_feat], dim=-1)
            ratio_pred = self.len_ratio_head(ratio_input).squeeze(-1)
            ratio_val = ratio_pred.unsqueeze(-1)
            r_scale = self.ratio_to_scale(ratio_val)
            r_shift = self.ratio_to_shift(ratio_val)
            if batch is not None:
                r_scale, r_shift = r_scale[batch], r_shift[batch]
            h_s = h_s * r_scale + r_shift

        if self.use_learnable_film:
            h_s = h_s * self.film_scale + self.film_shift

        node_logits = self.classifier(h_s).squeeze(-1)

        if self.use_length_ratio_cascade and ratio_pred is not None:
            return {'node_logits': node_logits, 'ratio_pred': ratio_pred}
        if self.use_ratio_aux:
            from torch_geometric.nn import global_mean_pool
            if batch is not None:
                graph_feat = global_mean_pool(h_s, batch)
            else:
                graph_feat = h_s.mean(dim=0, keepdim=True)
            ratio_pred = self.ratio_head(graph_feat).squeeze(-1)
            return {'node_logits': node_logits, 'ratio_pred': ratio_pred}

        return node_logits
