#!/usr/bin/env python3
"""
PNBind-v2 GVP + IPA 混合模型
==============================

架构：
    ESM3+ESM-MSA+ESMC+ESM2(8576) ──MLP──┐
                             ├─ GateFusion(384) ─┐
  Phys(97)           ──MLP──┘                    │
                                               ├─ 4×GVPLayer ─ K×IPA ─ Classifier
  node_vectors(N,2,3) ──Linear──(N,8,3) ───────┘
  edge_attr(E,27)    ──Linear──(E,384)
  edge_vec(E,3)      → (E,1,3)

IPA 设计说明：
  - Frame 由 node_vectors 的两个 Cα 差分向量 Gram-Schmidt 正交化构建
  - 注意力限制在 edge_index 邻域内，不做全局 O(N²) attention
  - 批内跨链隔离由 PyG Batch 的 edge_index 自动保证（已验证跨链边=0）
  - 每条边上：标量注意力得分 + 帧内查询/键点距离惩罚 → 局部几何感知
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from pnbind.models.pnbind_v2_gvp_model import GVPLayer, safe_norm


# ============================================================================
# Step 1: build_frames — Cα差分向量 → 局部旋转矩阵（含边界处理）
# ============================================================================

def build_frames(node_vectors: torch.Tensor, validate: bool = False) -> torch.Tensor:
    """
    从 node_vectors 构建每个残基的局部坐标框架旋转矩阵 R。

    node_vectors 的两个向量来源（pnbind_v2_features.py _extract_direction_vectors）：
      v0[i] = normalize(ca[i+1] - ca[i])   # 前向：指向下一个残基
      v1[i] = normalize(ca[i-1] - ca[i])   # 后向：指向上一个残基
    边界处理：链首 v0 存在但 v1=0；链尾 v1 存在但 v0=0。
    用零向量做 Gram-Schmidt 会产生 NaN，需要 fallback。

    Gram-Schmidt 正交化：
      e1 = normalize(v0)
      e2 = normalize(v1 - (v1·e1)*e1)      # v1 在 e1 法平面内的分量
      e3 = e1 × e2                           # 右手系第三轴

    返回：
      R: (N, 3, 3)，行向量正交基，R[i] = [e1, e2, e3]
         全局坐标 p 转局部坐标: local = (p - t) @ R^T = einsum('d,nd->n', p-t, R)
         局部坐标转全局坐标: global = local @ R + t = einsum('n,nd->d', local, R) + t

    Args:
        node_vectors: (N, 2, 3)
        validate: 是否检验 R^T R ≈ I（调试用，训练时关闭）
    """
    v0 = node_vectors[:, 0, :].clone()   # (N, 3) 前向向量
    v1 = node_vectors[:, 1, :].clone()   # (N, 3) 后向向量
    N  = v0.shape[0]
    device = v0.device

    # ── 边界 fallback ──────────────────────────────────────────────────────
    # v0 或 v1 为零向量时（链首/链尾），用另一侧向量填充
    # 链首：v1 ≈ 0，用 v0 的反方向补足
    v0_zero = v0.norm(dim=-1) < 1e-6   # (N,)
    v1_zero = v1.norm(dim=-1) < 1e-6   # (N,)

    # 链首（v1 为零）：v1 ← -v0（反向，保证不与 v0 共线）
    v1[v1_zero] = -v0[v1_zero]
    # 链尾（v0 为零）：v0 ← -v1
    v0[v0_zero] = -v1[v0_zero]
    # 极端情况：两者都为零（单残基链）→ 用标准基向量
    both_zero = v0_zero & v1_zero
    if both_zero.any():
        v0[both_zero] = torch.tensor([1.0, 0.0, 0.0], device=device)
        v1[both_zero] = torch.tensor([0.0, 1.0, 0.0], device=device)

    # ── Gram-Schmidt ────────────────────────────────────────────────────────
    e1 = F.normalize(v0, dim=-1, eps=1e-8)                   # (N, 3)
    proj = (v1 * e1).sum(-1, keepdim=True) * e1              # v1 在 e1 方向上的投影
    e2_raw = v1 - proj
    # 防止 v0 ∥ v1 导致 e2_raw ≈ 0（理论上Cα差分向量不会平行，但数值安全）
    parallel = e2_raw.norm(dim=-1) < 1e-6
    if parallel.any():
        # 用与 e1 不平行的标准基向量构造 e2
        fallback = torch.zeros_like(e1)
        # 选 e1 中绝对值最小的分量方向作为辅助方向
        _, min_idx = e1[parallel].abs().min(dim=-1)
        for j, idx in enumerate(min_idx):
            fallback[parallel.nonzero(as_tuple=False)[j, 0], idx] = 1.0
        e2_raw[parallel] = fallback[parallel] - (fallback[parallel] * e1[parallel]).sum(-1, keepdim=True) * e1[parallel]
    e2 = F.normalize(e2_raw, dim=-1, eps=1e-8)              # (N, 3)
    e3 = torch.linalg.cross(e1, e2)                          # (N, 3)

    # R: (N, 3, 3)，每行是一个正交基向量
    R = torch.stack([e1, e2, e3], dim=1)  # (N, 3, 3)

    if validate:
        # 验证正交性：R @ R^T 应接近 I
        I_approx = torch.bmm(R, R.transpose(1, 2))
        I_ref    = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1)
        err = (I_approx - I_ref).abs().max().item()
        assert err < 1e-4, f"build_frames: R^T R 正交性误差过大 {err:.2e}"

    return R


# ============================================================================
# Step 2: SimpleIPA — 图邻域内的轻量级 IPA 层
# ============================================================================

class SimpleIPA(nn.Module):
    """
    轻量级 Invariant Point Attention（基于 AlphaFold2 IPA 核心思想）。

    在每条图边 (src → dst) 上计算注意力得分：
      score = scalar_attn(Q_dst, K_src) + geometry_attn(qp_dst, kp_src)

    其中 geometry_attn 是"将 dst 的查询点和 src 的键点均变换到 dst 的局部坐标系，
    然后计算距离的负值"，保证对全局旋转/平移不变（SE(3)-invariant）。

    批内跨链隔离说明：
      PyG Batch.from_data_list() 合并时，各图的 edge_index 偏移后不含跨链边
      （已验证：cross-chain edges = 0/5215）。因此 IPA 限制在 edge_index 范围
      内即自动保证跨链不做 attention，无需额外 mask。

    Args:
        hidden_dim:    节点标量特征维度（默认 384）
        num_heads:     注意力头数
        num_qk_points: 每头帧内查询/键点数
        num_v_points:  每头帧内值向量点数
        dropout:       Dropout 率
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        num_heads: int = 4,
        num_qk_points: int = 4,
        num_v_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        self.H   = num_heads
        self.Hd  = hidden_dim // num_heads
        self.Qk  = num_qk_points
        self.Qv  = num_v_points

        # 标量 Q / K / V
        self.q_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_s = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 帧内点偏移（在各自残基的局部坐标系中预测，再转为全局坐标）
        self.q_pts = nn.Linear(hidden_dim, num_heads * num_qk_points * 3, bias=False)
        self.k_pts = nn.Linear(hidden_dim, num_heads * num_qk_points * 3, bias=False)
        self.v_pts = nn.Linear(hidden_dim, num_heads * num_v_points  * 3, bias=False)

        # 每头独立的几何项权重（可学习，AF2原文用 softplus 保证正值）
        self.log_pt_w = nn.Parameter(torch.zeros(num_heads))

        # 输出投影：标量聚合(H*Hd) + 值向量点模长(H*Qv) → hidden_dim
        self.out_proj = nn.Linear(num_heads * (self.Hd + num_v_points), hidden_dim, bias=False)

        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 缩放因子
        self.s_scale  = math.sqrt(self.Hd) ** -1
        # AF2 原文：pt_scale = sqrt(3 * Npts) 的倒数（3 for xyz）
        self.pt_scale = math.sqrt(3.0 * num_qk_points) ** -1

    # ── 辅助：局部坐标 → 全局坐标 ──────────────────────────────────────────
    @staticmethod
    def _local_to_global(local_pts, R, t):
        """
        local_pts: (N, K, 3)  在各自残基局部坐标系中的点
        R:         (N, 3, 3)  行向量旋转矩阵
        t:         (N, 3)     平移（Cα坐标）
        返回:      (N, K, 3)  全局坐标
        global = local @ R + t  （R 是行向量矩阵，左乘列向量等价于右乘行向量矩阵）
        """
        # einsum: 对每个节点 n，每个点 k，local[n,k,:] @ R[n,:,:] → (3,)
        return torch.einsum('nkd,nde->nke', local_pts, R) + t.unsqueeze(1)

    # ── 辅助：全局坐标 → dst 残基的局部坐标 ───────────────────────────────
    @staticmethod
    def _global_to_local(global_pts, R_dst, t_dst):
        """
        global_pts: (E, K, 3)  全局坐标
        R_dst:      (E, 3, 3)  dst 残基的行向量旋转矩阵
        t_dst:      (E, 3)     dst 残基的 Cα 坐标
        返回:       (E, K, 3)  在 dst 局部坐标系中的坐标
        local = (global - t) @ R^T
              = einsum('ekd, edf->ekf', global - t, R)
          （R 行向量矩阵，R^T = R.transpose(1,2)，但右乘 R 等价于左乘 R^T）
        """
        return torch.einsum('ekd,edf->ekf', global_pts - t_dst.unsqueeze(1), R_dst)

    def forward(
        self,
        h_s: torch.Tensor,         # (N, hidden_dim)  节点标量特征
        R: torch.Tensor,           # (N, 3, 3)         残基局部框架
        t: torch.Tensor,           # (N, 3)            Cα 坐标
        edge_index: torch.Tensor,  # (2, E)            图边（已保证无跨链边）
    ) -> torch.Tensor:
        """返回更新后的节点标量特征 (N, hidden_dim)，残差连接 + LayerNorm。"""
        N = h_s.shape[0]
        H, Hd, Qk, Qv = self.H, self.Hd, self.Qk, self.Qv
        src, dst = edge_index           # src 是 key/value 侧，dst 是 query 侧
        E = src.shape[0]

        # ── 1. 标量 QKV 投影 ───────────────────────────────────────────────
        Q = self.q_s(h_s).view(N, H, Hd)  # (N, H, Hd)
        K = self.k_s(h_s).view(N, H, Hd)
        V = self.v_s(h_s).view(N, H, Hd)

        # ── 2. 帧内点：局部偏移 → 全局坐标 ─────────────────────────────────
        # 每个残基在自己的局部坐标系里预测 Qk 个查询点偏移，转为全局坐标
        qp = self._local_to_global(self.q_pts(h_s).view(N, H * Qk, 3), R, t)  # (N, H*Qk, 3)
        kp = self._local_to_global(self.k_pts(h_s).view(N, H * Qk, 3), R, t)  # (N, H*Qk, 3)
        vp = self._local_to_global(self.v_pts(h_s).view(N, H * Qv, 3), R, t)  # (N, H*Qv, 3)

        # ── 3. 每条边上计算注意力得分 ─────────────────────────────────────
        # 3a. 标量注意力：Q_dst · K_src / sqrt(Hd)
        attn_s = (Q[dst] * K[src]).sum(-1) * self.s_scale  # (E, H)

        # 3b. 帧内几何注意力
        R_d = R[dst]  # (E, 3, 3)  dst 残基的局部框架
        t_d = t[dst]  # (E, 3)

        # 将 qp(dst) 和 kp(src) 均变换到 dst 的局部坐标系
        qp_l = self._global_to_local(qp[dst].view(E, H * Qk, 3), R_d, t_d).view(E, H, Qk, 3)
        kp_l = self._global_to_local(kp[src].view(E, H * Qk, 3), R_d, t_d).view(E, H, Qk, 3)

        # 距离惩罚：-0.5 * w * Σ_points ||qp - kp||² / scale
        dist2    = ((qp_l - kp_l) ** 2).sum(-1).sum(-1)   # (E, H)，对所有点求和
        pt_w     = F.softplus(self.log_pt_w)                # (H,)，每头独立权重
        attn_pt  = -0.5 * pt_w[None, :] * dist2 * self.pt_scale  # (E, H)

        attn_logits = attn_s + attn_pt  # (E, H)

        # ── 4. Scatter Softmax（对每个 dst 节点的入边做 softmax）────────────
        # 数值稳定：先减去每个 dst 节点的最大值
        max_l = torch.full((N, H), float('-inf'), dtype=h_s.dtype, device=h_s.device)
        max_l.scatter_reduce_(
            0, dst.unsqueeze(1).expand(E, H), attn_logits,
            reduce='amax', include_self=True
        )
        exp_l = torch.exp(attn_logits - max_l[dst].clamp(min=-1e9))  # (E, H)
        denom = torch.zeros(N, H, dtype=h_s.dtype, device=h_s.device)
        denom.index_add_(0, dst, exp_l)
        attn_w = exp_l / denom[dst].clamp(min=1e-8)   # (E, H)
        attn_w = self.dropout(attn_w)

        # ── 5. 聚合 ────────────────────────────────────────────────────────
        # 5a. 标量值聚合
        agg_s = torch.zeros(N, H, Hd, dtype=h_s.dtype, device=h_s.device)
        agg_s.index_add_(0, dst,
                         (attn_w.unsqueeze(-1) * V[src]).reshape(E, H * Hd)
                          .view(E, H, Hd))
        agg_s = agg_s.view(N, H * Hd)

        # 5b. 帧内值向量点聚合，取模长得旋转不变量
        #     将 vp(src) 变换到 dst 的局部坐标系后加权求和
        vp_l    = self._global_to_local(vp[src].view(E, H * Qv, 3), R_d, t_d).view(E, H, Qv, 3)
        agg_vp  = torch.zeros(N, H * Qv * 3, dtype=h_s.dtype, device=h_s.device)
        agg_vp.index_add_(0, dst,
                          (attn_w.view(E, H, 1, 1) * vp_l).view(E, H * Qv * 3))
        # 取模长：(N, H, Qv, 3) → (N, H, Qv) → (N, H*Qv)
        agg_vp_norms = safe_norm(agg_vp.view(N, H, Qv, 3), dim=-1).view(N, H * Qv)

        # ── 6. 输出投影 + 残差 + LayerNorm ────────────────────────────────
        out = self.out_proj(torch.cat([agg_s, agg_vp_norms], dim=-1))  # (N, hidden_dim)
        return self.norm(h_s + out)


# ============================================================================
# Gate Fusion（与 pnbind_v2_gvp_gate_fusion.py 完全一致）
# ============================================================================

class GateFusion(nn.Module):
    def __init__(self, feature_dim: int = 384):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, esm_feat, phys_feat):
        gate = self.gate_mlp(torch.cat([esm_feat, phys_feat], dim=-1))
        return gate * esm_feat + (1 - gate) * phys_feat


# ============================================================================
# Step 3: PNBindGVPIPA — GVP + IPA 完整模型
# ============================================================================

class PNBindGVPIPA(nn.Module):
    """
    PNBind-v2 GVP + IPA 混合模型。

    与 PNBindGVPGateFusion 的唯一区别：
      在第 4 层 GVP 之后、分类头之前，插入 num_ipa_layers 层 SimpleIPA。
      IPA 只使用标量特征（h_s），向量特征（h_v）在 GVP 之后直接丢弃。
      不改变任何 cache、数据 pipeline 或训练脚本。
    """

    def __init__(
        self,
        node_scalar_dim: int = 1558,   # 兼容旧参数，实际不使用
        node_vector_dim: int = 2,
        edge_scalar_dim: int = 27,
        edge_vector_dim: int = 1,
        hidden_scalar_dim: int = 384,
        hidden_vector_dim: int = 8,
        num_layers: int = 4,            # GVP 层数
        num_ipa_layers: int = 1,        # IPA 层数（1 或 2）
        ipa_heads: int = 4,
        ipa_qk_points: int = 4,
        ipa_v_points: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden_scalar_dim = hidden_scalar_dim

        # ESM 路径: 6016(ESM3+ESM-MSA+ESMC+ESM2) → hidden*2 → hidden
        self.esm3_encoder = nn.Sequential(
            nn.Linear(6016, hidden_scalar_dim * 2),
            nn.LayerNorm(hidden_scalar_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_scalar_dim * 2, hidden_scalar_dim),
            nn.LayerNorm(hidden_scalar_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 物理特征路径: 97 → 256 → 384  (26+3+12+6+20+30=97，含φ/ψ扭转角4维+表面几何3维)
        self.phys_encoder = nn.Sequential(
            nn.Linear(97, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_scalar_dim),
            nn.LayerNorm(hidden_scalar_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Gate Fusion
        self.gate_fusion = GateFusion(hidden_scalar_dim)

        # 向量投影: (N, 2, 3) → (N, hidden_vector_dim, 3)
        self.node_proj_v = nn.Linear(node_vector_dim, hidden_vector_dim)

        # 边标量投影: 27 → 384
        self.edge_proj_s = nn.Sequential(
            nn.Linear(edge_scalar_dim, hidden_scalar_dim),
            nn.LayerNorm(hidden_scalar_dim),
            nn.SiLU(),
        )

        # GVP 层
        self.gvp_layers = nn.ModuleList([
            GVPLayer(
                node_dims=(hidden_scalar_dim, hidden_vector_dim),
                edge_dims=(hidden_scalar_dim, edge_vector_dim),
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # IPA 层（GVP 之后，分类头之前）
        self.ipa_layers = nn.ModuleList([
            SimpleIPA(
                hidden_dim=hidden_scalar_dim,
                num_heads=ipa_heads,
                num_qk_points=ipa_qk_points,
                num_v_points=ipa_v_points,
                dropout=dropout,
            )
            for _ in range(num_ipa_layers)
        ])

        # 分类头（与 gate_fusion 版本相同）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_scalar_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        # ── Phase 1: 双路节点特征编码 ────────────────────────────────────
        h_esm  = self.esm3_encoder(data.esm_feat)   # (N, 384)
        h_phys = self.phys_encoder(data.phys_feat)  # (N, 384)

        # ── Phase 2: Gate Fusion → 标量节点特征 ─────────────────────────
        h_s = self.gate_fusion(h_esm, h_phys)       # (N, 384)

        # ── Phase 3: 向量节点特征投影 ────────────────────────────────────
        h_v = self.node_proj_v(
            data.node_vectors.transpose(1, 2)
        ).transpose(1, 2)                            # (N, 8, 3)

        # ── Phase 4: 边特征投影 ──────────────────────────────────────────
        edge_s = self.edge_proj_s(data.edge_attr)   # (E, 384)
        edge_v = data.edge_vec.unsqueeze(1)          # (E, 1, 3)

        # ── Phase 5: 4 × GVP 消息传递 ────────────────────────────────────
        for layer in self.gvp_layers:
            h_s, h_v = layer(
                h_s=h_s, h_v=h_v,
                edge_index=data.edge_index,
                edge_s=edge_s, edge_v=edge_v,
            )
        # h_v 在 IPA 中不使用，GVP 已将几何信息融入 h_s

        # ── Phase 6: 构建骨架局部框架（Cα差分 Gram-Schmidt）────────────
        R = build_frames(data.node_vectors)  # (N, 3, 3)
        t = data.pos                         # (N, 3)

        # ── Phase 7: K × IPA（图邻域内的帧内几何注意力）────────────────
        # 跨链隔离：edge_index 已保证无跨链边，无需额外 mask
        for ipa in self.ipa_layers:
            h_s = ipa(h_s, R, t, data.edge_index)

        # ── Phase 8: 分类头 ──────────────────────────────────────────────
        return self.classifier(h_s).squeeze(-1)   # (N,)
