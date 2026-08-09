#!/usr/bin/env python3
"""
PNBind-v2 GVP Gate Fusion 模型（多尺度边版本）
===============================================

输入格式（来自新版 graph cache）：
    data.esm_feat:     (N, 8576)  ESM3(1536) + ESM-MSA(768) + ESMC(1152) + ESM2(5120) 拼接嵌入
  data.phys_feat:    (N, 97)    物理特征（26 + 3 + 12 + 6 + 20 PSSM + 30 HHM）
  data.node_vectors: (N, 2, 3)  残基方向向量（供 GVP 等变建模）
  data.edge_index:   (2, E)     多尺度边（10Å中程 + 6Å短程 + 序列邻域）
  data.edge_attr:    (E, 27)    边标量（24 原始 + 3 类型 one-hot）
  data.edge_vec:     (E, 3)     Cα→Cα 单位方向向量（供 GVP 等变建模）

架构：
  ESM3+ESM-MSA+ESMC+ESM2(4736) ──MLP──┐
                          ├─ GateFusion(384) ─┐
  Phys(90)        ──MLP──┘                    │
                                               ├─ 4×GVPLayer ─ [IPA×K] ─ Classifier ─ logits(N,)
  node_vectors(N,2,3) ──Linear──(N,8,3) ───────┘
  edge_attr(E,27)    ──Linear──(E,384)
  edge_vec(E,3)      ─unsqueeze─(E,1,3)

IPA 说明：
  - 可选，由 num_ipa_layers 控制（0=禁用，与旧行为完全一致；1或2=启用）
  - 插在第 4 层 GVP 之后、分类头之前，仅使用标量特征 h_s
  - Frame 由 node_vectors Cα差分向量 Gram-Schmidt 正交化构建
  - 邻域内做帧内几何注意力（不做全局 L×L），跨链隔离由 edge_index 自动保证
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from pnbind.models.pnbind_v2_gvp_model import GVPLayer, safe_norm
from pnbind.models.pnbind_v2_gvp_ipa import build_frames, SimpleIPA


class GateFusion(nn.Module):
    """可学习门控：自适应融合 ESM3 语义特征和物理特征。"""

    def __init__(self, feature_dim: int = 384):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, esm_feat: torch.Tensor, phys_feat: torch.Tensor) -> torch.Tensor:
        gate = self.gate_mlp(torch.cat([esm_feat, phys_feat], dim=-1))
        return gate * esm_feat + (1 - gate) * phys_feat


class PNBindGVPGateFusion(nn.Module):
    """
    PNBind-v2 GVP Gate Fusion（多尺度边版本）

    接受新版 graph cache 格式：esm_feat / phys_feat / node_vectors /
    edge_attr(27维) / edge_vec 分开存储。
    """

    def __init__(
        self,
        node_scalar_dim: int = 1558,   # 兼容旧参数，实际不使用
        node_vector_dim: int = 2,
        edge_scalar_dim: int = 27,     # 24 + 3 one-hot
        edge_vector_dim: int = 1,
        hidden_scalar_dim: int = 384,
        hidden_vector_dim: int = 8,
        num_layers: int = 4,
        dropout: float = 0.15,
        # IPA 参数（num_ipa_layers=0 时行为与旧版完全一致）
        num_ipa_layers: int = 0,
        ipa_heads: int = 4,
        ipa_qk_points: int = 4,
        ipa_v_points: int = 4,
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

        # 物理特征路径: 97 → 256 → hidden  (26+3+12+6+20+30=97，含φ/ψ扭转角4维+表面几何3维)
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

        # 边标量投影: 27 → hidden
        self.edge_proj_s = nn.Sequential(
            nn.Linear(edge_scalar_dim, hidden_scalar_dim),
            nn.LayerNorm(hidden_scalar_dim),
            nn.SiLU(),
        )

        # GVP 消息传递层
        self.gvp_layers = nn.ModuleList([
            GVPLayer(
                node_dims=(hidden_scalar_dim, hidden_vector_dim),
                edge_dims=(hidden_scalar_dim, edge_vector_dim),
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # IPA 层（可选，插在 GVP 之后）
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

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_scalar_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        # Phase 1: 双路节点特征编码
        h_esm  = self.esm3_encoder(data.esm_feat)    # (N, hidden)
        h_phys = self.phys_encoder(data.phys_feat)   # (N, hidden)

        # Phase 2: Gate Fusion → 标量节点特征
        h_s = self.gate_fusion(h_esm, h_phys)        # (N, hidden)

        # Phase 3: 向量节点特征投影
        # (N, 2, 3) → (N, 3, 2) → Linear → (N, 3, Hv) → (N, Hv, 3)
        h_v = self.node_proj_v(
            data.node_vectors.transpose(1, 2)
        ).transpose(1, 2)                             # (N, Hv, 3)

        # Phase 4: 边特征投影
        edge_s = self.edge_proj_s(data.edge_attr)    # (E, hidden)
        edge_v = data.edge_vec.unsqueeze(1)           # (E, 1, 3)

        # Phase 5: GVP 消息传递
        for layer in self.gvp_layers:
            h_s, h_v = layer(
                h_s=h_s, h_v=h_v,
                edge_index=data.edge_index,
                edge_s=edge_s, edge_v=edge_v,
            )

        # Phase 6: IPA（可选）
        # 跨链隔离由 edge_index 自动保证（已验证批内跨链边=0）
        if self.ipa_layers:
            R = build_frames(data.node_vectors)   # (N, 3, 3)
            t = data.pos                          # (N, 3)
            for ipa in self.ipa_layers:
                h_s = ipa(h_s, R, t, data.edge_index)

        # Phase 7: 分类（仅用标量部分）
        return self.classifier(h_s).squeeze(-1)      # (N,)

