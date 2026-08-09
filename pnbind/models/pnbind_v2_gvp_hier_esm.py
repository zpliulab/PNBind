#!/usr/bin/env python3
"""
PNBind-v2 GVP + 层次化ESM融合塔 + IPA 模型
============================================
  
层次化 ESM 融合塔（HierESMFusion）设计：

  Layer 1: MSA FiLM 调制
    各路 ESM（ESM3/ESMC/ESM2）独立编码时，MSA 通过 scale+shift 进行条件调制
    → 进化保守信息在各路特征提取阶段就深度渗入

  Layer 2: 三路 ESM 互相 Cross-Attention（同质语义对齐）
    ESM3 作为主路 Query，分别 attend ESMC 和 ESM2
    → 三路语义空间内部对齐，互相吸收补充视角

  Layer 3: 结构特征 h_s 作为 Q，融合 ESM 作为 KV（结构查序列）
    GVP 输出的结构特征查询融合后的序列语义
    → 结构驱动的序列信息提取

  Layer 4: MSA Cross-Attention 最终校正
    多头 Cross-Attention（Q=输出特征, K=V=MSA）
    → 进化上下文对最终表示做精细校正

esm_feat layout（6016-dim）：
    [:1536]      → ESM3
    [1536:2304]  → ESM-MSA
    [2304:3456]  → ESMC
    [3456:]      → ESM2(2560)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from pnbind.models.pnbind_v2_gvp_model import GVPLayer, safe_norm
from pnbind.models.pnbind_v2_gvp_ipa import build_frames, SimpleIPA, GateFusion
from pnbind.models.gpsite_geo_featurizer import (
    get_geo_feat,
    GEO_NODE_DIM,
    GEO_EDGE_DIM,
)

# ── 维度常量 ──────────────────────────────────────────────────────────────
ESM3_DIM = 1536
MSA_DIM  = 768
ESMC_DIM = 1152
ESM2_DIM = 2560
# ESM3 output head logits (功能先验)
HEAD_RESIDUE_DIM = 32   # 精选 binding 相关 annotation logits
HEAD_SS8_DIM     = 11   # 二级结构 logits
HEAD_SASA_DIM    = 19   # SASA logits
HEAD_TOTAL_DIM   = HEAD_RESIDUE_DIM + HEAD_SS8_DIM + HEAD_SASA_DIM  # = 62


# ============================================================================
# 可学习层权重（Learnable Layer Weighting, à la MegSite）
# ============================================================================

class LearnableLayerWeighting(nn.Module):
    """
    残差软冻结的多层隐状态融合。
    
    输入: (K, N, D) — K 层 × N 残基 × D 维
    输出: (N, D) — 加权融合后的单一表示
    
    核心机制：在均匀平均（先验）和学习权重之间用可学习门控插值。
      output = α · learned_weighted + (1-α) · uniform_mean
    
    α = sigmoid(gate)，gate 初始化为 gate_init：
      gate_init=-2 → α≈0.12（强烈偏向均匀平均，近似冻结）
      gate_init= 0 → α=0.5（均匀和学习各半）
      gate_init= 2 → α≈0.88（偏向学习权重）
    
    训练后检查 gate 值可直观看出模型有多大程度信任学习到的层权重。
    """
    def __init__(self, num_layers: int = 4, gate_init: float = -2.0):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.gate = nn.Parameter(torch.tensor(gate_init))
    
    def forward(self, layer_hiddens: torch.Tensor) -> torch.Tensor:
        """
        layer_hiddens: (K, N, D) 或 batch 场景下 (K, N_total, D)
        返回: (N, D) 或 (N_total, D)
        """
        uniform = layer_hiddens.mean(dim=0)            # (N, D) 均匀平均先验
        w = F.softmax(self.layer_weights, dim=0)        # (K,)
        learned = (w[:, None, None] * layer_hiddens).sum(dim=0)  # (N, D)
        alpha = torch.sigmoid(self.gate)                # 标量 0→1
        return alpha * learned + (1.0 - alpha) * uniform



# ============================================================================
# 子模块 1：FiLM 条件调制层
# ============================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation。
    用条件向量 c 对特征 x 做 scale + shift：
        out = gamma(c) * LayerNorm(x) + beta(c)
    其中 gamma/beta 由 c 的线性变换生成。
    """
    def __init__(self, feat_dim: int, cond_dim: int):
        super().__init__()
        self.norm  = nn.LayerNorm(feat_dim)
        # 一次性生成 gamma 和 beta
        self.film  = nn.Linear(cond_dim, feat_dim * 2)
        # 关键: 用小值 xavier 初始化 weight 而非全零，打破梯度死锁
        # 全零 weight 会导致条件向量 c 的梯度为零，encoder 完全无法学习
        nn.init.xavier_uniform_(self.film.weight, gain=0.01)
        nn.init.ones_(self.film.bias[:feat_dim])   # gamma 初始化为 1
        nn.init.zeros_(self.film.bias[feat_dim:])  # beta  初始化为 0

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (N, feat_dim)
        c: (N, cond_dim)
        """
        params = self.film(c)                          # (N, feat_dim*2)
        gamma, beta = params.chunk(2, dim=-1)          # 各 (N, feat_dim)
        return gamma * self.norm(x) + beta


# ============================================================================
# 子模块 2：轻量级多头 Cross-Attention（残基位置对齐，非全局 N×N）
# ============================================================================

class PositionwiseCrossAttn(nn.Module):
    """
    残基位置对齐的 Cross-Attention：
      每个残基位置 i 的 Q 只 attend 自身位置的 K/V
      （即残基级别的多头线性融合，而非序列级全局 attention）
    
    若需要全局 N×N attention，请使用 GlobalCrossAttn（计算量 O(N²)）。
    当前设计：O(N)，轻量且与 GVP 图结构一致。
    """
    def __init__(self, q_dim: int, kv_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert q_dim % num_heads == 0
        self.H  = num_heads
        self.Hd = q_dim // num_heads
        self.scale = self.Hd ** -0.5

        self.q_proj   = nn.Linear(q_dim,  q_dim,  bias=False)
        self.k_proj   = nn.Linear(kv_dim, q_dim,  bias=False)
        self.v_proj   = nn.Linear(kv_dim, q_dim,  bias=False)
        self.out_proj = nn.Linear(q_dim,  q_dim,  bias=False)
        self.norm     = nn.LayerNorm(q_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q:  (N, q_dim)
        kv: (N, kv_dim)
        返回: (N, q_dim)，残差 + LayerNorm
        """
        N, H, Hd = q.shape[0], self.H, self.Hd
        Q = self.q_proj(q).view(N, H, Hd)    # (N, H, Hd)
        K = self.k_proj(kv).view(N, H, Hd)   # (N, H, Hd)
        V = self.v_proj(kv).view(N, H, Hd)   # (N, H, Hd)

        # 位置对齐：每残基自身 Q·K，sigmoid 门控（非 softmax）
        attn = torch.sigmoid((Q * K).sum(-1, keepdim=True) * self.scale)  # (N, H, 1)
        attn = self.dropout(attn)
        out  = (attn * V).reshape(N, H * Hd)  # (N, q_dim)
        out  = self.out_proj(out)
        return self.norm(q + out)


class GlobalCrossAttn(nn.Module):
    """
    全局 N×N Cross-Attention（用于 MSA 最终校正）：
      每个残基的 Q 可以 attend 所有残基的 MSA K/V
      这是合理的：MSA 捕捉的是全局进化协变，保守位点的信息可以
      跨残基传播（如共进化残基对）。
    计算量 O(N²)，N 通常 50-500，可接受。
    """
    def __init__(self, q_dim: int, kv_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert q_dim % num_heads == 0
        self.H  = num_heads
        self.Hd = q_dim // num_heads
        self.scale = self.Hd ** -0.5

        self.q_proj   = nn.Linear(q_dim,  q_dim,  bias=False)
        self.k_proj   = nn.Linear(kv_dim, q_dim,  bias=False)
        self.v_proj   = nn.Linear(kv_dim, q_dim,  bias=False)
        self.out_proj = nn.Linear(q_dim,  q_dim,  bias=False)
        self.norm     = nn.LayerNorm(q_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(
        self,
        q:   torch.Tensor,          # (N, q_dim)
        kv:  torch.Tensor,          # (N, kv_dim)
        batch: torch.Tensor = None, # (N,) batch 标识，用于跨链 mask
    ) -> torch.Tensor:
        N, H, Hd = q.shape[0], self.H, self.Hd
        Q = self.q_proj(q).view(N, H, Hd)    # (N, H, Hd)
        K = self.k_proj(kv).view(N, H, Hd)   # (N, H, Hd)
        V = self.v_proj(kv).view(N, H, Hd)   # (N, H, Hd)

        # 全局注意力：(N, H, Hd) × (N, H, Hd)^T → (N, H, N)
        # einsum: Q[i,h,d] * K[j,h,d] → scores[i,h,j]
        scores = torch.einsum('nhd,mhd->hnm', Q, K) * self.scale  # (H, N, N)

        # 跨链 mask：同一 batch 内才能 attend
        if batch is not None:
            # batch: (N,) → mask[i,j] = (batch[i] != batch[j])
            mask = (batch.unsqueeze(0) != batch.unsqueeze(1))  # (N, N)
            scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)   # (H, N, N)
        attn = self.dropout(attn)

        # 加权聚合：(H, N, N) × (N, H, Hd) → (N, H, Hd)
        out = torch.einsum('hnm,mhd->nhd', attn, V).reshape(N, H * Hd)
        out = self.out_proj(out)
        return self.norm(q + out)


# ============================================================================
# 层次化 ESM 融合塔（HierESMFusion） - CLEAN IMPLEMENTATION
# ============================================================================

class HierESMFusion(nn.Module):
    def __init__(self, hidden_dim: int = 384, num_heads: int = 8, dropout: float = 0.1, esm3_dim: int = ESM3_DIM):
        super().__init__()
        self._esm3_dim = esm3_dim

        # Layer 1: per-ESM projection + MSA FiLM
        self.esm3_linear = nn.Linear(ESM3_DIM, hidden_dim)
        self.esm3_film = FiLMLayer(hidden_dim, MSA_DIM)
        self.esm3_act = nn.Sequential(nn.SiLU(), nn.Dropout(dropout))

        self.esmc_linear = nn.Linear(ESMC_DIM, hidden_dim)
        self.esmc_film = FiLMLayer(hidden_dim, MSA_DIM)
        self.esmc_act = nn.Sequential(nn.SiLU(), nn.Dropout(dropout))

        self.esm2_linear = nn.Linear(ESM2_DIM, hidden_dim)
        self.esm2_film = FiLMLayer(hidden_dim, MSA_DIM)
        self.esm2_act = nn.Sequential(nn.SiLU(), nn.Dropout(dropout))

        self.msa_encoder = nn.Sequential(
            nn.Linear(MSA_DIM, MSA_DIM),
            nn.LayerNorm(MSA_DIM),
            nn.SiLU(),
        )

        # Layer 2: ESM3 cross-attend ESMC and ESM2
        self.cross_3_c = PositionwiseCrossAttn(hidden_dim, hidden_dim, num_heads, dropout)
        self.cross_3_2 = PositionwiseCrossAttn(hidden_dim, hidden_dim, num_heads, dropout)

        # New: MSA -> structure direct attention heads
        self.msa_to_hidden = nn.Linear(MSA_DIM, hidden_dim)
        self.struct_msa_attn_c = PositionwiseCrossAttn(hidden_dim, hidden_dim, num_heads, dropout)
        self.struct_msa_attn_2 = PositionwiseCrossAttn(hidden_dim, hidden_dim, num_heads, dropout)

        # Sequence fuse
        self.seq_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Layer 3: structure queries sequence
        self.struct_seq_attn = PositionwiseCrossAttn(hidden_dim, hidden_dim, num_heads, dropout)

        # Layer 4: MSA global correction
        self.msa_global_attn = GlobalCrossAttn(hidden_dim, MSA_DIM, num_heads, dropout)

        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_s: torch.Tensor, esm_feat: torch.Tensor, batch: torch.Tensor = None,
                msa_fused: torch.Tensor = None, esmc_fused: torch.Tensor = None,
                esm2_fused: torch.Tensor = None, esm3_override: torch.Tensor = None) -> torch.Tensor:
        # split esm features
        esm3 = esm_feat[:, :ESM3_DIM]  # 始终从 graph_cache 取 1536-dim
        esm_msa = msa_fused if msa_fused is not None else esm_feat[:, ESM3_DIM: ESM3_DIM + MSA_DIM]
        esmc = esmc_fused if esmc_fused is not None else esm_feat[:, ESM3_DIM + MSA_DIM: ESM3_DIM + MSA_DIM + ESMC_DIM]
        esm2 = esm2_fused if esm2_fused is not None else esm_feat[:, ESM3_DIM + MSA_DIM + ESMC_DIM:]

        msa_cond = self.msa_encoder(esm_msa)

        f3 = self.esm3_act(self.esm3_film(self.esm3_linear(esm3), msa_cond))
        fc = self.esmc_act(self.esmc_film(self.esmc_linear(esmc), msa_cond))
        f2 = self.esm2_act(self.esm2_film(self.esm2_linear(esm2), msa_cond))

        # Layer2: ESM3 absorb ESMC and ESM2
        f3 = self.cross_3_c(f3, fc)
        f3 = self.cross_3_2(f3, f2)
        seq_feat = self.seq_fuse(torch.cat([f3, fc, f2], dim=-1))

        # New direct MSA -> structure path
        msa_hidden = self.msa_to_hidden(esm_msa)
        msa_struct_c = self.struct_msa_attn_c(h_s, msa_hidden)
        msa_struct_2 = self.struct_msa_attn_2(h_s, msa_hidden)
        h_s = h_s + msa_struct_c + msa_struct_2

        # Layer3: structure queries fused sequence
        h_out = self.struct_seq_attn(h_s, seq_feat)

        # Layer4: final MSA global correction
        h_out = self.msa_global_attn(h_out, esm_msa, batch=batch)

        return self.out_norm(h_out)


# 恢复类声明：确保 PNBindGVPHierESM 作为完整模型类存在
class PNBindGVPHierESM(nn.Module):
    def __init__(
        self,
        node_scalar_dim:  int = 1558,
        node_vector_dim:  int = 2,
        edge_scalar_dim:  int = 27,
        edge_vector_dim:  int = 1,
        hidden_scalar_dim: int = 384,
        hidden_vector_dim: int = 8,
        num_layers:       int = 4,
        num_ipa_layers:   int = 1,
        ipa_heads:        int = 4,
        ipa_qk_points:    int = 4,
        ipa_v_points:     int = 4,
        hier_heads:       int = 8,
        dropout:          float = 0.3,
        esm3_dim:         int = ESM3_DIM,
        use_ratio_aux:    bool  = False,
        use_length_ratio_cascade: bool = False,
        use_learnable_film: bool = False,
        use_geo_feat:     bool  = False,
        enable_geo_edge:  bool  = False,
        freeze_hier_esm:  bool  = False,
        use_voxel_aux:    bool  = False,
        voxel_dim:        int   = 256,
        esm3_large_only:  bool  = False,
        esm3_large_dim:   int   = 6144,
        use_basic_gnn:    bool  = False,
        use_onehot_esm:   bool  = False,
    ):
        super().__init__()
        self.hidden_scalar_dim = hidden_scalar_dim
        self.esm3_dim = esm3_dim
        self.use_ratio_aux = use_ratio_aux
        self.use_length_ratio_cascade = use_length_ratio_cascade
        self.use_learnable_film = use_learnable_film
        self.use_geo_feat = use_geo_feat
        self.enable_geo_edge = enable_geo_edge
        self.freeze_hier_esm = freeze_hier_esm
        self.use_voxel_aux = use_voxel_aux
        self.voxel_dim = voxel_dim
        self.esm3_large_only = esm3_large_only
        self.esm3_large_dim = esm3_large_dim
        self.use_basic_gnn = use_basic_gnn
        self.use_onehot_esm = use_onehot_esm

        # Basic GNN layers (for ablation: replace GVP with simple message passing)
        if use_basic_gnn:
            self.gnn_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_scalar_dim * 2, hidden_scalar_dim * 2),
                    nn.LayerNorm(hidden_scalar_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_scalar_dim * 2, hidden_scalar_dim),
                    nn.LayerNorm(hidden_scalar_dim),
                ) for _ in range(num_layers)
            ])

        # ESM3 初始编码
        # 可学习多层融合（从 v2/v3 per-layer 加载时使用）
        self.esm3_layer_weighting = LearnableLayerWeighting(num_layers=4)
        self.msa_layer_weighting  = LearnableLayerWeighting(num_layers=4)
        self.esmc_layer_weighting = LearnableLayerWeighting(num_layers=4)
        self.esm2_layer_weighting = LearnableLayerWeighting(num_layers=4)

        # Contact map 编码器（ESM2 attention contact + MSA row attention）
        self.contact_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        # ESM3 v3 路径（LLW per-layer → 1536）
        self.esm3_encoder = nn.Sequential(
            nn.Linear(ESM3_DIM, hidden_scalar_dim * 2),
            nn.LayerNorm(hidden_scalar_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_scalar_dim * 2, hidden_scalar_dim),
            nn.LayerNorm(hidden_scalar_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # One-hot sequence projection used only by the w/o ESM ablation.
        if self.use_onehot_esm:
            self.onehot_encoder = nn.Sequential(
                nn.Linear(20, ESM3_DIM),
                nn.LayerNorm(ESM3_DIM),
                nn.SiLU(),
            )

        # ESM3-large(98B) → 1536 投影（bottleneck，替换 LLW 最后一层）
        if esm3_dim != ESM3_DIM:
            _neck = 512
            self.esm3_large_proj = nn.Sequential(
                nn.Linear(esm3_dim, _neck),
                nn.LayerNorm(_neck),
                nn.SiLU(),
                nn.Linear(_neck, ESM3_DIM),
                nn.LayerNorm(ESM3_DIM),
            )
        else:
            self.esm3_large_proj = None

        # ESM3-large only encoder (绕开 LLW，6144 → hidden 直接编码)
        if esm3_large_only:
            self.esm3_large_only_encoder = nn.Sequential(
                nn.Linear(esm3_large_dim, hidden_scalar_dim * 2),
                nn.LayerNorm(hidden_scalar_dim * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_scalar_dim * 2, hidden_scalar_dim),
                nn.LayerNorm(hidden_scalar_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
        else:
            self.esm3_large_only_encoder = None

        # 物理特征路径 (97 base + 184 GPSite geo if enabled)
        _phys_in_dim = 97 + GEO_NODE_DIM if self.use_geo_feat else 97
        self.phys_encoder = nn.Sequential(
            nn.Linear(_phys_in_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_scalar_dim),
            nn.LayerNorm(hidden_scalar_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # GPSite geometric edge features encoder (optional)
        if self.use_geo_feat:
            self.geo_edge_encoder = nn.Sequential(
                nn.Linear(GEO_EDGE_DIM, hidden_scalar_dim),
                nn.LayerNorm(hidden_scalar_dim),
                nn.SiLU(),
            )

        self.gate_fusion = GateFusion(hidden_scalar_dim)
        self.node_proj_v = nn.Linear(node_vector_dim, hidden_vector_dim)
        self.edge_proj_s = nn.Sequential(
            nn.Linear(edge_scalar_dim, hidden_scalar_dim),
            nn.LayerNorm(hidden_scalar_dim),
            nn.SiLU(),
        )

        self.gvp_layers = nn.ModuleList([
            GVPLayer(
                node_dims=(hidden_scalar_dim, hidden_vector_dim),
                edge_dims=(hidden_scalar_dim, edge_vector_dim),
                dropout=dropout,
            ) for _ in range(num_layers)
        ])

        self.hier_esm = HierESMFusion(
            hidden_dim=hidden_scalar_dim,
            num_heads=hier_heads,
            dropout=dropout,
            esm3_dim=esm3_dim,
        )

        self.ipa_layers = nn.ModuleList([
            SimpleIPA(
                hidden_dim=hidden_scalar_dim,
                num_heads=ipa_heads,
                num_qk_points=ipa_qk_points,
                num_v_points=ipa_v_points,
                dropout=dropout,
            ) for _ in range(num_ipa_layers)
        ])

        # ── ESM3 Head Logits 功能先验模块 ──────────────────────────────
        # ESM3 output heads 包含残基级功能注释预测（binding site / ss8 / sasa）
        # 这是 ESM3 从序列推断的"内在知识"，不依赖外部数据库
        # Gated Residual Add: h_out = h + sigmoid(gate) * encoder(head)
        # 直接残差注入，梯度传播路径短且清晰
        self.head_logits_encoder = nn.Sequential(
            nn.Linear(HEAD_TOTAL_DIM, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_scalar_dim),
            nn.LayerNorm(hidden_scalar_dim),
        )
        # 可学习门控：初始 sigmoid(-2) ≈ 0.12，保守注入
        self.head_gate = nn.Parameter(torch.tensor(-2.0))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_scalar_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # ── Voxel 辅助头 (Inverse-design supervision)：每残基预测 256 维 NA atom occupancy ──
        if use_voxel_aux:
            self.voxel_head = nn.Sequential(
                nn.Linear(hidden_scalar_dim, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(256, voxel_dim),
            )

        # ── 多任务辅助回归头：预测结合位点比例 (可选) ─────────────
        if self.use_ratio_aux:
            self.ratio_head = nn.Sequential(
                nn.Linear(hidden_scalar_dim, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        # ── 级联辅助：链长度→比例→调制主任务 (可选旁支) ──────────
        if self.use_length_ratio_cascade:
            # Step A: 链长度直接 FiLM 调制 h_s（全局上下文注入）
            #   每个残基都能感知"我属于一条长/短链"
            self.len_film_scale = nn.Linear(1, hidden_scalar_dim)
            self.len_film_shift = nn.Linear(1, hidden_scalar_dim)
            nn.init.zeros_(self.len_film_scale.weight)
            nn.init.ones_(self.len_film_scale.bias)    # scale≈1
            nn.init.zeros_(self.len_film_shift.weight)
            nn.init.zeros_(self.len_film_shift.bias)   # shift≈0

            # Step B: graph_feat + len_feat → ratio_pred（辅助监督）
            self.len_ratio_head = nn.Sequential(
                nn.Linear(hidden_scalar_dim + 1, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

            # Step C: ratio_pred → 再次 FiLM 调制 h_s（级联）
            self.ratio_to_scale = nn.Linear(1, hidden_scalar_dim)
            self.ratio_to_shift = nn.Linear(1, hidden_scalar_dim)
            nn.init.zeros_(self.ratio_to_scale.weight)
            nn.init.ones_(self.ratio_to_scale.bias)    # scale≈1
            nn.init.zeros_(self.ratio_to_shift.weight)
            nn.init.zeros_(self.ratio_to_shift.bias)   # shift≈0

        # ── 可学习残差 FiLM：纯参数向量，不依赖输入 (可选旁支) ───
        if self.use_learnable_film:
            self.film_scale = nn.Parameter(torch.ones(hidden_scalar_dim))
            self.film_shift = nn.Parameter(torch.zeros(hidden_scalar_dim))

    def forward(self, data: Data):
        # w/o ESM: replace every pretrained language-model input with a
        # learned projection of the residue one-hot sequence.  Other input
        # branches (evolutionary and geometric) are left unchanged.
        if self.use_onehot_esm:
            if not hasattr(data, 'onehot') or data.onehot is None:
                raise RuntimeError('use_onehot_esm=True requires data.onehot')
            esm3_init = self.onehot_encoder(data.onehot.float())
            h_esm = self.esm3_encoder(esm3_init)
            esm3_override = None
        # ESM3-large only 路径（绕开 LLW + 4 层加权）
        elif self.esm3_large_only and self.esm3_large_only_encoder is not None \
                and hasattr(data, 'esm3_large_feat') and data.esm3_large_feat is not None:
            h_esm = self.esm3_large_only_encoder(data.esm3_large_feat)  # (N, hidden)
            esm3_override = None
        else:
            # ESM3: LLW per-layer 融合，若有 large 则替换最后一层
            esm3_override = None
            if hasattr(data, 'esm3_layers') and data.esm3_layers is not None:
                layers = data.esm3_layers  # (K=4, N, 1536)
                # ESM3-large → 投影到 1536 维，替换最后一层
                if self.esm3_large_proj is not None and hasattr(data, 'esm3_large_feat') and data.esm3_large_feat is not None:
                    large_proj = self.esm3_large_proj(data.esm3_large_feat)  # (N, 1536)
                    layers = layers.clone()
                    layers[-1] = large_proj  # 替换第4层（最后一层）
                esm3_init = self.esm3_layer_weighting(layers)  # (N, 1536)
            else:
                esm3_init = data.esm_feat[:, :ESM3_DIM]
            h_esm = self.esm3_encoder(esm3_init)  # (N, hidden)
        
        # 若启用 GPSite 几何特征，将 geo_node_feat 拼接到 phys_feat 中
        phys_input = data.phys_feat
        if self.use_geo_feat and hasattr(data, 'geo_node_feat') and data.geo_node_feat is not None:
            phys_input = torch.cat([phys_input, data.geo_node_feat], dim=-1)  # (N, 97+184=281)
        h_phys = self.phys_encoder(phys_input)
        h_s = self.gate_fusion(h_esm, h_phys)

        # MSA/ESMC/ESM2: 若有 per-layer，用可学习层权重融合
        msa_fused = None
        esmc_fused = None
        esm2_fused = None
        if self.use_onehot_esm:
            # Explicit zeros prevent fallback to the cached 6016-d ESM tensor
            # inside HierESMFusion.
            N = data.pos.shape[0]
            msa_fused = torch.zeros(N, MSA_DIM, device=data.pos.device, dtype=h_s.dtype)
            esmc_fused = torch.zeros(N, ESMC_DIM, device=data.pos.device, dtype=h_s.dtype)
            esm2_fused = torch.zeros(N, ESM2_DIM, device=data.pos.device, dtype=h_s.dtype)
        elif hasattr(data, 'msa_layers') and data.msa_layers is not None:
            msa_fused = self.msa_layer_weighting(data.msa_layers)  # (N, 768)
        if (not self.use_onehot_esm) and hasattr(data, 'esmc_layers') and data.esmc_layers is not None:
            esmc_fused = self.esmc_layer_weighting(data.esmc_layers)  # (N, 1152)
        if (not self.use_onehot_esm) and hasattr(data, 'esm2_layers') and data.esm2_layers is not None:
            esm2_fused = self.esm2_layer_weighting(data.esm2_layers)  # (N, 1280→2560 pad)
            # ESM2 v2 是 1280 维但 graph_cache 中 ESM2 是 2560，需要 zero-pad
            N = esm2_fused.shape[0]
            esm2_fused = torch.cat([esm2_fused, torch.zeros(N, ESM2_DIM - esm2_fused.shape[1], device=esm2_fused.device)], dim=-1)

        # Contact map 增强边特征
        if hasattr(data, 'contact_map') and data.contact_map is not None:
            src, dst = data.edge_index[0], data.edge_index[1]
            contact_vals = data.contact_map[src, dst]  # (E,) or (E, 2)
            if contact_vals.dim() == 1:
                contact_vals = contact_vals.unsqueeze(-1).expand(-1, 2)
            contact_edge = self.contact_encoder(contact_vals).squeeze(-1)  # (E,)

        h_v = self.node_proj_v(data.node_vectors.transpose(1,2)).transpose(1,2)
        edge_s = self.edge_proj_s(data.edge_attr)
        edge_v = data.edge_vec.unsqueeze(1)
        
        # GPSite geometric edge features
        if self.enable_geo_edge and self.use_geo_feat and hasattr(data, 'coords_5ref') and data.coords_5ref is not None:
            _, geo_edge = get_geo_feat(data.coords_5ref, data.edge_index)
            geo_edge_s = self.geo_edge_encoder(geo_edge)  # (E, hidden)
            edge_s = edge_s + geo_edge_s

        # 将 contact map 信息注入边特征
        if hasattr(data, 'contact_map') and data.contact_map is not None:
            edge_s = edge_s + contact_edge.unsqueeze(-1)

        # ── GVP 或 Basic GNN 消息传递（消融开关）───────────────────────
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else None
        if self.use_basic_gnn:
            # Basic GNN: GraphSAGE-style mean aggregation (支持batch)
            src, dst = data.edge_index[0], data.edge_index[1]
            for layer in self.gnn_layers:
                # Aggregate neighbor hidden states with batch awareness
                neighbor_h = h_s[src]  # (E, hidden)
                if batch is not None and batch.max().item() > 0:
                    from torch_scatter import scatter_mean
                    # Compute batch-aware neighbor aggregation
                    h_agg = scatter_mean(neighbor_h, dst, dim=0, dim_size=h_s.size(0))
                else:
                    # Single graph: simple neighbor mean
                    # Compute neighbor indices for each node
                    ones = torch.ones(dst.size(0), device=h_s.device, dtype=h_s.dtype)
                    neighbor_counts = torch.zeros(h_s.size(0), device=h_s.device)
                    neighbor_counts.scatter_add_(0, dst, ones)
                    neighbor_counts[neighbor_counts == 0] = 1  # avoid div by zero
                    # Sum neighbor features
                    neighbor_sum = torch.zeros_like(h_s)
                    neighbor_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(neighbor_h), neighbor_h)
                    h_agg = neighbor_sum / neighbor_counts.unsqueeze(-1)
                # Combine self + neighbor: h_new = MLP([h_self; h_agg])
                h_s = layer(torch.cat([h_s, h_agg], dim=-1))
        else:
            h_v = self.node_proj_v(data.node_vectors.transpose(1,2)).transpose(1,2)
            for layer in self.gvp_layers:
                h_s, h_v = layer(h_s=h_s, h_v=h_v, edge_index=data.edge_index, edge_s=edge_s, edge_v=edge_v)

        hier_esm_feat = data.esm_feat
        if self.use_onehot_esm:
            # Keep the expected 6016-d layout while zeroing all other ESM
            # streams; the only sequence signal is the one-hot projection.
            hier_esm_feat = torch.cat([
                esm3_init,
                torch.zeros(ESM2_DIM + ESMC_DIM + MSA_DIM,
                            device=esm3_init.device, dtype=esm3_init.dtype)
                    .expand(esm3_init.shape[0], -1),
            ], dim=-1)
        h_s = self.hier_esm(h_s, hier_esm_feat, batch=batch,
                            msa_fused=msa_fused, esmc_fused=esmc_fused,
                            esm2_fused=esm2_fused, esm3_override=esm3_override)

        R = build_frames(data.node_vectors)
        t = data.pos
        for ipa in self.ipa_layers:
            h_s = ipa(h_s, R, t, data.edge_index)

        # ── ESM3 Head Logits 功能先验注入 ────────────────────────────
        if (not self.use_onehot_esm) and hasattr(data, 'head_logits') and data.head_logits is not None:
            head_enc = self.head_logits_encoder(data.head_logits)  # (N, hidden)
            h_s = h_s + torch.sigmoid(self.head_gate) * head_enc

        # ── 级联辅助：链长度→比例→调制主任务 (可选旁支，三步) ────
        ratio_pred = None
        if self.use_length_ratio_cascade:
            from torch_geometric.nn import global_mean_pool
            # ── 预计算链长度 ──
            if batch is not None:
                num_graphs = batch.max().item() + 1
                num_nodes = torch.zeros(num_graphs, device=h_s.device)
                num_nodes.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
            else:
                num_nodes = torch.tensor([h_s.shape[0]], device=h_s.device, dtype=torch.float)
            # 归一化链长度: log(N)/log(2000)，值域约 [0.5, 1.0]
            len_feat = (torch.log(num_nodes + 1.0) / math.log(2000.0)).unsqueeze(-1)  # (B, 1)

            # Step A: 链长度直接 FiLM 调制 h_s（全局上下文注入）
            len_scale = self.len_film_scale(len_feat)   # (B, hidden)
            len_shift = self.len_film_shift(len_feat)   # (B, hidden)
            if batch is not None:
                len_scale = len_scale[batch]             # (N, hidden)
                len_shift = len_shift[batch]             # (N, hidden)
            h_s = h_s * len_scale + len_shift

            # Step B: graph_feat + len_feat → ratio_pred（辅助监督）
            if batch is not None:
                graph_feat = global_mean_pool(h_s, batch)  # (B, hidden)
            else:
                graph_feat = h_s.mean(dim=0, keepdim=True)
            ratio_input = torch.cat([graph_feat, len_feat], dim=-1)  # (B, hidden+1)
            ratio_pred = self.len_ratio_head(ratio_input).squeeze(-1)  # (B,)

            # Step C: ratio_pred → 再次 FiLM 调制 h_s（级联反馈）
            ratio_val = ratio_pred.unsqueeze(-1)         # (B, 1)
            r_scale = self.ratio_to_scale(ratio_val)     # (B, hidden)
            r_shift = self.ratio_to_shift(ratio_val)     # (B, hidden)
            if batch is not None:
                r_scale = r_scale[batch]                 # (N, hidden)
                r_shift = r_shift[batch]                 # (N, hidden)
            h_s = h_s * r_scale + r_shift

        # ── 可学习残差 FiLM (可选旁支) ──────────────────────────
        if self.use_learnable_film:
            h_s = h_s * self.film_scale + self.film_shift

        node_logits = self.classifier(h_s).squeeze(-1)

        # ── Voxel 辅助预测 ─────────────────────────────────────
        voxel_pred = self.voxel_head(h_s) if self.use_voxel_aux else None

        # ── 多任务辅助回归：预测结合位点比例 (可选，两种模式) ─────
        if self.use_length_ratio_cascade and ratio_pred is not None:
            out = {'node_logits': node_logits, 'ratio_pred': ratio_pred}
            if voxel_pred is not None: out['voxel_pred'] = voxel_pred
            return out
        if self.use_ratio_aux:
            from torch_geometric.nn import global_mean_pool
            if batch is not None:
                graph_feat = global_mean_pool(h_s, batch)
            else:
                graph_feat = h_s.mean(dim=0, keepdim=True)
            ratio_pred = self.ratio_head(graph_feat).squeeze(-1)
            out = {'node_logits': node_logits, 'ratio_pred': ratio_pred}
            if voxel_pred is not None: out['voxel_pred'] = voxel_pred
            return out
        if voxel_pred is not None:
            return {'node_logits': node_logits, 'voxel_pred': voxel_pred}

        return node_logits
