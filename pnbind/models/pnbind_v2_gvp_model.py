#!/usr/bin/env python3
"""
PNBind-v2 GVP模型架构
===================

基于Geometric Vector Perceptron的蛋白质-核酸结合位点预测模型

架构:
├── Phase 1: 特征投影
│   ├── 节点标量: 1558 → 256
│   ├── 节点向量: (2, 3) → (8, 3)
│   ├── 边标量: 24 → 128
│   └── 边向量: (3,) → (1, 3)
├── Phase 2: GVP消息传递
│   └── GVPLayer × 4 (标量/向量双轨)
└── Phase 3: 分类头
    └── MLP: 256 → 128 → 1

关键约束:
- SE(3)等变性: 向量通道对旋转/平移保持等变
- 数值稳定性: 使用安全的模长计算,防止NaN
- 向量激活: 只对标量激活,向量保持线性

Author: PNBind-v2 Team
Date: 2026-03-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple
import math


# ============================================================================
# 工具函数: 数值稳定的向量操作
# ============================================================================

def safe_norm(vectors, dim=-1, keepdim=False, eps=1e-8):
    """
    安全的向量模长计算,防止梯度爆炸

    Args:
        vectors: (..., 3) 向量
        dim: 计算模长的维度
        keepdim: 是否保持维度
        eps: 数值稳定性常数

    Returns:
        norm: (...,) 或 (..., 1) 模长

    注意: 不要使用torch.norm,因为当向量趋近0时,
          ∂(√x)/∂x = 1/(2√x) → ∞ 会导致梯度爆炸!
    """
    norm_sq = torch.sum(vectors ** 2, dim=dim, keepdim=keepdim)
    norm = torch.sqrt(norm_sq + eps)
    return norm


def safe_normalize(vectors, dim=-1, eps=1e-8):
    """
    安全的向量归一化

    Args:
        vectors: (..., 3) 向量
        dim: 归一化的维度
        eps: 数值稳定性常数

    Returns:
        normalized: (..., 3) 归一化后的向量
    """
    norm = safe_norm(vectors, dim=dim, keepdim=True, eps=eps)
    return vectors / norm


# ============================================================================
# GVP Layer: 标量/向量双轨消息传递层
# ============================================================================

class GVPLayer(nn.Module):
    """
    Geometric Vector Perceptron Layer

    保持SE(3)等变性的图神经网络层,同时处理标量和向量特征

    输入:
        h_s: (N, hidden_s) 节点标量特征
        h_v: (N, hidden_v, 3) 节点向量特征
        edge_index: (2, E) 边连接
        edge_s: (E, edge_s) 边标量特征
        edge_v: (E, edge_v, 3) 边向量特征

    输出:
        h_s_new: (N, hidden_s) 更新后的标量
        h_v_new: (N, hidden_v, 3) 更新后的向量
    """

    def __init__(
        self,
        node_dims: Tuple[int, int],
        edge_dims: Tuple[int, int],
        dropout: float = 0.1,
    ):
        """
        Args:
            node_dims: (scalar_dim, vector_dim) 节点特征维度
            edge_dims: (scalar_dim, vector_dim) 边特征维度
            dropout: Dropout率
        """
        super().__init__()

        self.si, self.vi = node_dims  # 输入维度
        self.so, self.vo = node_dims  # 输出维度 (这里设为相同)
        self.se, self.ve = edge_dims  # 边维度
        self.dropout = dropout

        # 消息MLP (标量通道)
        # 输入: [h_s_i, h_s_j, edge_s, ||h_v_i||, ||h_v_j||, ||edge_v||]
        scalar_message_dim = 2 * self.si + self.se + self.vi + self.vi + self.ve
        self.message_mlp_s = nn.Sequential(
            nn.Linear(scalar_message_dim, self.so * 2),
            nn.LayerNorm(self.so * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.so * 2, self.so),
        )

        # 消息MLP (向量通道)
        # 输入向量: [h_v_i, h_v_j, edge_v]
        # 通过标量权重加权组合
        self.message_mlp_v = nn.Sequential(
            nn.Linear(self.si, self.vo * (2 * self.vi + self.ve)),
            nn.LayerNorm(self.vo * (2 * self.vi + self.ve)),
        )

        # 更新MLP (标量通道)
        self.update_mlp_s = nn.Sequential(
            nn.Linear(self.si + self.so, self.so),
            nn.LayerNorm(self.so),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 更新MLP (向量通道)
        self.update_mlp_v = nn.Sequential(
            nn.Linear(self.vi + self.vo, self.vo),
            nn.LayerNorm(self.vo),
        )

        # LayerNorm
        self.norm_s = nn.LayerNorm(self.so)
        self.norm_v = nn.LayerNorm(self.vo)

    def forward(
        self,
        h_s: torch.Tensor,
        h_v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            h_s: (N, hidden_s) 节点标量
            h_v: (N, hidden_v, 3) 节点向量
            edge_index: (2, E) 边索引
            edge_s: (E, edge_s) 边标量
            edge_v: (E, edge_v, 3) 边向量

        Returns:
            h_s_new: (N, hidden_s) 更新后的标量
            h_v_new: (N, hidden_v, 3) 更新后的向量
        """
        row, col = edge_index  # row=source, col=target

        # 提取边上的节点特征
        h_s_i = h_s[row]  # (E, hidden_s)
        h_s_j = h_s[col]  # (E, hidden_s)
        h_v_i = h_v[row]  # (E, hidden_v, 3)
        h_v_j = h_v[col]  # (E, hidden_v, 3)

        # 计算消息
        m_s, m_v = self.compute_message(h_s_i, h_s_j, h_v_i, h_v_j, edge_s, edge_v)

        # 聚合消息
        try:
            from torch_scatter import scatter_add
        except ImportError:
            # 回退到手动实现
            def scatter_add(src, index, dim=0, dim_size=None):
                if dim_size is None:
                    dim_size = index.max().item() + 1
                out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
                out.index_add_(dim, index, src)
                return out

        N = h_s.shape[0]
        m_s_agg = scatter_add(m_s, row, dim=0, dim_size=N)  # (N, hidden_s)
        m_v_agg = scatter_add(m_v, row, dim=0, dim_size=N)  # (N, hidden_v, 3)

        # 更新节点特征 (带残差连接)
        h_s_update = self.update_mlp_s(torch.cat([h_s, m_s_agg], dim=-1))
        h_s_new = self.norm_s(h_s + h_s_update)

        h_v_update = self.update_mlp_v(
            torch.cat([
                safe_norm(h_v, dim=-1),  # (N, hidden_v)
                safe_norm(m_v_agg, dim=-1),  # (N, hidden_v)
            ], dim=-1)
        )  # (N, hidden_v)

        # 向量更新: 用标量权重缩放向量
        h_v_new = h_v + m_v_agg * h_v_update.unsqueeze(-1)  # (N, hidden_v, 3)

        return h_s_new, h_v_new

    def compute_message(
        self,
        h_s_i: torch.Tensor,
        h_s_j: torch.Tensor,
        h_v_i: torch.Tensor,
        h_v_j: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算消息

        Args:
            h_s_i: (E, hidden_s) source节点标量
            h_s_j: (E, hidden_s) target节点标量
            h_v_i: (E, hidden_v, 3) source节点向量
            h_v_j: (E, hidden_v, 3) target节点向量
            edge_s: (E, edge_s) 边标量
            edge_v: (E, edge_v, 3) 边向量

        Returns:
            m_s: (E, hidden_s) 标量消息
            m_v: (E, hidden_v, 3) 向量消息
        """
        # 提取向量的标量不变量 (模长)
        v_i_norm = safe_norm(h_v_i, dim=-1)  # (E, hidden_v)
        v_j_norm = safe_norm(h_v_j, dim=-1)  # (E, hidden_v)
        edge_v_norm = safe_norm(edge_v, dim=-1)  # (E, edge_v)

        # 标量消息
        scalar_input = torch.cat([
            h_s_i,
            h_s_j,
            edge_s,
            v_i_norm,
            v_j_norm,
            edge_v_norm,
        ], dim=-1)
        m_s = self.message_mlp_s(scalar_input)  # (E, hidden_s)

        # 向量消息
        # 通过标量MLP生成权重,然后加权组合向量
        v_weights = self.message_mlp_v(h_s_i)  # (E, hidden_v * (2*hidden_v + edge_v))
        v_weights = v_weights.view(-1, self.vo, 2 * self.vi + self.ve)  # (E, hidden_v, 2*hidden_v+edge_v)

        # 拼接所有向量: [h_v_i, h_v_j, edge_v]
        all_vectors = torch.cat([
            h_v_i,  # (E, hidden_v, 3)
            h_v_j,  # (E, hidden_v, 3)
            edge_v,  # (E, edge_v, 3)
        ], dim=1)  # (E, 2*hidden_v + edge_v, 3)

        # 加权组合: (E, hidden_v, 2*hidden_v+edge_v) @ (E, 2*hidden_v+edge_v, 3)
        m_v = torch.einsum('evc,ecx->evx', v_weights, all_vectors)  # (E, hidden_v, 3)

        return m_s, m_v


# ============================================================================
# 完整模型: PNBindGVP
# ============================================================================

class PNBindGVP(nn.Module):
    """
    PNBind-v2 GVP模型

    基于Geometric Vector Perceptron的蛋白质-核酸结合位点预测

    输入:
        data.x: (N, 1558) 节点标量特征
        data.node_vectors: (N, 2, 3) 节点方向向量
        data.edge_index: (2, E) 边连接
        data.edge_attr: (E, 24) 边标量特征
        data.edge_vec: (E, 3) 边方向向量

    输出:
        logits: (N,) 每个残基的预测logit (raw, 不加sigmoid)
    """

    def __init__(
        self,
        node_scalar_dim: int = 1558,
        node_vector_dim: int = 2,
        edge_scalar_dim: int = 24,
        edge_vector_dim: int = 1,
        hidden_scalar_dim: int = 256,
        hidden_vector_dim: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            node_scalar_dim: 节点标量特征维度 (1558)
            node_vector_dim: 节点向量特征数量 (2个方向向量)
            edge_scalar_dim: 边标量特征维度 (24)
            edge_vector_dim: 边向量特征数量 (1个方向向量)
            hidden_scalar_dim: 隐藏标量维度 (256)
            hidden_vector_dim: 隐藏向量维度 (8)
            num_layers: GVP层数 (4)
            dropout: Dropout率 (0.1)
        """
        super().__init__()

        self.node_scalar_dim = node_scalar_dim
        self.node_vector_dim = node_vector_dim
        self.edge_scalar_dim = edge_scalar_dim
        self.edge_vector_dim = edge_vector_dim
        self.hidden_scalar_dim = hidden_scalar_dim
        self.hidden_vector_dim = hidden_vector_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # 输入投影
        self.node_proj_s = nn.Sequential(
            nn.Linear(node_scalar_dim, hidden_scalar_dim),
            nn.LayerNorm(hidden_scalar_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 向量投影: (N, 2, 3) → (N, 8, 3)
        self.node_proj_v = nn.Linear(node_vector_dim, hidden_vector_dim)

        self.edge_proj_s = nn.Sequential(
            nn.Linear(edge_scalar_dim, hidden_scalar_dim),
            nn.LayerNorm(hidden_scalar_dim),
            nn.SiLU(),
        )

        # edge_vec直接使用,不投影

        # GVP层
        self.gvp_layers = nn.ModuleList([
            GVPLayer(
                node_dims=(hidden_scalar_dim, hidden_vector_dim),
                edge_dims=(hidden_scalar_dim, edge_vector_dim),
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 分类头 (只用标量特征)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_scalar_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播

        Args:
            data: PyG Data对象,包含:
                - x: (N, 1558) 节点标量
                - node_vectors: (N, 2, 3) 节点向量
                - edge_index: (2, E) 边索引
                - edge_attr: (E, 24) 边标量
                - edge_vec: (E, 3) 边向量

        Returns:
            logits: (N,) 预测logits (raw, 不加sigmoid)
        """
        # 投影输入特征
        h_s = self.node_proj_s(data.x)  # (N, 256)

        # 向量投影: (N, 2, 3) → (N, 8, 3)
        # 需要转置: (N, 2, 3) → (N, 3, 2) → Linear → (N, 3, 8) → (N, 8, 3)
        h_v = data.node_vectors.transpose(1, 2)  # (N, 3, 2)
        h_v = self.node_proj_v(h_v)  # (N, 3, 8)
        h_v = h_v.transpose(1, 2)  # (N, 8, 3)

        edge_s = self.edge_proj_s(data.edge_attr)  # (E, 256)
        edge_v = data.edge_vec.unsqueeze(1)  # (E, 1, 3)

        # GVP消息传递
        for layer in self.gvp_layers:
            h_s, h_v = layer(h_s, h_v, data.edge_index, edge_s, edge_v)

        # 分类 (只用标量特征)
        logits = self.classifier(h_s).squeeze(-1)  # (N,)

        return logits


# ============================================================================
# 工具函数
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_gvp_layer():
    """测试GVP Layer"""
    print("=" * 80)
    print("测试 GVP Layer")
    print("=" * 80)

    # 模拟数据
    N, E = 100, 500
    hidden_s, hidden_v = 256, 8
    edge_s_dim, edge_v_dim = 128, 1

    h_s = torch.randn(N, hidden_s)
    h_v = torch.randn(N, hidden_v, 3)
    edge_index = torch.randint(0, N, (2, E))
    edge_s = torch.randn(E, edge_s_dim)
    edge_v = torch.randn(E, edge_v_dim, 3)

    # 创建层
    layer = GVPLayer(
        node_dims=(hidden_s, hidden_v),
        edge_dims=(edge_s_dim, edge_v_dim),
        dropout=0.1,
    )

    # 前向传播
    h_s_new, h_v_new = layer(h_s, h_v, edge_index, edge_s, edge_v)

    # 验证
    print(f"输入标量: {h_s.shape}")
    print(f"输入向量: {h_v.shape}")
    print(f"输出标量: {h_s_new.shape}")
    print(f"输出向量: {h_v_new.shape}")
    print(f"标量范围: [{h_s_new.min():.3f}, {h_s_new.max():.3f}]")
    print(f"向量范围: [{h_v_new.min():.3f}, {h_v_new.max():.3f}]")
    print(f"是否有NaN: {torch.isnan(h_s_new).any() or torch.isnan(h_v_new).any()}")

    assert h_s_new.shape == (N, hidden_s)
    assert h_v_new.shape == (N, hidden_v, 3)
    assert not torch.isnan(h_s_new).any()
    assert not torch.isnan(h_v_new).any()

    print("✓ GVP Layer测试通过!")


def test_full_model():
    """测试完整模型"""
    print("\n" + "=" * 80)
    print("测试完整模型")
    print("=" * 80)

    # 模拟数据
    N, E = 100, 500

    # 创建模拟的Data对象
    data = Data(
        x=torch.randn(N, 1558),
        node_vectors=torch.randn(N, 2, 3),
        edge_index=torch.randint(0, N, (2, E)),
        edge_attr=torch.randn(E, 24),
        edge_vec=torch.randn(E, 3),
        y=torch.randint(0, 2, (N,)).float(),
    )

    # 创建模型
    model = PNBindGVP(
        node_scalar_dim=1558,
        node_vector_dim=2,
        edge_scalar_dim=24,
        edge_vector_dim=1,
        hidden_scalar_dim=256,
        hidden_vector_dim=8,
        num_layers=4,
        dropout=0.1,
    )

    print(f"模型参数量: {count_parameters(model):,}")

    # 前向传播
    model.eval()
    with torch.no_grad():
        logits = model(data)

    # 验证
    print(f"输入节点数: {N}")
    print(f"输入边数: {E}")
    print(f"输出logits: {logits.shape}")
    print(f"logits范围: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"是否有NaN: {torch.isnan(logits).any()}")

    assert logits.shape == (N,)
    assert not torch.isnan(logits).any()

    print("✓ 完整模型测试通过!")


if __name__ == '__main__':
    # 运行测试
    test_gvp_layer()
    test_full_model()

    print("\n" + "=" * 80)
    print("🎉 所有测试通过!")
    print("=" * 80)
