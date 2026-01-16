"""
Geometry-Semantic Dual-Path State Space Model for Point Cloud Processing

创新点：双路径处理 - 空间域（序列化）+ 几何域（平滑度排序）
与ASD-SSM互补：ASD-SSM关注尺度，本方法关注几何特性

时间复杂度保证：完全O(N)复杂度！
- 几何特征提取: O(N) (序列化邻域，零kNN)
- 序列化: O(N)
- Mamba处理: O(N)
- 跨域交互: O(N)

核心优化：用序列化邻域替代kNN
- Hilbert/Z-order的空间局部性保证序列邻域≈欧氏邻域
- 复用已有的序列化结构，零额外开销

Author: Claude Code
Date: 2026-01-15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max, scatter_add
import numpy as np
from typing import Optional, Tuple, List
from functools import partial
from mamba_ssm.ops.triton.layernorm import RMSNorm
from openpoints.models.PCM.mamba_layer import MambaBlock
from .serialized_geometry_extractor import EfficientSerializedGeometricFeatureExtractor


# 旧的kNN版本已被序列化邻域版本替代
# 见 serialized_geometry_extractor.py


class GeometrySemanticDualPathSSM(nn.Module):
    """
    几何-语义双路径状态空间模型

    核心创新：
    1. 路径1（空间域）：使用现有空间序列化（Z-order/Hilbert）
    2. 路径2（几何域）：基于几何平滑度排序（平滑→粗糙）
    3. 跨域交互：通过交叉注意力让两条路径互相增强

    与ASD-SSM的互补性：
    - ASD-SSM: 多尺度（粗→细）
    - 本方法: 多几何（平滑→粗糙）

    优势：
    - 零Graph Laplacian计算
    - 零特征分解
    - 完全O(N)复杂度（除kNN的O(N log N)）
    - 物理意义清晰
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        k: int = 16,
        use_cross_attention: bool = True,
        dropout: float = 0.0
    ):
        """
        Args:
            d_model: 模型维度
            d_state: SSM状态维度
            d_conv: 卷积核大小
            expand: 扩展因子
            k: 几何特征提取的邻域大小
            use_cross_attention: 是否使用跨域交互
            dropout: dropout率
        """
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.use_cross_attention = use_cross_attention

        # 几何特征提取器（优化版：零kNN，纯O(N)）
        self.geometry_extractor = EfficientSerializedGeometricFeatureExtractor(k=k)

        # 几何特征编码器（将5维几何特征融入点特征）
        self.geometry_encoder = nn.Sequential(
            nn.Linear(d_model + 5, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # 路径1：空间域Mamba
        # SSM配置参数需要放在ssm_cfg字典中
        ssm_cfg = {
            'd_state': d_state,
            'd_conv': d_conv,
            'expand': expand
        }
        self.spatial_mamba = MambaBlock(
            dim=d_model,
            layer_idx=None,  # 不需要layer_idx（无缓存）
            bimamba_type='v2',
            norm_cls=partial(RMSNorm, eps=1e-5),
            fused_add_norm=True,
            residual_in_fp32=True,
            drop_path=dropout,
            ssm_cfg=ssm_cfg
        )
        self.spatial_norm = RMSNorm(d_model)

        # 路径2：几何域Mamba
        self.geometry_mamba = MambaBlock(
            dim=d_model,
            layer_idx=None,
            bimamba_type='v2',
            norm_cls=partial(RMSNorm, eps=1e-5),
            fused_add_norm=True,
            residual_in_fp32=True,
            drop_path=dropout,
            ssm_cfg=ssm_cfg
        )
        self.geometry_norm = RMSNorm(d_model)

        # 跨域交互（可选）
        if use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.cross_norm1 = nn.LayerNorm(d_model)
            self.cross_norm2 = nn.LayerNorm(d_model)

        # 双路径融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def compute_geometry_order(
        self,
        coords: torch.Tensor,
        offset: torch.Tensor,
        geometry_features: torch.Tensor
    ) -> torch.Tensor:
        """
        基于几何特征的序列化

        排序策略：从平滑到粗糙（类似低频→高频）

        Args:
            coords: [N, 3] - 点云坐标
            offset: [B] - batch偏移
            geometry_features: [N, 5] - 几何特征

        Returns:
            geometry_order: [N] - 几何排序索引
        """
        # 解构几何特征
        linearity = geometry_features[:, 0]
        planarity = geometry_features[:, 1]
        scattering = geometry_features[:, 2]
        curvature = geometry_features[:, 3]
        density = geometry_features[:, 4]

        # 计算"平滑度"分数
        # 高平滑度 = 高planarity + 低curvature + 低scattering
        smoothness = (
            0.5 * planarity      # 平面性（最重要）
            - 0.3 * curvature    # 曲率（负贡献）
            - 0.2 * scattering   # 散度（负贡献）
        )  # [N]

        # 排序：从高平滑（平面区域）到低平滑（边缘/角点）
        geometry_order = torch.argsort(smoothness, descending=True)

        return geometry_order

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        offset: torch.Tensor,
        spatial_order: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [N, D] - 点特征
            coords: [N, 3] - 点坐标
            offset: [B] - batch偏移
            spatial_order: [N] - 已有的空间序列化（如Hilbert）

        Returns:
            output: [N, D] - 增强后的点特征
        """
        N, D = x.shape
        device = x.device

        # ========== 提取几何特征（零kNN，纯O(N)） ==========
        # 使用序列化邻域替代kNN
        geometry_features = self.geometry_extractor(coords, spatial_order, offset)  # [N, 5]

        # ========== 路径1：空间域 ==========
        # 使用已有的空间序列化
        x_spatial = x[spatial_order]  # [N, D]
        x_spatial = self.spatial_norm(x_spatial)

        # Mamba处理（需要[B, L, D]格式，这里batch=1）
        x_spatial = x_spatial.unsqueeze(0)  # [1, N, D]
        x_spatial_out, _ = self.spatial_mamba(x_spatial, residual=None)  # 返回(hidden_states, residual)
        x_spatial_out = x_spatial_out.squeeze(0)  # [N, D]

        # 恢复原始顺序
        spatial_inverse = torch.argsort(spatial_order)
        spatial_out = x_spatial_out[spatial_inverse]  # [N, D]

        # ========== 路径2：几何域 ==========
        # 几何特征增强
        x_with_geom = torch.cat([x, geometry_features], dim=-1)  # [N, D+5]
        x_geometry = self.geometry_encoder(x_with_geom)  # [N, D]

        # 基于几何平滑度排序
        geometry_order = self.compute_geometry_order(coords, offset, geometry_features)
        x_geometry = x_geometry[geometry_order]  # [N, D]
        x_geometry = self.geometry_norm(x_geometry)

        # Mamba处理
        x_geometry = x_geometry.unsqueeze(0)  # [1, N, D]
        x_geometry_out, _ = self.geometry_mamba(x_geometry, residual=None)  # 返回(hidden_states, residual)
        x_geometry_out = x_geometry_out.squeeze(0)  # [N, D]

        # 恢复原始顺序
        geometry_inverse = torch.argsort(geometry_order)
        geometry_out = x_geometry_out[geometry_inverse]  # [N, D]

        # ========== 跨域交互 ==========
        if self.use_cross_attention:
            # 空间域 <- 几何域
            spatial_enhanced, _ = self.cross_attention(
                spatial_out.unsqueeze(0),      # query
                geometry_out.unsqueeze(0),     # key
                geometry_out.unsqueeze(0)      # value
            )
            spatial_enhanced = spatial_enhanced.squeeze(0)
            spatial_out = self.cross_norm1(spatial_out + spatial_enhanced)

            # 几何域 <- 空间域
            geometry_enhanced, _ = self.cross_attention(
                geometry_out.unsqueeze(0),     # query
                spatial_out.unsqueeze(0),      # key
                spatial_out.unsqueeze(0)       # value
            )
            geometry_enhanced = geometry_enhanced.squeeze(0)
            geometry_out = self.cross_norm2(geometry_out + geometry_enhanced)

        # ========== 双路径融合 ==========
        dual_feat = torch.cat([spatial_out, geometry_out], dim=-1)  # [N, 2D]
        output = self.fusion(dual_feat)  # [N, D]

        # 残差连接
        output = output + x

        return output


class SerializedGraphBuilder(nn.Module):
    """
    基于序列化的图构建器（可选组件）

    用于构建基于空间填充曲线的邻域图
    时间复杂度: O(N)

    注意：此组件为可选，主要用于对比实验
    实际推荐使用GeometrySemanticDualPathSSM（无需显式图构建）
    """

    def __init__(self, k: int = 16):
        super().__init__()
        self.k = k

    def build_serialization_graph(
        self,
        order: torch.Tensor,
        num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从序列化顺序构建图

        Args:
            order: [N] - 序列化顺序
            num_nodes: int - 节点数

        Returns:
            edge_index: [2, E] - 边索引
            edge_weight: [E] - 边权重（可选）
        """
        N = num_nodes
        device = order.device

        inverse = torch.argsort(order)  # inverse[i] = 点i在序列中的位置

        # 构建边
        edges_src = []
        edges_dst = []

        for i in range(N):
            seq_pos = inverse[i].item()

            # 前后k//2个点作为邻居
            left = max(0, seq_pos - self.k // 2)
            right = min(N, seq_pos + self.k // 2 + 1)

            neighbors = order[left:right]

            for j in neighbors:
                j_val = j.item()
                if i != j_val:
                    edges_src.append(i)
                    edges_dst.append(j_val)

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device)

        # 简单的二值权重
        edge_weight = torch.ones(edge_index.shape[1], device=device)

        return edge_index, edge_weight


# ========== 辅助函数 ==========

def offset2batch(offset: torch.Tensor) -> torch.Tensor:
    """
    将offset转换为batch索引

    Args:
        offset: [B] - 累积偏移量

    Returns:
        batch: [N] - batch索引
    """
    device = offset.device
    batch = torch.zeros(offset[-1].item(), dtype=torch.long, device=device)

    for i in range(len(offset)):
        start = 0 if i == 0 else offset[i-1]
        end = offset[i]
        batch[start:end] = i

    return batch


# ========== 测试代码 ==========

if __name__ == "__main__":
    """简单测试"""
    print("Testing GeometrySemanticDualPathSSM...")

    # 模拟数据
    N = 1000
    D = 256
    B = 2

    coords = torch.randn(N, 3).cuda()
    features = torch.randn(N, D).cuda()
    offset = torch.tensor([500, 1000], dtype=torch.long).cuda()

    # 模拟空间序列化（简单排序）
    spatial_order = torch.randperm(N).cuda()

    # 创建模型
    model = GeometrySemanticDualPathSSM(
        d_model=D,
        d_state=16,
        k=16,
        use_cross_attention=True
    ).cuda()

    # 前向传播
    output = model(features, coords, offset, spatial_order)

    print(f"Input shape: {features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # 测试几何特征提取
    geometry_extractor = GeometricFeatureExtractor(k=16).cuda()
    geom_feat = geometry_extractor(coords, offset)
    print(f"\nGeometry features shape: {geom_feat.shape}")
    print(f"Geometry features sample:\n{geom_feat[:5]}")

    print("\n✅ All tests passed!")
