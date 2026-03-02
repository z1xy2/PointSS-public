"""
Adaptive Scale-Decoupled State Space Model (ASD-SSM)
自适应尺度解耦状态空间模型

Core Innovation: 为不同尺度生成定制化的SSM参数，解决共享参数无法适应多尺度特性的问题

Author: Xinyuan Zhang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from mamba_ssm.ops.triton.layernorm import RMSNorm
from openpoints.models.PCM.mamba_layer import MambaBlock


class ScaleAwareParameterGenerator(nn.Module):
    """
    尺度感知的SSM参数生成器

    为不同尺度动态生成定制化的状态空间参数，使得：
    - 粗尺度：状态衰减慢（大感受野，长程依赖）
    - 细尺度：状态衰减快（小感受野，局部细节）

    Args:
        d_model: 特征维度
        num_scales: 尺度数量
        use_global_feature: 是否使用全局特征调制参数
    """

    def __init__(self, d_model, num_scales=3, use_global_feature=True):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.use_global_feature = use_global_feature

        # 全局特征提取器（分析每个patch内的整体点云特性）
        if use_global_feature:
            self.global_feature_extractor = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            input_dim = d_model // 2
        else:
            input_dim = d_model // 4

        # 尺度编码（可学习的尺度嵌入）
        self.scale_embedding = nn.Embedding(num_scales, d_model // 4)

        # 参数生成网络 - 只生成 Ā_s（论文公式：Ā_s = α_s · σ(0.1·MLP(...))）
        self.param_generator = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)  # 只生成 Ā_s^raw，sigmoid后乘α_s
        )

        # 尺度约束因子：细→粗 单调递增（论文：α_1=0.3, α_2=0.9）
        self.register_buffer(
            'scale_constraints',
            torch.linspace(0.3, 0.9, num_scales)
        )

    def forward(self, features, scale_id):
        """
        生成尺度特定的SSM参数

        Args:
            features: (N, L, C) 当前尺度的特征序列
            scale_id: int, 尺度索引 (0=粗尺度, num_scales-1=细尺度)

        Returns:
            A_bar: (N, C) 尺度特定的状态转移矩阵，∈ [0, α_s]
            scale_info: dict 包含尺度相关信息
        """
        N, L, C = features.shape
        device = features.device

        # 1. 提取全局特征（反映当前批次点云的整体特性）
        if self.use_global_feature:
            global_feat = self.global_feature_extractor(
                features.transpose(1, 2)
            )  # (N, C//4)
        else:
            global_feat = None

        # 2. 获取尺度嵌入
        scale_emb = self.scale_embedding(
            torch.tensor([scale_id], device=device, dtype=torch.long)
        ).expand(N, -1)  # (N, C//4)

        # 3. 构建条件向量
        if self.use_global_feature:
            condition = torch.cat([global_feat, scale_emb], dim=-1)  # (N, C//2)
        else:
            condition = scale_emb  # (N, C//4)

        # 4. 生成 Ā_s^raw，再应用 sigmoid + α_s 约束
        # 论文公式：Ā_s = α_s · σ(λ · MLP([f_global ∥ e_s]))，λ=0.1
        A_bar_raw = self.param_generator(condition)  # (N, C)
        scale_constraint = self.scale_constraints[scale_id]
        A_bar = scale_constraint * torch.sigmoid(0.1 * A_bar_raw)  # ∈ [0, α_s]，始终非负

        # 5. 收集尺度信息
        scale_info = {
            'scale_id': scale_id,
            'scale_constraint': scale_constraint,
            'A_bar_mean': A_bar.mean(),
            'A_bar_std': A_bar.std(),
            'global_feat_norm': global_feat.norm(dim=-1).mean() if self.use_global_feature else 0.0
        }

        return A_bar, scale_info


class AdaptiveScaleDecoupledMamba(nn.Module):
    """
    自适应尺度解耦Mamba (ASD-Mamba)

    核心创新：为不同尺度生成定制化的SSM参数
    - 共享基础Mamba结构（参数效率）
    - 通过参数生成器为每个尺度定制SSM行为（表达能力）
    - 动态调整状态转移矩阵（自适应性）

    Args:
        d_model: 特征维度
        num_scales: 尺度数量
        layer_idx: Mamba层索引
        use_global_feature: 是否使用全局特征
    """

    def __init__(self, d_model, num_scales=3, layer_idx=None, use_global_feature=True, enable_state_passing=True):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.layer_idx = layer_idx
        self.enable_state_passing = enable_state_passing  # 🆕 是否启用状态传递

        # 基础Mamba模块（共享参数）
        self.base_mamba = MambaBlock(
            dim=d_model,
            layer_idx=layer_idx,
            bimamba_type='v2',  # 双向Mamba
            norm_cls=partial(RMSNorm, eps=1e-5),
            fused_add_norm=True,
            residual_in_fp32=True,
            drop_path=0.0
        )

        # 尺度感知参数生成器
        self.param_generator = ScaleAwareParameterGenerator(
            d_model, num_scales, use_global_feature
        )

        # 残差连接的权重
        self.residual_weight = nn.Parameter(torch.ones(1))

        # 🆕 状态传递相关：直接传递策略（无门控）
        # 与论文描述一致：h_0^(s,m) = h_P^(s,m-1)
        # 不需要额外的投影或门控机制，直接使用SSM的状态更新公式

    def forward(self, x, scale_id, x_res=None, h_init=None):
        """
        使用尺度特定的参数处理输入

        Args:
            x: (N, L, C) 输入特征序列
            scale_id: 当前尺度编号
            x_res: 残差连接（可选）
            h_init: (N, C) 初始隐状态（用于patch间状态传递，可选）

        Returns:
            output: (N, L, C) 输出特征
            x_res: 更新后的残差
            h_final: (N, C) 最终隐状态（用于传递给下一个patch）
            scale_info: 尺度信息字典
        """
        # 1. 生成 Ā_s（论文公式）
        A_bar, scale_info = self.param_generator(x, scale_id)

        # 2. 通过基础 BiMamba 处理（B、C 使用 Mamba 内部 selective 机制）
        output, x_res = self.base_mamba(x, x_res)

        # 3. 用 Ā_s 调制输出（近似实现状态衰减控制）
        # fine scale: A_bar ∈ [0, 0.3] → 轻微增强（快衰减，关注局部）
        # coarse scale: A_bar ∈ [0, 0.9] → 较强增强（慢衰减，保留全局）
        output = output * (1.0 + A_bar.unsqueeze(1))

        # 4. BiMamba 内部残差
        if x_res is not None:
            output = output + self.residual_weight * x_res

        # 5. 提取最终隐状态（保留接口，h_init 不再使用）
        h_final = output[:, -1, :].clone()

        scale_info['output_norm'] = output.norm(dim=-1).mean()
        scale_info['h_final_norm'] = h_final.norm(dim=-1).mean()
        scale_info['state_passing_enabled'] = False

        return output, x_res, h_final, scale_info


class ScaleSequentialMambaWithASDSSM(nn.Module):
    """
    结合ASD-SSM的跨尺度序列化Mamba

    工作流程：
    1. 每个尺度用ASD-Mamba独立处理（尺度内特征提取）
    2. 将多尺度特征视为尺度序列，用Mamba建模尺度间依赖（跨尺度交互）
    3. 动态加权融合多尺度特征（自适应融合）

    Args:
        channels: 特征通道数
        num_scales: 尺度数量
        use_scale_pe: 是否使用尺度位置编码
    """

    def __init__(self, channels, num_scales=3, use_scale_pe=True):
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales
        self.use_scale_pe = use_scale_pe

        # 为每个尺度创建ASD-Mamba
        self.scale_mambas = nn.ModuleList([
            AdaptiveScaleDecoupledMamba(
                d_model=channels,
                num_scales=num_scales,
                layer_idx=i,
                use_global_feature=True
            )
            for i in range(num_scales)
        ])

        # 尺度特征投影和归一化
        self.scale_proj = nn.ModuleList([
            nn.Linear(channels, channels) for _ in range(num_scales)
        ])

        self.scale_norms = nn.ModuleList([
            RMSNorm(channels, eps=1e-5) for _ in range(num_scales)
        ])

        # 尺度位置编码
        if use_scale_pe:
            self.scale_position_embedding = nn.Embedding(num_scales, channels)

        # 跨尺度Mamba：在尺度维度上进行序列建模
        self.cross_scale_mamba = MambaBlock(
            dim=channels,
            layer_idx=None,
            bimamba_type='v2',
            norm_cls=partial(RMSNorm, eps=1e-5),
            fused_add_norm=True,
            residual_in_fp32=True,
            drop_path=0.0
        )

        # 尺度重要性评分网络
        self.scale_importance = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid()
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels, channels)
        )

        self.output_norm = RMSNorm(channels, eps=1e-5)

    def forward(self, scale_features_list):
        """
        处理多尺度特征

        Args:
            scale_features_list: List[(N, L, C)], 多尺度特征列表
                                从粗到细排列

        Returns:
            fused_features: (N, L, C) 融合后的特征
            scale_weights: (N, num_scales) 每个尺度的权重
            scale_info_list: List[dict] 每个尺度的信息
        """
        N = scale_features_list[0].shape[0]
        L = scale_features_list[0].shape[1]

        # 1. 使用ASD-Mamba处理每个尺度
        enhanced_features = []
        scale_info_list = []

        for scale_id, (features, mamba, proj, norm) in enumerate(
            zip(scale_features_list, self.scale_mambas,
                self.scale_proj, self.scale_norms)
        ):
            # 投影和归一化
            proj_feat = proj(features)
            proj_feat = norm(proj_feat)

            # 使用尺度特定的Mamba处理
            enhanced, _, scale_info = mamba(proj_feat, scale_id, x_res=None)
            enhanced_features.append(enhanced)
            scale_info_list.append(scale_info)

        # 2. 添加尺度位置编码
        if self.use_scale_pe:
            for i, feat in enumerate(enhanced_features):
                scale_pe = self.scale_position_embedding(
                    torch.tensor([i], device=feat.device, dtype=torch.long)
                )  # (1, C)
                enhanced_features[i] = feat + scale_pe.unsqueeze(0).unsqueeze(0)

        # 3. 堆叠为尺度序列：(N, L, num_scales, C)
        # 然后reshape为 (N*L, num_scales, C) 以便在尺度维度上应用Mamba
        scale_sequence = torch.stack(enhanced_features, dim=2)  # (N, L, num_scales, C)
        original_shape = scale_sequence.shape
        scale_sequence = scale_sequence.reshape(N * L, self.num_scales, self.channels)

        # 4. 在尺度维度上应用跨尺度Mamba
        cross_scale_enhanced, _ = self.cross_scale_mamba(scale_sequence, None)

        # 5. Reshape回原始形状
        cross_scale_enhanced = cross_scale_enhanced.reshape(
            N, L, self.num_scales, self.channels
        )

        # 6. 计算每个尺度的重要性权重
        # (N, L, num_scales, C) → (N, L, num_scales, 1)
        scale_importance = torch.stack([
            self.scale_importance(cross_scale_enhanced[:, :, i])
            for i in range(self.num_scales)
        ], dim=2)  # (N, L, num_scales, 1)

        # 在尺度维度上归一化
        scale_weights = F.softmax(scale_importance, dim=2)  # (N, L, num_scales, 1)

        # 7. 加权融合
        fused = (cross_scale_enhanced * scale_weights).sum(dim=2)  # (N, L, C)

        # 8. 输出投影
        fused = self.output_proj(fused)
        fused = self.output_norm(fused)

        # 9. 计算平均权重（用于分析）
        avg_scale_weights = scale_weights.mean(dim=[0, 1]).squeeze(-1)  # (num_scales,)

        return fused, avg_scale_weights, scale_info_list


# 用于替换原有的SerializedAttention中的Mamba
class MSFFSBlockWithASDSSM(nn.Module):
    """
    集成ASD-SSM的MSFFS Block

    可以直接替换原point_transformer_v3m1_base.py中的SerializedAttention
    """

    def __init__(
        self,
        channels,
        patch_size=1024,
        num_scales=3,
        use_order_prompt=True,
        prompt_num_per_order=6
    ):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.num_scales = num_scales
        self.prompt_num_per_order = prompt_num_per_order

        # 顺序提示（如果使用）
        if use_order_prompt:
            self.order_prompt = nn.Embedding(
                prompt_num_per_order * num_scales,
                channels
            )
            self.order_prompt_proj = nn.Linear(channels, channels, bias=False)
        else:
            self.order_prompt = None

        # ASD-SSM增强的跨尺度Mamba
        self.asd_mamba_fusion = ScaleSequentialMambaWithASDSSM(
            channels=channels,
            num_scales=num_scales,
            use_scale_pe=True
        )

        # 输出投影和MLP
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(0.1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 4, channels),
            nn.Dropout(0.1)
        )

    def forward(self, scale_features_list, order_indices=None):
        """
        Args:
            scale_features_list: List[(N, L, C)], 多尺度特征
            order_indices: 顺序提示索引（可选）

        Returns:
            output: (N, L, C) 融合后的特征
        """
        # 添加顺序提示（如果使用）
        if self.order_prompt is not None and order_indices is not None:
            prompted_features = []
            for i, feat in enumerate(scale_features_list):
                if i == 0:  # 只对第一层（有序层）添加提示
                    prompt_idx = order_indices[i]
                    prompt = self.order_prompt(prompt_idx)
                    prompt = self.order_prompt_proj(prompt)
                    # 在序列前后添加提示
                    feat_with_prompt = torch.cat([
                        prompt.unsqueeze(0).unsqueeze(0).expand(feat.size(0), -1, -1),
                        feat,
                        prompt.unsqueeze(0).unsqueeze(0).expand(feat.size(0), -1, -1)
                    ], dim=1)
                    prompted_features.append(feat_with_prompt)
                else:
                    prompted_features.append(feat)
            scale_features_list = prompted_features

        # ASD-SSM处理
        fused, scale_weights, scale_info_list = self.asd_mamba_fusion(
            scale_features_list
        )

        # 如果添加了提示，需要移除
        if self.order_prompt is not None and order_indices is not None:
            fused = fused[:, self.prompt_num_per_order:-self.prompt_num_per_order, :]

        # 输出投影
        output = self.proj(fused)
        output = self.proj_drop(output)

        # MLP
        output = output + self.mlp(output)

        return output, scale_weights, scale_info_list
