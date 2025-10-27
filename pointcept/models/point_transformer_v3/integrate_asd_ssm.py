"""
集成ASD-SSM到现有PointSS架构的辅助脚本

使用方法：
1. 替换SerializedAttention中的Mamba模块
2. 在配置文件中启用ASD-SSM
3. 训练时收集和可视化尺度信息

Author: Xinyuan Zhang
"""

import torch
import torch.nn as nn
from .asd_ssm import (
    AdaptiveScaleDecoupledMamba,
    ScaleSequentialMambaWithASDSSM,
    MSFFSBlockWithASDSSM
)


def replace_mamba_with_asd_ssm(model, num_scales=3):
    """
    将模型中的普通Mamba替换为ASD-SSM

    Args:
        model: 原始PointTransformerV3模型
        num_scales: 尺度数量

    Returns:
        修改后的模型
    """
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        if hasattr(module, 'mamba0') and hasattr(module, 'mamba1'):
            # 这是SerializedAttention模块
            channels = module.channels

            # 创建ASD-SSM模块
            asd_mamba0 = AdaptiveScaleDecoupledMamba(
                d_model=channels,
                num_scales=num_scales,
                layer_idx=0,
                use_global_feature=True
            )

            asd_mamba1 = AdaptiveScaleDecoupledMamba(
                d_model=channels,
                num_scales=num_scales,
                layer_idx=1,
                use_global_feature=True
            )

            # 替换原有的Mamba
            module.mamba0 = asd_mamba0
            module.mamba1 = asd_mamba1

            print(f"Replaced Mamba in {name} with ASD-SSM")

    return model


class ASDSSMWrapper(nn.Module):
    """
    包装器：将ASD-SSM集成到现有的SerializedAttention中

    🆕 支持Patch间状态传递：
    - 所有尺度均启用状态传递，直接将前一Patch的最终状态作为当前Patch的初始状态
    - 符合标准SSM公式：h_0^(s,m) = h_P^(s,m-1)
    - Patch内部点云打乱，在利用序列化顺序的同时尊重点云的无序性

    使用方法：
    在point_transformer_v3m1_base.py的SerializedAttention.__init__中添加：
    ```python
    from .integrate_asd_ssm import ASDSSMWrapper
    self.asd_ssm_wrapper = ASDSSMWrapper(channels, num_scales=3)
    ```

    在forward中替换原来的mamba调用：
    ```python
    # 原来：
    # x, x_res = self.mamba0(x, x_res)

    # 改为：
    # x, x_res, scale_info = self.asd_ssm_wrapper(x, scale_id=0, x_res=x_res)
    ```
    """

    def __init__(self, channels, num_scales=3, enable_state_passing_scales=None):
        """
        Args:
            channels: 特征通道数
            num_scales: 尺度数量
            enable_state_passing_scales: 哪些尺度启用状态传递（默认所有尺度启用）
        """
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales
        # 🆕 默认所有尺度都启用状态传递
        if enable_state_passing_scales is None:
            enable_state_passing_scales = list(range(num_scales))
        self.enable_state_passing_scales = enable_state_passing_scales

        # 创建多个ASD-Mamba，对应不同的尺度，所有尺度都启用状态传递
        self.asd_mambas = nn.ModuleList([
            AdaptiveScaleDecoupledMamba(
                d_model=channels,
                num_scales=num_scales,
                layer_idx=i,
                use_global_feature=True,
                enable_state_passing=(i in enable_state_passing_scales)  # 默认全部启用
            )
            for i in range(num_scales)
        ])

        # 尺度信息收集器
        self.scale_info_history = []

    def forward(self, x, scale_id=0, x_res=None):
        """
        使用ASD-SSM处理输入，支持patch间状态传递

        Args:
            x: (N_patches, L, C) 输入特征，N_patches是patch数量
            scale_id: 当前尺度ID
            x_res: 残差

        Returns:
            x: 输出特征
            x_res: 更新的残差
            scale_info: 尺度信息
        """
        # 选择对应尺度的Mamba
        scale_id = min(scale_id, self.num_scales - 1)
        asd_mamba = self.asd_mambas[scale_id]

        N_patches, L, C = x.shape

        # 🆕 对于启用状态传递的尺度，逐个patch处理并传递状态
        if scale_id in self.enable_state_passing_scales and N_patches > 1:
            outputs = []
            h_state = None  # 初始隐状态为None

            for i in range(N_patches):
                # 处理当前patch
                x_patch = x[i:i+1]  # (1, L, C)

                # 前向传播，传入前一个patch的隐状态
                out_patch, x_res, h_state, scale_info = asd_mamba(
                    x_patch, scale_id, x_res, h_init=h_state
                )
                outputs.append(out_patch)

                # h_state会自动传递到下一个patch

            # 合并所有patch的输出
            x = torch.cat(outputs, dim=0)  # (N_patches, L, C)

        else:
            # 不启用状态传递的尺度，或者只有一个patch，直接处理
            x, x_res, h_final, scale_info = asd_mamba(x, scale_id, x_res, h_init=None)

        # 收集信息（用于可视化和分析）
        if self.training:
            self.scale_info_history.append(scale_info)

        return x, x_res, scale_info

    def get_scale_statistics(self):
        """
        获取尺度统计信息（用于分析和可视化）
        """
        if len(self.scale_info_history) == 0:
            return None

        stats = {}
        for key in self.scale_info_history[0].keys():
            values = [info[key] for info in self.scale_info_history]
            stats[key] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'history': values
            }

        return stats

    def reset_statistics(self):
        """重置统计信息"""
        self.scale_info_history = []


class ScaleInfoLogger:
    """
    尺度信息记录器

    用于训练过程中记录和可视化ASD-SSM的尺度特性
    """

    def __init__(self, log_dir='./logs/asd_ssm'):
        self.log_dir = log_dir
        self.scale_info_buffer = []

        import os
        os.makedirs(log_dir, exist_ok=True)

    def log_scale_info(self, scale_info_list, epoch, batch_idx):
        """
        记录尺度信息

        Args:
            scale_info_list: List[dict], 每个尺度的信息
            epoch: 当前epoch
            batch_idx: 当前batch索引
        """
        import json

        log_entry = {
            'epoch': epoch,
            'batch': batch_idx,
            'scales': scale_info_list
        }

        self.scale_info_buffer.append(log_entry)

        # 每100个batch保存一次
        if len(self.scale_info_buffer) >= 100:
            self.save_buffer()

    def save_buffer(self):
        """保存缓冲区到文件"""
        import json
        import time

        if len(self.scale_info_buffer) == 0:
            return

        filename = f"{self.log_dir}/scale_info_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(self.scale_info_buffer, f, indent=2)

        print(f"Saved {len(self.scale_info_buffer)} scale info entries to {filename}")
        self.scale_info_buffer = []

    def visualize_scale_characteristics(self, save_path=None):
        """
        可视化尺度特性

        生成图表展示：
        1. 不同尺度的状态衰减速度（scale_constraint）
        2. 参数偏移量的分布（delta_A_mean, delta_A_std）
        3. 时间步长的变化（delta_t_mean）
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib not available, skip visualization")
            return

        if len(self.scale_info_buffer) == 0:
            print("No scale info to visualize")
            return

        # 收集数据
        num_scales = len(self.scale_info_buffer[0]['scales'])
        scale_constraints = []
        delta_a_means = [[] for _ in range(num_scales)]
        delta_t_means = [[] for _ in range(num_scales)]

        for entry in self.scale_info_buffer:
            for i, scale_info in enumerate(entry['scales']):
                if i == 0:
                    scale_constraints.append(scale_info['scale_constraint'])
                delta_a_means[i].append(scale_info['delta_A_mean'])
                delta_t_means[i].append(scale_info['delta_t_mean'])

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 尺度约束（状态衰减速度）
        axes[0, 0].bar(range(num_scales), scale_constraints[:num_scales])
        axes[0, 0].set_xlabel('Scale ID')
        axes[0, 0].set_ylabel('Scale Constraint (State Decay Speed)')
        axes[0, 0].set_title('State Decay Speed Across Scales')
        axes[0, 0].set_xticks(range(num_scales))
        axes[0, 0].set_xticklabels([f'Scale {i}\n({"Coarse" if i==0 else "Medium" if i==num_scales//2 else "Fine"})'
                                     for i in range(num_scales)])

        # 2. 参数偏移量随训练的变化
        for i in range(num_scales):
            axes[0, 1].plot(delta_a_means[i], label=f'Scale {i}', alpha=0.7)
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Delta A Mean')
        axes[0, 1].set_title('Parameter Offset Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 时间步长的分布
        for i in range(num_scales):
            axes[1, 0].plot(delta_t_means[i], label=f'Scale {i}', alpha=0.7)
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Delta t Mean')
        axes[1, 0].set_title('Time Step Size Evolution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 参数偏移量的箱线图
        axes[1, 1].boxplot([delta_a_means[i] for i in range(num_scales)],
                           labels=[f'Scale {i}' for i in range(num_scales)])
        axes[1, 1].set_ylabel('Delta A Mean Distribution')
        axes[1, 1].set_title('Parameter Offset Distribution Across Scales')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.savefig(f"{self.log_dir}/scale_characteristics.png", dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {self.log_dir}/scale_characteristics.png")

        plt.close()


# 使用示例
"""
# 在训练脚本中：

from pointcept.models.point_transformer_v3.integrate_asd_ssm import (
    replace_mamba_with_asd_ssm,
    ScaleInfoLogger
)

# 1. 替换模型中的Mamba
model = replace_mamba_with_asd_ssm(model, num_scales=3)

# 2. 创建日志记录器
scale_logger = ScaleInfoLogger(log_dir='./logs/asd_ssm')

# 3. 在训练循环中记录信息
for epoch in range(num_epochs):
    for batch_idx, data in enumerate(train_loader):
        output = model(data)

        # 如果模型返回了scale_info
        if hasattr(output, 'scale_info_list'):
            scale_logger.log_scale_info(
                output.scale_info_list,
                epoch,
                batch_idx
            )

        # ... 正常的训练步骤

    # 每个epoch结束后可视化
    scale_logger.visualize_scale_characteristics(
        save_path=f'./logs/asd_ssm/epoch_{epoch}_scales.png'
    )
"""
