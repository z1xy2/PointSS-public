"""
Mamba参数共享策略消融实验 - 参数量与推理时间测试

测试三种策略（K=3固定）：
1. 所有分量共享Mamba（1个MambaBlock服务3个频段）
2. 低阶/高阶各自共享（2组：T0+T1共享一个，T2单独一个）
3. 每个分量独立Mamba（3个独立MambaBlock，当前设计）

用法：
    python benchmark_mamba_sharing.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pointcept.models.point_transformer_v3.chebyshev_spectral_ssm import (
    ChebyshevSpectralSSM,
    ChebConv,
    SerializedWindowGraphBuilder,
)
from mamba_ssm.ops.triton.layernorm import RMSNorm
from openpoints.models.PCM.mamba_layer import MambaBlock


class ChebyshevSSM_SharedMamba(nn.Module):
    """所有分量共享同一个Mamba"""

    def __init__(self, d_model, cheb_K=3, k_neighbors=16, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.cheb_K = cheb_K

        self.graph_builder = SerializedWindowGraphBuilder(window_size=128, k_neighbors=k_neighbors)
        self.cheb_conv = ChebConv(d_model, d_model, K=cheb_K)

        ssm_cfg = {'d_state': d_state, 'd_conv': 4, 'expand': 2}
        # 只有1个Mamba，所有分量共享
        self.shared_mamba = MambaBlock(
            dim=d_model, layer_idx=None, bimamba_type='v2',
            norm_cls=partial(RMSNorm, eps=1e-5), fused_add_norm=True,
            residual_in_fp32=True, drop_path=0, ssm_cfg=ssm_cfg
        )
        self.freq_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(cheb_K)])
        self.freq_fusion_weights = nn.Parameter(torch.ones(cheb_K) / cheb_K)
        self.norm_input = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )

    def forward(self, x, coords, offset, spatial_order):
        N, D = x.shape
        edge_index, edge_weight = self.graph_builder.build_window_graph(coords, spatial_order, offset)
        if edge_index.shape[1] == 0:
            return x
        x_norm = self.norm_input(x)
        Tx_list = self.cheb_conv.chebyshev_basis(x_norm, edge_index, edge_weight, lambda_max=2.0, K=self.cheb_K)
        inverse_order = torch.argsort(spatial_order)
        freq_outputs = []
        for k in range(self.cheb_K):
            x_freq = Tx_list[k][spatial_order].unsqueeze(0)
            out, _ = self.shared_mamba(x_freq, residual=None)  # 共享同一个Mamba
            out = self.freq_norms[k](out.squeeze(0)[inverse_order])
            freq_outputs.append(out)
        freq_stack = torch.stack(freq_outputs, dim=0)
        weights = torch.softmax(self.freq_fusion_weights, dim=0)
        x_multi = torch.einsum('k,knd->nd', weights, freq_stack)
        output = self.fusion(torch.cat([x, x_multi], dim=-1)) + x
        return output


class ChebyshevSSM_GroupedMamba(nn.Module):
    """低阶(T0,T1)共享一个Mamba，高阶(T2)单独一个"""

    def __init__(self, d_model, cheb_K=3, k_neighbors=16, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.cheb_K = cheb_K

        self.graph_builder = SerializedWindowGraphBuilder(window_size=128, k_neighbors=k_neighbors)
        self.cheb_conv = ChebConv(d_model, d_model, K=cheb_K)

        ssm_cfg = {'d_state': d_state, 'd_conv': 4, 'expand': 2}
        # 2组Mamba：低阶组 + 高阶组
        self.low_order_mamba = MambaBlock(
            dim=d_model, layer_idx=None, bimamba_type='v2',
            norm_cls=partial(RMSNorm, eps=1e-5), fused_add_norm=True,
            residual_in_fp32=True, drop_path=0, ssm_cfg=ssm_cfg
        )
        self.high_order_mamba = MambaBlock(
            dim=d_model, layer_idx=None, bimamba_type='v2',
            norm_cls=partial(RMSNorm, eps=1e-5), fused_add_norm=True,
            residual_in_fp32=True, drop_path=0, ssm_cfg=ssm_cfg
        )
        self.freq_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(cheb_K)])
        self.freq_fusion_weights = nn.Parameter(torch.ones(cheb_K) / cheb_K)
        self.norm_input = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )

    def forward(self, x, coords, offset, spatial_order):
        N, D = x.shape
        edge_index, edge_weight = self.graph_builder.build_window_graph(coords, spatial_order, offset)
        if edge_index.shape[1] == 0:
            return x
        x_norm = self.norm_input(x)
        Tx_list = self.cheb_conv.chebyshev_basis(x_norm, edge_index, edge_weight, lambda_max=2.0, K=self.cheb_K)
        inverse_order = torch.argsort(spatial_order)
        freq_outputs = []
        for k in range(self.cheb_K):
            x_freq = Tx_list[k][spatial_order].unsqueeze(0)
            mamba = self.low_order_mamba if k < 2 else self.high_order_mamba
            out, _ = mamba(x_freq, residual=None)
            out = self.freq_norms[k](out.squeeze(0)[inverse_order])
            freq_outputs.append(out)
        freq_stack = torch.stack(freq_outputs, dim=0)
        weights = torch.softmax(self.freq_fusion_weights, dim=0)
        x_multi = torch.einsum('k,knd->nd', weights, freq_stack)
        output = self.fusion(torch.cat([x, x_multi], dim=-1)) + x
        return output


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 与 benchmark_chebyshev_k.py 一致的层级配置
LEVEL_CONFIGS = [
    (32,  2, 80000),
    (64,  2, 40000),
    (128, 2, 20000),
]

NUM_WARMUP = 10
NUM_RUNS = 50


def benchmark_strategy(strategy_name, model_cls, K=3):
    """测量一种共享策略在所有level上的总参数和总耗时"""
    device = torch.device("cuda")
    total_params = 0
    total_time = 0.0

    for channels, num_blocks, num_points in LEVEL_CONFIGS:
        model = model_cls(d_model=channels, cheb_K=K, k_neighbors=16).to(device)
        model.eval()

        level_params = count_params(model)
        total_params += level_params * num_blocks

        coords = torch.randn(num_points, 3, device=device)
        features = torch.randn(num_points, channels, device=device)
        offset = torch.tensor([num_points], dtype=torch.long, device=device)
        spatial_order = torch.randperm(num_points, device=device)

        with torch.no_grad():
            for _ in range(NUM_WARMUP):
                _ = model(features, coords, offset, spatial_order)
        torch.cuda.synchronize()

        timings = []
        with torch.no_grad():
            for _ in range(NUM_RUNS):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(features, coords, offset, spatial_order)
                end.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end))

        mean_time = np.mean(timings)
        total_time += mean_time * num_blocks

        print(f"    Level (ch={channels}, pts={num_points}): "
              f"{mean_time:.1f} ms/block x {num_blocks} = {mean_time * num_blocks:.1f} ms, "
              f"params={level_params:,}")

        del model, coords, features, offset, spatial_order
        torch.cuda.empty_cache()

    return total_params, total_time


def main():
    strategies = [
        ("所有分量共享Mamba",            ChebyshevSSM_SharedMamba),
        ("低阶/高阶各自共享（2组）",      ChebyshevSSM_GroupedMamba),
        ("每个分量独立Mamba",            ChebyshevSpectralSSM),
    ]

    print("=" * 65)
    print("Mamba参数共享策略消融实验 (K=3)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    results = []
    base_params = None

    for name, cls in strategies:
        print(f"\n--- {name} ---")
        if cls == ChebyshevSpectralSSM:
            # ChebyshevSpectralSSM 的构造函数签名不同，需要适配
            class Wrapper:
                @staticmethod
                def __call__(**kwargs):
                    return ChebyshevSpectralSSM(
                        d_model=kwargs['d_model'],
                        cheb_K=kwargs['cheb_K'],
                        k_neighbors=kwargs['k_neighbors'],
                    )
            # 用lambda包装
            def make_model(d_model, cheb_K, k_neighbors):
                return ChebyshevSpectralSSM(d_model=d_model, cheb_K=cheb_K, k_neighbors=k_neighbors)
            params, time_ms = benchmark_strategy(name, make_model)
        else:
            params, time_ms = benchmark_strategy(name, cls)

        if base_params is None:
            base_params = params
        ratio = params / base_params
        results.append((name, ratio, time_ms, params))
        print(f"  总参数: {params:,} ({ratio:.2f}x)")
        print(f"  总耗时: {time_ms:.1f} ms")

    # 汇总
    print("\n" + "=" * 65)
    print("汇总")
    print("=" * 65)
    print(f"{'策略':<28} {'参数量':>10} {'耗时(ms)':>10} {'mIoU':>8}")
    print("-" * 60)
    for name, ratio, time_ms, _ in results:
        miou = "70.8" if "独立" in name else "??"
        print(f"{name:<28} {ratio:.2f}x{' ':>5} {time_ms:.1f}{' ':>5} {miou:>8}")


if __name__ == "__main__":
    main()
