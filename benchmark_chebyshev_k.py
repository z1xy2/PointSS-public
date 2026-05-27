"""
Chebyshev多项式阶数K消融实验 - 推理时间与参数量测试

用法：
    python benchmark_chebyshev_k.py

按实际模型配置测量：
    - 编码器各level的通道数和点数不同
    - Chebyshev在前3个编码层启用（论文设定），共6个block
    - 参数量以"全模型占比"形式报告
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pointcept.models.point_transformer_v3.chebyshev_spectral_ssm import (
    ChebyshevSpectralSSM,
)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# 模型配置（与 semseg-pt-v3m1-0-test.py 一致）
# 论文设定：前3个编码层启用Chebyshev
LEVEL_CONFIGS = [
    # (channels, num_blocks, approx_points)
    # stride=(2,2,2,2), 起始约80000点
    (32,  2, 80000),   # enc level 0
    (64,  2, 40000),   # enc level 1
    (128, 2, 20000),   # enc level 2
]

K_NEIGHBORS = 16
NUM_WARMUP = 10
NUM_RUNS = 50


def benchmark_k(K: int):
    """测试单个K值：在所有启用Chebyshev的level上分别测量，然后求和"""
    device = torch.device("cuda")

    total_params = 0
    total_time = 0.0

    for channels, num_blocks, num_points in LEVEL_CONFIGS:
        # 每个level的Chebyshev模块（同一level内各block共享相同配置）
        model = ChebyshevSpectralSSM(
            d_model=channels,
            cheb_K=K,
            window_size=128,
            k_neighbors=K_NEIGHBORS,
            d_state=16,
            dropout=0.0,
        ).to(device)
        model.eval()

        level_params = count_parameters(model)
        total_params += level_params * num_blocks

        # 模拟输入
        coords = torch.randn(num_points, 3, device=device)
        features = torch.randn(num_points, channels, device=device)
        offset = torch.tensor([num_points], dtype=torch.long, device=device)
        spatial_order = torch.randperm(num_points, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(NUM_WARMUP):
                _ = model(features, coords, offset, spatial_order)
        torch.cuda.synchronize()

        # 测量单个block的推理时间
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
        # 该level有num_blocks个block，每个block调用一次Chebyshev
        total_time += mean_time * num_blocks

        print(f"    Level (ch={channels}, pts={num_points}): "
              f"{mean_time:.1f} ms/block x {num_blocks} blocks = {mean_time * num_blocks:.1f} ms")

        del model, coords, features, offset, spatial_order
        torch.cuda.empty_cache()

    return total_params, total_time


def main():
    K_values = [1, 2, 3, 4, 5]

    print("=" * 65)
    print("Chebyshev 阶数K 消融 - 按实际模型层级配置测量")
    print(f"启用Chebyshev的层级: {len(LEVEL_CONFIGS)} 层, "
          f"共 {sum(c[1] for c in LEVEL_CONFIGS)} 个block")
    print(f"各层: {[(ch, pts) for ch, _, pts in LEVEL_CONFIGS]}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    results = []
    base_params = None

    for K in K_values:
        print(f"\n--- K={K} ---")
        cheb_params, cheb_time = benchmark_k(K)

        if base_params is None:
            base_params = cheb_params

        param_ratio = cheb_params / base_params
        results.append((K, param_ratio, cheb_time, cheb_params))

        print(f"  Chebyshev总参数: {cheb_params:,} ({param_ratio:.2f}x)")
        print(f"  Chebyshev总耗时: {cheb_time:.1f} ms")

    # 汇总
    print("\n" + "=" * 65)
    print("汇总（Chebyshev模块的额外开销）")
    print("=" * 65)
    print(f"{'K':<4} {'参数量(相对K=1)':>16} {'Chebyshev耗时(ms)':>20} {'mIoU':>8}")
    print("-" * 52)
    for K, ratio, time_ms, _ in results:
        miou = "TODO" if K != 3 else "70.8"
        print(f"{K:<4} {ratio:.2f}x{' ':>12} {time_ms:.1f}{' ':>14} {miou:>8}")

    # LaTeX
    print("\n" + "=" * 65)
    print("LaTeX表格行")
    print("=" * 65)
    for K, ratio, time_ms, _ in results:
        miou = r"\textcolor{red}{TODO}" if K != 3 else "70.8"
        if K == 3:
            print(rf"\textbf{{{K}}} & \textbf{{{ratio:.2f}$\times$}} & \textbf{{{time_ms:.1f}}} & \textbf{{{miou}}} \\")
        else:
            print(rf"{K} & {ratio:.2f}$\times$ & {time_ms:.1f} & {miou} \\")


if __name__ == "__main__":
    main()
