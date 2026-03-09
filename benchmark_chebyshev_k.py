"""
Chebyshev多项式阶数K消融实验 - 推理时间与参数量测试

用法：
    python benchmark_chebyshev_k.py

测试内容：
    对K=1,2,3,4,5分别测量：
    1. ChebyshevSpectralSSM模块的参数量
    2. 单次前向推理时间（ms）

注意：
    - 使用CUDA Event计时，比time.time()更精确
    - 包含warmup阶段，排除首次运行的编译开销
    - 模拟S3DIS典型场景的点数规模
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
    """统计可训练参数量"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def benchmark_single_k(
    K: int,
    d_model: int = 256,
    num_points: int = 80000,
    k_neighbors: int = 16,
    num_warmup: int = 10,
    num_runs: int = 50,
):
    """
    测试单个K值的推理时间和参数量

    Args:
        K: Chebyshev多项式阶数
        d_model: 特征维度（与模型配置一致）
        num_points: 点数（模拟S3DIS单场景）
        k_neighbors: 邻居数
        num_warmup: warmup次数
        num_runs: 正式测试次数
    """
    device = torch.device("cuda")

    # 构建模块
    model = ChebyshevSpectralSSM(
        d_model=d_model,
        cheb_K=K,
        window_size=128,
        k_neighbors=k_neighbors,
        d_state=16,
        dropout=0.0,
    ).to(device)
    model.eval()

    params = count_parameters(model)

    # 模拟输入数据（单batch）
    coords = torch.randn(num_points, 3, device=device)
    features = torch.randn(num_points, d_model, device=device)
    offset = torch.tensor([num_points], dtype=torch.long, device=device)
    spatial_order = torch.randperm(num_points, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(features, coords, offset, spatial_order)
    torch.cuda.synchronize()

    # 正式测试
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = model(features, coords, offset, spatial_order)
            end_event.record()

            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))  # ms

    timings = np.array(timings)
    mean_time = timings.mean()
    std_time = timings.std()

    # 清理显存
    del model, coords, features, offset, spatial_order
    torch.cuda.empty_cache()

    return params, mean_time, std_time


def main():
    K_values = [1, 2, 3, 4, 5]
    d_model = 256  # 与enc_channels第4层一致
    num_points = 80000  # S3DIS单场景的典型点数

    print("=" * 65)
    print("Chebyshev多项式阶数K 消融实验 - 推理时间基准测试")
    print(f"配置: d_model={d_model}, num_points={num_points}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    results = []
    # K=1时的参数量作为基准
    base_params = None

    for K in K_values:
        print(f"\n测试 K={K} ...")
        params, mean_time, std_time = benchmark_single_k(
            K=K, d_model=d_model, num_points=num_points
        )
        if base_params is None:
            base_params = params
        param_ratio = params / base_params
        results.append((K, param_ratio, mean_time, std_time, params))
        print(f"  参数量: {params:,} ({param_ratio:.2f}x)")
        print(f"  推理时间: {mean_time:.1f} +/- {std_time:.1f} ms")

    # 汇总表格
    print("\n" + "=" * 65)
    print("汇总结果（可直接填入论文表格）")
    print("=" * 65)
    print(f"{'K':<6} {'参数量':>10} {'推理时间 (ms)':>16} {'mIoU (%)':>10}")
    print("-" * 45)
    for K, ratio, mean_t, std_t, _ in results:
        miou = "TODO" if K != 3 else "70.8"
        print(f"{K:<6} {ratio:.2f}x{' ':>5} {mean_t:.1f}{' ':>10} {miou:>10}")

    # LaTeX格式输出
    print("\n" + "=" * 65)
    print("LaTeX表格行（复制粘贴用）")
    print("=" * 65)
    for K, ratio, mean_t, std_t, _ in results:
        miou = r"\textcolor{red}{TODO}" if K != 3 else "70.8"
        bold = K == 3
        if bold:
            print(
                rf"\textbf{{{K}}} & \textbf{{{ratio:.2f}$\times$}} & \textbf{{{mean_t:.1f}}} & \textbf{{{miou}}} \\"
            )
        else:
            print(rf"{K} & {ratio:.2f}$\times$ & {mean_t:.1f} & {miou} \\")


if __name__ == "__main__":
    main()
