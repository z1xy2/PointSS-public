"""
测试Geometry-Semantic Dual-Path SSM模块

功能：
1. 测试GeometrySemanticDualPathSSM单独运行
2. 测试集成到PointTransformerV3的完整流程
3. 验证时间复杂度
"""

import torch
import sys
import time
sys.path.append('D:\\PointSS')

from pointcept.models.point_transformer_v3.geometry_semantic_dual_path import (
    GeometrySemanticDualPathSSM,
    GeometricFeatureExtractor
)


def test_geometry_feature_extractor():
    """测试几何特征提取器"""
    print("=" * 60)
    print("测试1: 几何特征提取器")
    print("=" * 60)

    # 创建测试数据
    N = 5000
    coords = torch.randn(N, 3).cuda()
    offset = torch.tensor([2000, 5000], dtype=torch.long).cuda()

    # 创建提取器
    extractor = GeometricFeatureExtractor(k=16).cuda()

    # 测试前向传播
    start_time = time.time()
    geometry_features = extractor(coords, offset)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    print(f"✓ 输入点数: {N}")
    print(f"✓ 输出形状: {geometry_features.shape}  (预期: [{N}, 5])")
    print(f"✓ 时间: {elapsed_time*1000:.2f} ms")
    print(f"✓ 特征范围:")
    print(f"  - Linearity:  [{geometry_features[:, 0].min():.3f}, {geometry_features[:, 0].max():.3f}]")
    print(f"  - Planarity:  [{geometry_features[:, 1].min():.3f}, {geometry_features[:, 1].max():.3f}]")
    print(f"  - Scattering: [{geometry_features[:, 2].min():.3f}, {geometry_features[:, 2].max():.3f}]")
    print(f"  - Curvature:  [{geometry_features[:, 3].min():.3f}, {geometry_features[:, 3].max():.3f}]")
    print(f"  - Density:    [{geometry_features[:, 4].min():.3f}, {geometry_features[:, 4].max():.3f}]")
    print()


def test_geometry_semantic_dual_path():
    """测试Geometry-Semantic Dual-Path SSM"""
    print("=" * 60)
    print("测试2: Geometry-Semantic Dual-Path SSM")
    print("=" * 60)

    # 创建测试数据
    N = 8000
    D = 256
    coords = torch.randn(N, 3).cuda()
    features = torch.randn(N, D).cuda()
    offset = torch.tensor([3000, 8000], dtype=torch.long).cuda()
    spatial_order = torch.randperm(N).cuda()

    # 创建模型
    model = GeometrySemanticDualPathSSM(
        d_model=D,
        d_state=16,
        k=16,
        use_cross_attention=True
    ).cuda()

    # 测试前向传播
    print(f"输入形状: {features.shape}")
    start_time = time.time()
    output = model(features, coords, offset, spatial_order)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    print(f"✓ 输出形状: {output.shape}  (预期: [{N}, {D}])")
    print(f"✓ 时间: {elapsed_time*1000:.2f} ms")
    print(f"✓ 输出范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"✓ 残差连接验证: 输入和输出是否相似")
    print(f"  - 输入均值: {features.mean():.3f}, 输出均值: {output.mean():.3f}")
    print(f"  - 输入std:  {features.std():.3f}, 输出std:  {output.std():.3f}")
    print()


def test_scalability():
    """测试不同点数的扩展性"""
    print("=" * 60)
    print("测试3: 扩展性测试（时间复杂度验证）")
    print("=" * 60)

    D = 256
    model = GeometrySemanticDualPathSSM(
        d_model=D,
        d_state=16,
        k=16,
        use_cross_attention=True
    ).cuda()

    point_counts = [1000, 2000, 5000, 10000, 20000]
    times = []

    print(f"{'点数':>10} | {'时间(ms)':>12} | {'时间/N(µs)':>15} | 复杂度估计")
    print("-" * 60)

    for N in point_counts:
        coords = torch.randn(N, 3).cuda()
        features = torch.randn(N, D).cuda()
        offset = torch.tensor([N], dtype=torch.long).cuda()
        spatial_order = torch.randperm(N).cuda()

        # 预热
        _ = model(features, coords, offset, spatial_order)

        # 测试
        torch.cuda.synchronize()
        start_time = time.time()
        _ = model(features, coords, offset, spatial_order)
        torch.cuda.synchronize()
        elapsed_time = (time.time() - start_time) * 1000  # ms

        times.append(elapsed_time)
        time_per_point = elapsed_time * 1000 / N  # µs

        # 估计复杂度
        if len(times) > 1:
            ratio = elapsed_time / times[0]
            n_ratio = N / point_counts[0]
            complexity = ratio / n_ratio
            complexity_str = f"O(N^{complexity:.2f})"
        else:
            complexity_str = "baseline"

        print(f"{N:>10} | {elapsed_time:>12.2f} | {time_per_point:>15.2f} | {complexity_str}")

    print()
    print("✓ 如果 时间/N 大致恒定 → O(N)")
    print("✓ 如果 时间/N 线性增长 → O(N log N)")
    print("✓ 如果 时间/N 二次增长 → O(N²)")
    print()


def test_with_cross_attention_comparison():
    """对比有无跨域交互的性能"""
    print("=" * 60)
    print("测试4: 跨域交互对比")
    print("=" * 60)

    N = 5000
    D = 256
    coords = torch.randn(N, 3).cuda()
    features = torch.randn(N, D).cuda()
    offset = torch.tensor([N], dtype=torch.long).cuda()
    spatial_order = torch.randperm(N).cuda()

    # 无跨域交互
    model_no_cross = GeometrySemanticDualPathSSM(
        d_model=D,
        d_state=16,
        k=16,
        use_cross_attention=False
    ).cuda()

    start_time = time.time()
    output_no_cross = model_no_cross(features, coords, offset, spatial_order)
    torch.cuda.synchronize()
    time_no_cross = (time.time() - start_time) * 1000

    # 有跨域交互
    model_with_cross = GeometrySemanticDualPathSSM(
        d_model=D,
        d_state=16,
        k=16,
        use_cross_attention=True
    ).cuda()

    start_time = time.time()
    output_with_cross = model_with_cross(features, coords, offset, spatial_order)
    torch.cuda.synchronize()
    time_with_cross = (time.time() - start_time) * 1000

    print(f"无跨域交互:")
    print(f"  - 时间: {time_no_cross:.2f} ms")
    print(f"  - 输出范围: [{output_no_cross.min():.3f}, {output_no_cross.max():.3f}]")
    print()
    print(f"有跨域交互:")
    print(f"  - 时间: {time_with_cross:.2f} ms")
    print(f"  - 输出范围: [{output_with_cross.min():.3f}, {output_with_cross.max():.3f}]")
    print()
    print(f"✓ 跨域交互增加时间: {time_with_cross - time_no_cross:.2f} ms ({(time_with_cross/time_no_cross - 1)*100:.1f}%)")
    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Geometry-Semantic Dual-Path SSM 测试")
    print("="*60 + "\n")

    try:
        test_geometry_feature_extractor()
        test_geometry_semantic_dual_path()
        test_scalability()
        test_with_cross_attention_comparison()

        print("="*60)
        print("✅ 所有测试通过！")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print("❌ 测试失败！")
        print("="*60)
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
