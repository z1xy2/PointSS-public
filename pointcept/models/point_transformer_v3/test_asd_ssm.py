"""
ASD-SSM测试脚本

用于验证ASD-SSM模块的正确性和性能

Usage:
    python test_asd_ssm.py
"""

import torch
import torch.nn as nn
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pointcept.models.point_transformer_v3.asd_ssm import (
    ScaleAwareParameterGenerator,
    AdaptiveScaleDecoupledMamba,
    ScaleSequentialMambaWithASDSSM,
    MSFFSBlockWithASDSSM
)


def test_scale_parameter_generator():
    """测试尺度参数生成器"""
    print("\n" + "="*50)
    print("Test 1: ScaleAwareParameterGenerator")
    print("="*50)

    d_model = 256
    num_scales = 3
    batch_size = 2
    seq_len = 128

    generator = ScaleAwareParameterGenerator(d_model, num_scales, use_global_feature=True)
    print(f"✓ Created generator with d_model={d_model}, num_scales={num_scales}")

    # 测试不同尺度
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"✓ Input shape: {x.shape}")

    for scale_id in range(num_scales):
        delta_A, delta_B, delta_C, delta_t, scale_info = generator(x, scale_id)

        print(f"\nScale {scale_id}:")
        print(f"  - delta_A shape: {delta_A.shape}")
        print(f"  - delta_B shape: {delta_B.shape}")
        print(f"  - delta_C shape: {delta_C.shape}")
        print(f"  - delta_t shape: {delta_t.shape}")
        print(f"  - scale_constraint: {scale_info['scale_constraint']:.4f}")
        print(f"  - delta_A_mean: {scale_info['delta_A_mean']:.6f}")
        print(f"  - delta_t_mean: {scale_info['delta_t_mean']:.6f}")

        # 验证约束：粗尺度应该有更大的约束因子
        if scale_id == 0:
            first_constraint = scale_info['scale_constraint']
        elif scale_id == num_scales - 1:
            last_constraint = scale_info['scale_constraint']
            assert first_constraint > last_constraint, \
                "Coarse scale should have larger constraint than fine scale"

    print("\n✅ ScaleAwareParameterGenerator test passed!")


def test_asd_mamba():
    """测试ASD-Mamba模块"""
    print("\n" + "="*50)
    print("Test 2: AdaptiveScaleDecoupledMamba")
    print("="*50)

    d_model = 128  # 使用较小的维度以加快测试
    num_scales = 3
    batch_size = 2
    seq_len = 64

    asd_mamba = AdaptiveScaleDecoupledMamba(
        d_model=d_model,
        num_scales=num_scales,
        layer_idx=0,
        use_global_feature=True
    )
    print(f"✓ Created ASD-Mamba with d_model={d_model}")

    x = torch.randn(batch_size, seq_len, d_model)
    print(f"✓ Input shape: {x.shape}")

    # 测试不同尺度
    for scale_id in range(num_scales):
        output, x_res, scale_info = asd_mamba(x, scale_id, x_res=None)

        print(f"\nScale {scale_id}:")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output norm: {scale_info['output_norm']:.6f}")

        # 验证输出形状
        assert output.shape == x.shape, \
            f"Output shape {output.shape} doesn't match input shape {x.shape}"

    print("\n✅ AdaptiveScaleDecoupledMamba test passed!")


def test_scale_sequential_mamba():
    """测试跨尺度序列化Mamba"""
    print("\n" + "="*50)
    print("Test 3: ScaleSequentialMambaWithASDSSM")
    print("="*50)

    channels = 128
    num_scales = 3
    batch_size = 2
    seq_len = 64

    ss_mamba = ScaleSequentialMambaWithASDSSM(
        channels=channels,
        num_scales=num_scales,
        use_scale_pe=True
    )
    print(f"✓ Created SS-Mamba with channels={channels}, num_scales={num_scales}")

    # 创建多尺度特征
    scale_features = [
        torch.randn(batch_size, seq_len, channels)
        for _ in range(num_scales)
    ]
    print(f"✓ Created {num_scales} scale features, each with shape {scale_features[0].shape}")

    # 前向传播
    fused, scale_weights, scale_info_list = ss_mamba(scale_features)

    print(f"\nResults:")
    print(f"  - Fused output shape: {fused.shape}")
    print(f"  - Scale weights shape: {scale_weights.shape}")
    print(f"  - Scale weights: {scale_weights.detach().cpu().numpy()}")

    # 验证
    assert fused.shape == (batch_size, seq_len, channels), \
        f"Fused shape {fused.shape} incorrect"
    assert scale_weights.shape == (num_scales,), \
        f"Scale weights shape {scale_weights.shape} incorrect"
    assert len(scale_info_list) == num_scales, \
        f"Should have {num_scales} scale info entries"

    # 验证权重和为1
    assert torch.allclose(scale_weights.sum(), torch.tensor(1.0), atol=1e-5), \
        "Scale weights should sum to 1"

    print("\n✅ ScaleSequentialMambaWithASDSSM test passed!")


def test_msffs_block():
    """测试完整的MSFFS Block"""
    print("\n" + "="*50)
    print("Test 4: MSFFSBlockWithASDSSM")
    print("="*50)

    channels = 128
    patch_size = 1024
    num_scales = 3
    batch_size = 2
    seq_len = 64

    msffs_block = MSFFSBlockWithASDSSM(
        channels=channels,
        patch_size=patch_size,
        num_scales=num_scales,
        use_order_prompt=True,
        prompt_num_per_order=6
    )
    print(f"✓ Created MSFFS Block")

    # 创建多尺度特征
    scale_features = [
        torch.randn(batch_size, seq_len, channels)
        for _ in range(num_scales)
    ]

    # 前向传播
    output, scale_weights, scale_info_list = msffs_block(
        scale_features,
        order_indices=None
    )

    print(f"\nResults:")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Scale weights: {scale_weights.detach().cpu().numpy()}")

    # 验证
    assert output.shape == (batch_size, seq_len, channels), \
        f"Output shape {output.shape} incorrect"

    print("\n✅ MSFFSBlockWithASDSSM test passed!")


def test_gradient_flow():
    """测试梯度流"""
    print("\n" + "="*50)
    print("Test 5: Gradient Flow")
    print("="*50)

    channels = 128
    num_scales = 3
    batch_size = 2
    seq_len = 64

    ss_mamba = ScaleSequentialMambaWithASDSSM(
        channels=channels,
        num_scales=num_scales,
        use_scale_pe=True
    )

    # 创建多尺度特征（需要梯度）
    scale_features = [
        torch.randn(batch_size, seq_len, channels, requires_grad=True)
        for _ in range(num_scales)
    ]

    # 前向传播
    fused, scale_weights, scale_info_list = ss_mamba(scale_features)

    # 计算损失
    loss = fused.sum()

    # 反向传播
    loss.backward()

    # 检查梯度
    print("Checking gradients:")
    for i, feat in enumerate(scale_features):
        if feat.grad is not None:
            grad_norm = feat.grad.norm().item()
            print(f"  - Scale {i} gradient norm: {grad_norm:.6f}")
            assert grad_norm > 0, f"Gradient for scale {i} is zero!"
        else:
            raise AssertionError(f"No gradient for scale {i}")

    # 检查参数梯度
    has_param_grad = False
    for name, param in ss_mamba.named_parameters():
        if param.grad is not None and param.grad.norm() > 0:
            has_param_grad = True
            break

    assert has_param_grad, "No parameter has gradient!"
    print("  ✓ Parameters have gradients")

    print("\n✅ Gradient flow test passed!")


def test_memory_efficiency():
    """测试显存效率"""
    print("\n" + "="*50)
    print("Test 6: Memory Efficiency")
    print("="*50)

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping memory test")
        return

    device = torch.device('cuda')
    channels = 256
    num_scales = 3
    batch_size = 4
    seq_len = 1024

    # 清空缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    ss_mamba = ScaleSequentialMambaWithASDSSM(
        channels=channels,
        num_scales=num_scales,
        use_scale_pe=True
    ).to(device)

    # 创建输入
    scale_features = [
        torch.randn(batch_size, seq_len, channels, device=device)
        for _ in range(num_scales)
    ]

    # 前向传播
    initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB

    fused, _, _ = ss_mamba(scale_features)
    loss = fused.sum()

    forward_memory = torch.cuda.memory_allocated() / 1024**2
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    print(f"Memory usage:")
    print(f"  - Initial: {initial_memory:.2f} MB")
    print(f"  - After forward: {forward_memory:.2f} MB")
    print(f"  - Peak: {peak_memory:.2f} MB")

    # 反向传播
    loss.backward()

    backward_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  - After backward: {backward_memory:.2f} MB")

    print("\n✅ Memory efficiency test completed!")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print(" "*20 + "ASD-SSM Test Suite")
    print("="*70)

    try:
        test_scale_parameter_generator()
        test_asd_mamba()
        test_scale_sequential_mamba()
        test_msffs_block()
        test_gradient_flow()
        test_memory_efficiency()

        print("\n" + "="*70)
        print("🎉 All tests passed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        print("\n" + "="*70)
        print("❌ Test failed!")
        print("="*70)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    run_all_tests()
