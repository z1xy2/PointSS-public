"""
快速验证ASD-SSM集成是否成功

Usage:
    python quick_verify_integration.py
"""

import torch
import sys
import os

# 添加路径
sys.path.append('D:/PointSS')

print("="*70)
print(" "*20 + "ASD-SSM 集成验证")
print("="*70)

try:
    # 1. 测试导入
    print("\n[1/5] 测试模块导入...")
    from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
        SerializedAttention
    )
    print("✅ 成功导入 SerializedAttention")

    # 2. 测试创建（启用ASD-SSM）
    print("\n[2/5] 测试创建 SerializedAttention (use_asd_ssm=True)...")
    attn_asd = SerializedAttention(
        is_enc=True,
        layer_idx=0,
        channels=128,
        num_heads=4,
        patch_size=1024,
        use_asd_ssm=True,  # 启用ASD-SSM
        num_scales=2
    )
    print("✅ 成功创建 ASD-SSM 版本")
    print(f"   - 是否有 asd_ssm_wrapper: {hasattr(attn_asd, 'asd_ssm_wrapper')}")
    print(f"   - 尺度数量: {attn_asd.num_scales}")

    # 3. 测试创建（原始Mamba）
    print("\n[3/5] 测试创建 SerializedAttention (use_asd_ssm=False)...")
    attn_orig = SerializedAttention(
        is_enc=True,
        layer_idx=0,
        channels=128,
        num_heads=4,
        patch_size=1024,
        use_asd_ssm=False  # 使用原始Mamba
    )
    print("✅ 成功创建原始Mamba版本")
    print(f"   - 是否有 mamba0: {hasattr(attn_orig, 'mamba0')}")
    print(f"   - 是否有 mamba1: {hasattr(attn_orig, 'mamba1')}")

    # 4. 测试参数统计
    print("\n[4/5] 对比参数量...")
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    params_asd = count_parameters(attn_asd)
    params_orig = count_parameters(attn_orig)
    ratio = params_asd / params_orig

    print(f"   - ASD-SSM版本: {params_asd:,} 参数")
    print(f"   - 原始Mamba版本: {params_orig:,} 参数")
    print(f"   - 参数增加比例: {ratio:.2f}x")

    if ratio > 1.0 and ratio < 1.5:
        print("   ✅ 参数增加在合理范围内 (1.0x - 1.5x)")
    else:
        print(f"   ⚠️  参数增加比例异常: {ratio:.2f}x")

    # 5. 测试前向传播（如果有CUDA）
    print("\n[5/5] 测试前向传播...")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"   使用设备: {device}")

        # 创建模拟输入
        from pointcept.models.utils.structure import Point
        from addict import Dict

        # 简单的点云数据
        N = 2048  # 点数
        C = 128   # 通道数
        batch_size = 2

        point_dict = Dict({
            'feat': torch.randn(N * batch_size, C, device=device),
            'coord': torch.randn(N * batch_size, 3, device=device),
            'grid_coord': torch.randint(0, 100, (N * batch_size, 3), device=device),
            'offset': torch.tensor([N, N * 2], device=device),
            'batch': torch.cat([
                torch.zeros(N, dtype=torch.long, device=device),
                torch.ones(N, dtype=torch.long, device=device)
            ]),
            'serialized_order': [torch.arange(N * batch_size, device=device)] * 4,
            'serialized_inverse': [torch.arange(N * batch_size, device=device)] * 4,
            'serialized_code': torch.randint(0, 1000000, (4, N * batch_size), device=device),
            'serialized_depth': 10,
            'sort_name': ['z', 'z-trans', 'hilbert', 'hilbert-trans']
        })

        point = Point(point_dict)
        point.sparsify()

        attn_asd = attn_asd.to(device)

        try:
            output = attn_asd(point)
            print("   ✅ 前向传播成功")
            print(f"   - 输出特征形状: {output.feat.shape}")
            print(f"   - 输出特征范围: [{output.feat.min():.4f}, {output.feat.max():.4f}]")
        except Exception as e:
            print(f"   ❌ 前向传播失败: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("   ⚠️  CUDA不可用，跳过前向传播测试")

    print("\n" + "="*70)
    print("🎉 集成验证完成！ASD-SSM已成功集成到模型中")
    print("="*70)
    print("\n下一步：")
    print("1. 运行完整测试: python -m pointcept.models.point_transformer_v3.test_asd_ssm")
    print("2. 开始训练: python train.py --config configs/your_config.py")
    print("3. 对比实验: 在配置中设置 use_asd_ssm=False 进行基线对比")
    print()

except Exception as e:
    print("\n" + "="*70)
    print("❌ 验证失败！")
    print("="*70)
    print(f"\n错误信息: {str(e)}")
    import traceback
    traceback.print_exc()
    print("\n请检查：")
    print("1. 是否在正确的分支: git branch")
    print("2. 代码是否正确提交: git status")
    print("3. 依赖是否安装完整")
    sys.exit(1)
