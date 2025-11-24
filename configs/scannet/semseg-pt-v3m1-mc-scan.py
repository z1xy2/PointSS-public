"""
Multi-Curve Fused Adaptive Scan 配置示例

基于ScanNet语义分割任务的配置
"""

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 12  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = False
enable_amp = True
evaluate = True

# model settings
model = dict(
    type="PT-v3m1",
    in_channels=6,
    num_classes=20,
    patch_embed_depth=1,
    patch_embed_channels=48,
    patch_embed_groups=6,
    patch_embed_neighbours=16,
    enc_depths=(2, 2, 6, 2),
    enc_channels=(96, 192, 384, 512),
    enc_groups=(12, 24, 48, 64),
    enc_neighbours=(16, 16, 16, 16),
    dec_depths=(1, 1, 1, 1),
    dec_channels=(48, 96, 192, 384),
    dec_groups=(6, 12, 24, 48),
    dec_neighbours=(16, 16, 16, 16),
    grid_sizes=(0.06, 0.12, 0.24, 0.48),
    attn_qkv_bias=True,
    pe_type="rpe",
    pe_in_attn=False,
    pe_multiplier=False,
    pe_bias=True,
    attn_drop_rate=0.0,
    drop_path_rate=0.3,
    enable_checkpoint=False,
    unpool_backend="interp",
    pdnorm_bn=False,
    pdnorm_ln=False,
    pdnorm_decouple=True,
    pdnorm_adaptive=False,
    pdnorm_affine=True,
    pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),

    # ========== 🆕 多曲线融合自适应扫描配置 ==========
    # 启用多曲线自适应扫描
    use_multi_curve_scan=True,

    # 重要性权重（0-1之间）
    # - 0.0-0.2: 几乎纯空间排序
    # - 0.3-0.5: 平衡空间和语义（推荐）
    # - 0.6-1.0: 强调语义重要性
    mc_scan_importance_weight=0.3,

    # 曲线权重是否可学习
    # True: 模型自动学习每条曲线的重要性
    # False: 使用均匀权重
    mc_scan_learnable_weights=True,

    # 是否使用重要性预测
    # True: 根据点特征预测重要性
    # False: 仅使用空间曲线
    mc_scan_use_importance=True,
    # ================================================

    # 序列化顺序（使用4种曲线以获得最佳效果）
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    shuffle_orders=True,  # 训练时随机打乱

    # 其他配置保持不变
    enable_rpe=False,
    enable_flash=True,
    upcast_attention=False,
    upcast_softmax=False,
)

# scheduler settings
epoch = 800
eval_epoch = 800
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="blocks", lr=0.0002)]

# dataset settings
dataset_type = "ScanNetDataset"
data_root = "data/scannet"

data = dict(
    num_classes=20,
    ignore_index=-1,
    names=[
        "wall", "floor", "cabinet", "bed", "chair",
        "sofa", "table", "door", "window", "bookshelf",
        "picture", "counter", "desk", "curtain", "refridgerator",
        "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            dict(
                type="RandomRotate",
                angle=[-1, 1],
                axis="z",
                center=[0, 0, 0],
                p=0.5,
            ),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment"),
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment"),
                return_grid_coord=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
    ),
)

# ========== 训练脚本示例 ==========
"""
# 训练
sh scripts/train.sh -p python -d scannet -c semseg-pt-v3m1-mc-scan -n mc_scan_exp

# 测试
sh scripts/test.sh -p python -d scannet -n mc_scan_exp -w model_best

# 查看学到的曲线权重（训练完成后）
python tools/analyze_curve_weights.py --checkpoint exp/scannet/mc_scan_exp/model_best.pth
"""