"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import datetime
from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath

import logging
try:
    import flash_attn
except ImportError:
    flash_attn = None
from openpoints.models.PCM.mamba_layer import MambaBlock
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out

def cyclic_counter(max_value):
    """生成一个循环递增的数列，从0开始到max_value-1，然后重置为0。"""
    count = 0
    while True:
        yield count
        count += 1
        if count == max_value:
            count = 0



class SerializedAttention(PointModule):
    enc_order_prompt_proj=[]
    dec_order_prompt_proj=[]
    order_prompt=None
    per_layer_prompt_indexe=None
    prompt_num_per_order = 6 #位置编码数量
    order_name={"z":0, "z-trans":1, "hilbert":2, "hilbert-trans":3}
    @staticmethod
    def static_init(enc_channels, dec_channels,
                    use_order_prompt=True,  #是否使用位置编码
                    mamba_layers_orders=["z", "z-trans", "hilbert", "hilbert-trans"]):
        prompt_num_per_order=SerializedAttention.prompt_num_per_order
        if use_order_prompt:
            for i in enc_channels:
                SerializedAttention.enc_order_prompt_proj.append(nn.Linear(384, i, bias=False).cuda())  # 将位置编码feat映射为out_channel
            #使用位置編碼


            # learnable embeddings for per order, channel is 384
            unique_order = list(set(mamba_layers_orders))#将mamba_layers_orders打乱排序，感觉没啥用
            overall_prompt_nums = len(unique_order) * prompt_num_per_order
            # Embedding层将一个数字直接转化为你想要的维度的向量，就是order_prompt
            SerializedAttention.order_prompt = nn.Embedding(overall_prompt_nums, 384).cuda()
            #{'hilbert': [6, 12], 'hilbert-trans': [18, 24], 'z': [12, 18], 'z-trans': [0, 6]} 每次运行还不太一样
            #{'hilbert': [0, 6], 'hilbert-trans': [6, 12], 'z': [18, 24], 'z-trans': [12, 18]}
            order2idx = {order: [i * prompt_num_per_order, (i + 1) * prompt_num_per_order]
                         for i, order in enumerate(unique_order)}  # 为每个order分配6个索引
            SerializedAttention.per_layer_prompt_indexe = []
            for order in mamba_layers_orders:
                SerializedAttention.per_layer_prompt_indexe.append(order2idx[order])  # 根据order在数组中的顺序，将索引存入数组


            for i in dec_channels:
                SerializedAttention.dec_order_prompt_proj.append(nn.Linear(384, i, bias=False).cuda())  # 将位置编码feat映射为out_channel



    def __init__(
        self,
        is_enc,
        layer_idx,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        area_num=2,

    ):
        super().__init__()
        assert channels % num_heads == 0
        self.is_enc=is_enc
        self.layer_idx=layer_idx
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        self.area_num = area_num
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.current_patch_size = patch_size
            self.attn_drop = attn_drop


        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.current_patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        1
        self.proj = torch.nn.Linear(3*channels, channels)#进行了修改

        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.mlp=MLP(channels,channels)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None
        self.mamba0 = MambaBlock(dim=channels, layer_idx=None,
                                # mamba_layer_idx存储缓存 而只有_get_states_from_cache用到layer_idx，
                                # 但_get_states_from_cache从未执行过,因此layer_idx放心写none。
                                bimamba_type='v2',
                                norm_cls=partial( RMSNorm, eps=1e-5,), fused_add_norm=True,
                                residual_in_fp32=True,
                                drop_path=0)  # drop_path就是随机drop，先不drop试试效果
        self.mamba1 = MambaBlock(dim=channels, layer_idx=None,
                                # mamba_layer_idx存储缓存 而只有_get_states_from_cache用到layer_idx，
                                # 但_get_states_from_cache从未执行过,因此layer_idx放心写none。
                                bimamba_type='v2',
                                norm_cls=partial( RMSNorm, eps=1e-5,), fused_add_norm=True,
                                residual_in_fp32=True,
                                drop_path=0)  # drop_path就是随机drop，先不drop试试效果
        self.mamba2 = MambaBlock(dim=channels, layer_idx=None,
                                # mamba_layer_idx存储缓存 而只有_get_states_from_cache用到layer_idx，
                                # 但_get_states_from_cache从未执行过,因此layer_idx放心写none。
                                bimamba_type='v2',
                                norm_cls=partial( RMSNorm, eps=1e-5,), fused_add_norm=True,
                                residual_in_fp32=True,
                                drop_path=0)  # drop_path就是随机drop，先不drop试试效果
    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.current_patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    def create_filled_tensor(self,length, range_end):#todo:转为cpu形式，再验证一下正确性
        # 计算基本的重复次数和额外需要分配的元素数
        base_count = length // range_end
        remainder = length % range_end

        # 创建一个包含重复次数的数组
        repeats = torch.ones(range_end, dtype=torch.long).to(range_end.device) * base_count

        # 为前remainder个元素增加额外的重复次数
        repeats[:remainder] += 1

        # 使用arange和repeat_interleave生成最终的张量
        final_tensor = torch.repeat_interleave(torch.arange(range_end).to(range_end.device), repeats)

        return final_tensor

    @torch.no_grad()
    def get_padding_and_inverse( self,point, config_patch_size):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        # 检查键值：函数首先检查point字典中是否已经存在pad、unpad和cu_seqlens_key键。如果这些键都存在，则不需要重新计算它们。
        if (
                pad_key not in point.keys()
                or unpad_key not in point.keys()
                or cu_seqlens_key not in point.keys()
        ):  # 否则分别计算pad、unpad和cu_seqlens_key

            offset = point.offset
            bincount = offset2bincount(offset)  # 每一个batch中的点数量


            # ====填充计算：======
            bincount_pad = (  # 对每个batch的元素数量进行向上取整到最近的patch_size的倍数，确保每个块的大小都是patch_size的整数倍。
                    torch.div(
                        bincount + config_patch_size - 1,
                        config_patch_size,
                        rounding_mode="trunc",
                    )
                    * config_patch_size
            )  # bincount_pad shape与bincount相同
            # only pad point when num of points larger than patch_size 一个布尔掩码，指示哪些块的元素数量大于patch_size。
            mask_pad = bincount > config_patch_size  # 该位置点数是否大于self.patch_size True or False
            # 通过mask_pad调整bincount_pad，确保仅对元素数量超过patch_size的块进行填充，元素不超过。
            bincount_pad = ~mask_pad * config_patch_size + mask_pad * bincount_pad
            # bincount_pad记录了填充后一个batch的点的数量。如果某一个batch没有超过patch_size的点
            # 那么bincount_pad和bincount是相等的，也就是该batch不需要被填充
            # 否则bincount_pad就是patch整数填充后的点数量
            # ====偏移量和填充索引的计算======
            # 偏移量和填充索引的计算：_offset和_offset_pad分别计算原始偏移量和经过填充调整后的偏移量。
            _offset = nn.functional.pad(offset, (1, 0))  # 前面添个0，得到前面带0的offset
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))  # 得到添加pad后，且前面带0的offset
            # pad和unpad分别计算填充后的索引和从填充索引到原始索引的映射。
            pad = torch.arange(_offset_pad[-1], device=offset.device)  # pad为填充后的点云总量，value=[0,1,...,填充后点数量-1]
            unpad = torch.arange(_offset[-1], device=offset.device)  # 不填充的点云总量，value=[0,1,...,原点数量-1]
            cu_seqlens = []
            # 对每个点云批次进行填充（padding）和逆填充（unpadding）的调整，同时计算每个批次经过调整后的序列长度。
            for i in range(len(offset)):  # 每一个batch遍历。是一个一维数组，表示每个点云批次的起始索引。
                # 这一步调整逆填充（unpad）数组，确保经过填充后的序列能够反映回原始序列的位置。（？？？？）
                # 对于每个批次，它基于原始偏移（_offset）和经过填充调整后的偏移（_offset_pad）之间的差值，
                # _offset[i] : _offset[i + 1]取第i个batch的点，加上该batch填充了的点
                unpad[_offset[i]: _offset[i + 1]] += _offset_pad[i] - _offset[i]  # 计算unpad_key，是统一对batch的操作，不需要额外修改，每个batch都会用到
                if(bincount[i]<config_patch_size):#如果这一batch的点数量不足以形成一个patch，那就将该batch的点重复。
                    count_per_element =bincount[i]
                    this_batch_pad_count= _offset_pad[i + 1] - _offset_pad[i]
                    result = self.create_filled_tensor(this_batch_pad_count, count_per_element)
                    pad[_offset_pad[i ]:_offset_pad[i + 1]]=result+_offset_pad[i]
                else:
                    # 如果bincount[i]（原始批次中的点数）与bincount_pad[i]（调整后的点数）不相等，说明该batch的点数不能被patch_size整除，需要特殊处理：
                    # 将倒数第二个patch_size的后面一部分索引填充到最后一个patch_size的padding的部分，因为这些点是最邻近的，见思源
                    if bincount[i] != bincount_pad[i]:#如果需要填充,填充与原版ptv3略有区别
                        # pad[  # 填充位的点云变为
                        #     _offset_pad[i + 1]
                        #     - current_patch_size
                        #     + (bincount[i] % current_patch_size): _offset_pad[i + 1]
                        #     ] = pad[
                        #
                        #         _offset_pad[i + 1]
                        #         - 2 * current_patch_size
                        #         + (bincount[i] % current_patch_size): _offset_pad[i + 1]
                        #                                                 - current_patch_size
                        # ]
                        new_pad=pad.clone()#新建，用于为pad赋值。
                        # 最后一组填充仍然保持顺序
                        pad[
                        _offset_pad[i + 1]
                        - config_patch_size
                            : _offset_pad[i + 1]
                            ] = new_pad[

                                _offset_pad[i + 1]
                                - 2 * config_patch_size
                                + (bincount[i] % config_patch_size): _offset_pad[i + 1]
                                                        - config_patch_size + (bincount[i] % config_patch_size)
                        ]


                # 更新填充（pad）数组，调整填充后序列中的索引，以确保它们正确反映原始数据中的位置。原本的pad，除了第一个batch，后面的batch都无法索引原数组，
                # 因为pad的原因，每个batch都是pad过后的编号，比如每个batch都是整数patch_size开头。
                # 我们要对每个batch减去前几个batch造成的填充差值，之后就可以正常索引原数组了
                pad[_offset_pad[i]: _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                # 对于每个批次，计算经过填充调整后的序列中，每个patch_size区间的起始索引。这些起始索引被存储在cu_seqlens列表中，表示每个序列块的开始位置，这对于后续处理序列化数据非常重要。
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=config_patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )

            point[pad_key] = pad
            point[unpad_key] = unpad

            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.current_patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )
        bincount = offset2bincount(point.offset)  # 每一个batch中的点数量


        H = self.num_heads
        K = self.current_patch_size
        C = self.channels #通道数
        #已完成：动态patch，每个batch的点数量都大于patch_size时，patch_size为我们预先设置好的最大值，也就是self.patch_size
        #已完成：有任何一个batch的点连一个patch_size都不到，那么patch_size就是这个batch的点数。
        #===========顺序特征================
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point,K)

        # 取出padding过后的order顺序，可以索引原点云，得到pad后的有序点云
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        ordered_and_padded_points_feat=point.feat[order] #排序后和padding后的点云特征。
        x = ordered_and_padded_points_feat.reshape(-1, K, C)  # 将排好序和pad的点云转换为最初点云的顺序（没排序没pad）
        x_res=None
        #进行order prompt,todo：顺序提示需要给出当前编码顺序
        sort_name=point.sort_name[self.order_index]# 取出第i个顺序提示的名称
        layer_order_prompt_indexes = self.per_layer_prompt_indexe[self.order_name[sort_name]]  # 根据名称取出第i个顺序提示的prompt的index

        layer_order_prompt = self.order_prompt.weight[  # order_prompt.weight:[54,384]，54是所有的顺序提示的数量，384是每个顺序提示的维度
                             layer_order_prompt_indexes[0]: layer_order_prompt_indexes[1]]  # 将这组的位置提示的MLP（权重）取出来
        if self.is_enc:
            layer_order_prompt_proj = self.enc_order_prompt_proj[self.layer_idx]
        else:
            layer_order_prompt_proj = self.dec_order_prompt_proj[self.layer_idx]
        layer_order_prompt=layer_order_prompt_proj(layer_order_prompt)
        layer_order_prompt = layer_order_prompt.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = torch.cat([layer_order_prompt, x, layer_order_prompt], dim=1)  # 很神奇，将顺序提示当做点给添加进去了，前面六个点，后面六个点，实际都是顺序提示
        x,x_res=self.mamba0(x,x_res) #todo：看x_res如何修改,这个不好改，还要在下采样里改，先不改试试。
        x = x[:, self.prompt_num_per_order:-self.prompt_num_per_order]  # 出来时将前后的顺序提示去掉
        x=x.reshape(-1, C) #返回为非patch的情况
        feat0 = x[inverse]#将排好序和pad的点云转换为最初点云的顺序（没排序没pad）

        #=============================引入随机全局特征4================================
        del point["pad"]
        del point["unpad"]
        del point["cu_seqlens_key"]
        pad_4, unpad_4, cu_seqlens_4 = self.get_padding_and_inverse(point, 3*K)

        # 取出padding过后的order顺序，可以索引原点云，得到pad后的有序点云
        order = point.serialized_order[self.order_index][pad_4]
        inverse = unpad_4[point.serialized_inverse[self.order_index]]
        ordered_and_padded_points_feat=point.feat[order] #排序后和padding后的点云特征。
        x = ordered_and_padded_points_feat.reshape(-1, 3*K, C)  # 将排好序和pad的点云转换为最初点云的顺序（没排序没pad）
        x_res=None

        #打乱同patch的特征
        shuffle_one_patch_indices = torch.randperm(x.shape[1]).cuda()#打乱后的下标
        inverse_shuffle_one_patch_indices = torch.empty_like(shuffle_one_patch_indices)
        inverse_shuffle_one_patch_indices[shuffle_one_patch_indices] = torch.arange(x.shape[1]).cuda()#打乱后将点云返还原来顺序，所需的反向下标
        x_shuffled = x.index_select(dim=1, index=shuffle_one_patch_indices)

        x_shuffled,x_res=self.mamba1(x_shuffled,x_res) #todo：看x_res如何修改,这个不好改，还要在下采样里改，先不改试试。
        x=x_shuffled.index_select(dim=1, index=inverse_shuffle_one_patch_indices)

        x=x.reshape(-1, C) #返回为非patch的情况
        feat1 = x[inverse]#将排好序和pad的点云转换为最初点云的顺序（没排序没pad）

        # =============================引入随机全局特征16================================
        del point["pad"]
        del point["unpad"]
        del point["cu_seqlens_key"]
        pad_16, unpad_16, cu_seqlens_16 = self.get_padding_and_inverse(point, 6 * K)

        # 取出padding过后的order顺序，可以索引原点云，得到pad后的有序点云
        order = point.serialized_order[self.order_index][pad_16]
        inverse = unpad_16[point.serialized_inverse[self.order_index]]
        ordered_and_padded_points_feat = point.feat[order]  # 排序后和padding后的点云特征。
        x = ordered_and_padded_points_feat.reshape(-1, 6 * K, C)  # 将排好序和pad的点云转换为最初点云的顺序（没排序没pad）
        x_res = None

        # 打乱同patch的特征
        shuffle_one_patch_indices = torch.randperm(x.shape[1]).cuda()  # 打乱后的下标
        inverse_shuffle_one_patch_indices = torch.empty_like(shuffle_one_patch_indices)
        inverse_shuffle_one_patch_indices[shuffle_one_patch_indices] = torch.arange(
            x.shape[1]).cuda()  # 打乱后将点云返还原来顺序，所需的反向下标
        x_shuffled = x.index_select(dim=1, index=shuffle_one_patch_indices)

        x_shuffled, x_res = self.mamba2(x_shuffled, x_res)  # todo：看x_res如何修改,这个不好改，还要在下采样里改，先不改试试。
        x = x_shuffled.index_select(dim=1, index=inverse_shuffle_one_patch_indices)

        x = x.reshape(-1, C)  # 返回为非patch的情况
        feat2 = x[inverse]  # 将排好序和pad的点云转换为最初点云的顺序（没排序没pad）

        point["pad_key"]=pad
        point["unpad_key"]=unpad
        point["cu_seqlens_key"]=cu_seqlens

        feat=torch.concat([feat0,feat1,feat2],1)

        # ffn
        feat = self.proj(feat)

        feat=self.mlp(feat)

        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    time_cost=0
    count=0
    def __init__(
        self,
        is_enc,
        layer_idx,#第几号编码器/解码器
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
            drop_path_rate=0.1,#新增
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))

        #================================修改的代码=============================
        self.latecyLogger=logging.getLogger("b3-p128-trans")

        #自添加mamba
        self.attn = SerializedAttention(
            is_enc,
            layer_idx,
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        # x, x_res = self.mamba(x, x_res)

        time_start = datetime.datetime.now()
        point=self.attn(point)
        time_end = datetime.datetime.now()
        time_latecy=time_end - time_start

        self.count+=1
        self.latecyLogger.info(time_latecy)

        point = self.drop_path(point)
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]
            sort_name = [point.sort_name[perm[i]] for i in range(len(perm))]
        # print(sort_name)
        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
            sort_name=sort_name
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module("PT-v3m1")
class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        #===============创建记录运行时间的日志==========================
        self.latecyLogger=logging.getLogger("b3-p128-trans")
        self.latecyLogger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        fh = logging.FileHandler("b3-p128-trans", "w")
        fh.setFormatter(formatter)
        self.latecyLogger.addHandler(fh)

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        # 初始化SerializedAttention
        SerializedAttention.static_init(enc_channels, dec_channels)
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )


            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        True,
                        s,
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            False,
                            s,
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        # else:
        #     point.feat = torch_scatter.segment_csr(
        #         src=point.feat,
        #         indptr=nn.functional.pad(point.offset, (1, 0)),
        #         reduce="mean",
        #     )
        return point
