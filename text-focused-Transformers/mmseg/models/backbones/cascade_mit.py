import math
import warnings

import numpy
import cv2
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmseg.ops import resize
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmcv.runner import BaseModule, ModuleList, Sequential

from ..builder import BACKBONES
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw


class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(MultiheadAttention):
    """An implementation of Efficient Multi-head Attention of Segformer.

    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        # handle the BC-breaking from https://github.com/open-mmlab/mmcv/pull/1418 # noqa
        from mmseg import digit_version, mmcv_version
        if mmcv_version < digit_version('1.3.17'):
            warnings.warn('The legacy version of forward function in'
                          'EfficientMultiheadAttention is deprecated in'
                          'mmcv>=1.3.17 and will no longer support in the'
                          'future. Please upgrade your mmcv.')
            self.forward = self.legacy_forward

    def forward(self, x, hw_shape, identity=None, attn_mask=None):

        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)

        out = self.attn(query=x_q, key=x_kv, value=x_kv, attn_mask=attn_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

    def legacy_forward(self, x, hw_shape, identity=None, attn_mask=None):
        """multi head attention forward in mmcv version < 1.3.17."""

        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # `need_weights=True` will let nn.MultiHeadAttention
        # `return attn_output, attn_output_weights.sum(dim=1) / num_heads`
        # The `attn_output_weights.sum(dim=1)` may cause cuda error. So, we set
        # `need_weights=False` to ignore `attn_output_weights.sum(dim=1)`.
        # This issue - `https://github.com/pytorch/pytorch/issues/37583` report
        # the error that large scale tensor sum operation may cause cuda error.
        out = self.attn(query=x_q, key=x_kv, value=x_kv, need_weights=False, attn_mask=attn_mask)[0]

        return identity + self.dropout_layer(self.proj_drop(out))


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Segformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default:None.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 sr_ratio=1,
                 with_cp=False):
        super(TransformerEncoderLayer, self).__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

        self.with_cp = with_cp

    def forward(self, x, hw_shape, attn_mask=None):

        def _inner_forward(x):
            x = self.attn(self.norm1(x), hw_shape, identity=x, attn_mask=attn_mask)
            x = self.ffn(self.norm2(x), hw_shape, identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


@BACKBONES.register_module()
class CascadeMixVisionTransformer(BaseModule):
    """The backbone of Segformer.

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=in_channels[i],
                embed_dims=embed_dims_i[i],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=7, stride=4, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Conv2d(2*num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(2*num_heads[2]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(2*num_heads[1]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv5 = nn.Conv2d(2*num_heads[0]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False)

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )


    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer, self).init_weights()

    def forward(self, x):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.bn1(self.conv1(x))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
        x4_ = self.conv2(torch.cat([x4, x4_], dim=1))  # 256 * 16 * 32

        x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
        x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64
        x3_ = self.conv3(torch.cat([x3, x3_], dim=1))  # 160 * 32 * 64

        x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
        x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
        x2_ = self.conv4(torch.cat([x2, x2_], dim=1))  # 64 * 64 * 128

        x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
        x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
        x1_ = self.conv5(torch.cat([x1, x1_], dim=1))  # 32 * 128 * 256

        outs = [x1_, x2_, x3_, x4_]

        return outs

@BACKBONES.register_module()
class CascadeMixVisionTransformer_V1(BaseModule):
    """The backbone of Segformer.

    加了det的损失

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer_V1, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=in_channels[i],
                embed_dims=embed_dims_i[i],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=7, stride=4, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[3]*self.embed_dims)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*num_heads[2]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[2] * self.embed_dims)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(2*num_heads[1]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[1] * self.embed_dims)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(2*num_heads[0]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[0] * self.embed_dims)
        )

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )

        # 用于做文本区域检测
        self.out_det_1 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[0], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_2 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[1], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_3 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_4 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3]*4, self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.det_cls = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[3], 2, kernel_size=1),
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer_V1, self).init_weights()

    def forward(self, x):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.bn1(self.conv1(x))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
        x4_ = self.conv2(torch.cat([x4, x4_], dim=1))  # 256 * 16 * 32

        x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
        x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64
        x3_ = self.conv3(torch.cat([x3, x3_], dim=1))  # 160 * 32 * 64

        x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
        x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
        x2_ = self.conv4(torch.cat([x2, x2_], dim=1))  # 64 * 64 * 128

        x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
        x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
        x1_ = self.conv5(torch.cat([x1, x1_], dim=1))  # 32 * 128 * 256

        outs = [x1_, x2_, x3_, x4_]

        if self.training:
            # 处理下采样特征用于文本区域检测
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]
            # for item in det_feat:
            #     print(item.shape)
            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)
            return outs, det_res
        else:
            return outs

@BACKBONES.register_module()
class CascadeMixVisionTransformer_V2(BaseModule):
    """The backbone of Segformer.

    只保留高层的信息用于文本区域的检测 前面的两层不额外使用

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer_V2, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=in_channels[i],
                embed_dims=embed_dims_i[i],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=7, stride=4, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[3]*self.embed_dims)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*num_heads[2]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[2] * self.embed_dims)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(2*num_heads[1]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[1] * self.embed_dims)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(2*num_heads[0]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[0] * self.embed_dims)
        )

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )

        # 用于做文本区域检测
        # self.out_det_1 = nn.Sequential(
        #     nn.Conv2d(self.embed_dims * num_heads[0], self.embed_dims * num_heads[3], kernel_size=1),
        #     nn.BatchNorm2d(self.embed_dims * num_heads[3])
        # )
        # self.out_det_2 = nn.Sequential(
        #     nn.Conv2d(self.embed_dims * num_heads[1], self.embed_dims * num_heads[3], kernel_size=1),
        #     nn.BatchNorm2d(self.embed_dims * num_heads[3])
        # )
        self.out_det_3 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_4 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3]*2, self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.det_cls = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[3], 2, kernel_size=1),
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer_V2, self).init_weights()

    def forward(self, x):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.bn1(self.conv1(x))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
        x4_ = self.conv2(torch.cat([x4, x4_], dim=1))  # 256 * 16 * 32

        x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
        x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64
        x3_ = self.conv3(torch.cat([x3, x3_], dim=1))  # 160 * 32 * 64

        x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
        x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
        x2_ = self.conv4(torch.cat([x2, x2_], dim=1))  # 64 * 64 * 128

        x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
        x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
        x1_ = self.conv5(torch.cat([x1, x1_], dim=1))  # 32 * 128 * 256

        outs = [x1_, x2_, x3_, x4_]

        if self.training:
            # 处理下采样特征用于文本区域检测
            det_feat = [
                # resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                # resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]
            # for item in det_feat:
            #     print(item.shape)
            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)
            return outs, det_res
        else:
            return outs

@BACKBONES.register_module()
class CascadeMixVisionTransformer_V3(BaseModule):
    """The backbone of Segformer.

    将融合特征处的卷积核换位1

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer_V3, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=in_channels[i],
                embed_dims=embed_dims_i[i],
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=7, stride=4, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[3]*self.embed_dims)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*num_heads[2]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[2] * self.embed_dims)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(2*num_heads[1]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[1] * self.embed_dims)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(2*num_heads[0]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[0] * self.embed_dims)
        )

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )

        # 用于做文本区域检测
        self.out_det_1 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[0], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_2 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[1], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_3 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_4 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3]*4, self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.det_cls = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[3], 2, kernel_size=1),
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer_V3, self).init_weights()

    def forward(self, x):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.bn1(self.conv1(x))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
        x4_ = self.conv2(torch.cat([x4, x4_], dim=1))  # 256 * 16 * 32

        x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
        x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64
        x3_ = self.conv3(torch.cat([x3, x3_], dim=1))  # 160 * 32 * 64

        x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
        x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
        x2_ = self.conv4(torch.cat([x2, x2_], dim=1))  # 64 * 64 * 128

        x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
        x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
        x1_ = self.conv5(torch.cat([x1, x1_], dim=1))  # 32 * 128 * 256

        outs = [x1_, x2_, x3_, x4_]

        if self.training:
            # 处理下采样特征用于文本区域检测
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]
            # for item in det_feat:
            #     print(item.shape)
            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)
            return outs, det_res
        else:
            return outs

@BACKBONES.register_module()
class CascadeMixVisionTransformer_V4(BaseModule):
    """The backbone of Segformer.

    加了det的损失-并且把文本区域的单独做transformer encoder

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer_V4, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=in_channels[i],
                embed_dims=embed_dims_i[i],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=7, stride=4, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Sequential(
            nn.Conv2d(3*num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[3]*self.embed_dims)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3*num_heads[2]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[2] * self.embed_dims)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(3*num_heads[1]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[1] * self.embed_dims)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(3*num_heads[0]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[0] * self.embed_dims)
        )

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )

        # 用于做文本区域检测
        self.out_det_1 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[0], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_2 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[1], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_3 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_4 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3]*4, self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.det_cls = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[3], 2, kernel_size=1),
        )

        # 用于全部文本区域的自注意力
        self.text_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
            )
        self.text_sa_bn_1 = nn.BatchNorm2d(embed_dims_i[0])

        self.text_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )
        self.text_sa_bn_2 = nn.BatchNorm2d(embed_dims_i[1])


        self.text_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )
        self.text_sa_bn_3 = nn.BatchNorm2d(embed_dims_i[2])


        self.text_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )
        self.text_sa_bn_4 = nn.BatchNorm2d(embed_dims_i[3])


        # self.num = 0

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer_V4, self).init_weights()

    def calculate_mask(self, det_gt, shape, ratio, l):
        h, w = shape
        # print(h, w)
        det_gt_q = torch.nn.functional.interpolate(det_gt.float(), size=(h, w)).long().squeeze(1).flatten(1) # b, 1, h, w
        det_gt_kv = torch.nn.functional.interpolate(det_gt.float(), size=(h//ratio, w//ratio)).long().squeeze(1).flatten(1)
        # print('det_gt_q:', set(list(numpy.array(det_gt_q.cpu()).reshape(-1))))
        # print('det_gt_kv:', set(list(numpy.array(det_gt_kv.cpu()).reshape(-1))))
        # print('det_gt_q:', det_gt_q.shape)
        # print(torch.sum(det_gt_kv))
        mask = (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1)).repeat(self.num_heads[l], 1, 1)
        mask = mask.float() * -1e10
        # return (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1))
        return mask

    def soft_argmax(self, x, beta=1e10):
        x_shape = x.shape
        x = x.view(x.shape[0], x.shape[1], -1) #[bs,c,h*w]
        L = x.shape[1]
        soft_max = nn.functional.softmax(x*beta,dim=1)
        indices = torch.arange(start=0, end=L).unsqueeze(0).unsqueeze(2).to(x.device)
        soft_argmax = soft_max * indices
        indices = soft_argmax.sum(dim=1)  #[bs,c]
        indices = indices.view(x_shape[0], x_shape[2], x_shape[3])
        return indices

    def forward(self, x, det_gt=None):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.bn1(self.conv1(x))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        # print('mit:', set(list(numpy.array(det_gt.cpu()).reshape(-1))))

        if self.training:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            # 单纯将预测的分割图引入
            # _, det_res_max = torch.max(det_res, dim=1)
            # det_res_max = det_res_max.unsqueeze(1)

            # 将soft-argmax分割图引入
            det_res_max = self.soft_argmax(det_res)
            det_res_max = det_res_max.long().unsqueeze(1)
            # print('det_res_max:', det_res_max)

            x1_mask = self.calculate_mask(det_res_max, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
            x4_ = self.conv2(torch.cat([x4, x4_, x4_text_sa], dim=1))  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_sa], dim=1))  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_sa], dim=1))  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_sa], dim=1))  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]
            return outs, det_res
        else:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]
            # for item in det_feat:
            #     print(item.shape)
            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)
            # print('det_res:', det_res.shape)

            # from PIL import Image
            _, det_res = torch.max(det_res, dim=1)
            det_res = det_res.unsqueeze(1)
            # for i in range(det_res.shape[0]):
            #     img_tensor = (det_res[i].squeeze() * 255).cpu().numpy()
            #     print('img_tensor:', img_tensor.shape)
            #     img = Image.fromarray(img_tensor.astype('uint8')).convert('L')
            #     img.save('/home/yuhaiyang/mmsegmentation/imgs/'+str(self.num)+'.jpg')
            #
            #     print('x[i]:', x[i].shape)
            #     img_ = (x[i].squeeze() * 255).cpu().numpy().transpose(1,2,0)
            #     img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
            #     img_.save('/home/yuhaiyang/mmsegmentation/imgs/' + str(self.num) + '_.jpg')
            #     self.num += 1
                # exit()

            x1_mask = self.calculate_mask(det_res, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
            x4_ = self.conv2(torch.cat([x4, x4_, x4_text_sa], dim=1))  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False)  # 256 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)  # 160 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_sa], dim=1))  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_sa], dim=1))  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_sa], dim=1))  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]

            return outs

@BACKBONES.register_module()
class CascadeMixVisionTransformer_V4_1(BaseModule):
    """The backbone of Segformer.

    加了det的损失-并且把文本区域的单独做transformer encoder-残差中不加入原来x的特征

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer_V4_1, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=in_channels[i],
                embed_dims=embed_dims_i[i],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=7, stride=4, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Sequential(
            nn.Conv2d(3*num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[3]*self.embed_dims)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3*num_heads[2]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[2] * self.embed_dims)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(3*num_heads[1]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[1] * self.embed_dims)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(3*num_heads[0]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[0] * self.embed_dims)
        )

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )

        # 用于做文本区域检测
        self.out_det_1 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[0], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_2 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[1], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_3 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_4 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3]*4, self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.det_cls = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[3], 2, kernel_size=1),
        )

        # 用于全部文本区域的自注意力
        self.text_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
            )
        self.text_sa_bn_1 = nn.BatchNorm2d(embed_dims_i[0])

        self.text_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )
        self.text_sa_bn_2 = nn.BatchNorm2d(embed_dims_i[1])


        self.text_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )
        self.text_sa_bn_3 = nn.BatchNorm2d(embed_dims_i[2])


        self.text_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )
        self.text_sa_bn_4 = nn.BatchNorm2d(embed_dims_i[3])


    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer_V4_1, self).init_weights()

    def calculate_mask(self, det_gt, shape, ratio, l):
        h, w = shape
        # print(h, w)
        det_gt_q = torch.nn.functional.interpolate(det_gt.float(), size=(h, w)).long().squeeze(1).flatten(1) # b, 1, h, w
        det_gt_kv = torch.nn.functional.interpolate(det_gt.float(), size=(h//ratio, w//ratio)).long().squeeze(1).flatten(1)
        # print('det_gt_q:', set(list(numpy.array(det_gt_q.cpu()).reshape(-1))))
        # print('det_gt_kv:', set(list(numpy.array(det_gt_kv.cpu()).reshape(-1))))
        # print('det_gt_q:', det_gt_q.shape)
        # print(torch.sum(det_gt_kv))
        mask = (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1)).repeat(self.num_heads[l], 1, 1)
        mask = mask.float() * -1e10
        # return (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1))
        return mask

    def soft_argmax(self, x, beta=1e10):
        x_shape = x.shape
        x = x.view(x.shape[0], x.shape[1], -1) #[bs,c,h*w]
        L = x.shape[1]
        soft_max = nn.functional.softmax(x*beta,dim=1)
        indices = torch.arange(start=0, end=L).unsqueeze(0).unsqueeze(2).to(x.device)
        soft_argmax = soft_max * indices
        indices = soft_argmax.sum(dim=1)  #[bs,c]
        indices = indices.view(x_shape[0], x_shape[2], x_shape[3])
        return indices

    def forward(self, x, det_gt=None):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.bn1(self.conv1(x))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        # print('mit:', set(list(numpy.array(det_gt.cpu()).reshape(-1))))

        if self.training:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            # 单纯将预测的分割图引入
            # _, det_res_max = torch.max(det_res, dim=1)
            # det_res_max = det_res_max.unsqueeze(1)

            # 将soft-argmax分割图引入
            det_res_max = self.soft_argmax(det_res)
            det_res_max = det_res_max.long().unsqueeze(1)
            # print('det_res_max:', det_res_max)

            x1_mask = self.calculate_mask(det_res_max, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)


            x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
            x4_ = self.conv2(torch.cat([x4_, x4_text_sa], dim=1))  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64
            x3_ = self.conv3(torch.cat([x3_, x3_text_sa], dim=1))  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
            x2_ = self.conv4(torch.cat([x2_, x2_text_sa], dim=1))  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
            x1_ = self.conv5(torch.cat([x1_, x1_text_sa], dim=1))  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]
            return outs, det_res
        else:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]
            # for item in det_feat:
            #     print(item.shape)
            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)
            # print('det_res:', det_res.shape)

            # from PIL import Image
            _, det_res = torch.max(det_res, dim=1)
            det_res = det_res.unsqueeze(1)
            # for i in range(det_res.shape[0]):
            #     img_tensor = (det_res[i].squeeze() * 255).cpu().numpy()
            #     print('img_tensor:', img_tensor.shape)
            #     img = Image.fromarray(img_tensor.astype('uint8')).convert('L')
            #     img.save('/home/yuhaiyang/mmsegmentation/imgs/'+str(self.num)+'.jpg')
            #
            #     print('x[i]:', x[i].shape)
            #     img_ = (x[i].squeeze() * 255).cpu().numpy().transpose(1,2,0)
            #     img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
            #     img_.save('/home/yuhaiyang/mmsegmentation/imgs/' + str(self.num) + '_.jpg')
            #     self.num += 1
                # exit()

            x1_mask = self.calculate_mask(det_res, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)
            # print(x1_text_sa.shape)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)
            # print(x2_text_sa.shape)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)
            # print(x3_text_sa.shape)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)
            # print(x4_text_sa.shape)
            # quit()

            x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
            x4_ = self.conv2(torch.cat([x4_, x4_text_sa], dim=1))  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False)  # 256 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)  # 160 * 32 * 64
            x3_ = self.conv3(torch.cat([x3_, x3_text_sa], dim=1))  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
            x2_ = self.conv4(torch.cat([x2_, x2_text_sa], dim=1))  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
            x1_ = self.conv5(torch.cat([x1_, x1_text_sa], dim=1))  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]

            return outs

@BACKBONES.register_module()
class CascadeMixVisionTransformer_V5(BaseModule):
    """The backbone of Segformer.

    加了det的损失-并且把文本Instance的单独做transformer encoder

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer_V5, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=in_channels[i],
                embed_dims=embed_dims_i[i],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=7, stride=4, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Sequential(
            nn.Conv2d(3*num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[3]*self.embed_dims)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3*num_heads[2]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[2] * self.embed_dims)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(3*num_heads[1]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[1] * self.embed_dims)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(3*num_heads[0]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[0] * self.embed_dims)
        )

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )

        # 用于做文本区域检测
        self.out_det_1 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[0], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_2 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[1], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_3 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_4 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3]*4, self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.det_cls = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[3], 2, kernel_size=1),
        )

        # 用于全部文本区域的自注意力
        self.text_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
            )
        self.text_sa_bn = nn.BatchNorm2d(embed_dims_i[0])

        self.text_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )

        self.text_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )

        self.text_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer_V5, self).init_weights()

    def calculate_mask(self, det_gt, shape, ratio, l):
        h, w = shape
        det_gt_q = torch.nn.functional.interpolate(det_gt.float(), size=(h, w)).long().squeeze(1).flatten(1) # b, 1, h, w
        det_gt_kv = torch.nn.functional.interpolate(det_gt.float(), size=(h//ratio, w//ratio)).long().squeeze(1).flatten(1)
        # print('det_gt_q:', set(list(numpy.array(det_gt_q.cpu()).reshape(-1))))
        # print('det_gt_kv:', set(list(numpy.array(det_gt_kv.cpu()).reshape(-1))))
        # print('det_gt_q:', det_gt_q.shape)
        # print(torch.sum(det_gt_kv))
        mask = (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1)).repeat(self.num_heads[l], 1, 1)
        mask = mask.float() * -1e10
        # return (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1))
        return mask

    def forward(self, x, det_gt=None):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.bn1(self.conv1(x))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        # print('mit:', set(list(numpy.array(det_gt.cpu()).reshape(-1))))

        if self.training:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max.shape[0]):
                img = det_res_max[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))

            x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
            x4_ = self.conv2(torch.cat([x4, x4_, x4_text_sa], dim=1))  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_sa], dim=1))  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_sa], dim=1))  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_sa], dim=1))  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]
            return outs, det_res
        else:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max.shape[0]):
                img = det_res_max[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            # print(x1_text_sa.shape)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            # print(x2_text_sa.shape)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            # print(x3_text_sa.shape)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            # print(x4_text_sa.shape)
            # quit()

            x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
            x4_ = self.conv2(torch.cat([x4, x4_, x4_text_sa], dim=1))  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False)  # 256 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)  # 160 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_sa], dim=1))  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_sa], dim=1))  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_sa], dim=1))  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]

            return outs

@BACKBONES.register_module()
class CascadeMixVisionTransformer_V6(BaseModule):
    """The backbone of Segformer.

    加了det的损失-并且把文本Instance的单独做transformer encoder 采用soft-argmax

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer_V6, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=in_channels[i],
                embed_dims=embed_dims_i[i],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=7, stride=4, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Sequential(
            nn.Conv2d(3*num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[3]*self.embed_dims)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3*num_heads[2]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[2] * self.embed_dims)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(3*num_heads[1]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[1] * self.embed_dims)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(3*num_heads[0]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[0] * self.embed_dims)
        )

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )

        # 用于做文本区域检测
        self.out_det_1 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[0], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_2 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[1], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_3 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_4 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3]*4, self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.det_cls = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[3], 2, kernel_size=1),
        )

        # 用于全部文本区域的自注意力
        self.text_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
            )
        self.text_sa_bn_1 = nn.BatchNorm2d(embed_dims_i[0])

        self.text_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )
        self.text_sa_bn_2 = nn.BatchNorm2d(embed_dims_i[1])

        self.text_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )
        self.text_sa_bn_3 = nn.BatchNorm2d(embed_dims_i[2])

        self.text_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )
        self.text_sa_bn_4 = nn.BatchNorm2d(embed_dims_i[3])

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer_V6, self).init_weights()

    def calculate_mask(self, det_gt, shape, ratio, l):
        h, w = shape
        det_gt_q = torch.nn.functional.interpolate(det_gt.float(), size=(h, w)).long().squeeze(1).flatten(1) # b, 1, h, w
        det_gt_kv = torch.nn.functional.interpolate(det_gt.float(), size=(h//ratio, w//ratio)).long().squeeze(1).flatten(1)
        # print('det_gt_q:', set(list(numpy.array(det_gt_q.cpu()).reshape(-1))))
        # print('det_gt_kv:', set(list(numpy.array(det_gt_kv.cpu()).reshape(-1))))
        # print('det_gt_q:', det_gt_q.shape)
        # print(torch.sum(det_gt_kv))
        mask = (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1)).repeat(self.num_heads[l], 1, 1)
        mask = mask.float() * -1e10
        # return (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1))
        return mask

    def soft_argmax(self, x, beta=1e10):
        x_shape = x.shape
        x = x.view(x.shape[0], x.shape[1], -1) #[bs,c,h*w]
        L = x.shape[1]
        soft_max = nn.functional.softmax(x*beta,dim=1)
        indices = torch.arange(start=0, end=L).unsqueeze(0).unsqueeze(2).to(x.device)
        soft_argmax = soft_max * indices
        indices = soft_argmax.sum(dim=1)  #[bs,c]
        indices = indices.view(x_shape[0], x_shape[2], x_shape[3])
        return indices

    def forward(self, x, det_gt=None):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.bn1(self.conv1(x))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        # print('mit:', set(list(numpy.array(det_gt.cpu()).reshape(-1))))

        if self.training:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            det_res_max = self.soft_argmax(det_res)
            det_res_max = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max.shape[0]):
                img = det_res_max[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
            x4_ = self.conv2(torch.cat([x4, x4_, x4_text_sa], dim=1))  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_sa], dim=1))  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_sa], dim=1))  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_sa], dim=1))  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]
            return outs, det_res
        else:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max.shape[0]):
                img = det_res_max[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
            x4_ = self.conv2(torch.cat([x4, x4_, x4_text_sa], dim=1))  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False)  # 256 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)  # 160 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_sa], dim=1))  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_sa], dim=1))  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_sa], dim=1))  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]

            return outs

@BACKBONES.register_module()
class CascadeMixVisionTransformer_V7(BaseModule):
    """The backbone of Segformer.

    加了det的损失-
    并且把文本Instance的单独做transformer encoder 采用soft-argmax
    把文本区域单独做transformer encoder 采用soft-argmax

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer_V7, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=in_channels[i],
                embed_dims=embed_dims_i[i],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=7, stride=4, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Sequential(
            nn.Conv2d(3*num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[3]*self.embed_dims)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3*num_heads[2]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[2] * self.embed_dims)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(3*num_heads[1]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[1] * self.embed_dims)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(3*num_heads[0]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[0] * self.embed_dims)
        )

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )

        # 用于做文本区域检测
        self.out_det_1 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[0], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_2 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[1], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_3 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_4 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3]*4, self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.det_cls = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[3], 2, kernel_size=1),
        )

        # 文本区域的自注意力
        self.text_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
            )
        self.text_sa_bn_1 = nn.BatchNorm2d(embed_dims_i[0])

        self.text_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )
        self.text_sa_bn_2 = nn.BatchNorm2d(embed_dims_i[1])

        self.text_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )
        self.text_sa_bn_3 = nn.BatchNorm2d(embed_dims_i[2])

        self.text_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )
        self.text_sa_bn_4 = nn.BatchNorm2d(embed_dims_i[3])

        # 文本instance区域
        self.instance_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
        )
        self.instance_sa_bn_1 = nn.BatchNorm2d(embed_dims_i[0])

        self.instance_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )
        self.instance_sa_bn_2 = nn.BatchNorm2d(embed_dims_i[1])

        self.instance_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )
        self.instance_sa_bn_3 = nn.BatchNorm2d(embed_dims_i[2])

        self.instance_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )
        self.instance_sa_bn_4 = nn.BatchNorm2d(embed_dims_i[3])

        self.fuse_text_instance_1 = nn.Sequential(
            nn.Conv2d(2*embed_dims_i[0], embed_dims_i[0], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[0]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_2 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[1], embed_dims_i[1], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[1]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_3 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[2], embed_dims_i[2], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[2]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_4 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[3], embed_dims_i[3], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[3]),
            nn.Sigmoid()
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer_V7, self).init_weights()

    def calculate_mask(self, det_gt, shape, ratio, l):
        h, w = shape
        det_gt_q = torch.nn.functional.interpolate(det_gt.float(), size=(h, w)).long().squeeze(1).flatten(1) # b, 1, h, w
        det_gt_kv = torch.nn.functional.interpolate(det_gt.float(), size=(h//ratio, w//ratio)).long().squeeze(1).flatten(1)
        mask = (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1)).repeat(self.num_heads[l], 1, 1)
        mask = mask.float() * -1e10
        return mask

    def soft_argmax(self, x, beta=1e10):
        x_shape = x.shape
        x = x.view(x.shape[0], x.shape[1], -1) #[bs,c,h*w]
        L = x.shape[1]
        soft_max = nn.functional.softmax(x*beta,dim=1)
        indices = torch.arange(start=0, end=L).unsqueeze(0).unsqueeze(2).to(x.device)
        soft_argmax = soft_max * indices
        indices = soft_argmax.sum(dim=1)  #[bs,c]
        indices = indices.view(x_shape[0], x_shape[2], x_shape[3])
        return indices

    def forward(self, x, det_gt=None):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.bn1(self.conv1(x))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        # print('mit:', set(list(numpy.array(det_gt.cpu()).reshape(-1))))

        if self.training:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
            x4_ = self.conv2(torch.cat([x4, x4_, x4_text_instance], dim=1))  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_instance], dim=1))  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_instance], dim=1))  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_instance], dim=1))  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]
            return outs, det_res
        else:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            _, det_res_max = torch.max(det_res, dim=1)
            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
            x4_ = self.conv2(torch.cat([x4, x4_, x4_text_instance], dim=1))  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False)  # 256 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)  # 160 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_instance], dim=1))  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_instance], dim=1))  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_instance], dim=1))  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]

            return outs

@BACKBONES.register_module()
class CascadeMixVisionTransformer_V8(BaseModule):
    """The backbone of Segformer.

    加了det的损失-
    并且把文本Instance的单独做transformer encoder 采用soft-argmax
    把文本区域单独做transformer encoder 采用soft-argmax
    改进1 drop-rate

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer_V8, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=in_channels[i],
                embed_dims=embed_dims_i[i],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=7, stride=4, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Sequential(
            nn.Conv2d(3*num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[3]*self.embed_dims)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3*num_heads[2]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[2] * self.embed_dims)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(3*num_heads[1]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[1] * self.embed_dims)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(3*num_heads[0]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[0] * self.embed_dims)
        )

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )

        # 用于做文本区域检测
        self.out_det_1 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[0], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_2 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[1], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_3 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_4 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3]*4, self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.det_cls = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[3], 2, kernel_size=1),
        )

        # 文本区域的自注意力
        self.text_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
            drop_path_rate=0.1,
        )
        self.text_sa_bn_1 = nn.BatchNorm2d(embed_dims_i[0])

        self.text_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
            drop_path_rate=0.1,
        )
        self.text_sa_bn_2 = nn.BatchNorm2d(embed_dims_i[1])

        self.text_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
            drop_path_rate=0.1,
        )
        self.text_sa_bn_3 = nn.BatchNorm2d(embed_dims_i[2])

        self.text_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
            drop_path_rate=0.1,
        )
        self.text_sa_bn_4 = nn.BatchNorm2d(embed_dims_i[3])

        # 文本instance区域
        self.instance_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
            drop_path_rate=0.1,
        )
        self.instance_sa_bn_1 = nn.BatchNorm2d(embed_dims_i[0])

        self.instance_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
            drop_path_rate=0.1,
        )
        self.instance_sa_bn_2 = nn.BatchNorm2d(embed_dims_i[1])

        self.instance_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
            drop_path_rate=0.1,
        )
        self.instance_sa_bn_3 = nn.BatchNorm2d(embed_dims_i[2])

        self.instance_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
            drop_path_rate=0.1,
        )
        self.instance_sa_bn_4 = nn.BatchNorm2d(embed_dims_i[3])

        self.fuse_text_instance_1 = nn.Sequential(
            nn.Conv2d(2*embed_dims_i[0], embed_dims_i[0], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[0]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_2 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[1], embed_dims_i[1], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[1]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_3 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[2], embed_dims_i[2], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[2]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_4 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[3], embed_dims_i[3], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[3]),
            nn.Sigmoid()
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer_V8, self).init_weights()

    def calculate_mask(self, det_gt, shape, ratio, l):
        h, w = shape
        det_gt_q = torch.nn.functional.interpolate(det_gt.float(), size=(h, w)).long().squeeze(1).flatten(1) # b, 1, h, w
        det_gt_kv = torch.nn.functional.interpolate(det_gt.float(), size=(h//ratio, w//ratio)).long().squeeze(1).flatten(1)
        mask = (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1)).repeat(self.num_heads[l], 1, 1)
        mask = mask.float() * -1e10
        return mask

    def soft_argmax(self, x, beta=1e10):
        x_shape = x.shape
        x = x.view(x.shape[0], x.shape[1], -1) #[bs,c,h*w]
        L = x.shape[1]
        soft_max = nn.functional.softmax(x*beta,dim=1)
        indices = torch.arange(start=0, end=L).unsqueeze(0).unsqueeze(2).to(x.device)
        soft_argmax = soft_max * indices
        indices = soft_argmax.sum(dim=1)  #[bs,c]
        indices = indices.view(x_shape[0], x_shape[2], x_shape[3])
        return indices

    def forward(self, x, det_gt=None):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.bn1(self.conv1(x))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        # print('mit:', set(list(numpy.array(det_gt.cpu()).reshape(-1))))

        if self.training:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
            x4_ = self.conv2(torch.cat([x4, x4_, x4_text_instance], dim=1))  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_instance], dim=1))  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_instance], dim=1))  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_instance], dim=1))  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]
            return outs, det_res
        else:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            _, det_res_max = torch.max(det_res, dim=1)
            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
            x4_ = self.conv2(torch.cat([x4, x4_, x4_text_instance], dim=1))  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False)  # 256 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)  # 160 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_instance], dim=1))  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_instance], dim=1))  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_instance], dim=1))  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]

            return outs

@BACKBONES.register_module()
class CascadeMixVisionTransformer_V9(BaseModule):
    """The backbone of Segformer.

    加了det的损失-
    并且把文本Instance的单独做transformer encoder 采用soft-argmax
    把文本区域单独做transformer encoder 采用soft-argmax
    改进2 用layernorm

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer_V9, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=in_channels[i],
                embed_dims=embed_dims_i[i],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=7, stride=4, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Sequential(
            nn.Conv2d(3*num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[3]*self.embed_dims)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3*num_heads[2]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[2] * self.embed_dims)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(3*num_heads[1]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[1] * self.embed_dims)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(3*num_heads[0]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[0] * self.embed_dims)
        )

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )

        # 用于做文本区域检测
        self.out_det_1 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[0], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_2 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[1], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_3 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_4 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3]*4, self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.det_cls = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[3], 2, kernel_size=1),
        )

        # 文本区域的自注意力
        self.text_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
            # drop_path_rate=0.1,
        )
        self.text_sa_bn_1 = nn.LayerNorm(embed_dims_i[0])

        self.text_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
            # drop_path_rate=0.1,
        )
        self.text_sa_bn_2 = nn.LayerNorm(embed_dims_i[1])

        self.text_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
            # drop_path_rate=0.1,
        )
        self.text_sa_bn_3 = nn.LayerNorm(embed_dims_i[2])

        self.text_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
            # drop_path_rate=0.1,
        )
        self.text_sa_bn_4 = nn.LayerNorm(embed_dims_i[3])

        # 文本instance区域
        self.instance_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
            # drop_path_rate=0.1,
        )
        self.instance_sa_bn_1 = nn.LayerNorm(embed_dims_i[0])

        self.instance_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
            # drop_path_rate=0.1,
        )
        self.instance_sa_bn_2 = nn.LayerNorm(embed_dims_i[1])

        self.instance_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
            # drop_path_rate=0.1,
        )
        self.instance_sa_bn_3 = nn.LayerNorm(embed_dims_i[2])

        self.instance_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
            # drop_path_rate=0.1,
        )
        self.instance_sa_bn_4 = nn.LayerNorm(embed_dims_i[3])

        self.fuse_text_instance_1 = nn.Sequential(
            nn.Conv2d(2*embed_dims_i[0], embed_dims_i[0], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[0]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_2 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[1], embed_dims_i[1], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[1]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_3 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[2], embed_dims_i[2], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[2]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_4 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[3], embed_dims_i[3], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[3]),
            nn.Sigmoid()
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer_V9, self).init_weights()

    def calculate_mask(self, det_gt, shape, ratio, l):
        h, w = shape
        det_gt_q = torch.nn.functional.interpolate(det_gt.float(), size=(h, w)).long().squeeze(1).flatten(1) # b, 1, h, w
        det_gt_kv = torch.nn.functional.interpolate(det_gt.float(), size=(h//ratio, w//ratio)).long().squeeze(1).flatten(1)
        mask = (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1)).repeat(self.num_heads[l], 1, 1)
        mask = mask.float() * -1e10
        return mask

    def soft_argmax(self, x, beta=1e10):
        x_shape = x.shape
        x = x.view(x.shape[0], x.shape[1], -1) #[bs,c,h*w]
        L = x.shape[1]
        soft_max = nn.functional.softmax(x*beta,dim=1)
        indices = torch.arange(start=0, end=L).unsqueeze(0).unsqueeze(2).to(x.device)
        soft_argmax = soft_max * indices
        indices = soft_argmax.sum(dim=1)  #[bs,c]
        indices = indices.view(x_shape[0], x_shape[2], x_shape[3])
        return indices

    def forward(self, x, det_gt=None):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.bn1(self.conv1(x))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        # print('mit:', set(list(numpy.array(det_gt.cpu()).reshape(-1))))

        if self.training:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
            x4_ = self.conv2(torch.cat([x4, x4_, x4_text_instance], dim=1))  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_instance], dim=1))  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_instance], dim=1))  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_instance], dim=1))  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]
            return outs, det_res
        else:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            _, det_res_max = torch.max(det_res, dim=1)
            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = through_layer(self.layers[3], x4)  # 256 * 16 * 32
            x4_ = self.conv2(torch.cat([x4, x4_, x4_text_instance], dim=1))  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False)  # 256 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)  # 160 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_instance], dim=1))  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_instance], dim=1))  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_instance], dim=1))  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]

            return outs

@BACKBONES.register_module()
class CascadeMixVisionTransformer_V10(BaseModule):
    """The backbone of Segformer.

    加了det的损失-
    并且把文本Instance的单独做transformer encoder 采用soft-argmax
    把文本区域单独做transformer encoder 采用soft-argmax
    修改特征融合方式

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer_V10, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        # in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=embed_dims_i[i],
                embed_dims=embed_dims_i[i],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=7, stride=4, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[3]*self.embed_dims)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*num_heads[2]*self.embed_dims + num_heads[3]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[2] * self.embed_dims)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(2*num_heads[1]*self.embed_dims + num_heads[2]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[1] * self.embed_dims)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(2*num_heads[0]*self.embed_dims + num_heads[1]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[0] * self.embed_dims)
        )

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )

        # 用于做文本区域检测
        self.out_det_1 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[0], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_2 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[1], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_3 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_4 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3]*4, self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.det_cls = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[3], 2, kernel_size=1),
        )

        # 文本区域的自注意力
        self.text_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
            )
        self.text_sa_bn_1 = nn.BatchNorm2d(embed_dims_i[0])

        self.text_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )
        self.text_sa_bn_2 = nn.BatchNorm2d(embed_dims_i[1])

        self.text_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )
        self.text_sa_bn_3 = nn.BatchNorm2d(embed_dims_i[2])

        self.text_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )
        self.text_sa_bn_4 = nn.BatchNorm2d(embed_dims_i[3])

        # 文本instance区域
        self.instance_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
        )
        self.instance_sa_bn_1 = nn.BatchNorm2d(embed_dims_i[0])

        self.instance_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )
        self.instance_sa_bn_2 = nn.BatchNorm2d(embed_dims_i[1])

        self.instance_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )
        self.instance_sa_bn_3 = nn.BatchNorm2d(embed_dims_i[2])

        self.instance_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )
        self.instance_sa_bn_4 = nn.BatchNorm2d(embed_dims_i[3])

        self.fuse_text_instance_1 = nn.Sequential(
            nn.Conv2d(2*embed_dims_i[0], embed_dims_i[0], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[0]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_2 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[1], embed_dims_i[1], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[1]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_3 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[2], embed_dims_i[2], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[2]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_4 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[3], embed_dims_i[3], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[3]),
            nn.Sigmoid()
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer_V10, self).init_weights()

    def calculate_mask(self, det_gt, shape, ratio, l):
        h, w = shape
        det_gt_q = torch.nn.functional.interpolate(det_gt.float(), size=(h, w)).long().squeeze(1).flatten(1) # b, 1, h, w
        det_gt_kv = torch.nn.functional.interpolate(det_gt.float(), size=(h//ratio, w//ratio)).long().squeeze(1).flatten(1)
        mask = (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1)).repeat(self.num_heads[l], 1, 1)
        mask = mask.float() * -1e10
        return mask

    def soft_argmax(self, x, beta=1e10):
        x_shape = x.shape
        x = x.view(x.shape[0], x.shape[1], -1) #[bs,c,h*w]
        L = x.shape[1]
        soft_max = nn.functional.softmax(x*beta,dim=1)
        indices = torch.arange(start=0, end=L).unsqueeze(0).unsqueeze(2).to(x.device)
        soft_argmax = soft_max * indices
        indices = soft_argmax.sum(dim=1)  #[bs,c]
        indices = indices.view(x_shape[0], x_shape[2], x_shape[3])
        return indices

    def forward(self, x, det_gt=None):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.bn1(self.conv1(x))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        # print('mit:', set(list(numpy.array(det_gt.cpu()).reshape(-1))))

        if self.training:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = self.conv2(torch.cat([x4, x4_text_instance], dim=1))  # 256 * 16 * 32
            x4_ = through_layer(self.layers[3], x4_)  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_instance], dim=1))  # 160 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_instance], dim=1))  # 64 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_instance], dim=1))  # 32 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]
            return outs, det_res
        else:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            _, det_res_max = torch.max(det_res, dim=1)
            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = self.conv2(torch.cat([x4, x4_text_instance], dim=1))  # 256 * 16 * 32
            x4_ = through_layer(self.layers[3], x4_)  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False)  # 256 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_instance], dim=1))  # 160 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_instance], dim=1))  # 64 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_instance], dim=1))  # 32 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]

            return outs

@BACKBONES.register_module()
class CascadeMixVisionTransformer_V11(BaseModule):
    """The backbone of Segformer.

    加了det的损失-
    并且把文本Instance的单独做transformer encoder 采用soft-argmax
    把文本区域单独做transformer encoder 采用soft-argmax
    修改特征融合方式-不融合连接

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer_V11, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        # in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=embed_dims_i[i],
                embed_dims=embed_dims_i[i],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=7, stride=4, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[3]*self.embed_dims)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_heads[2]*self.embed_dims + num_heads[3]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[2] * self.embed_dims)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_heads[1]*self.embed_dims + num_heads[2]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[1] * self.embed_dims)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(num_heads[0]*self.embed_dims + num_heads[1]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[0] * self.embed_dims)
        )

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )

        # 用于做文本区域检测
        self.out_det_1 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[0], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_2 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[1], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_3 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_4 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3]*4, self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.det_cls = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[3], 2, kernel_size=1),
        )

        # 文本区域的自注意力
        self.text_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
            )
        self.text_sa_bn_1 = nn.BatchNorm2d(embed_dims_i[0])

        self.text_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )
        self.text_sa_bn_2 = nn.BatchNorm2d(embed_dims_i[1])

        self.text_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )
        self.text_sa_bn_3 = nn.BatchNorm2d(embed_dims_i[2])

        self.text_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )
        self.text_sa_bn_4 = nn.BatchNorm2d(embed_dims_i[3])

        # 文本instance区域
        self.instance_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
        )
        self.instance_sa_bn_1 = nn.BatchNorm2d(embed_dims_i[0])

        self.instance_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )
        self.instance_sa_bn_2 = nn.BatchNorm2d(embed_dims_i[1])

        self.instance_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )
        self.instance_sa_bn_3 = nn.BatchNorm2d(embed_dims_i[2])

        self.instance_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )
        self.instance_sa_bn_4 = nn.BatchNorm2d(embed_dims_i[3])

        self.fuse_text_instance_1 = nn.Sequential(
            nn.Conv2d(2*embed_dims_i[0], embed_dims_i[0], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[0]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_2 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[1], embed_dims_i[1], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[1]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_3 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[2], embed_dims_i[2], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[2]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_4 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[3], embed_dims_i[3], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[3]),
            nn.Sigmoid()
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer_V11, self).init_weights()

    def calculate_mask(self, det_gt, shape, ratio, l):
        h, w = shape
        det_gt_q = torch.nn.functional.interpolate(det_gt.float(), size=(h, w)).long().squeeze(1).flatten(1) # b, 1, h, w
        det_gt_kv = torch.nn.functional.interpolate(det_gt.float(), size=(h//ratio, w//ratio)).long().squeeze(1).flatten(1)
        mask = (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1)).repeat(self.num_heads[l], 1, 1)
        mask = mask.float() * -1e10
        return mask

    def soft_argmax(self, x, beta=1e10):
        x_shape = x.shape
        x = x.view(x.shape[0], x.shape[1], -1) #[bs,c,h*w]
        L = x.shape[1]
        soft_max = nn.functional.softmax(x*beta,dim=1)
        indices = torch.arange(start=0, end=L).unsqueeze(0).unsqueeze(2).to(x.device)
        soft_argmax = soft_max * indices
        indices = soft_argmax.sum(dim=1)  #[bs,c]
        indices = indices.view(x_shape[0], x_shape[2], x_shape[3])
        return indices

    def forward(self, x, det_gt=None):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.bn1(self.conv1(x))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        # print('mit:', set(list(numpy.array(det_gt.cpu()).reshape(-1))))

        if self.training:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = self.conv2(torch.cat([x4_text_instance], dim=1))  # 256 * 16 * 32
            x4_ = through_layer(self.layers[3], x4_)  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
            x3_ = self.conv3(torch.cat([x3_, x3_text_instance], dim=1))  # 160 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = self.conv4(torch.cat([x2_, x2_text_instance], dim=1))  # 64 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = self.conv5(torch.cat([x1_, x1_text_instance], dim=1))  # 32 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]
            return outs, det_res
        else:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            _, det_res_max = torch.max(det_res, dim=1)
            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = self.conv2(torch.cat([x4_text_instance], dim=1))  # 256 * 16 * 32
            x4_ = through_layer(self.layers[3], x4_)  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False)  # 256 * 32 * 64
            x3_ = self.conv3(torch.cat([x3_, x3_text_instance], dim=1))  # 160 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = self.conv4(torch.cat([x2_, x2_text_instance], dim=1))  # 64 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = self.conv5(torch.cat([x1_, x1_text_instance], dim=1))  # 32 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]

            return outs

@BACKBONES.register_module()
class CascadeMixVisionTransformer_V12(BaseModule):
    """The backbone of Segformer.

    加了det的损失-
    并且把文本Instance的单独做transformer encoder 采用soft-argmax
    把文本区域单独做transformer encoder 采用soft-argmax
    修改特征融合方式-LN

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer_V12, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        # in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=embed_dims_i[i],
                embed_dims=embed_dims_i[i],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=7, stride=4, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[3]*self.embed_dims)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*num_heads[2]*self.embed_dims + num_heads[3]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[2] * self.embed_dims)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(2*num_heads[1]*self.embed_dims + num_heads[2]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[1] * self.embed_dims)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(2*num_heads[0]*self.embed_dims + num_heads[1]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[0] * self.embed_dims)
        )

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )

        # 用于做文本区域检测
        self.out_det_1 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[0], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_2 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[1], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_3 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_4 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3]*4, self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.det_cls = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[3], 2, kernel_size=1),
        )

        # 文本区域的自注意力
        self.text_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
            )
        self.text_sa_bn_1 = nn.LayerNorm(embed_dims_i[0])

        self.text_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )
        self.text_sa_bn_2 = nn.LayerNorm(embed_dims_i[1])

        self.text_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )
        self.text_sa_bn_3 = nn.LayerNorm(embed_dims_i[2])

        self.text_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )
        self.text_sa_bn_4 = nn.LayerNorm(embed_dims_i[3])

        # 文本instance区域
        self.instance_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
        )
        self.instance_sa_bn_1 = nn.LayerNorm(embed_dims_i[0])

        self.instance_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )
        self.instance_sa_bn_2 = nn.LayerNorm(embed_dims_i[1])

        self.instance_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )
        self.instance_sa_bn_3 = nn.LayerNorm(embed_dims_i[2])

        self.instance_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )
        self.instance_sa_bn_4 = nn.LayerNorm(embed_dims_i[3])

        self.fuse_text_instance_1 = nn.Sequential(
            nn.Conv2d(2*embed_dims_i[0], embed_dims_i[0], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[0]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_2 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[1], embed_dims_i[1], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[1]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_3 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[2], embed_dims_i[2], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[2]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_4 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[3], embed_dims_i[3], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[3]),
            nn.Sigmoid()
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer_V12, self).init_weights()

    def calculate_mask(self, det_gt, shape, ratio, l):
        h, w = shape
        det_gt_q = torch.nn.functional.interpolate(det_gt.float(), size=(h, w)).long().squeeze(1).flatten(1) # b, 1, h, w
        det_gt_kv = torch.nn.functional.interpolate(det_gt.float(), size=(h//ratio, w//ratio)).long().squeeze(1).flatten(1)
        mask = (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1)).repeat(self.num_heads[l], 1, 1)
        mask = mask.float() * -1e10
        return mask

    def soft_argmax(self, x, beta=1e10):
        x_shape = x.shape
        x = x.view(x.shape[0], x.shape[1], -1) #[bs,c,h*w]
        L = x.shape[1]
        soft_max = nn.functional.softmax(x*beta,dim=1)
        indices = torch.arange(start=0, end=L).unsqueeze(0).unsqueeze(2).to(x.device)
        soft_argmax = soft_max * indices
        indices = soft_argmax.sum(dim=1)  #[bs,c]
        indices = indices.view(x_shape[0], x_shape[2], x_shape[3])
        return indices

    def forward(self, x, det_gt=None):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.bn1(self.conv1(x))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        # print('mit:', set(list(numpy.array(det_gt.cpu()).reshape(-1))))

        if self.training:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = self.conv2(torch.cat([x4, x4_text_instance], dim=1))  # 256 * 16 * 32
            x4_ = through_layer(self.layers[3], x4_)  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_instance], dim=1))  # 160 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_instance], dim=1))  # 64 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_instance], dim=1))  # 32 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]
            return outs, det_res
        else:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            _, det_res_max = torch.max(det_res, dim=1)
            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = self.conv2(torch.cat([x4, x4_text_instance], dim=1))  # 256 * 16 * 32
            x4_ = through_layer(self.layers[3], x4_)  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False)  # 256 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_instance], dim=1))  # 160 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_instance], dim=1))  # 64 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_instance], dim=1))  # 32 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]

            return outs

@BACKBONES.register_module()
class CascadeMixVisionTransformer_V13(BaseModule):
    """The backbone of Segformer.

    加了det的损失-
    并且把文本Instance的单独做transformer encoder 采用soft-argmax
    把文本区域单独做transformer encoder 采用soft-argmax
    修改特征融合方式-修改最初的特征提取

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer_V13, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        # in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=embed_dims_i[i],
                embed_dims=embed_dims_i[i],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[3]*self.embed_dims)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*num_heads[2]*self.embed_dims + num_heads[3]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[2] * self.embed_dims)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(2*num_heads[1]*self.embed_dims + num_heads[2]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[1] * self.embed_dims)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(2*num_heads[0]*self.embed_dims + num_heads[1]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[0] * self.embed_dims)
        )

        self.layer0 = nn.Sequential(
            ResNetBlock(self.embed_dims, self.embed_dims, stride=2),
            ResNetBlock(self.embed_dims, self.embed_dims),
            ResNetBlock(self.embed_dims, self.embed_dims, stride=2),
            ResNetBlock(self.embed_dims, self.embed_dims),
        )

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )

        # 用于做文本区域检测
        self.out_det_1 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[0], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_2 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[1], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_3 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_4 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3]*4, self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.det_cls = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[3], 2, kernel_size=1),
        )

        # 文本区域的自注意力
        self.text_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
            )
        self.text_sa_bn_1 = nn.BatchNorm2d(embed_dims_i[0])

        self.text_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )
        self.text_sa_bn_2 = nn.BatchNorm2d(embed_dims_i[1])

        self.text_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )
        self.text_sa_bn_3 = nn.BatchNorm2d(embed_dims_i[2])

        self.text_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )
        self.text_sa_bn_4 = nn.BatchNorm2d(embed_dims_i[3])

        # 文本instance区域
        self.instance_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
        )
        self.instance_sa_bn_1 = nn.BatchNorm2d(embed_dims_i[0])

        self.instance_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )
        self.instance_sa_bn_2 = nn.BatchNorm2d(embed_dims_i[1])

        self.instance_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )
        self.instance_sa_bn_3 = nn.BatchNorm2d(embed_dims_i[2])

        self.instance_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )
        self.instance_sa_bn_4 = nn.BatchNorm2d(embed_dims_i[3])

        self.fuse_text_instance_1 = nn.Sequential(
            nn.Conv2d(2*embed_dims_i[0], embed_dims_i[0], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[0]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_2 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[1], embed_dims_i[1], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[1]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_3 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[2], embed_dims_i[2], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[2]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_4 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[3], embed_dims_i[3], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[3]),
            nn.Sigmoid()
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer_V13, self).init_weights()

    def calculate_mask(self, det_gt, shape, ratio, l):
        h, w = shape
        det_gt_q = torch.nn.functional.interpolate(det_gt.float(), size=(h, w)).long().squeeze(1).flatten(1) # b, 1, h, w
        det_gt_kv = torch.nn.functional.interpolate(det_gt.float(), size=(h//ratio, w//ratio)).long().squeeze(1).flatten(1)
        mask = (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1)).repeat(self.num_heads[l], 1, 1)
        mask = mask.float() * -1e10
        return mask

    def soft_argmax(self, x, beta=1e10):
        x_shape = x.shape
        x = x.view(x.shape[0], x.shape[1], -1) #[bs,c,h*w]
        L = x.shape[1]
        soft_max = nn.functional.softmax(x*beta,dim=1)
        indices = torch.arange(start=0, end=L).unsqueeze(0).unsqueeze(2).to(x.device)
        soft_argmax = soft_max * indices
        indices = soft_argmax.sum(dim=1)  #[bs,c]
        indices = indices.view(x_shape[0], x_shape[2], x_shape[3])
        return indices

    def forward(self, x, det_gt=None):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.layer0(self.bn1(self.conv1(x)))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        # print('mit:', set(list(numpy.array(det_gt.cpu()).reshape(-1))))

        if self.training:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = self.conv2(torch.cat([x4, x4_text_instance], dim=1))  # 256 * 16 * 32
            x4_ = through_layer(self.layers[3], x4_)  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_instance], dim=1))  # 160 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_instance], dim=1))  # 64 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_instance], dim=1))  # 32 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]
            return outs, det_res
        else:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            _, det_res_max = torch.max(det_res, dim=1)
            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = self.conv2(torch.cat([x4, x4_text_instance], dim=1))  # 256 * 16 * 32
            x4_ = through_layer(self.layers[3], x4_)  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False)  # 256 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_instance], dim=1))  # 160 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_instance], dim=1))  # 64 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_instance], dim=1))  # 32 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]

            return outs

@BACKBONES.register_module()
class CascadeMixVisionTransformer_V14(BaseModule):
    """The backbone of Segformer.

    加了det的损失-
    并且把文本Instance的单独做transformer encoder 采用soft-argmax
    把文本区域单独做transformer encoder 采用soft-argmax
    修改特征融合方式-全部LN

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(CascadeMixVisionTransformer_V14, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.inchannels = in_channels
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        # in_channels = [self.embed_dims*2, self.embed_dims*5, self.embed_dims*8, self.embed_dims*8]
        embed_dims_i = [self.embed_dims*1, self.embed_dims*2, self.embed_dims*5, self.embed_dims*8]
        for i, num_layer in enumerate(num_layers):
            patch_embed = PatchEmbed(
                in_channels=embed_dims_i[i],
                embed_dims=embed_dims_i[i],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i[i],
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i[i],
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i[i])[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.conv1 = nn.Conv2d(self.inchannels, self.embed_dims, kernel_size=7, stride=4, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.embed_dims)
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*num_heads[3]*self.embed_dims, num_heads[3]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[3]*self.embed_dims)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*num_heads[2]*self.embed_dims + num_heads[3]*self.embed_dims, num_heads[2]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[2] * self.embed_dims)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(2*num_heads[1]*self.embed_dims + num_heads[2]*self.embed_dims, num_heads[1]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[1] * self.embed_dims)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(2*num_heads[0]*self.embed_dims + num_heads[1]*self.embed_dims, num_heads[0]*self.embed_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_heads[0] * self.embed_dims)
        )

        self.layer1 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[0], self.embed_dims*num_heads[1], stride=2),
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[1])
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[1], self.embed_dims*num_heads[2], stride=2),
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[2])
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], stride=2),
            ResNetBlock(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3])
        )

        # 用于做文本区域检测
        self.out_det_1 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[0], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_2 = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[1], self.embed_dims * num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_3 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[2], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.out_det_4 = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3], self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims*num_heads[3]*4, self.embed_dims*num_heads[3], kernel_size=1),
            nn.BatchNorm2d(self.embed_dims * num_heads[3])
        )
        self.det_cls = nn.Sequential(
            nn.Conv2d(self.embed_dims * num_heads[3], 2, kernel_size=1),
        )

        # 文本区域的自注意力
        self.text_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
            )
        self.text_sa_bn_1 = nn.BatchNorm2d(embed_dims_i[0])

        self.text_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )
        self.text_sa_bn_2 = nn.BatchNorm2d(embed_dims_i[1])

        self.text_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )
        self.text_sa_bn_3 = nn.BatchNorm2d(embed_dims_i[2])

        self.text_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )
        self.text_sa_bn_4 = nn.BatchNorm2d(embed_dims_i[3])

        # 文本instance区域
        self.instance_sa_1 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[0],
            num_heads=num_heads[0],
            feedforward_channels=mlp_ratio * embed_dims_i[0],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0],
        )
        self.instance_sa_bn_1 = nn.BatchNorm2d(embed_dims_i[0])

        self.instance_sa_2 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[1],
            num_heads=num_heads[1],
            feedforward_channels=mlp_ratio * embed_dims_i[1],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1],
        )
        self.instance_sa_bn_2 = nn.BatchNorm2d(embed_dims_i[1])

        self.instance_sa_3 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[2],
            num_heads=num_heads[2],
            feedforward_channels=mlp_ratio * embed_dims_i[2],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2],
        )
        self.instance_sa_bn_3 = nn.BatchNorm2d(embed_dims_i[2])

        self.instance_sa_4 = TransformerEncoderLayer(
            embed_dims=embed_dims_i[3],
            num_heads=num_heads[3],
            feedforward_channels=mlp_ratio * embed_dims_i[3],
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3],
        )
        self.instance_sa_bn_4 = nn.BatchNorm2d(embed_dims_i[3])

        self.fuse_text_instance_1 = nn.Sequential(
            nn.Conv2d(2*embed_dims_i[0], embed_dims_i[0], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[0]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_2 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[1], embed_dims_i[1], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[1]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_3 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[2], embed_dims_i[2], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[2]),
            nn.Sigmoid()
        )

        self.fuse_text_instance_4 = nn.Sequential(
            nn.Conv2d(2 * embed_dims_i[3], embed_dims_i[3], kernel_size=1),
            nn.BatchNorm2d(embed_dims_i[3]),
            nn.Sigmoid()
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(CascadeMixVisionTransformer_V14, self).init_weights()

    def calculate_mask(self, det_gt, shape, ratio, l):
        h, w = shape
        det_gt_q = torch.nn.functional.interpolate(det_gt.float(), size=(h, w)).long().squeeze(1).flatten(1) # b, 1, h, w
        det_gt_kv = torch.nn.functional.interpolate(det_gt.float(), size=(h//ratio, w//ratio)).long().squeeze(1).flatten(1)
        mask = (det_gt_q.unsqueeze(2) == det_gt_kv.unsqueeze(1)).repeat(self.num_heads[l], 1, 1)
        mask = mask.float() * -1e10
        return mask

    def soft_argmax(self, x, beta=1e10):
        x_shape = x.shape
        x = x.view(x.shape[0], x.shape[1], -1) #[bs,c,h*w]
        L = x.shape[1]
        soft_max = nn.functional.softmax(x*beta,dim=1)
        indices = torch.arange(start=0, end=L).unsqueeze(0).unsqueeze(2).to(x.device)
        soft_argmax = soft_max * indices
        indices = soft_argmax.sum(dim=1)  #[bs,c]
        indices = indices.view(x_shape[0], x_shape[2], x_shape[3])
        return indices

    def forward(self, x, det_gt=None):

        def through_layer(layer, x):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            return x

        x1 = self.bn1(self.conv1(x))  # 32 * 128 * 256
        x2 = self.layer1(x1)  # 64 * 64 * 128
        x3 = self.layer2(x2)  # 160 * 32 * 64
        x4 = self.layer3(x3)  # 256 * 16 * 32

        # print('mit:', set(list(numpy.array(det_gt.cpu()).reshape(-1))))

        if self.training:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = self.conv2(torch.cat([x4, x4_text_instance], dim=1))  # 256 * 16 * 32
            x4_ = through_layer(self.layers[3], x4_)  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False) # 256 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_instance], dim=1))  # 160 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)   # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_instance], dim=1))  # 64 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_instance], dim=1))  # 32 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]
            return outs, det_res
        else:
            det_feat = [
                resize(self.out_det_1(x1), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_2(x2), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_3(x3), size=x1.shape[2:], mode='bilinear', align_corners=False),
                resize(self.out_det_4(x4), size=x1.shape[2:], mode='bilinear', align_corners=False),
            ]

            det_fuse_feat = self.fusion_conv(torch.cat(det_feat, dim=1))
            det_res = self.det_cls(det_fuse_feat)

            _, det_res_max = torch.max(det_res, dim=1)
            det_res_max = self.soft_argmax(det_res)
            # instance 部分
            det_res_max_instance = det_res_max.long()
            # _, det_res_max = torch.max(det_res, dim=1)
            det_res_max_new = []
            for i in range(det_res_max_instance.shape[0]):
                img = det_res_max_instance[i].byte().cpu().detach().numpy()
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = img.copy()
                counter = 1
                for i in range(len(contours)):
                    cv2.drawContours(img_copy, contours, i, counter, -1)
                    counter += 1
                det_res_max_new.append(torch.Tensor(img_copy).unsqueeze(0))
            det_res_max_instance = torch.cat(det_res_max_new, dim=0).unsqueeze(1).cuda()

            x1_mask = self.calculate_mask(det_res_max_instance, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_instance, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_instance, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_instance, x4.shape[2:], self.sr_ratios[3], 3)

            x1_instance_sa = self.instance_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_instance_sa = nlc_to_nchw(x1_instance_sa, (x1.shape[2], x1.shape[3]))
            x1_instance_sa = self.instance_sa_bn_1(x1_instance_sa)

            x2_instance_sa = self.instance_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_instance_sa = nlc_to_nchw(x2_instance_sa, (x2.shape[2], x2.shape[3]))
            x2_instance_sa = self.instance_sa_bn_2(x2_instance_sa)

            x3_instance_sa = self.instance_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_instance_sa = nlc_to_nchw(x3_instance_sa, (x3.shape[2], x3.shape[3]))
            x3_instance_sa = self.instance_sa_bn_3(x3_instance_sa)

            x4_instance_sa = self.instance_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_instance_sa = nlc_to_nchw(x4_instance_sa, (x4.shape[2], x4.shape[3]))
            x4_instance_sa = self.instance_sa_bn_4(x4_instance_sa)

            # text 部分
            det_res_max_text = det_res_max.long().unsqueeze(1)

            x1_mask = self.calculate_mask(det_res_max_text, x1.shape[2:], self.sr_ratios[0], 0)
            x2_mask = self.calculate_mask(det_res_max_text, x2.shape[2:], self.sr_ratios[1], 1)
            x3_mask = self.calculate_mask(det_res_max_text, x3.shape[2:], self.sr_ratios[2], 2)
            x4_mask = self.calculate_mask(det_res_max_text, x4.shape[2:], self.sr_ratios[3], 3)

            x1_text_sa = self.text_sa_1(x1.flatten(2).transpose(1, 2), (x1.shape[2], x1.shape[3]), x1_mask)
            x1_text_sa = nlc_to_nchw(x1_text_sa, (x1.shape[2], x1.shape[3]))
            x1_text_sa = self.text_sa_bn_1(x1_text_sa)

            x2_text_sa = self.text_sa_2(x2.flatten(2).transpose(1, 2), (x2.shape[2], x2.shape[3]), x2_mask)
            x2_text_sa = nlc_to_nchw(x2_text_sa, (x2.shape[2], x2.shape[3]))
            x2_text_sa = self.text_sa_bn_2(x2_text_sa)

            x3_text_sa = self.text_sa_3(x3.flatten(2).transpose(1, 2), (x3.shape[2], x3.shape[3]), x3_mask)
            x3_text_sa = nlc_to_nchw(x3_text_sa, (x3.shape[2], x3.shape[3]))
            x3_text_sa = self.text_sa_bn_3(x3_text_sa)

            x4_text_sa = self.text_sa_4(x4.flatten(2).transpose(1, 2), (x4.shape[2], x4.shape[3]), x4_mask)
            x4_text_sa = nlc_to_nchw(x4_text_sa, (x4.shape[2], x4.shape[3]))
            x4_text_sa = self.text_sa_bn_4(x4_text_sa)

            x1_text_instance_w = self.fuse_text_instance_1(torch.cat([x1_text_sa, x1_instance_sa], dim=1))
            x1_text_instance = x1_text_instance_w * x1_text_sa + (1 - x1_text_instance_w) * x1_instance_sa

            x2_text_instance_w = self.fuse_text_instance_2(torch.cat([x2_text_sa, x2_instance_sa], dim=1))
            x2_text_instance = x2_text_instance_w * x2_text_sa + (1 - x2_text_instance_w) * x2_instance_sa

            x3_text_instance_w = self.fuse_text_instance_3(torch.cat([x3_text_sa, x3_instance_sa], dim=1))
            x3_text_instance = x3_text_instance_w * x3_text_sa + (1 - x3_text_instance_w) * x3_instance_sa

            x4_text_instance_w = self.fuse_text_instance_4(torch.cat([x4_text_sa, x4_instance_sa], dim=1))
            x4_text_instance = x4_text_instance_w * x4_text_sa + (1 - x4_text_instance_w) * x4_instance_sa

            x4_ = self.conv2(torch.cat([x4, x4_text_instance], dim=1))  # 256 * 16 * 32
            x4_ = through_layer(self.layers[3], x4_)  # 256 * 16 * 32

            x3_ = resize(input=x4_, size=x3.shape[2:], mode='bilinear', align_corners=False)  # 256 * 32 * 64
            x3_ = self.conv3(torch.cat([x3, x3_, x3_text_instance], dim=1))  # 160 * 32 * 64
            x3_ = through_layer(self.layers[2], x3_)  # 160 * 32 * 64

            x2_ = resize(input=x3_, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 160 * 64 * 128
            x2_ = self.conv4(torch.cat([x2, x2_, x2_text_instance], dim=1))  # 64 * 64 * 128
            x2_ = through_layer(self.layers[1], x2_)  # 64 * 64 * 128

            x1_ = resize(input=x2_, size=x1.shape[2:], mode='bilinear', align_corners=False)  # 64 * 128 * 256
            x1_ = self.conv5(torch.cat([x1, x1_, x1_text_instance], dim=1))  # 32 * 128 * 256
            x1_ = through_layer(self.layers[0], x1_)  # 32 * 128 * 256

            outs = [x1_, x2_, x3_, x4_]

            return outs