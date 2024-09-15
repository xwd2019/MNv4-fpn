import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import make_divisible


def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU6())
    return conv


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, act=False, squeeze_excitation=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=3, stride=stride))
        if squeeze_excitation:
            self.block.add_module('conv_3x3',
                                  conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, act=act))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 start_dw_kernel_size,
                 middle_dw_kernel_size,
                 middle_dw_downsample,
                 stride,
                 expand_ratio
                 ):
        """An inverted bottleneck block with optional depthwises.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
        """
        super().__init__()
        # Starting depthwise conv.
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        # Expansion with 1x1 convs.
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
                                      groups=expand_filters)
        # Projection with 1x1 convs.
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)

        # Ending depthwise conv.
        # this not used
        # _end_dw_kernel_size = 0
        # self._end_dw = conv_2d(oup, oup, kernel_size=_end_dw_kernel_size, stride=stride, groups=inp, act=False)

    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        x = self._proj_conv(x)
        # print("_proj_conv", x.shape)
        return x


class MultiQueryAttentionLayerWithDownSampling(nn.Module):
    def __init__(self, inp, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides,
                 dw_kernel_size=3, dropout=0.0):
        """Multi Query Attention with spatial downsampling.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py

        3 parameters are introduced for the spatial downsampling:
        1. kv_strides: downsampling factor on Key and Values only.
        2. query_h_strides: vertical strides on Query only.
        3. query_w_strides: horizontal strides on Query only.

        This is an optimized version.
        1. Projections in Attention is explict written out as 1x1 Conv2D.
        2. Additional reshapes are introduced to bring a up to 3x speed up.
        """
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.dw_kernel_size = dw_kernel_size
        self.dropout = dropout

        self.head_dim = key_dim // num_heads

        if self.query_h_strides > 1 or self.query_w_strides > 1:
            self._query_downsampling_norm = nn.BatchNorm2d(inp)
        self._query_proj = conv_2d(inp, num_heads * key_dim, 1, 1, norm=False, act=False)

        if self.kv_strides > 1:
            self._key_dw_conv = conv_2d(inp, inp, dw_kernel_size, kv_strides, groups=inp, norm=True, act=False)
            self._value_dw_conv = conv_2d(inp, inp, dw_kernel_size, kv_strides, groups=inp, norm=True, act=False)
        self._key_proj = conv_2d(inp, key_dim, 1, 1, norm=False, act=False)
        self._value_proj = conv_2d(inp, key_dim, 1, 1, norm=False, act=False)

        self._output_proj = conv_2d(num_heads * key_dim, inp, 1, 1, norm=False, act=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size, seq_length, _, _ = x.size()
        if self.query_h_strides > 1 or self.query_w_strides > 1:
            q = F.avg_pool2d(self.query_h_stride, self.query_w_stride)
            q = self._query_downsampling_norm(q)
            q = self._query_proj(q)
        else:
            q = self._query_proj(x)
        px = q.size(2)
        q = q.view(batch_size, self.num_heads, -1, self.key_dim)  # [batch_size, num_heads, seq_length, key_dim]

        if self.kv_strides > 1:
            k = self._key_dw_conv(x)
            k = self._key_proj(k)
            v = self._value_dw_conv(x)
            v = self._value_proj(v)
        else:
            k = self._key_proj(x)
            v = self._value_proj(x)
        k = k.view(batch_size, 1, self.key_dim, -1)  # [batch_size, 1, key_dim, seq_length]
        v = v.view(batch_size, 1, -1, self.key_dim)  # [batch_size, 1, seq_length, key_dim]

        # calculate attn score
        attn_score = torch.matmul(q, k) / (self.head_dim ** 0.5)
        attn_score = self.dropout(attn_score)
        attn_score = F.softmax(attn_score, dim=-1)

        context = torch.matmul(attn_score, v)
        context = context.view(batch_size, self.num_heads * self.key_dim, px, px)
        output = self._output_proj(context)
        return output


class MNV4LayerScale(nn.Module):
    def __init__(self, inp, init_value):
        """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py

        As used in MobileNetV4.

        Attributes:
            init_value (float): value to initialize the diagonal matrix of LayerScale.
        """
        super().__init__()
        self.init_value = init_value
        self._gamma = nn.Parameter(self.init_value * torch.ones(inp, 1, 1))

    def forward(self, x):
        return x * self._gamma


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(
            self,
            inp,
            num_heads,
            key_dim,
            value_dim,
            query_h_strides,
            query_w_strides,
            kv_strides,
            use_layer_scale,
            use_multi_query,
            use_residual=True
    ):
        super().__init__()
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.use_layer_scale = use_layer_scale
        self.use_multi_query = use_multi_query
        self.use_residual = use_residual

        self._input_norm = nn.BatchNorm2d(inp)
        if self.use_multi_query:
            self.multi_query_attention = MultiQueryAttentionLayerWithDownSampling(
                inp, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides
            )
        else:
            self.multi_head_attention = nn.MultiheadAttention(inp, num_heads, kdim=key_dim)

        if self.use_layer_scale:
            self.layer_scale_init_value = 1e-5
            self.layer_scale = MNV4LayerScale(inp, self.layer_scale_init_value)

    def forward(self, x):
        # Not using CPE, skipped
        # input norm
        shortcut = x
        x = self._input_norm(x)
        # multi query
        if self.use_multi_query:
            x = self.multi_query_attention(x)
        else:
            x = self.multi_head_attention(x, x)
        # layer scale
        if self.use_layer_scale:
            x = self.layer_scale(x)
        # use residual
        if self.use_residual:
            x = x + shortcut
        return x


class MN4(nn.Module):
    def __init__(self):
        # MobileNetV4ConvMedium is implemented
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6())
        self.layer1 = nn.Sequential(InvertedResidual(32, 48, 2, 4.0, True))
        self.layer2 = nn.Sequential(UniversalInvertedBottleneckBlock(48, 80, 3, 5, True, 2, 4),
                                    UniversalInvertedBottleneckBlock(80, 80, 3, 3, True, 1, 2))
        self.layer3 = nn.Sequential(UniversalInvertedBottleneckBlock(80, 160, 3, 5, True, 2, 6),
                                    UniversalInvertedBottleneckBlock(160, 160, 3, 3, True, 1, 4),
                                    UniversalInvertedBottleneckBlock(160, 160, 3, 3, True, 1, 4),
                                    UniversalInvertedBottleneckBlock(160, 160, 3, 5, True, 1, 4),
                                    UniversalInvertedBottleneckBlock(160, 160, 3, 3, True, 1, 4),
                                    UniversalInvertedBottleneckBlock(160, 160, 3, 0, True, 1, 4),
                                    UniversalInvertedBottleneckBlock(160, 160, 0, 0, True, 1, 2),
                                    UniversalInvertedBottleneckBlock(160, 160, 3, 0, True, 1, 4))
        self.layer4 = nn.Sequential(UniversalInvertedBottleneckBlock(160, 256, 5, 5, True, 2, 6),
                                    UniversalInvertedBottleneckBlock(256, 256, 5, 5, True, 1, 4),
                                    UniversalInvertedBottleneckBlock(256, 256, 3, 5, True, 1, 4),
                                    UniversalInvertedBottleneckBlock(256, 256, 3, 5, True, 1, 4),
                                    UniversalInvertedBottleneckBlock(256, 256, 0, 0, True, 1, 4),
                                    UniversalInvertedBottleneckBlock(256, 256, 3, 0, True, 1, 4),
                                    UniversalInvertedBottleneckBlock(256, 256, 3, 5, True, 1, 2),
                                    UniversalInvertedBottleneckBlock(256, 256, 5, 5, True, 1, 4),
                                    UniversalInvertedBottleneckBlock(256, 256, 0, 0, True, 1, 4),
                                    UniversalInvertedBottleneckBlock(256, 256, 0, 0, True, 1, 4),
                                    UniversalInvertedBottleneckBlock(256, 256, 5, 0, True, 1, 2))

    def forward(self, x):  # torch.Size([N, 3, 480, 640])
        x0 = self.conv0(x)  # torch.Size([N, 32, 240, 320])
        x1 = self.layer1(x0)  # torch.Size([N, 48, 120, 160])
        x2 = self.layer2(x1)  # torch.Size([N, 80, 60, 80])
        seg_out1 = x2  # torch.Size([N, 80, 60, 80])
        x3 = self.layer3(x2)  # torch.Size([N, 160, 30, 40])
        seg_out2 = x3  # torch.Size([N, 160, 30, 40])
        x4 = self.layer4(x3)  # torch.Size([N, 256, 15, 20])
        seg_out3 = x4  # torch.Size([N, 256, 15, 20])

        return seg_out1, seg_out2, seg_out3
