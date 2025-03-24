# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# --------------------------------------------------------
# TinyViT 模型架构
# 版权所有 (c) 2022 Microsoft
# 改编自 LeViT 和 Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# 构建 TinyViT 模型
# --------------------------------------------------------

import itertools
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from ultralytics.nn.modules import LayerNorm2d
from ultralytics.utils.instance import to_2tuple


class Conv2d_BN(torch.nn.Sequential):
    """
    一个顺序容器，先执行 2D 卷积，再执行批量归一化。

    属性:
        c (torch.nn.Conv2d): 2D 卷积层。
        bn (torch.nn.BatchNorm2d): 批量归一化层。

    方法:
        __init__: 用指定的参数初始化 Conv2d_BN。

    参数:
        a (int): 输入通道数。
        b (int): 输出通道数。
        ks (int): 卷积核大小。默认值为 1。
        stride (int): 卷积步幅。默认值为 1。
        pad (int): 卷积填充。默认值为 0。
        dilation (int): 卷积膨胀因子。默认值为 1。
        groups (int): 卷积的组数。默认值为 1。
        bn_weight_init (float): 批量归一化权重的初始值。默认值为 1。

    示例:
        >>> conv_bn = Conv2d_BN(3, 64, ks=3, stride=1, pad=1)
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output = conv_bn(input_tensor)
        >>> print(output.shape)
    """

    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        """用 2D 卷积和批量归一化初始化一个顺序容器。"""
        super().__init__()
        self.add_module("c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)


class PatchEmbed(nn.Module):
    """
    将图像嵌入为补丁，并将它们投影到指定的嵌入维度。

    属性:
        patches_resolution (Tuple[int, int]): 嵌入后的补丁分辨率。
        num_patches (int): 补丁总数。
        in_chans (int): 输入通道数。
        embed_dim (int): 嵌入维度。
        seq (nn.Sequential): 用于补丁嵌入的卷积和激活层的序列。

    方法:
        forward: 通过补丁嵌入序列处理输入张量。

    示例:
        >>> import torch
        >>> patch_embed = PatchEmbed(in_chans=3, embed_dim=96, resolution=224, activation=nn.GELU)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = patch_embed(x)
        >>> print(output.shape)
    """

    def __init__(self, in_chans, embed_dim, resolution, activation):
        """通过卷积层初始化补丁嵌入，用于将图像转换为补丁并进行投影。"""
        super().__init__()
        img_size: Tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        """通过补丁嵌入序列处理输入张量，将图像转换为补丁嵌入。"""
        return self.seq(x)


class MBConv(nn.Module):
    """
    移动倒置瓶颈卷积层（MBConv），是 EfficientNet 架构的一部分。

    属性:
        in_chans (int): 输入通道数。
        hidden_chans (int): 隐藏通道数。
        out_chans (int): 输出通道数。
        conv1 (Conv2d_BN): 第一个卷积层。
        act1 (nn.Module): 第一个激活函数。
        conv2 (Conv2d_BN): 深度卷积层。
        act2 (nn.Module): 第二个激活函数。
        conv3 (Conv2d_BN): 最后的卷积层。
        act3 (nn.Module): 第三个激活函数。
        drop_path (nn.Module): Drop path 层（推理时为恒等映射）。

    方法:
        forward: 执行 MBConv 层的前向传播。

    示例:
        >>> in_chans, out_chans = 32, 64
        >>> mbconv = MBConv(in_chans, out_chans, expand_ratio=4, activation=nn.ReLU, drop_path=0.1)
        >>> x = torch.randn(1, in_chans, 56, 56)
        >>> output = mbconv(x)
        >>> print(output.shape)
        torch.Size([1, 64, 56, 56])
    """

    def __init__(self, in_chans, out_chans, expand_ratio, activation, drop_path):
        """用指定的输入/输出通道、扩展比例和激活函数初始化 MBConv 层。"""
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans, ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()

        self.conv3 = Conv2d_BN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        # 注意：`DropPath` 仅在训练时需要。
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()

    def forward(self, x):
        """实现 MBConv 的前向传播，应用卷积和跳跃连接。"""
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        return self.act3(x)

1
class PatchMerging(nn.Module):
    """
    在特征图中合并相邻的补丁并投影到一个新的维度。

    该类实现了一个补丁合并操作，通过组合空间信息并调整特征维度来实现。它使用一系列的卷积层和批归一化来完成这一操作。

    属性说明:
        input_resolution (Tuple[int, int]): 输入特征图的分辨率（高度，宽度）。
        dim (int): 输入特征图的维度。
        out_dim (int): 合并和投影后的输出维度。
        act (nn.Module): 卷积之间使用的激活函数。
        conv1 (Conv2d_BN): 第一个卷积层，用于维度投影。
        conv2 (Conv2d_BN): 第二个卷积层，用于空间合并。
        conv3 (Conv2d_BN): 第三个卷积层，用于最终的投影。

    方法说明:
        forward: 将补丁合并操作应用于输入张量。

    示例用法:
        >>> input_resolution = (56, 56)
        >>> patch_merging = PatchMerging(input_resolution, dim=64, out_dim=128, activation=nn.ReLU)
        >>> x = torch.randn(4, 64, 56, 56)
        >>> output = patch_merging(x)
        >>> print(output.shape)
    """

    def __init__(self, input_resolution, dim, out_dim, activation):
        """初始化 PatchMerging 模块，用于在特征图中合并和投影相邻的补丁。"""
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c = 1 if out_dim in {320, 448, 576} else 2
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        """将补丁合并和维度投影应用于输入特征图。"""
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x.flatten(2).transpose(1, 2)


class ConvLayer(nn.Module):
    """
    特征提取的卷积层，包含多个 MobileNetV3 风格的倒残差卷积（MBConv）。

    该层可以选择性地对输出应用下采样操作，并支持梯度检查点技术。

    属性说明:
        dim (int): 输入和输出的维度。
        input_resolution (Tuple[int, int]): 输入图像的分辨率。
        depth (int): MBConv 层的数量。
        use_checkpoint (bool): 是否使用梯度检查点以节省内存。
        blocks (nn.ModuleList): MBConv 层的列表。
        downsample (Optional[Callable]): 用于下采样输出的函数。

    方法说明:
        forward: 通过卷积层处理输入。

    示例用法:
        >>> input_tensor = torch.randn(1, 64, 56, 56)
        >>> conv_layer = ConvLayer(64, (56, 56), depth=3, activation=nn.ReLU)
        >>> output = conv_layer(input_tensor)
        >>> print(output.shape)
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        activation,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        out_dim=None,
        conv_expand_ratio=4.0,
    ):
        """
        初始化 ConvLayer，配置给定的维度和设置。

        该层由多个 MobileNetV3 风格的倒残差卷积（MBConv）组成，并可以选择性地对输出应用下采样。

        参数说明:
            dim (int): 输入和输出的维度。
            input_resolution (Tuple[int, int]): 输入图像的分辨率。
            depth (int): MBConv 层的数量。
            activation (Callable): 每个卷积后的激活函数。
            drop_path (float | List[float]): Drop path 的比率。可以是单一浮动值，也可以是一个浮动值列表（对应每个MBConv）。
            downsample (Optional[Callable]): 用于下采样输出的函数。若为 None，则跳过下采样。
            use_checkpoint (bool): 是否使用梯度检查点以节省内存。
            out_dim (Optional[int]): 输出的维度。若为 None，则与 `dim` 相同。
            conv_expand_ratio (float): MBConv 层的扩展比例。

        示例用法:
            >>> input_tensor = torch.randn(1, 64, 56, 56)
            >>> conv_layer = ConvLayer(64, (56, 56), depth=3, activation=nn.ReLU)
            >>> output = conv_layer(input_tensor)
            >>> print(output.shape)
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 构建 MBConv 层
        self.blocks = nn.ModuleList(
            [
                MBConv(
                    dim,
                    dim,
                    conv_expand_ratio,
                    activation,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )

        # 补丁合并层
        self.downsample = (
            None
            if downsample is None
            else downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        )

    def forward(self, x):
        """通过卷积层处理输入，应用 MBConv 层和可选的下采样操作。"""
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x if self.downsample is None else self.downsample(x)
1

class Mlp(nn.Module):
    """
    多层感知器 (MLP) 模块，用于 transformer 架构。

    该模块应用层归一化、两个全连接层（中间有激活函数），并使用丢弃层。通常用于基于 Transformer 的架构中。

    属性：
        norm (nn.LayerNorm): 应用于输入的层归一化。
        fc1 (nn.Linear): 第一个全连接层。
        fc2 (nn.Linear): 第二个全连接层。
        act (nn.Module): 应用于第一个全连接层后的激活函数。
        drop (nn.Dropout): 应用于激活函数后的丢弃层。

    方法：
        forward: 对输入张量应用 MLP 操作。

    示例:
        >>> import torch
        >>> from torch import nn
        >>> mlp = Mlp(in_features=256, hidden_features=512, out_features=256, act_layer=nn.GELU, drop=0.1)
        >>> x = torch.randn(32, 100, 256)
        >>> output = mlp(x)
        >>> print(output.shape)
        torch.Size([32, 100, 256])
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        """初始化一个具有可配置输入、隐藏和输出维度的多层感知器。"""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """应用 MLP 操作：层归一化、全连接层、激活函数和丢弃层。"""
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(torch.nn.Module):
    """
    具有空间感知和可训练注意力偏置的多头注意力模块。

    该模块实现了一个多头注意力机制，支持空间感知，基于空间分辨率应用注意力偏置。它包括每个唯一的空间位置偏移的可训练注意力偏置。

    属性：
        num_heads (int): 注意力头的数量。
        scale (float): 注意力分数的缩放因子。
        key_dim (int): 键（key）和查询（query）的维度。
        nh_kd (int): 注意力头数和键维度的乘积。
        d (int): 值（value）向量的维度。
        dh (int): 值维度和注意力头数的乘积。
        attn_ratio (float): 影响值向量维度的注意力比率。
        norm (nn.LayerNorm): 应用于输入的层归一化。
        qkv (nn.Linear): 用于计算查询、键和值投影的线性层。
        proj (nn.Linear): 用于最终投影的线性层。
        attention_biases (nn.Parameter): 可学习的注意力偏置。
        attention_bias_idxs (Tensor): 注意力偏置的索引。
        ab (Tensor): 推理过程中缓存的注意力偏置，训练时删除。

    方法：
        train: 设置模块为训练模式并处理 'ab' 属性。
        forward: 执行注意力机制的前向传播。

    示例:
        >>> attn = Attention(dim=256, key_dim=64, num_heads=8, resolution=(14, 14))
        >>> x = torch.randn(1, 196, 256)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 196, 256])
    """

    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=(14, 14),
    ):
        """
        初始化具有空间感知的多头注意力模块。

        该模块实现了一个多头注意力机制，支持空间感知，基于空间分辨率应用注意力偏置。它包括每个唯一的空间位置偏移的可训练注意力偏置。

        参数：
            dim (int): 输入和输出的维度。
            key_dim (int): 键和查询的维度。
            num_heads (int): 注意力头的数量，默认是 8。
            attn_ratio (float): 注意力比率，影响值向量的维度，默认是 4。
            resolution (Tuple[int, int]): 输入特征图的空间分辨率，默认是 (14, 14)。

        异常：
            AssertionError: 如果 'resolution' 不是长度为 2 的元组。

        示例:
            >>> attn = Attention(dim=256, key_dim=64, num_heads=8, resolution=(14, 14))
            >>> x = torch.randn(1, 196, 256)
            >>> output = attn(x)
            >>> print(output.shape)
            torch.Size([1, 196, 256])
        """
        super().__init__()

        assert isinstance(resolution, tuple) and len(resolution) == 2, "'resolution' 参数不是长度为 2 的元组"
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(N, N), persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        """执行具有空间感知和可训练注意力偏置的多头注意力机制。"""
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x
        """应用具有空间感知和可训练注意力偏差的多头自注意力。"""
        B, N, _ = x.shape  # B, N, C

        # 归一化
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        self.ab = self.ab.to(self.attention_biases.device)

        attn = (q @ k.transpose(-2, -1)) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        return self.proj(x)


class TinyViTBlock(nn.Module):
    """
    TinyViT 块，应用自注意力和局部卷积于输入。

    该块是 TinyViT 架构的关键组件，结合自注意力机制和局部卷积，
    高效处理输入特征。

    属性：
        dim (int): 输入和输出的维度。
        input_resolution (Tuple[int, int]): 输入特征图的空间分辨率。
        num_heads (int): 注意力头的数量。
        window_size (int): 注意力窗口的大小。
        mlp_ratio (float): MLP 隐藏层维度与嵌入维度的比例。
        drop_path (nn.Module): 随机深度层，推理时为恒等函数。
        attn (Attention): 自注意力模块。
        mlp (Mlp): 多层感知机模块。
        local_conv (Conv2d_BN): 深度卷积的局部卷积层。

    方法：
        forward: 通过 TinyViT 块处理输入。
        extra_repr: 返回包含块参数的额外信息的字符串。

    示例：
        >>> input_tensor = torch.randn(1, 196, 192)
        >>> block = TinyViTBlock(dim=192, input_resolution=(14, 14), num_heads=3)
        >>> output = block(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 196, 192])
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        local_conv_size=3,
        activation=nn.GELU,
    ):
        """
        初始化 TinyViT 块，包含自注意力和局部卷积。

        该块是 TinyViT 架构的关键组件，结合自注意力机制和局部卷积，
        高效处理输入特征。

        参数：
            dim (int): 输入和输出特征的维度。
            input_resolution (Tuple[int, int]): 输入特征图的空间分辨率（高度，宽度）。
            num_heads (int): 注意力头的数量。
            window_size (int): 注意力窗口的大小。必须大于 0。
            mlp_ratio (float): MLP 隐藏层维度与嵌入维度的比例。
            drop (float): Dropout 比例。
            drop_path (float): 随机深度比率。
            local_conv_size (int): 局部卷积的卷积核大小。
            activation (torch.nn.Module): MLP 的激活函数。

        引发：
            AssertionError: 如果 window_size 不大于 0。
            AssertionError: 如果 dim 不能被 num_heads 整除。

        示例：
            >>> block = TinyViTBlock(dim=192, input_resolution=(14, 14), num_heads=3)
            >>> input_tensor = torch.randn(1, 196, 192)
            >>> output = block(input_tensor)
            >>> print(output.shape)
            torch.Size([1, 196, 192])
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, "window_size 必须大于 0"
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        # 注意：`DropPath` 仅在训练时需要。
        # self.drop_path = DropPath(drop_path) 如果 drop_path > 0. 否则为 nn.Identity()
        self.drop_path = nn.Identity()

        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        """应用自注意力、局部卷积和 MLP 操作到输入张量上。"""
        h, w = self.input_resolution
        b, hw, c = x.shape  # batch, height*width, channels
        assert hw == h * w, "输入特征的尺寸不正确"
        res_x = x
        if h == self.window_size and w == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(b, h, w, c)
            pad_b = (self.window_size - h % self.window_size) % self.window_size
            pad_r = (self.window_size - w % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = h + pad_b, w + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size

            # 窗口分割
            x = (
                x.view(b, nH, self.window_size, nW, self.window_size, c)
                .transpose(2, 3)
                .reshape(b * nH * nW, self.window_size * self.window_size, c)
            )
            x = self.attn(x)

            # 窗口反转
            x = x.view(b, nH, nW, self.window_size, self.window_size, c).transpose(2, 3).reshape(b, pH, pW, c)
            if padding:
                x = x[:, :h, :w].contiguous()

            x = x.view(b, hw, c)

        x = res_x + self.drop_path(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.local_conv(x)
        x = x.view(b, c, hw).transpose(1, 2)

        return x + self.drop_path(self.mlp(x))

    def extra_repr(self) -> str:
        """
        返回 TinyViTBlock 参数的字符串表示。

        该方法提供了一个格式化的字符串，包含 TinyViTBlock 的关键信息，包括其维度、输入分辨率、注意力头数、窗口大小和 MLP 比率。

        返回:
            (str): 包含块参数的格式化字符串。

        示例:
            >>> block = TinyViTBlock(dim=192, input_resolution=(14, 14), num_heads=3, window_size=7, mlp_ratio=4.0)
            >>> print(block.extra_repr())
            dim=192, input_resolution=(14, 14), num_heads=3, window_size=7, mlp_ratio=4.0
        """
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
        )


class BasicLayer(nn.Module):
    """
    TinyViT 架构中的基本层，表示一个阶段的层。

    该类表示 TinyViT 模型中的单个层，由多个 TinyViT 块和一个可选的下采样操作组成。

    属性:
        dim (int): 输入和输出特征的维度。
        input_resolution (Tuple[int, int]): 输入特征图的空间分辨率。
        depth (int): 该层中的 TinyViT 块数。
        use_checkpoint (bool): 是否使用梯度检查点以节省内存。
        blocks (nn.ModuleList): 组成该层的 TinyViT 块列表。
        downsample (nn.Module | None): 层末的下采样层（如果指定）。

    方法:
        forward: 通过该层的块和可选的下采样处理输入。
        extra_repr: 返回包含该层参数的字符串以供打印。

    示例:
        >>> input_tensor = torch.randn(1, 3136, 192)
        >>> layer = BasicLayer(dim=192, input_resolution=(56, 56), depth=2, num_heads=3, window_size=7)
        >>> output = layer(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 784, 384])
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        local_conv_size=3,
        activation=nn.GELU,
        out_dim=None,
    ):
        """
        初始化 TinyViT 架构中的 BasicLayer。

        该层由多个 TinyViT 块和一个可选的下采样操作组成，旨在处理特定分辨率和维度的特征图。

        参数:
            dim (int): 输入和输出特征的维度。
            input_resolution (Tuple[int, int]): 输入特征图的空间分辨率（高度，宽度）。
            depth (int): 该层中的 TinyViT 块数。
            num_heads (int): 每个 TinyViT 块中的注意力头数。
            window_size (int): 用于注意力计算的局部窗口大小。
            mlp_ratio (float): MLP 隐藏维度与嵌入维度的比率。
            drop (float): Dropout 比例。
            drop_path (float | List[float]): 随机深度率。可以是一个浮动值，或每个块的浮动列表。
            downsample (nn.Module | None): 该层末的下采样层。None 表示跳过下采样。
            use_checkpoint (bool): 是否使用梯度检查点以节省内存。
            local_conv_size (int): 每个 TinyViT 块中的局部卷积的卷积核大小。
            activation (nn.Module): 用于 MLP 的激活函数。
            out_dim (int | None): 下采样后的输出维度。如果为 None，则与 `dim` 相同。

        抛出:
            ValueError: 如果 `drop_path` 为列表且其长度与 `depth` 不匹配。

        示例:
            >>> layer = BasicLayer(dim=96, input_resolution=(56, 56), depth=2, num_heads=3, window_size=7)
            >>> x = torch.randn(1, 56 * 56, 96)
            >>> output = layer(x)
            >>> print(output.shape)
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 构建块
        self.blocks = nn.ModuleList(
            [
                TinyViTBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    local_conv_size=local_conv_size,
                    activation=activation,
                )
                for i in range(depth)
            ]
        )

        # 补丁合并层
        self.downsample = (
            None
            if downsample is None
            else downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        )

    def forward(self, x):
        """通过 TinyViT 块和可选的下采样处理输入。"""
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x if self.downsample is None else self.downsample(x)

    def extra_repr(self) -> str:
        """返回包含该层参数的字符串，以供打印。"""
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TinyViT(nn.Module):
    """
    TinyViT: 一种紧凑的视觉 Transformer 架构，用于高效的图像分类和特征提取。

    该类实现了 TinyViT 模型，它结合了视觉 Transformer 和卷积神经网络的元素，以提高视觉任务的效率和性能。

    属性说明:
        img_size (int): 输入图像的大小。
        num_classes (int): 分类的类别数。
        depths (List[int]): 每个阶段的块数量。
        num_layers (int): 网络中的总层数。
        mlp_ratio (float): MLP 隐藏维度与嵌入维度的比率。
        patch_embed (PatchEmbed): 用于补丁嵌入的模块。
        patches_resolution (Tuple[int, int]): 嵌入补丁的分辨率。
        layers (nn.ModuleList): 网络层的列表。
        norm_head (nn.LayerNorm): 用于分类头的层归一化。
        head (nn.Linear): 最终分类的线性层。
        neck (nn.Sequential): 用于特征精炼的 Neck 模块。

    方法说明:
        set_layer_lr_decay: 设置按层次的学习率衰减。
        _init_weights: 初始化线性层和归一化层的权重。
        no_weight_decay_keywords: 返回不使用权重衰减的参数关键字。
        forward_features: 通过特征提取层处理输入。
        forward: 通过整个网络进行前向传播。

    示例用法:
        >>> model = TinyViT(img_size=224, num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> features = model.forward_features(x)
        >>> print(features.shape)
        torch.Size([1, 256, 64, 64])
    """

    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dims=(96, 192, 384, 768),
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_sizes=(7, 7, 14, 7),
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=1.0,
    ):
        """
        初始化 TinyViT 模型。

        该构造函数设置了 TinyViT 架构，包括补丁嵌入、多个注意力和卷积块层，以及分类头。

        参数说明:
            img_size (int): 输入图像的大小，默认为 224。
            in_chans (int): 输入通道的数量，默认为 3。
            num_classes (int): 分类的类别数，默认为 1000。
            embed_dims (Tuple[int, int, int, int]): 每个阶段的嵌入维度，默认为 (96, 192, 384, 768)。
            depths (Tuple[int, int, int, int]): 每个阶段的块数量，默认为 (2, 2, 6, 2)。
            num_heads (Tuple[int, int, int, int]): 每个阶段的注意力头数，默认为 (3, 6, 12, 24)。
            window_sizes (Tuple[int, int, int, int]): 每个阶段的窗口大小，默认为 (7, 7, 14, 7)。
            mlp_ratio (float): MLP 隐藏维度与嵌入维度的比率，默认为 4.0。
            drop_rate (float): Dropout 比率，默认为 0.0。
            drop_path_rate (float): 随机深度比率，默认为 0.1。
            use_checkpoint (bool): 是否使用检查点以节省内存，默认为 False。
            mbconv_expand_ratio (float): MBConv 层的扩展比例，默认为 4.0。
            local_conv_size (int): 局部卷积的核大小，默认为 3。
            layer_lr_decay (float): 层次学习率衰减因子，默认为 1.0。

        示例用法:
            >>> model = TinyViT(img_size=224, num_classes=1000)
            >>> x = torch.randn(1, 3, 224, 224)
            >>> output = model(x)
            >>> print(output.shape)
            torch.Size([1, 1000])
        """
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(
            in_chans=in_chans, embed_dim=embed_dims[0], resolution=img_size, activation=activation
        )

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 随机深度衰减规则

        # 构建层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(
                dim=embed_dims[i_layer],
                input_resolution=(
                    patches_resolution[0] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                    patches_resolution[1] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                ),
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                out_dim=embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
                activation=activation,
            )
            if i_layer == 0:
                layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs,
                )
            self.layers.append(layer)

        # 分类头
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        # 初始化权重
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dims[-1],
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

    def set_layer_lr_decay(self, layer_lr_decay):
        """为 TinyViT 模型设置按层次的学习率衰减。"""
        decay_rate = layer_lr_decay

        # 层 -> 块（深度）
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

        def _set_lr_scale(m, scale):
            """根据层的深度为模型中的每一层设置学习率缩放。"""
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            """检查模块的参数中是否存在学习率缩放属性。"""
            for p in m.parameters():
                assert hasattr(p, "lr_scale"), p.param_name

        self.apply(_check_lr_scale)

    @staticmethod
    def _init_weights(m):
        """初始化 TinyViT 模型中的线性层和归一化层的权重。"""
        if isinstance(m, nn.Linear):
            # 注意：这个初始化仅在训练时需要。
            # trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """返回一组不应该使用权重衰减的参数关键词。"""
        return {"attention_biases"}

    def forward_features(self, x):
        """通过特征提取层处理输入，返回空间特征。"""
        x = self.patch_embed(x)  # x 输入形状为 (N, C, H, W)

        x = self.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        batch, _, channel = x.shape
        x = x.view(batch, self.patches_resolution[0] // 4, self.patches_resolution[1] // 4, channel)
        x = x.permute(0, 3, 1, 2)
        return self.neck(x)

    def forward(self, x):
        """执行 TinyViT 模型的前向传播，从输入图像中提取特征。"""
        return self.forward_features(x)

    def set_imgsz(self, imgsz=[1024, 1024]):
        """
        设置图像尺寸，使模型兼容不同的图像尺寸。

        参数：
            imgsz (Tuple[int, int]): 输入图像的尺寸。
        """
        imgsz = [s // 4 for s in imgsz]
        self.patches_resolution = imgsz
        for i, layer in enumerate(self.layers):
            input_resolution = (
                imgsz[0] // (2 ** (i - 1 if i == 3 else i)),
                imgsz[1] // (2 ** (i - 1 if i == 3 else i)),
            )
            layer.input_resolution = input_resolution
            if layer.downsample is not None:
                layer.downsample.input_resolution = input_resolution
            if isinstance(layer, BasicLayer):
                for b in layer.blocks:
                    b.input_resolution = input_resolution
