# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy
import math
from functools import partial
from typing import Any, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ultralytics.nn.modules import MLP, LayerNorm2d, MLPBlock

from .transformer import Attention, TwoWayAttentionBlock, TwoWayTransformer
from .utils import add_decomposed_rel_pos, apply_rotary_enc, compute_axial_cis, window_partition, window_unpartition


class DropPath(nn.Module):
    """
    实现神经网络中的随机深度正则化（Stochastic Depth）以用于训练。

    属性：
        drop_prob (float): 在训练过程中丢弃路径的概率。
        scale_by_keep (bool): 是否根据保持概率对输出进行缩放。

    方法：
        forward: 在训练期间应用随机深度到输入张量，并可以选择性地进行缩放。

    示例：
        >>> drop_path = DropPath(drop_prob=0.2, scale_by_keep=True)
        >>> x = torch.randn(32, 64, 224, 224)
        >>> output = drop_path(x)
    """

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        """初始化 DropPath 模块，用于训练中的随机深度正则化。"""
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """在训练期间对输入张量应用随机深度，并可以选择性地进行缩放。"""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class MaskDownSampler(nn.Module):
    """
    用于有效处理输入掩码的掩码下采样和嵌入模块。

    此类实现了一个掩码下采样器，逐步减少输入掩码的空间维度，同时使用卷积层、层归一化和激活函数
    扩展其通道维度。

    属性：
        encoder (nn.Sequential): 一个包含卷积层、层归一化和激活函数的顺序容器，用于下采样和嵌入掩码。

    方法：
        forward: 下采样并将输入掩码编码为嵌入维度的通道。

    示例：
        >>> mask_downsampler = MaskDownSampler(embed_dim=256, kernel_size=4, stride=4, padding=0, total_stride=16)
        >>> input_mask = torch.randn(1, 1, 256, 256)
        >>> output = mask_downsampler(input_mask)
        >>> print(output.shape)
        torch.Size([1, 256, 16, 16])
    """

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=nn.GELU,
    ):
        """初始化一个掩码下采样模块，用于逐步下采样并扩展通道。"""
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))

    def forward(self, x):
        """通过卷积层和 LayerNorm2d 对输入掩码进行下采样并编码为嵌入维度的通道。"""
        return self.encoder(x)


class CXBlock(nn.Module):
    """
    ConvNeXt 块用于卷积神经网络中的高效特征提取。

    此块实现了 ConvNeXt 架构的修改版，提供了在特征提取方面的更高性能和灵活性。

    属性：
        dwconv (nn.Conv2d): 深度卷积或标准的 2D 卷积层。
        norm (LayerNorm2d): 应用到通道上的层归一化。
        pwconv1 (nn.Linear): 第一个点卷积，使用线性层实现。
        act (nn.GELU): GELU 激活函数。
        pwconv2 (nn.Linear): 第二个点卷积，使用线性层实现。
        gamma (nn.Parameter | None): 用于层缩放的可学习缩放参数。
        drop_path (nn.Module): DropPath 层，用于随机深度正则化。

    方法：
        forward: 通过 ConvNeXt 块处理输入张量。

    示例：
        >>> import torch
        >>> x = torch.randn(1, 64, 56, 56)
        >>> block = CXBlock(dim=64, kernel_size=7, padding=3)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 64, 56, 56])
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        """
        初始化 ConvNeXt 块，用于卷积神经网络中的高效特征提取。

        此块实现了 ConvNeXt 架构的修改版，提供了在特征提取方面的更高性能和灵活性。

        参数：
            dim (int): 输入通道数。
            kernel_size (int): 卷积核的大小。
            padding (int): 卷积的填充大小。
            drop_path (float): 随机深度率。
            layer_scale_init_value (float): 层缩放的初始值。
            use_dwconv (bool): 是否使用深度卷积。

        示例：
            >>> block = CXBlock(dim=64, kernel_size=7, padding=3)
            >>> x = torch.randn(1, 64, 32, 32)
            >>> output = block(x)
            >>> print(output.shape)
            torch.Size([1, 64, 32, 32])
        """
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # 深度卷积
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 点卷积/1x1 卷积，用线性层实现
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """将 ConvNeXt 块的操作应用于输入张量，包括卷积和残差连接。"""
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Fuser(nn.Module):
    """
    通过多个层融合特征的模块。

    该类将一系列相同的层应用于输入张量，且可选择性地首先对输入进行投影。

    属性：
        proj (nn.Module): 可选的输入投影层。如果不需要投影，则为恒等映射。
        layers (nn.ModuleList): 一系列要顺序应用的相同层。

    方法：
        forward: 将 fuser 应用于输入张量。

    示例：
        >>> layer = CXBlock(dim=256)
        >>> fuser = Fuser(layer, num_layers=3, dim=256, input_projection=True)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = fuser(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        """
        初始化 Fuser 模块，通过多个层进行特征融合。

        该模块创建多个相同的层，并可选择性地应用输入投影。

        参数：
            layer (nn.Module): 要在 fuser 中复制的层。
            num_layers (int): 要复制层的次数。
            dim (int | None): 如果使用输入投影，则为输入投影的维度。
            input_projection (bool): 是否使用输入投影。

        示例：
            >>> layer = nn.Linear(64, 64)
            >>> fuser = Fuser(layer, num_layers=3, dim=64, input_projection=True)
            >>> input_tensor = torch.randn(1, 64)
            >>> output = fuser(input_tensor)
        """
        super().__init__()
        self.proj = nn.Identity()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        """将一系列层应用于输入张量，必要时首先进行投影。"""
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class SAM2TwoWayAttentionBlock(TwoWayAttentionBlock):
    """
    一个双向注意力模块，用于执行自注意力和交叉注意力。

    该模块扩展了 TwoWayAttentionBlock，包含四个主要组件：在稀疏输入上进行的自注意力，来自稀疏到密集输入的交叉注意力，在稀疏输入上的 MLP 块，以及来自密集到稀疏输入的交叉注意力。

    属性：
        self_attn (Attention): 用于查询的自注意力层。
        norm1 (nn.LayerNorm): 第一个注意力模块后的层归一化。
        cross_attn_token_to_image (Attention): 从查询到键的交叉注意力层。
        norm2 (nn.LayerNorm): 第二个注意力模块后的层归一化。
        mlp (MLP): 用于转换查询嵌入的 MLP 块。
        norm3 (nn.LayerNorm): MLP 块后的层归一化。
        norm4 (nn.LayerNorm): 第三个注意力模块后的层归一化。
        cross_attn_image_to_token (Attention): 从键到查询的交叉注意力层。
        skip_first_layer_pe (bool): 是否跳过第一层的位置信息编码。

    方法：
        forward: 通过注意力模块和 MLP 处理输入。

    示例：
        >>> block = SAM2TwoWayAttentionBlock(embedding_dim=256, num_heads=8)
        >>> sparse_input = torch.randn(1, 100, 256)
        >>> dense_input = torch.randn(1, 256, 16, 16)
        >>> sparse_output, dense_output = block(sparse_input, dense_input)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        初始化一个 SAM2TwoWayAttentionBlock，用于执行自注意力和交叉注意力。

        该模块扩展了 TwoWayAttentionBlock，包含四个主要组件：在稀疏输入上进行的自注意力，来自稀疏到密集输入的交叉注意力，在稀疏输入上的 MLP 块，以及来自密集到稀疏输入的交叉注意力。

        参数：
            embedding_dim (int): 嵌入的通道维度。
            num_heads (int): 注意力层中头的数量。
            mlp_dim (int): MLP 块的隐藏维度。
            activation (Type[nn.Module]): MLP 块的激活函数。
            attention_downsample_rate (int): 注意力计算的下采样率。
            skip_first_layer_pe (bool): 是否跳过第一层的位置信息编码。

        示例：
            >>> block = SAM2TwoWayAttentionBlock(embedding_dim=256, num_heads=8, mlp_dim=2048)
            >>> sparse_inputs = torch.randn(1, 100, 256)
            >>> dense_inputs = torch.randn(1, 256, 32, 32)
            >>> sparse_outputs, dense_outputs = block(sparse_inputs, dense_inputs)
        """
        super().__init__(embedding_dim, num_heads, mlp_dim, activation, attention_downsample_rate, skip_first_layer_pe)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, num_layers=2, act=activation)


class SAM2TwoWayTransformer(TwoWayTransformer):
    """
    一个双向变换器模块，用于同时关注图像和查询点。

    该类扩展了 TwoWayTransformer，实现了一个专门的变换器解码器，使用带有提供的位置编码的查询来关注输入图像。它特别适用于目标检测、图像分割和点云处理等任务。

    属性：
        depth (int): 变换器中的层数。
        embedding_dim (int): 输入嵌入的通道维度。
        num_heads (int): 多头注意力中的头数。
        mlp_dim (int): MLP 块的内部通道维度。
        layers (nn.ModuleList): 由 SAM2TwoWayAttentionBlock 层组成的变换器。
        final_attn_token_to_image (Attention): 从查询到图像的最终注意力层。
        norm_final_attn (nn.LayerNorm): 应用于最终查询的层归一化。

    方法：
        forward: 通过变换器处理输入的图像嵌入和查询嵌入。

    示例：
        >>> transformer = SAM2TwoWayTransformer(depth=5, embedding_dim=256, num_heads=8, mlp_dim=2048)
        >>> image_embedding = torch.randn(1, 256, 64, 64)
        >>> query_embedding = torch.randn(1, 100, 256)
        >>> output = transformer(image_embedding, query_embedding)
        >>> print(output[0].shape, output[1].shape)
        torch.Size([1, 100, 256]) torch.Size([1, 256, 64, 64])
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        初始化一个 SAM2TwoWayTransformer 实例。

        该变换器解码器使用带有提供的位置编码的查询来关注输入图像。它设计用于目标检测、图像分割和点云处理等任务。

        参数：
            depth (int): 变换器中的层数。
            embedding_dim (int): 输入嵌入的通道维度。
            num_heads (int): 多头注意力中的头数，必须能够整除 embedding_dim。
            mlp_dim (int): MLP 块的内部通道维度。
            activation (Type[nn.Module]): MLP 块中使用的激活函数。
            attention_downsample_rate (int): 注意力计算的下采样率。

        示例：
            >>> transformer = SAM2TwoWayTransformer(depth=5, embedding_dim=256, num_heads=8, mlp_dim=2048)
            >>> transformer
            SAM2TwoWayTransformer(
              (layers): ModuleList(
                (0-4): 5 x SAM2TwoWayAttentionBlock(...)
              )
              (final_attn_token_to_image): Attention(...)
              (norm_final_attn): LayerNorm(...)
            )
        """
        super().__init__(depth, embedding_dim, num_heads, mlp_dim, activation, attention_downsample_rate)
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                SAM2TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )


class RoPEAttention(Attention):
    """
    实现变压器架构中注意力机制的旋转位置编码（Rotary Position Encoding, RoPE）。

    这个类通过结合旋转位置编码（RoPE）扩展了基本的 Attention 类，以增强注意力机制的位置信息感知能力。

    属性：
        compute_cis (Callable): 用于计算旋转编码的轴向复数的函数。
        freqs_cis (Tensor): 预先计算好的旋转编码的频率张量。
        rope_k_repeat (bool): 标记是否重复查询的 RoPE 以匹配键的长度，针对交叉注意力与记忆匹配需要使用。

    方法：
        forward: 应用旋转位置编码并计算查询、键和值张量之间的注意力。

    示例：
        >>> rope_attn = RoPEAttention(embedding_dim=256, num_heads=8, rope_theta=10000.0, feat_sizes=(32, 32))
        >>> q = torch.randn(1, 1024, 256)
        >>> k = torch.randn(1, 1024, 256)
        >>> v = torch.randn(1, 1024, 256)
        >>> output = rope_attn(q, k, v)
        >>> print(output.shape)
        torch.Size([1, 1024, 256])
    """

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        rope_k_repeat=False,
        feat_sizes=(32, 32),  # [w, h] 用于步幅为16的特征，在512分辨率下
        **kwargs,
    ):
        """初始化 RoPEAttention，通过旋转位置编码增强位置信息感知能力。"""
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta)
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat  # 重复查询的RoPE以匹配键的长度，交叉注意力时需要使用

    def forward(self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0) -> Tensor:
        """应用旋转位置编码并计算查询、键和值张量之间的注意力。"""
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 拆分为多个头
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # 应用旋转位置编码
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        # 注意力计算
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # 获取输出
        out = attn @ v

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    """对张量应用池化和可选的归一化，处理空间维度的排列。"""
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    """
    实现多尺度自注意力（Multiscale Self-Attention），并可选地对查询进行池化以提高特征提取效率。

    这个类提供了一个灵活的多尺度注意力实现，允许通过池化对查询特征进行下采样。它旨在增强模型在视觉任务中捕捉多尺度信息的能力。

    属性：
        dim (int): 输入特征图的维度。
        dim_out (int): 注意力模块的输出维度。
        num_heads (int): 注意力头的数量。
        scale (float): 点积注意力的缩放因子。
        q_pool (nn.Module | None): 可选的池化模块，用于查询特征。
        qkv (nn.Linear): 查询、键和值的线性投影。
        proj (nn.Linear): 输出投影。

    方法：
        forward: 对输入张量应用多尺度注意力。

    示例：
        >>> import torch
        >>> from torch import nn
        >>> x = torch.randn(1, 64, 64, 256)
        >>> msa = MultiScaleAttention(dim=256, dim_out=256, num_heads=8)
        >>> output = msa(x)
        >>> print(output.shape)
        torch.Size([1, 64, 64, 256])
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        """初始化多尺度注意力，并可选地对查询特征进行池化以提高特征提取效率。"""
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用多尺度注意力，提取多尺度特征。"""
        B, H, W, _ = x.shape
        # qkv 的形状为 (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v 的形状为 (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # 查询池化（用于阶段变化时下采样）
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # 下采样后的形状
            q = q.reshape(B, H * W, self.num_heads, -1)

        # PyTorch 的 SDPA 期望输入为 [B, nheads, H*W, C]，因此需要转置
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # 转置回原形状
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    """
    一个具有窗口划分和查询池化的多尺度注意力块，用于高效的视觉变换器。

    该类实现了一个多尺度注意力机制，具有可选的窗口划分和下采样，设计用于视觉变换器架构。

    属性：
        dim (int): 块的输入维度。
        dim_out (int): 块的输出维度。
        norm1 (nn.Module): 第一个归一化层。
        window_size (int): 划分窗口的大小。
        pool (nn.Module | None): 用于查询下采样的池化层。
        q_stride (Tuple[int, int] | None): 查询池化的步幅。
        attn (MultiScaleAttention): 多尺度注意力模块。
        drop_path (nn.Module): 用于正则化的 DropPath 层。
        norm2 (nn.Module): 第二个归一化层。
        mlp (MLP): 多层感知机模块。
        proj (nn.Linear | None): 用于维度不匹配的投影层。

    方法：
        forward: 通过多尺度块处理输入张量。

    示例：
        >>> block = MultiScaleBlock(dim=256, dim_out=512, num_heads=8, window_size=7)
        >>> x = torch.randn(1, 56, 56, 256)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 28, 28, 512])
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        """初始化一个具有窗口划分和可选查询池化的多尺度注意力块。"""
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            act=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过多尺度注意力和 MLP 处理输入，具有可选的窗口划分和下采样。"""
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # 跳跃连接
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # 窗口划分
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # 窗口注意力 + 查询池化（如果阶段改变）
        x = self.attn(x)
        if self.q_stride:
            # 由于查询池化，形状发生了变化
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # 反向窗口划分
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PositionEmbeddingSine(nn.Module):
    """
    用于生成 2D 输入（如图像）的正弦位置嵌入的模块。

    该类实现了用于 2D 空间位置的正弦位置编码，可以在基于变换器的计算机视觉任务中使用。

    属性：
        num_pos_feats (int): 位置特征的数量（嵌入维度的一半）。
        temperature (int): 正弦函数的温度参数。
        normalize (bool): 是否归一化位置嵌入。
        scale (float): 当 normalize 为 True 时，位置嵌入的缩放因子。
        cache (Dict): 用于存储预计算的嵌入的缓存。

    方法：
        _encode_xy: 使用正弦和余弦函数对 2D 位置进行编码。
        encode_boxes: 将框坐标和尺寸编码为位置嵌入。
        encode_points: 使用正弦位置嵌入对 2D 点坐标进行编码。
        forward: 为 2D 输入生成正弦位置嵌入。

    示例：
        >>> pos_emb = PositionEmbeddingSine(num_pos_feats=128)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> embeddings = pos_emb(x)
        >>> print(embeddings.shape)
        torch.Size([1, 256, 224, 224])
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        """初始化用于 2D 图像输入的正弦位置嵌入。"""
        super().__init__()
        assert num_pos_feats % 2 == 0, "期望模型宽度为偶数"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("如果传递了 scale，normalize 应该为 True")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}

    def _encode_xy(self, x, y):
        """使用正弦/余弦函数对 2D 位置进行编码，以生成变换器的位置信息。"""
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        """将框坐标和尺寸编码为目标检测的位置信息。"""
        pos_x, pos_y = self._encode_xy(x, y)
        return torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)

    encode = encode_boxes  # 向后兼容

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        """对 2D 点使用正弦嵌入进行编码，并附加标签。"""
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        return torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """为 2D 输入（如图像）生成正弦位置嵌入。"""
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos


class PositionEmbeddingRandom(nn.Module):
    """
    使用随机空间频率进行位置编码。

    这个类使用随机空间频率为输入坐标生成位置嵌入。它特别适用于需要位置信息的基于变换器的模型。

    属性：
        positional_encoding_gaussian_matrix (torch.Tensor): 用于编码的随机值缓冲区。

    方法：
        _pe_encoding: 对归一化到 [0,1] 的点进行位置编码。
        forward: 为指定大小的网格生成位置编码。
        forward_with_coords: 对未归一化到 [0,1] 的点进行位置编码。

    示例：
        >>> pe = PositionEmbeddingRandom(num_pos_feats=64)
        >>> size = (32, 32)
        >>> encoding = pe(size)
        >>> print(encoding.shape)
        torch.Size([128, 32, 32])
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        """初始化用于变换器的随机空间频率位置嵌入。"""
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats)))

        # 设置非确定性，以避免 forward() 时出现 'cumsum_cuda_kernel does not have a deterministic implementation' 错误
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """使用随机空间频率对归一化到 [0,1] 的坐标进行编码。"""
        # 假设 coords 位于 [0, 1]^2 平方区域，形状为 d_1 x ... x d_n x 2
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # 输出形状为 d_1 x ... x d_n x C
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """为使用随机空间频率的网格生成位置编码。"""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """对输入坐标进行位置编码，基于给定图像大小将其归一化到 [0,1]。"""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class Block(nn.Module):
    """
    支持窗口注意力和残差传播的变换器块。

    这个类实现了一个变换器块，可以使用全局或窗口化自注意力，后跟一个前馈网络。它支持相对位置编码，设计用于视觉变换器架构。

    属性：
        norm1 (nn.Module): 第一个归一化层。
        attn (REAttention): 带有可选相对位置编码的自注意力层。
        norm2 (nn.Module): 第二个归一化层。
        mlp (MLPBlock): 多层感知机块。
        window_size (int): 注意力窗口的大小。如果为 0，则使用全局注意力。

    方法：
        forward: 通过变换器块处理输入。

    示例：
        >>> import torch
        >>> block = Block(dim=256, num_heads=8, window_size=7)
        >>> x = torch.randn(1, 56, 56, 256)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 56, 56, 256])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        初始化一个变换器块，支持可选的窗口注意力和相对位置编码。

        该构造函数设置了一个可以使用全局或窗口化自注意力的变换器块，后跟一个前馈网络。它支持相对位置编码，设计用于视觉变换器架构。

        参数：
            dim (int): 输入通道的数量。
            num_heads (int): 自注意力层中头的数量。
            mlp_ratio (float): MLP 隐藏维度与嵌入维度的比例。
            qkv_bias (bool): 如果为 True，则为查询、键、值投影添加可学习的偏置。
            norm_layer (Type[nn.Module]): 使用的归一化层类型。
            act_layer (Type[nn.Module]): MLP 块中使用的激活函数类型。
            use_rel_pos (bool): 如果为 True，则在注意力中使用相对位置编码。
            rel_pos_zero_init (bool): 如果为 True，则将相对位置参数初始化为零。
            window_size (int): 注意力窗口的大小。如果为 0，则使用全局注意力。
            input_size (Optional[Tuple[int, int]]): 计算相对位置参数大小时的输入分辨率。

        示例：
            >>> block = Block(dim=256, num_heads=8, window_size=7)
            >>> x = torch.randn(1, 56, 56, 256)
            >>> output = block(x)
            >>> print(output.shape)
            torch.Size([1, 56, 56, 256])
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = REAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过带有可选窗口化自注意力和残差连接的变换器块处理输入。"""
        shortcut = x
        x = self.norm1(x)
        # 窗口分区
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # 逆窗口分区
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class REAttention(nn.Module):
    """
    旋转嵌入注意力模块，用于变压器架构中的高效自注意力计算。

    该类实现了一个多头注意力机制，采用旋转位置编码，专为视觉变压器模型设计。它支持可选的查询池化和窗口划分，以提高大规模输入的处理效率。

    属性：
        compute_cis (Callable): 用于计算旋转编码的轴向复数的函数。
        freqs_cis (Tensor): 预计算的旋转编码频率张量。
        rope_k_repeat (bool): 标记是否重复查询的RoPE以匹配键的长度，针对交叉注意力与记忆匹配需要使用。
        q_proj (nn.Linear): 查询的线性投影。
        k_proj (nn.Linear): 键的线性投影。
        v_proj (nn.Linear): 值的线性投影。
        out_proj (nn.Linear): 输出投影。
        num_heads (int): 注意力头的数量。
        internal_dim (int): 用于注意力计算的内部维度。

    方法：
        forward: 应用旋转位置编码并计算查询、键和值张量之间的注意力。

    示例：
        >>> rope_attn = REAttention(embedding_dim=256, num_heads=8, rope_theta=10000.0, feat_sizes=(32, 32))
        >>> q = torch.randn(1, 1024, 256)
        >>> k = torch.randn(1, 1024, 256)
        >>> v = torch.randn(1, 1024, 256)
        >>> output = rope_attn(q, k, v)
        >>> print(output.shape)
        torch.Size([1, 1024, 256])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        初始化相对位置注意力模块，适用于基于变压器的架构。

        该模块实现了多头注意力机制，带有可选的相对位置编码，专为视觉任务中的变压器模型设计。

        参数：
            dim (int): 输入通道数。
            num_heads (int): 注意力头的数量。默认为8。
            qkv_bias (bool): 如果为True，则在查询、键、值投影中添加可学习的偏置项。默认为True。
            use_rel_pos (bool): 如果为True，则使用相对位置编码。默认为False。
            rel_pos_zero_init (bool): 如果为True，则将相对位置参数初始化为零。默认为True。
            input_size (Tuple[int, int] | None): 用于计算相对位置参数大小的输入分辨率。
                如果使用相对位置编码，则需要提供该参数。默认为None。

        示例：
            >>> attention = REAttention(dim=256, num_heads=8, input_size=(32, 32))
            >>> x = torch.randn(1, 32, 32, 256)
            >>> output = attention(x)
            >>> print(output.shape)
            torch.Size([1, 32, 32, 256])
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "如果使用相对位置编码，必须提供输入大小。"
            # 初始化相对位置嵌入
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用多头注意力并对输入张量进行可选的相对位置编码。"""
        B, H, W, _ = x.shape
        # qkv 形状为 (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v 形状为 (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return self.proj(x)


class PatchEmbed(nn.Module):
    """
    图像到补丁嵌入模块，用于视觉变压器架构。

    该模块使用卷积层将输入图像转换为一系列补丁嵌入。
    通常用作视觉变压器架构中的第一层，将图像数据转换为适合后续变压器块的格式。

    属性：
        proj (nn.Conv2d): 用于将图像补丁投影到嵌入空间的卷积层。

    方法：
        forward: 对输入张量应用补丁嵌入。

    示例：
        >>> patch_embed = PatchEmbed(kernel_size=(16, 16), stride=(16, 16), in_chans=3, embed_dim=768)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = patch_embed(x)
        >>> print(output.shape)
        torch.Size([1, 768, 14, 14])
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        初始化 PatchEmbed 模块，用于将图像补丁转换为嵌入。

        该模块通常作为视觉变压器架构中的第一层，将图像数据转换为适合后续变压器块的格式。

        参数：
            kernel_size (Tuple[int, int]): 用于提取补丁的卷积核大小。
            stride (Tuple[int, int]): 卷积操作的步幅。
            padding (Tuple[int, int]): 在卷积前对输入应用的填充。
            in_chans (int): 输入图像通道数。
            embed_dim (int): 输出补丁嵌入的维度。

        示例：
            >>> patch_embed = PatchEmbed(kernel_size=(16, 16), stride=(16, 16), in_chans=3, embed_dim=768)
            >>> x = torch.randn(1, 3, 224, 224)
            >>> output = patch_embed(x)
            >>> print(output.shape)
            torch.Size([1, 768, 14, 14])
        """
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过卷积计算补丁嵌入并转置结果张量。"""
        return self.proj(x).permute(0, 2, 3, 1)  # B C H W -> B H W C
