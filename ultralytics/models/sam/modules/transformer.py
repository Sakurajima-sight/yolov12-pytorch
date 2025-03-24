# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
from typing import Tuple, Type

import torch
from torch import Tensor, nn

from ultralytics.nn.modules import MLPBlock


class TwoWayTransformer(nn.Module):
    """
    一个双向变换器模块，用于同时关注图像和查询点。

    该类实现了一个特殊的变换器解码器，它使用查询和提供的位置编码同时关注输入图像。
    适用于目标检测、图像分割和点云处理等任务。

    属性:
        depth (int): 变换器中的层数。
        embedding_dim (int): 输入嵌入的通道维度。
        num_heads (int): 多头注意力的头数。
        mlp_dim (int): MLP块的内部通道维度。
        layers (nn.ModuleList): 组成变换器的TwoWayAttentionBlock层列表。
        final_attn_token_to_image (Attention): 从查询到图像的最终注意力层。
        norm_final_attn (nn.LayerNorm): 应用于最终查询的层归一化。

    方法:
        forward: 通过变换器处理图像和点嵌入。

    示例:
        >>> transformer = TwoWayTransformer(depth=6, embedding_dim=256, num_heads=8, mlp_dim=2048)
        >>> image_embedding = torch.randn(1, 256, 32, 32)
        >>> image_pe = torch.randn(1, 256, 32, 32)
        >>> point_embedding = torch.randn(1, 100, 256)
        >>> output_queries, output_image = transformer(image_embedding, image_pe, point_embedding)
        >>> print(output_queries.shape, output_image.shape)
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
        初始化一个双向变换器，用于同时关注图像和查询点。

        参数:
            depth (int): 变换器中的层数。
            embedding_dim (int): 输入嵌入的通道维度。
            num_heads (int): 多头注意力的头数。必须能整除embedding_dim。
            mlp_dim (int): MLP块的内部通道维度。
            activation (Type[nn.Module]): MLP块中使用的激活函数。
            attention_downsample_rate (int): 注意力机制的下采样率。

        属性:
            depth (int): 变换器中的层数。
            embedding_dim (int): 输入嵌入的通道维度。
            num_heads (int): 多头注意力的头数。
            mlp_dim (int): MLP块的内部通道维度。
            layers (nn.ModuleList): 组成变换器的TwoWayAttentionBlock层列表。
            final_attn_token_to_image (Attention): 从查询到图像的最终注意力层。
            norm_final_attn (nn.LayerNorm): 应用于最终查询的层归一化。

        示例:
            >>> transformer = TwoWayTransformer(depth=6, embedding_dim=256, num_heads=8, mlp_dim=2048)
            >>> image_embedding = torch.randn(1, 256, 32, 32)
            >>> image_pe = torch.randn(1, 256, 32, 32)
            >>> point_embedding = torch.randn(1, 100, 256)
            >>> output_queries, output_image = transformer(image_embedding, image_pe, point_embedding)
            >>> print(output_queries.shape, output_image.shape)
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        通过双向变换器处理图像和点嵌入。

        参数:
            image_embedding (torch.Tensor): 要注意的图像，形状为(B, embedding_dim, H, W)。
            image_pe (torch.Tensor): 要添加到图像的位置信息，形状与image_embedding相同。
            point_embedding (torch.Tensor): 要添加到查询点的嵌入，形状为(B, N_points, embedding_dim)。

        返回:
            (Tuple[torch.Tensor, torch.Tensor]): 处理后的point_embedding和image_embedding。

        示例:
            >>> transformer = TwoWayTransformer(depth=6, embedding_dim=256, num_heads=8, mlp_dim=2048)
            >>> image_embedding = torch.randn(1, 256, 32, 32)
            >>> image_pe = torch.randn(1, 256, 32, 32)
            >>> point_embedding = torch.randn(1, 100, 256)
            >>> output_queries, output_image = transformer(image_embedding, image_pe, point_embedding)
            >>> print(output_queries.shape, output_image.shape)
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # 准备查询
        queries = point_embedding
        keys = image_embedding

        # 应用变换器块和最终的层归一化
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # 应用从点到图像的最终注意力层
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    """
    一个用于同时关注图像和查询点的双向注意力块。

    该类实现了一个专门的变换器块，主要包含四个层次：稀疏输入的自注意力、稀疏输入到密集输入的交叉注意力、稀疏输入的MLP块以及密集输入到稀疏输入的交叉注意力。

    属性：
        self_attn (Attention): 查询的自注意力层。
        norm1 (nn.LayerNorm): 自注意力后的层归一化。
        cross_attn_token_to_image (Attention): 从查询到键的交叉注意力层。
        norm2 (nn.LayerNorm): token到图像注意力后的层归一化。
        mlp (MLPBlock): 用于转换查询嵌入的MLP块。
        norm3 (nn.LayerNorm): MLP块后的层归一化。
        norm4 (nn.LayerNorm): 图像到token注意力后的层归一化。
        cross_attn_image_to_token (Attention): 从键到查询的交叉注意力层。
        skip_first_layer_pe (bool): 是否跳过第一层的位置信息编码。

    方法：
        forward: 对查询和键的嵌入应用自注意力和交叉注意力。

    示例：
        >>> embedding_dim, num_heads = 256, 8
        >>> block = TwoWayAttentionBlock(embedding_dim, num_heads)
        >>> queries = torch.randn(1, 100, embedding_dim)
        >>> keys = torch.randn(1, 1000, embedding_dim)
        >>> query_pe = torch.randn(1, 100, embedding_dim)
        >>> key_pe = torch.randn(1, 1000, embedding_dim)
        >>> processed_queries, processed_keys = block(queries, keys, query_pe, key_pe)
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
        初始化一个双向注意力块，用于同时关注图像和查询点。

        该块实现了一个专门的变换器层，包含四个主要部分：稀疏输入的自注意力、稀疏输入到密集输入的交叉注意力、稀疏输入的MLP块和密集输入到稀疏输入的交叉注意力。

        参数：
            embedding_dim (int): 嵌入的通道维度。
            num_heads (int): 注意力层中的注意力头数。
            mlp_dim (int): MLP块的隐藏维度。
            activation (Type[nn.Module]): MLP块的激活函数。
            attention_downsample_rate (int): 注意力机制的下采样率。
            skip_first_layer_pe (bool): 是否跳过第一层的位置信息编码。

        示例：
            >>> embedding_dim, num_heads = 256, 8
            >>> block = TwoWayAttentionBlock(embedding_dim, num_heads)
            >>> queries = torch.randn(1, 100, embedding_dim)
            >>> keys = torch.randn(1, 1000, embedding_dim)
            >>> query_pe = torch.randn(1, 100, embedding_dim)
            >>> key_pe = torch.randn(1, 1000, embedding_dim)
            >>> processed_queries, processed_keys = block(queries, keys, query_pe, key_pe)
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor) -> Tuple[Tensor, Tensor]:
        """应用双向注意力处理查询和键的嵌入，进行变换器块处理。"""
        # 自注意力块
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # 交叉注意力块，tokens 关注图像嵌入
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP块
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # 交叉注意力块，图像嵌入关注 tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    一个具有下采样功能的注意力层，用于投影后的嵌入大小。

    该类实现了一个多头注意力机制，可以选择性地对查询、键和值的内部维度进行下采样。

    属性：
        embedding_dim (int): 输入嵌入的维度。
        kv_in_dim (int): 键和值输入的维度。
        internal_dim (int): 下采样后的内部维度。
        num_heads (int): 注意力头数。
        q_proj (nn.Linear): 查询的线性投影。
        k_proj (nn.Linear): 键的线性投影。
        v_proj (nn.Linear): 值的线性投影。
        out_proj (nn.Linear): 输出的线性投影。

    方法：
        _separate_heads: 将输入张量分离成多个注意力头。
        _recombine_heads: 将分离的注意力头重新组合成一个张量。
        forward: 对给定的查询、键和值张量计算注意力输出。

    示例：
        >>> attn = Attention(embedding_dim=256, num_heads=8, downsample_rate=2)
        >>> q = torch.randn(1, 100, 256)
        >>> k = v = torch.randn(1, 50, 256)
        >>> output = attn(q, k, v)
        >>> print(output.shape)
        torch.Size([1, 100, 256])
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        kv_in_dim: int = None,
    ) -> None:
        """
        初始化注意力模块，指定维度和设置。

        该类实现了一个多头注意力机制，支持可选的对查询、键和值的内部维度进行下采样。

        参数：
            embedding_dim (int): 输入嵌入的维度。
            num_heads (int): 注意力头数。
            downsample_rate (int): 内部维度下采样的倍数，默认是1。
            kv_in_dim (int | None): 键和值输入的维度。如果为None，则使用embedding_dim。

        异常：
            AssertionError: 如果num_heads不能整除内部维度（embedding_dim / downsample_rate）。

        示例：
            >>> attn = Attention(embedding_dim=256, num_heads=8, downsample_rate=2)
            >>> q = torch.randn(1, 100, 256)
            >>> k = v = torch.randn(1, 50, 256)
            >>> output = attn(q, k, v)
            >>> print(output.shape)
            torch.Size([1, 100, 256])
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads 必须整除 embedding_dim。"

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    @staticmethod
    def _separate_heads(x: Tensor, num_heads: int) -> Tensor:
        """将输入张量分离成指定数量的注意力头。"""
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    @staticmethod
    def _recombine_heads(x: Tensor) -> Tensor:
        """将分离的注意力头重新组合成一个张量。"""
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """对查询、键和值张量应用多头注意力，支持下采样。"""
        # 输入投影
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 分离成注意力头
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # 注意力计算
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # 获取输出
        out = attn @ v
        out = self._recombine_heads(out)
        return self.out_proj(out)
