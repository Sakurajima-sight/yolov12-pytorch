# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy
from typing import Optional

import torch
from torch import Tensor, nn

from .blocks import RoPEAttention


class MemoryAttentionLayer(nn.Module):
    """
    实现带有自注意力和交叉注意力机制的记忆注意力层，用于神经网络。

    该类结合自注意力、交叉注意力和前馈组件来处理输入张量，并生成基于记忆的注意力输出。

    属性:
        d_model (int): 模型的维度。
        dim_feedforward (int): 前馈网络的维度。
        dropout_value (float): 正则化的 dropout 比例。
        self_attn (RoPEAttention): 使用 RoPE（旋转位置嵌入）的自注意力机制。
        cross_attn_image (RoPEAttention): 用于图像处理的交叉注意力机制。
        linear1 (nn.Linear): 前馈网络的第一层线性层。
        linear2 (nn.Linear): 前馈网络的第二层线性层。
        norm1 (nn.LayerNorm): 自注意力输出的层归一化。
        norm2 (nn.LayerNorm): 交叉注意力输出的层归一化。
        norm3 (nn.LayerNorm): 前馈网络输出的层归一化。
        dropout1 (nn.Dropout): 自注意力后的 dropout 层。
        dropout2 (nn.Dropout): 交叉注意力后的 dropout 层。
        dropout3 (nn.Dropout): 前馈网络后的 dropout 层。
        activation (nn.ReLU): 前馈网络的激活函数。
        pos_enc_at_attn (bool): 是否在自注意力中添加位置编码。
        pos_enc_at_cross_attn_queries (bool): 是否在交叉注意力查询中添加位置编码。
        pos_enc_at_cross_attn_keys (bool): 是否在交叉注意力键中添加位置编码。

    方法:
        forward: 对输入张量执行完整的记忆注意力操作。
        _forward_sa: 对输入张量执行自注意力操作。
        _forward_ca: 执行目标张量和记忆张量之间的交叉注意力。

    示例:
        >>> layer = MemoryAttentionLayer(d_model=256, dim_feedforward=2048, dropout=0.1)
        >>> tgt = torch.randn(1, 100, 256)
        >>> memory = torch.randn(1, 100, 64)
        >>> pos = torch.randn(1, 100, 256)
        >>> query_pos = torch.randn(1, 100, 256)
        >>> output = layer(tgt, memory, pos, query_pos)
        >>> print(output.shape)
        torch.Size([1, 100, 256])
    """

    def __init__(
        self,
        d_model: int = 256,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pos_enc_at_attn: bool = False,
        pos_enc_at_cross_attn_keys: bool = True,
        pos_enc_at_cross_attn_queries: bool = False,
    ):
        """初始化一个包含自注意力、交叉注意力和前馈组件的记忆注意力层。"""
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = RoPEAttention(embedding_dim=256, num_heads=1, downsample_rate=1)
        self.cross_attn_image = RoPEAttention(
            rope_k_repeat=True,
            embedding_dim=256,
            num_heads=1,
            downsample_rate=1,
            kv_in_dim=64,
        )

        # 前馈网络模型的实现
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        # 位置编码添加的位置
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        """对输入张量执行自注意力操作，使用位置编码和 RoPE 注意力机制。"""
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        """在目标张量和记忆张量之间执行交叉注意力操作，使用 RoPEAttention 机制。"""
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # 交叉注意力
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        """使用自注意力、交叉注意力和 MLP 处理输入张量，以实现基于记忆的注意力。"""
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    """
    用于处理序列数据的记忆注意力模块，结合自注意力和交叉注意力机制。

    该类实现了一个多层注意力机制，将自注意力和交叉注意力相结合，用于处理序列数据，
    特别适用于类似 Transformer 的架构。

    属性说明:
        d_model (int): 模型隐藏状态的维度。
        layers (nn.ModuleList): MemoryAttentionLayer 模块的列表。
        num_layers (int): 注意力层的数量。
        norm (nn.LayerNorm): 应用于输出的层归一化。
        pos_enc_at_input (bool): 是否在输入时应用位置编码。
        batch_first (bool): 输入张量是否是 batch-first 格式。

    方法说明:
        forward: 通过注意力层处理输入张量。

    示例用法:
        >>> d_model = 256
        >>> layer = MemoryAttentionLayer(d_model)
        >>> attention = MemoryAttention(d_model, pos_enc_at_input=True, layer=layer, num_layers=3)
        >>> curr = torch.randn(10, 32, d_model)  # (seq_len, batch_size, d_model)
        >>> memory = torch.randn(20, 32, d_model)  # (mem_len, batch_size, d_model)
        >>> curr_pos = torch.randn(10, 32, d_model)
        >>> memory_pos = torch.randn(20, 32, d_model)
        >>> output = attention(curr, memory, curr_pos, memory_pos)
        >>> print(output.shape)
        torch.Size([10, 32, 256])
    """

    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # 层是否期望 batch-first 格式的输入？
    ):
        """初始化 MemoryAttention 模块，配置注意力处理的层和归一化。"""
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,  # 自注意力输入
        memory: torch.Tensor,  # 交叉注意力输入
        curr_pos: Optional[Tensor] = None,  # 自注意力输入的位置信息编码
        memory_pos: Optional[Tensor] = None,  # 交叉注意力输入的位置信息编码
        num_obj_ptr_tokens: int = 0,  # 对象指针 *tokens* 的数量
    ):
        """通过多个注意力层处理输入张量，应用自注意力和交叉注意力机制。"""
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert curr.shape[1] == memory.shape[1], "curr 和 memory 的 batch size 必须相同"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # 转换为 batch first 格式
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # 转换回 seq first 格式
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output
