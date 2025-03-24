# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Tuple

import torch
import torch.nn.functional as F


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    选择给定帧索引的最近条件帧。

    参数：
        frame_idx (int): 当前帧索引。
        cond_frame_outputs (Dict[int, Any]): 以帧索引为键的条件帧输出字典。
        max_cond_frame_num (int): 要选择的最大条件帧数量。

    返回：
        (Tuple[Dict[int, Any], Dict[int, Any]]): 返回一个包含两个字典的元组：
            - selected_outputs: 从cond_frame_outputs中选择的项目。
            - unselected_outputs: 未选择的cond_frame_outputs中的项目。

    示例：
        >>> frame_idx = 5
        >>> cond_frame_outputs = {1: "a", 3: "b", 7: "c", 9: "d"}
        >>> max_cond_frame_num = 2
        >>> selected, unselected = select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num)
        >>> print(selected)
        {3: 'b', 7: 'c'}
        >>> print(unselected)
        {1: 'a', 9: 'd'}
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "我们应该允许使用2个或更多的条件帧"
        selected_outputs = {}

        # 选择`frame_idx`之前最近的条件帧（如果有的话）
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # 选择`frame_idx`之后最近的条件帧（如果有的话）
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # 添加其他时间上最近的条件帧，直到达到`max_cond_frame_num`个条件帧。
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs}

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """生成给定位置和维度的1D正弦位置嵌入。"""
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def init_t_xy(end_x: int, end_y: int):
    """初始化指定维度网格的1D和2D坐标张量。"""
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    """计算2D空间位置的轴向复指数位置编码（用于网格）。"""
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """调整频率张量的形状，以便与输入张量进行广播，确保维度兼容。"""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    """应用旋转位置编码到查询和键张量，使用复数频率成分。"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        # 没有键张量，因而不需要旋转
        return xq_out.type_as(xq).to(xq.device), xk
    # 如果需要，重复频率以匹配键张量的序列长度
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def window_partition(x, window_size):
    """
    将输入张量分割成不重叠的窗口，如果需要的话进行填充。

    参数：
        x (torch.Tensor): 输入张量，形状为(B, H, W, C)。
        window_size (int): 每个窗口的大小。

    返回：
        (Tuple[torch.Tensor, Tuple[int, int]]): 返回一个包含以下内容的元组：
            - windows (torch.Tensor): 分割后的窗口，形状为(B * num_windows, window_size, window_size, C)。
            - (Hp, Wp) (Tuple[int, int]): 在分割之前的填充高度和宽度。

    示例：
        >>> x = torch.randn(1, 16, 16, 3)
        >>> windows, (Hp, Wp) = window_partition(x, window_size=4)
        >>> print(windows.shape, Hp, Wp)
        torch.Size([16, 4, 4, 3]) 16 16
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    将分块的序列还原为原始序列并去除填充。

    此函数逆转窗口化过程，通过去除窗口化过程中添加的填充，重建原始输入。

    参数：
        windows (torch.Tensor): 窗口化序列的输入张量，形状为 (B * num_windows, window_size,
            window_size, C)，其中 B 是批量大小，num_windows 是窗口数量，window_size 是每个窗口的大小，C 是通道数。
        window_size (int): 每个窗口的大小。
        pad_hw (Tuple[int, int]): 输入在窗口化前的填充高度和宽度 (Hp, Wp)。
        hw (Tuple[int, int]): 输入在填充和窗口化之前的原始高度和宽度 (H, W)。

    返回：
        (torch.Tensor): 还原后的序列，形状为 (B, H, W, C)，其中 B 是批量大小，H 和 W 是原始高度和宽度，C 是通道数。

    示例：
        >>> windows = torch.rand(32, 8, 8, 64)  # 32 个大小为 8x8，通道数为 64 的窗口
        >>> pad_hw = (16, 16)  # 填充后的高度和宽度
        >>> hw = (15, 14)  # 原始的高度和宽度
        >>> x = window_unpartition(windows, window_size=8, pad_hw=pad_hw, hw=hw)
        >>> print(x.shape)
        torch.Size([1, 15, 14, 64])
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    根据查询和键的大小提取相对位置嵌入。

    参数：
        q_size (int): 查询的大小。
        k_size (int): 键的大小。
        rel_pos (torch.Tensor): 形状为 (L, C) 的相对位置嵌入，其中 L 是最大相对距离，C 是嵌入维度。

    返回：
        (torch.Tensor): 根据相对位置提取的嵌入，形状为 (q_size, k_size, C)。

    示例：
        >>> q_size, k_size = 8, 16
        >>> rel_pos = torch.randn(31, 64)  # 31 = 2 * max(8, 16) - 1
        >>> extracted_pos = get_rel_pos(q_size, k_size, rel_pos)
        >>> print(extracted_pos.shape)
        torch.Size([8, 16, 64])
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # 如果需要，插值相对位置。
    if rel_pos.shape[0] != max_rel_dist:
        # 插值相对位置。
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # 如果查询和键的形状不同，则缩放坐标。
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    将分解的相对位置嵌入添加到注意力图中。

    此函数计算并应用分解的相对位置嵌入，增强注意力机制，通过结合查询和键位置之间的空间关系。

    参数：
        attn (torch.Tensor): 注意力图，形状为 (B, q_h * q_w, k_h * k_w)。
        q (torch.Tensor): 注意力层中的查询张量，形状为 (B, q_h * q_w, C)。
        rel_pos_h (torch.Tensor): 高度轴的相对位置嵌入，形状为 (Lh, C)。
        rel_pos_w (torch.Tensor): 宽度轴的相对位置嵌入，形状为 (Lw, C)。
        q_size (Tuple[int, int]): 查询 q 的空间序列大小，形状为 (q_h, q_w)。
        k_size (Tuple[int, int]): 键 k 的空间序列大小，形状为 (k_h, k_w)。

    返回：
        (torch.Tensor): 添加了相对位置嵌入后的注意力图，形状为 (B, q_h * q_w, k_h * k_w)。

    示例：
        >>> B, C, q_h, q_w, k_h, k_w = 1, 64, 8, 8, 8, 8
        >>> attn = torch.rand(B, q_h * q_w, k_h * k_w)
        >>> q = torch.rand(B, q_h * q_w, C)
        >>> rel_pos_h = torch.rand(2 * max(q_h, k_h) - 1, C)
        >>> rel_pos_w = torch.rand(2 * max(q_w, k_w) - 1, C)
        >>> q_size, k_size = (q_h, q_w), (k_h, k_w)
        >>> updated_attn = add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size)
        >>> print(updated_attn.shape)
        torch.Size([1, 64, 64])

    参考文献：
        https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
        B, q_h * q_w, k_h * k_w
    )

    return attn
