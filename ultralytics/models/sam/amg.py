# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
from itertools import product
from typing import Any, Generator, List, Tuple

import numpy as np
import torch


def is_box_near_crop_edge(
    boxes: torch.Tensor, crop_box: List[int], orig_box: List[int], atol: float = 20.0
) -> torch.Tensor:
    """判断边界框是否接近裁剪图像区域的边缘，使用指定的容差值。"""
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    boxes = uncrop_boxes_xyxy(boxes, crop_box).float()
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    """根据指定的批量大小，从输入参数中生成数据批次，以便高效处理。"""
    assert args and all(len(a) == len(args[0]) for a in args), "批次迭代必须具有相同大小的输入。"
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def calculate_stability_score(masks: torch.Tensor, mask_threshold: float, threshold_offset: float) -> torch.Tensor:
    """
    计算一批掩膜的稳定性分数。

    稳定性分数是通过在预测掩膜的对数值上应用高低阈值来生成二进制掩膜，进而计算其IoU。

    参数：
        masks (torch.Tensor): 一批预测掩膜的对数。
        mask_threshold (float): 用于生成二进制掩膜的阈值。
        threshold_offset (float): 用于创建高低二进制掩膜的阈值偏移。

    返回：
        (torch.Tensor): 每个掩膜的稳定性分数。

    注意：
        - 一个掩膜总是被包含在另一个掩膜内部。
        - 通过避免不必要的类型转换节省内存。

    示例：
        >>> masks = torch.rand(10, 256, 256)  # 一批10个掩膜
        >>> mask_threshold = 0.5
        >>> threshold_offset = 0.1
        >>> stability_scores = calculate_stability_score(masks, mask_threshold, threshold_offset)
    """
    intersections = (masks > (mask_threshold + threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    unions = (masks > (mask_threshold - threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    return intersections / unions


def build_point_grid(n_per_side: int) -> np.ndarray:
    """为图像分割任务生成一个均匀分布的2D点网格，范围为[0,1]x[0,1]。"""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    return np.stack([points_x, points_y], axis=-1).reshape(-1, 2)


def build_all_layer_point_grids(n_per_side: int, n_layers: int, scale_per_layer: int) -> List[np.ndarray]:
    """为多个裁剪层生成点网格，层之间具有不同的尺度和密度。"""
    return [build_point_grid(int(n_per_side / (scale_per_layer**i))) for i in range(n_layers + 1)]


def generate_crop_boxes(
    im_size: Tuple[int, ...], n_layers: int, overlap_ratio: float
) -> Tuple[List[List[int]], List[int]]:
    """为多尺度图像处理生成不同大小的裁剪框，并带有分层重叠区域。"""
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # 原始图像
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        """将边界框裁剪为输入图像的大小。"""
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # XYWH格式的裁剪框
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    """通过将裁剪框的偏移量加到边界框的坐标中来解裁剪边界框。"""
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    # 检查boxes是否具有通道维度
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset


def uncrop_points(points: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    """通过将裁剪框的偏移量添加到坐标来还原点的位置。"""
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0]], device=points.device)
    # 检查点是否有通道维度
    if len(points.shape) == 3:
        offset = offset.unsqueeze(1)
    return points + offset


def uncrop_masks(masks: torch.Tensor, crop_box: List[int], orig_h: int, orig_w: int) -> torch.Tensor:
    """通过将掩膜填充回原始图像大小，处理坐标变换来还原掩膜。"""
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # 坐标变换掩膜
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = (x0, pad_x - x0, y0, pad_y - y0)
    return torch.nn.functional.pad(masks, pad, value=0)


def remove_small_regions(mask: np.ndarray, area_thresh: float, mode: str) -> Tuple[np.ndarray, bool]:
    """根据面积阈值和模式去除掩膜中小的离散区域或孔洞。"""
    import cv2  # type: ignore

    assert mode in {"holes", "islands"}, f"提供的模式 {mode} 无效"
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # 第0行是背景标签
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if not small_regions:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        # 如果每个区域都小于阈值，则保留最大的区域
        fill_labels = [i for i in range(n_labels) if i not in fill_labels] or [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """计算二进制掩膜周围的边界框，处理空掩膜和各种输入形状。"""
    # torch.max 在空输入时会报错，遇到这种情况直接跳过
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # 将形状标准化为 CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    masks = masks.flatten(0, -3) if len(shape) > 2 else masks.unsqueeze(0)
    # 获取上边和下边
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # 获取左边和右边
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # 如果掩膜为空，则右边界会位于左边界左侧。
    # 用 [0, 0, 0, 0] 替换这些框
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # 恢复到原始形状
    return out.reshape(*shape[:-2], 4) if len(shape) > 2 else out[0]
