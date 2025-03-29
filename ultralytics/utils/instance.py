# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import abc
from itertools import repeat
from numbers import Number
from typing import List

import numpy as np

from .ops import ltwh2xywh, ltwh2xyxy, resample_segments, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh


def _ntuple(n):
    """来自 PyTorch 内部的函数。"""

    def parse(x):
        """解析边界框格式之间的转换，如 XYWH 和 LTWH。"""
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
to_4tuple = _ntuple(4)

# `xyxy` 表示左上角和右下角
# `xywh` 表示中心 x、中心 y 和宽度、高度（YOLO 格式）
# `ltwh` 表示左上角和宽度、高度（COCO 格式）
_formats = ["xyxy", "xywh", "ltwh"]

__all__ = ("Bboxes", "Instances")  # 元组或列表


class Bboxes:
    """
    用于处理边界框的类。

    该类支持多种边界框格式，如 'xyxy'、'xywh' 和 'ltwh'。
    边界框数据应以 numpy 数组提供。

    属性：
        bboxes (numpy.ndarray): 存储边界框的 2D numpy 数组。
        format (str): 边界框的格式（'xyxy'、'xywh' 或 'ltwh'）。

    注意：
        该类不处理边界框的归一化或反归一化。
    """

    def __init__(self, bboxes, format="xyxy") -> None:
        """使用指定格式的边界框数据初始化 Bboxes 类。"""
        assert format in _formats, f"无效的边界框格式：{format}，格式必须是 {_formats} 中的一个"
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format
        # self.normalized = normalized

    def convert(self, format):
        """将边界框格式从一种类型转换为另一种类型。"""
        assert format in _formats, f"无效的边界框格式：{format}，格式必须是 {_formats} 中的一个"
        if self.format == format:
            return
        elif self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else:
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self):
        """返回框的面积。"""
        return (
            (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])  # xyxy 格式
            if self.format == "xyxy"
            else self.bboxes[:, 3] * self.bboxes[:, 2]  # xywh 或 ltwh 格式
        )

    # def denormalize(self, w, h):
    #    if not self.normalized:
    #         return
    #    assert (self.bboxes <= 1.0).all()
    #    self.bboxes[:, 0::2] *= w
    #    self.bboxes[:, 1::2] *= h
    #    self.normalized = False
    #
    # def normalize(self, w, h):
    #     if self.normalized:
    #         return
    #     assert (self.bboxes > 1.0).any()
    #     self.bboxes[:, 0::2] /= w
    #     self.bboxes[:, 1::2] /= h
    #     self.normalized = True

    def mul(self, scale):
        """
        将边界框坐标乘以缩放因子。

        参数：
            scale (int | tuple | list): 用于四个坐标的缩放因子。
                如果是 int，则所有坐标应用相同的缩放因子。
        """
        if isinstance(scale, Number):
            scale = to_4tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset):
        """
        向边界框坐标添加偏移量。

        参数：
            offset (int | tuple | list): 用于四个坐标的偏移量。
                如果是 int，则所有坐标应用相同的偏移量。
        """
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self):
        """返回边界框的数量。"""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: List["Bboxes"], axis=0) -> "Bboxes":
        """
        将多个 Bboxes 对象合并为一个 Bboxes 对象。

        参数：
            boxes_list (List[Bboxes]): 要合并的 Bboxes 对象列表。
            axis (int, 可选): 沿着哪个轴进行合并，默认为 0。

        返回：
            Bboxes: 一个包含合并后边界框的新 Bboxes 对象。

        注意：
            输入应该是一个 Bboxes 对象列表或元组。
        """
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty(0))
        assert all(isinstance(box, Bboxes) for box in boxes_list)

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))

def __getitem__(self, index) -> "Bboxes":
    """
    使用索引检索特定的边界框或一组边界框。

    参数:
        index (int, slice, 或 np.ndarray): 用于选择所需边界框的索引、切片或布尔数组。

    返回:
        Bboxes: 包含所选边界框的新Bboxes对象。

    异常:
        AssertionError: 如果索引的边界框未形成二维矩阵。

    注意:
        使用布尔索引时，请确保提供一个与边界框数量相同长度的布尔数组。
    """
    if isinstance(index, int):
        return Bboxes(self.bboxes[index].reshape(1, -1))
    b = self.bboxes[index]
    assert b.ndim == 2, f"Indexing on Bboxes with {index} failed to return a matrix!"
    return Bboxes(b)


class Instances:
    """
    包含检测到的图像中对象的边界框、分割和关键点。

    属性:
        _bboxes (Bboxes): 用于处理边界框操作的内部对象。
        keypoints (ndarray): 形状为 [N, 17, 3] 的关键点(x, y, 可见性)。默认值为 None。
        normalized (bool): 标志，指示边界框坐标是否已归一化。
        segments (ndarray): 形状为 [N, 1000, 2] 的分割数组，经过重采样后。

    参数:
        bboxes (ndarray): 形状为 [N, 4] 的边界框数组。
        segments (list | ndarray, 可选): 对象分割的列表或数组。默认为 None。
        keypoints (ndarray, 可选): 形状为 [N, 17, 3] 的关键点数组，格式为(x, y, 可见性)。默认为 None。
        bbox_format (str, 可选): 边界框的格式（'xywh' 或 'xyxy'）。默认为 'xywh'。
        normalized (bool, 可选): 边界框坐标是否归一化。默认为 True。

    示例:
        ```python
        # 创建一个Instances对象
        instances = Instances(
            bboxes=np.array([[10, 10, 30, 30], [20, 20, 40, 40]]),
            segments=[np.array([[5, 5], [10, 10]]), np.array([[15, 15], [20, 20]])],
            keypoints=np.array([[[5, 5, 1], [10, 10, 1]], [[15, 15, 1], [20, 20, 1]]]),
        )
        ```

    注意:
        边界框格式为 'xywh' 或 'xyxy'，由 `bbox_format` 参数决定。
        该类不执行输入验证，假设输入是有效的。
    """

    def __init__(self, bboxes, segments=None, keypoints=None, bbox_format="xywh", normalized=True) -> None:
        """
        使用边界框、分割和关键点初始化对象。

        参数:
            bboxes (np.ndarray): 边界框，形状为 [N, 4]。
            segments (list | np.ndarray, 可选): 分割掩码。默认为 None。
            keypoints (np.ndarray, 可选): 关键点，形状为 [N, 17, 3]，格式为(x, y, 可见性)。默认为 None。
            bbox_format (str, 可选): 边界框格式。默认为 "xywh"。
            normalized (bool, 可选): 坐标是否归一化。默认为 True。
        """
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.keypoints = keypoints
        self.normalized = normalized
        self.segments = segments

    def convert_bbox(self, format):
        """转换边界框格式。"""
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self):
        """计算边界框的面积。"""
        return self._bboxes.areas()

    def scale(self, scale_w, scale_h, bbox_only=False):
        """类似于 denormalize 函数，但没有归一化标志。"""
        self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))
        if bbox_only:
            return
        self.segments[..., 0] *= scale_w
        self.segments[..., 1] *= scale_h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= scale_w
            self.keypoints[..., 1] *= scale_h

    def denormalize(self, w, h):
        """将边界框、分割和关键点从归一化坐标反归一化到实际尺寸。"""
        if not self.normalized:
            return
        self._bboxes.mul(scale=(w, h, w, h))
        self.segments[..., 0] *= w
        self.segments[..., 1] *= h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= w
            self.keypoints[..., 1] *= h
        self.normalized = False

    def normalize(self, w, h):
        """将边界框、分割和关键点归一化到图像尺寸。"""
        if self.normalized:
            return
        self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
        self.segments[..., 0] /= w
        self.segments[..., 1] /= h
        if self.keypoints is not None:
            self.keypoints[..., 0] /= w
            self.keypoints[..., 1] /= h
        self.normalized = True

    def add_padding(self, padw, padh):
        """处理矩形和拼接情况。"""
        assert not self.normalized, "您应该使用绝对坐标添加填充。"
        self._bboxes.add(offset=(padw, padh, padw, padh))
        self.segments[..., 0] += padw
        self.segments[..., 1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh

    def __getitem__(self, index) -> "Instances":
        """
        通过索引检索特定的实例或一组实例。

        参数：
            index (int, slice, 或 np.ndarray)：索引、切片或布尔数组，用于选择所需的实例。

        返回：
            Instances: 一个新的 Instances 对象，包含所选的边界框、分段和关键点（如果存在）。

        注意：
            使用布尔索引时，请确保提供的布尔数组长度与实例的数量相同。
        """
        segments = self.segments[index] if len(self.segments) else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index]
        bbox_format = self._bboxes.format
        return Instances(
            bboxes=bboxes,
            segments=segments,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=self.normalized,
        )

    def flipud(self, h):
        """垂直翻转边界框、分段和关键点的坐标。"""
        if self._bboxes.format == "xyxy":
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        else:
            self.bboxes[:, 1] = h - self.bboxes[:, 1]
        self.segments[..., 1] = h - self.segments[..., 1]
        if self.keypoints is not None:
            self.keypoints[..., 1] = h - self.keypoints[..., 1]

    def fliplr(self, w):
        """水平翻转边界框和分段的顺序。"""
        if self._bboxes.format == "xyxy":
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        else:
            self.bboxes[:, 0] = w - self.bboxes[:, 0]
        self.segments[..., 0] = w - self.segments[..., 0]
        if self.keypoints is not None:
            self.keypoints[..., 0] = w - self.keypoints[..., 0]

    def clip(self, w, h):
        """将边界框、分段和关键点的值裁剪到图像边界内。"""
        ori_format = self._bboxes.format
        self.convert_bbox(format="xyxy")
        self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
        self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
        if ori_format != "xyxy":
            self.convert_bbox(format=ori_format)
        self.segments[..., 0] = self.segments[..., 0].clip(0, w)
        self.segments[..., 1] = self.segments[..., 1].clip(0, h)
        if self.keypoints is not None:
            self.keypoints[..., 0] = self.keypoints[..., 0].clip(0, w)
            self.keypoints[..., 1] = self.keypoints[..., 1].clip(0, h)

    def remove_zero_area_boxes(self):
        """移除零面积的边界框，即裁剪后可能出现宽度或高度为零的框。"""
        good = self.bbox_areas > 0
        if not all(good):
            self._bboxes = self._bboxes[good]
            if len(self.segments):
                self.segments = self.segments[good]
            if self.keypoints is not None:
                self.keypoints = self.keypoints[good]
        return good

    def update(self, bboxes, segments=None, keypoints=None):
        """更新实例变量。"""
        self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
        if segments is not None:
            self.segments = segments
        if keypoints is not None:
            self.keypoints = keypoints

    def __len__(self):
        """返回实例列表的长度。"""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: List["Instances"], axis=0) -> "Instances":
        """
        将一组 Instances 对象连接成一个单一的 Instances 对象。

        参数：
            instances_list (List[Instances]): 要连接的 Instances 对象列表。
            axis (int, 可选): 数组连接的轴，默认为 0。

        返回：
            Instances: 一个新的 Instances 对象，包含连接后的边界框、分段和关键点（如果存在）。

        注意：
            列表中的 `Instances` 对象应具有相同的属性，例如边界框的格式、是否存在关键点以及坐标是否归一化。
        """
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty(0))
        assert all(isinstance(instance, Instances) for instance in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        use_keypoint = instances_list[0].keypoints is not None
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized

        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
        seg_len = [b.segments.shape[1] for b in instances_list]
        if len(set(seg_len)) > 1:  # 如果分段长度不同，则重新采样
            max_len = max(seg_len)
            cat_segments = np.concatenate(
                [
                    resample_segments(list(b.segments), max_len)
                    if len(b.segments)
                    else np.zeros((0, max_len, 2), dtype=np.float32)  # 重新生成空的分段
                    for b in instances_list
                ],
                axis=axis,
            )
        else:
            cat_segments = np.concatenate([b.segments for b in instances_list], axis=axis)
        cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None
        return cls(cat_boxes, cat_segments, cat_keypoints, bbox_format, normalized)

    @property
    def bboxes(self):
        """返回边界框。"""
        return self._bboxes.bboxes

