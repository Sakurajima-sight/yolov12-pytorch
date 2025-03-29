# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import math
import re
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import batch_probiou


class Profile(contextlib.ContextDecorator):
    """
    YOLOv8 计时器类。可以通过 @Profile() 装饰器使用，也可以通过 'with Profile():' 作为上下文管理器使用。

    示例:
        ```python
        from ultralytics.utils.ops import Profile

        with Profile(device=device) as dt:
            pass  # 在这里执行耗时操作

        print(dt)  # 打印 "Elapsed time is 9.5367431640625e-07 s"
        ```
    """

    def __init__(self, t=0.0, device: torch.device = None):
        """
        初始化 Profile 类。

        参数:
            t (float): 初始时间，默认为 0.0。
            device (torch.device): 用于模型推理的设备，默认为 None (cpu)。
        """
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """开始计时。"""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):  # noqa
        """停止计时。"""
        self.dt = self.time() - self.start  # 时间差
        self.t += self.dt  # 累加时间差

    def __str__(self):
        """返回一个可读字符串，表示计时器累计的经过时间。"""
        return f"Elapsed time is {self.t} s"

    def time(self):
        """获取当前时间。"""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()


def segment2box(segment, width=640, height=640):
    """
    将 1 个分割标签转换为 1 个框标签，应用图像内的约束，即将 (xy1, xy2, ...) 转换为 (xyxy)。

    参数:
        segment (torch.Tensor): 分割标签
        width (int): 图像的宽度，默认为 640
        height (int): 图像的高度，默认为 640

    返回:
        (np.ndarray): 分割区域的最小和最大 x 和 y 值。
    """
    x, y = segment.T  # 分割的 x 和 y 坐标
    # 如果 3 个边界坐标中的任意 3 个超出了图像边界，先裁剪坐标
    if np.array([x.min() < 0, y.min() < 0, x.max() > width, y.max() > height]).sum() >= 3:
        x = x.clip(0, width)
        y = y.clip(0, height)
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x = x[inside]
    y = y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )  # 返回格式为 xyxy


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    将边界框（默认为 xyxy 格式）从图像原始尺寸（img1_shape）缩放到目标图像尺寸（img0_shape）。

    参数:
        img1_shape (tuple): 原始图像的尺寸，格式为 (高度, 宽度)。
        boxes (torch.Tensor): 图像中的物体边界框，格式为 (x1, y1, x2, y2)
        img0_shape (tuple): 目标图像的尺寸，格式为 (高度, 宽度)。
        ratio_pad (tuple): 用于缩放框的 (比例, 填充) 元组。如果未提供，则根据两幅图像之间的尺寸差计算比例和填充。
        padding (bool): 如果为 True，假设边界框基于 YOLO 风格的图像增强。如果为 False，则进行常规缩放。
        xywh (bool): 是否使用 xywh 格式的框，默认为 False。

    返回:
        boxes (torch.Tensor): 缩放后的边界框，格式为 (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # 如果 ratio_pad 为 None，从 img0_shape 计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = 原始尺寸 / 目标尺寸
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # 宽高填充
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x 轴填充
        boxes[..., 1] -= pad[1]  # y 轴填充
        if not xywh:
            boxes[..., 2] -= pad[0]  # x 轴填充
            boxes[..., 3] -= pad[1]  # y 轴填充
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def make_divisible(x, divisor):
    """
    返回能被给定除数整除的最接近的整数。

    参数:
        x (int): 需要整除的数。
        divisor (int | torch.Tensor): 除数。

    返回:
        (int): 能被除数整除的最接近的整数。
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # 转为整数
    return math.ceil(x / divisor) * divisor


def nms_rotated(boxes, scores, threshold=0.45):
    """
    使用 probiou 和 fast-nms 对旋转的边界框进行 NMS。

    参数:
        boxes (torch.Tensor): 旋转边界框，形状为 (N, 5)，格式为 xywhr。
        scores (torch.Tensor): 置信度分数，形状为 (N,)。
        threshold (float, optional): IoU 阈值。默认值为 0.45。

    返回:
        (torch.Tensor): 保留的框的索引（经过 NMS 后）。
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes).triu_(diagonal=1)
    pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
    return sorted_idx[pick]


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # 类别数（可选）
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
):
    """
    对一组边界框执行非极大值抑制 (NMS)，支持掩码和每个框多个标签。

    参数:
        prediction (torch.Tensor): 形状为 (batch_size, num_classes + 4 + num_masks, num_boxes) 的张量
            包含预测的框、类别和掩码。张量应该是模型输出的格式，例如 YOLO。
        conf_thres (float): 置信度阈值，低于该阈值的框将被过滤掉。
            有效值在 0.0 到 1.0 之间。
        iou_thres (float): IoU 阈值，低于该阈值的框将在 NMS 过程中被过滤掉。
            有效值在 0.0 到 1.0 之间。
        classes (List[int]): 要考虑的类别索引列表。如果为 None，则考虑所有类别。
        agnostic (bool): 如果为 True，模型将忽略类别数，所有类别将作为一个类别考虑。
        multi_label (bool): 如果为 True，每个框可以有多个标签。
        labels (List[List[Union[int, float, torch.Tensor]]]): 一个列表的列表，每个内列表包含给定图像的先验标签。
            列表应为 dataloader 输出的格式，每个标签是一个元组，格式为 (class_index, x1, y1, x2, y2)。
        max_det (int): 在 NMS 后保留的最大框数。
        nc (int, optional): 模型输出的类别数。此后的索引将被视为掩码。
        max_time_img (float): 处理一张图像的最大时间（秒）。
        max_nms (int): 输入到 torchvision.ops.nms() 的最大框数。
        max_wh (int): 最大的框宽度和高度（像素）。
        in_place (bool): 如果为 True，输入的预测张量将在原地修改。
        rotated (bool): 如果使用旋转边界框（OBB）进行 NMS。

    返回:
        (List[torch.Tensor]): 一个长度为 batch_size 的列表，其中每个元素是一个形状为
            (num_boxes, 6 + num_masks) 的张量，包含保留的框，列包括
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...)。
    """
    import torchvision  # 加速 'import ultralytics'

    # 检查
    assert 0 <= conf_thres <= 1, f"无效的置信度阈值 {conf_thres}，有效值范围是 0.0 到 1.0"
    assert 0 <= iou_thres <= 1, f"无效的 IoU 阈值 {iou_thres}，有效值范围是 0.0 到 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 模型验证输出，output = (inference_out, loss_out)
        prediction = prediction[0]  # 选择只包含推理输出的部分
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6:  # 端到端模型 (BNC, 即 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, 即 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # 类别数
    nm = prediction.shape[1] - nc - 4  # 掩码数
    mi = 4 + nc  # 掩码开始索引
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # 候选框

    # 设置
    time_limit = 2.0 + max_time_img * bs  # 超过此时间（秒）退出
    multi_label &= nc > 1  # 每个框多个标签（每张图像多用 0.5ms）

    prediction = prediction.transpose(-1, -2)  # 将形状从 (1,84,6300) 转换为 (1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh 转换为 xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh 转换为 xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # 图像索引，图像推理
        # 应用约束
        x = x[xc[xi]]  # 置信度

        # 如果启用自动标签，则合并先验标签
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # 框
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # 类别
            x = torch.cat((x, v), 0)

        # 如果没有剩余框，处理下一张图像
        if not x.shape[0]:
            continue

        # 检测矩阵 nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # 只选择最佳类别
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # 按类别过滤
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        # 检查形状
        n = x.shape[0]  # 框的数量
        if not n:  # 没有框
            continue
        if n > max_nms:  # 超过最大框数
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # 按置信度排序并去除多余框

        # 批量 NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 类别
        scores = x[:, 4]  # 置信度
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # 框（按类别偏移）
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # 限制检测框数量

        # # 实验性功能
        # merge = False  # 使用合并 NMS
        # if merge and (1 < n < 3E3):  # 合并 NMS（使用加权均值合并框）
        #     # 更新框的值为 boxes(i,4) = weights(i,n) * boxes(n,4)
        #     from .metrics import box_iou
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # IoU 矩阵
        #     weights = iou * scores[None]  # 框的权重
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # 合并后的框
        #     redundant = True  # 需要冗余的检测
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # 需要冗余

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS 时间限制 {time_limit:.3f}s 超过")
            break  # 超过时间限制

    return output


def clip_boxes(boxes, shape):
    """
    对边界框列表进行裁剪，使其适应给定的图像大小（高度，宽度）。

    参数：
        boxes (torch.Tensor): 要裁剪的边界框。
        shape (tuple): 图像的大小（高度，宽度）。

    返回：
        (torch.Tensor | numpy.ndarray): 裁剪后的边界框。
    """
    if isinstance(boxes, torch.Tensor):  # 对于单个元素裁剪速度更快（警告：Apple MPS bug 在 inplace .clamp_()）
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array（批量处理更快）
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def clip_coords(coords, shape):
    """
    将线段坐标裁剪到图像边界内。

    参数：
        coords (torch.Tensor | numpy.ndarray): 一组线段坐标。
        shape (tuple): 一个包含图像大小的元组，格式为（高度，宽度）。

    返回：
        (torch.Tensor | numpy.ndarray): 裁剪后的坐标
    """
    if isinstance(coords, torch.Tensor):  # 对于单个元素裁剪速度更快（警告：Apple MPS bug 在 inplace .clamp_()）
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])  # y
    else:  # np.array（批量处理更快）
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords


def scale_image(masks, im0_shape, ratio_pad=None):
    """
    将掩膜图像调整为原始图像大小。

    参数：
        masks (np.ndarray): 已调整和填充的掩膜/图像，形状为 [h, w, num] 或 [h, w, 3]。
        im0_shape (tuple): 原始图像的形状。
        ratio_pad (tuple): 填充与原始图像的比例。

    返回：
        masks (np.ndarray): 调整后的掩膜，形状为 [h, w, num]。
    """
    # 从 im1_shape 到 im0_shape 调整坐标（xyxy）
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # 从 im0_shape 计算
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # 宽高填充
    else:
        # gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"masks shape 的长度" 应该是 2 或 3，但得到的是 {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks


def xyxy2xywh(x):
    """
    将边界框坐标从（x1，y1，x2，y2）格式转换为（x，y，宽度，高度）格式，其中（x1，y1）是
    左上角，（x2，y2）是右下角。

    参数：
        x (np.ndarray | torch.Tensor): 输入的边界框坐标，格式为（x1，y1，x2，y2）。

    返回：
        y (np.ndarray | torch.Tensor): 转换后的边界框坐标，格式为（x，y，宽度，高度）。
    """
    assert x.shape[-1] == 4, f"输入的最后一个维度应该是 4，但实际形状是 {x.shape}"
    y = empty_like(x)  # 比 clone/copy 更快
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x 中心
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y 中心
    y[..., 2] = x[..., 2] - x[..., 0]  # 宽度
    y[..., 3] = x[..., 3] - x[..., 1]  # 高度
    return y


def xywh2xyxy(x):
    """
    将边界框坐标从（x，y，宽度，高度）格式转换为（x1，y1，x2，y2）格式，其中（x1，y1）是
    左上角，（x2，y2）是右下角。注意：每 2 个通道的操作比每个通道操作更快。

    参数：
        x (np.ndarray | torch.Tensor): 输入的边界框坐标，格式为（x，y，宽度，高度）。

    返回：
        y (np.ndarray | torch.Tensor): 转换后的边界框坐标，格式为（x1，y1，x2，y2）。
    """
    assert x.shape[-1] == 4, f"输入的最后一个维度应该是 4，但实际形状是 {x.shape}"
    y = empty_like(x)  # 比 clone/copy 更快
    xy = x[..., :2]  # 中心
    wh = x[..., 2:] / 2  # 宽高的一半
    y[..., :2] = xy - wh  # 左上角 xy
    y[..., 2:] = xy + wh  # 右下角 xy
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    将归一化的边界框坐标转换为像素坐标。

    参数：
        x (np.ndarray | torch.Tensor): 边界框坐标。
        w (int): 图像的宽度，默认为 640。
        h (int): 图像的高度，默认为 640。
        padw (int): 填充宽度，默认为 0。
        padh (int): 填充高度，默认为 0。

    返回：
        y (np.ndarray | torch.Tensor): 以 [x1, y1, x2, y2] 格式表示的边界框坐标，其中
            x1, y1 是左上角坐标，x2, y2 是右下角坐标。
    """
    assert x.shape[-1] == 4, f"输入形状的最后一维应为 4，但输入的形状为 {x.shape}"
    y = empty_like(x)  # 比 clone/copy 更快
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # 左上角 x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # 左上角 y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # 右下角 x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # 右下角 y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """
    将边界框坐标从 (x1, y1, x2, y2) 格式转换为 (x, y, 宽度, 高度, 归一化) 格式。
    x, y, 宽度和高度被归一化到图像尺寸。

    参数：
        x (np.ndarray | torch.Tensor): 输入的边界框坐标，格式为 (x1, y1, x2, y2)。
        w (int): 图像的宽度，默认为 640。
        h (int): 图像的高度，默认为 640。
        clip (bool): 如果为 True，边界框将被裁剪到图像边界内，默认为 False。
        eps (float): 边界框宽度和高度的最小值，默认为 0.0。

    返回：
        y (np.ndarray | torch.Tensor): 转换后的边界框坐标，格式为 (x, y, 宽度, 高度, 归一化)。
    """
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))
    assert x.shape[-1] == 4, f"输入形状的最后一维应为 4，但输入的形状为 {x.shape}"
    y = empty_like(x)  # 比 clone/copy 更快
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x 中心
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y 中心
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # 宽度
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # 高度
    return y


def xywh2ltwh(x):
    """
    将边界框格式从 [x, y, w, h] 转换为 [x1, y1, w, h] 格式，
    其中 x1, y1 是左上角坐标。

    参数：
        x (np.ndarray | torch.Tensor): 输入的边界框坐标，格式为 xywh。

    返回：
        y (np.ndarray | torch.Tensor): 转换后的边界框坐标，格式为 xyltwh。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # 左上角 x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # 左上角 y
    return y


def xyxy2ltwh(x):
    """
    将 nx4 的边界框从 [x1, y1, x2, y2] 格式转换为 [x1, y1, w, h] 格式，
    其中 xy1 是左上角，xy2 是右下角。

    参数：
        x (np.ndarray | torch.Tensor): 输入的边界框坐标，格式为 xyxy。

    返回：
        y (np.ndarray | torch.Tensor): 转换后的边界框坐标，格式为 xyltwh。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # 宽度
    y[..., 3] = x[..., 3] - x[..., 1]  # 高度
    return y


def ltwh2xywh(x):
    """
    将 nx4 的边界框从 [x1, y1, w, h] 格式转换为 [x, y, w, h] 格式，
    其中 xy1 是左上角，xy 是中心。

    参数：
        x (torch.Tensor): 输入的边界框坐标。

    返回：
        y (np.ndarray | torch.Tensor): 转换后的边界框坐标，格式为 xywh。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # 中心 x
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # 中心 y
    return y


def xyxyxyxy2xywhr(x):
    """
    将批量的有向边界框（OBB）从 [xy1, xy2, xy3, xy4] 转换为 [xywh, 旋转角度] 格式。
    旋转角度的值以弧度表示，范围从 0 到 pi/2。

    参数：
        x (numpy.ndarray | torch.Tensor): 输入的边界框角点，格式为 [xy1, xy2, xy3, xy4]，形状为 (n, 8)。

    返回：
        (numpy.ndarray | torch.Tensor): 转换后的数据，格式为 [cx, cy, w, h, 旋转角度]，形状为 (n, 5)。
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        # 注意：使用 cv2.minAreaRect 来获得准确的 xywhr，
        # 特别是一些物体在数据加载时被裁剪。
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)


def xywhr2xyxyxyxy(x):
    """
    将批量的有向边界框（OBB）从 [xywh, 旋转角度] 转换为 [xy1, xy2, xy3, xy4] 格式。
    旋转角度应以弧度表示，范围从 0 到 pi/2。

    参数：
        x (numpy.ndarray | torch.Tensor): 以 [cx, cy, w, h, 旋转角度] 格式表示的边界框，形状为 (n, 5) 或 (b, n, 5)。

    返回：
        (numpy.ndarray | torch.Tensor): 转换后的角点，形状为 (n, 4, 2) 或 (b, n, 4, 2)。
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def ltwh2xyxy(x):
    """
    将边界框从 [x1, y1, w, h] 转换为 [x1, y1, x2, y2]，其中 xy1 为左上角，xy2 为右下角。

    参数：
        x (np.ndarray | torch.Tensor): 输入图像

    返回：
        y (np.ndarray | torch.Tensor): 边界框的 xyxy 坐标。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]  # 宽度
    y[..., 3] = x[..., 3] + x[..., 1]  # 高度
    return y


def segments2boxes(segments):
    """
    将分割标签转换为框标签，即从 (cls, xy1, xy2, ...) 转换为 (cls, xywh)。

    参数：
        segments (list): 分割的列表，每个分割是一个点的列表，每个点是一个 x, y 坐标

    返回：
        (np.ndarray): 边界框的 xywh 坐标。
    """
    boxes = []
    for s in segments:
        x, y = s.T  # 分割的 xy 坐标
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    """
    输入一个分割列表 (n,2)，返回一个分割列表，其中每个分割都有 n 个点。

    参数：
        segments (list): 一个 (n,2) 数组的列表，其中 n 是每个分割中的点数。
        n (int): 每个分割要重新采样到的点数。默认为 1000。

    返回：
        segments (list): 重新采样后的分割。
    """
    for i, s in enumerate(segments):
        if len(s) == n:
            continue
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n - len(s) if len(s) < n else n)
        xp = np.arange(len(s))
        x = np.insert(x, np.searchsorted(x, xp), xp) if len(s) < n else x
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # 分割的 xy 坐标
    return segments


def crop_mask(masks, boxes):
    """
    获取一个掩码和一个边界框，并返回裁剪到该边界框的掩码。

    参数：
        masks (torch.Tensor): [n, h, w] 形状的掩码张量
        boxes (torch.Tensor): [n, 4] 形状的边界框坐标，表示为相对坐标

    返回：
        (torch.Tensor): 裁剪到边界框的掩码。
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 形状(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # 行 形状(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # 列 形状(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    使用掩码头的输出应用掩码到边界框。

    参数：
        protos (torch.Tensor): 形状为 [mask_dim, mask_h, mask_w] 的张量。
        masks_in (torch.Tensor): 形状为 [n, mask_dim] 的张量，其中 n 是 NMS 后的掩码数量。
        bboxes (torch.Tensor): 形状为 [n, 4] 的张量，其中 n 是 NMS 后的掩码数量。
        shape (tuple): 表示输入图像大小的元组 (h, w)。
        upsample (bool): 一个标志，指示是否将掩码上采样到原始图像大小。默认为 False。

    返回：
        (torch.Tensor): 形状为 [n, h, w] 的二进制掩码张量，其中 n 是 NMS 后的掩码数量，h 和 w 是输入图像的高度和宽度。
            掩码已应用于边界框。
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # CHW
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    return masks.gt_(0.0)


def process_mask_native(protos, masks_in, bboxes, shape):
    """
    获取掩码头的输出，并在上采样后将其裁剪到边界框。

    参数：
        protos (torch.Tensor): [mask_dim, mask_h, mask_w]
        masks_in (torch.Tensor): [n, mask_dim]，n 是 NMS 后的掩码数量。
        bboxes (torch.Tensor): [n, 4]，n 是 NMS 后的掩码数量。
        shape (tuple): 输入图像的大小 (h, w)。

    返回：
        masks (torch.Tensor): 返回的掩码，形状为 [h, w, n]。
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = scale_masks(masks[None], shape)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.0)


def scale_masks(masks, shape, padding=True):
    """
    将分割掩码调整为目标大小。

    参数：
        masks (torch.Tensor): (N, C, H, W)。
        shape (tuple): 高度和宽度。
        padding (bool): 如果为 True，假设框是基于图像通过 YOLO 样式进行增强的。如果为 False，则进行常规的
            重新缩放。
    """
    mh, mw = masks.shape[2:]
    gain = min(mh / shape[0], mw / shape[1])  # gain = 旧尺寸 / 新尺寸
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # 宽高填充
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)  # y, x
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    masks = masks[..., top:bottom, left:right]

    masks = F.interpolate(masks, shape, mode="bilinear", align_corners=False)  # NCHW
    return masks


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    将分割坐标 (xy) 从 img1_shape 缩放到 img0_shape。

    参数:
        img1_shape (tuple): 坐标所在图像的尺寸。
        coords (torch.Tensor): 要缩放的坐标，形状为 n,2。
        img0_shape (tuple): 应用分割的目标图像尺寸。
        ratio_pad (tuple): 图像大小与填充后图像大小的比率。
        normalize (bool): 如果为 True，坐标将被归一化到 [0, 1] 范围。默认为 False。
        padding (bool): 如果为 True，假设边界框是基于 YOLO 风格的图像增强。如果为 False，则进行常规的重新缩放。

    返回:
        coords (torch.Tensor): 缩放后的坐标。
    """
    if ratio_pad is None:  # 从 img0_shape 计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = 原始尺寸 / 新尺寸
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # 宽高填充
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x 填充
        coords[..., 1] -= pad[1]  # y 填充
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # 宽度
        coords[..., 1] /= img0_shape[0]  # 高度
    return coords


def regularize_rboxes(rboxes):
    """
    将旋转的边界框正规化到 [0, pi/2] 范围内。

    参数:
        rboxes (torch.Tensor): 输入的边界框，形状为 (N, 5)，xywhr 格式。

    返回:
        (torch.Tensor): 正规化后的边界框。
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)
    # 如果 h >= w，则交换宽度和高度及角度
    w_ = torch.where(w > h, w, h)
    h_ = torch.where(w > h, h, w)
    t = torch.where(w > h, t, t + math.pi / 2) % math.pi
    return torch.stack([x, y, w_, h_, t], dim=-1)  # 返回正规化后的边界框


def masks2segments(masks, strategy="all"):
    """
    将一组掩码 (n,h,w) 转换为一组分割 (n,xy)。

    参数:
        masks (torch.Tensor): 模型的输出，形状为 (batch_size, 160, 160) 的张量。
        strategy (str): 'all' 或 'largest'。默认为 'all'。

    返回:
        segments (List): 分割掩码的列表。
    """
    from ultralytics.data.converter import merge_multi_segment

    segments = []
    for x in masks.int().cpu().numpy().astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "all":  # 合并并连接所有分割
                c = (
                    np.concatenate(merge_multi_segment([x.reshape(-1, 2) for x in c]))
                    if len(c) > 1
                    else c[0].reshape(-1, 2)
                )
            elif strategy == "largest":  # 选择最大的分割
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # 未找到分割
        segments.append(c.astype("float32"))
    return segments


def convert_torch2numpy_batch(batch: torch.Tensor) -> np.ndarray:
    """
    将一批 FP32 的 torch 张量（值在 0.0 到 1.0 之间）转换为 NumPy uint8 数组（值在 0 到 255 之间），
    并从 BCHW 格式转换为 BHWC 格式。

    参数:
        batch (torch.Tensor): 输入的张量批次，形状为 (Batch, Channels, Height, Width)，数据类型为 torch.float32。

    返回:
        (np.ndarray): 输出的 NumPy 数组批次，形状为 (Batch, Height, Width, Channels)，数据类型为 uint8。
    """
    return (batch.permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()


def clean_str(s):
    """
    清理字符串，将特殊字符替换为 '_' 字符。

    参数:
        s (str): 需要替换特殊字符的字符串。

    返回:
        (str): 字符串，特殊字符被替换为下划线 _。
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def empty_like(x):
    """创建一个与输入形状相同，数据类型为 float32 的空张量或 NumPy 数组。"""
    return (
        torch.empty_like(x, dtype=torch.float32) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=np.float32)
    )
