# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""模型验证指标。"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics.utils import LOGGER, SimpleClass, TryExcept, plt_settings

OKS_SIGMA = (
    np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
    / 10.0
)


def bbox_ioa(box1, box2, iou=False, eps=1e-7):
    """
    计算给定 box1 和 box2 的交集与 box2 面积的比值。框的格式为 x1y1x2y2。

    参数:
        box1 (np.ndarray): 形状为 (n, 4) 的 numpy 数组，表示 n 个边界框。
        box2 (np.ndarray): 形状为 (m, 4) 的 numpy 数组，表示 m 个边界框。
        iou (bool): 如果为 True，则计算标准的 IoU，否则返回交集面积与 box2 面积的比值。
        eps (float, optional): 用于避免除零的小值，默认值为 1e-7。

    返回:
        (np.ndarray): 形状为 (n, m) 的 numpy 数组，表示交集与 box2 面积的比值。
    """
    # 获取边界框的坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # 交集面积
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # box2 面积
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # 交集与 box2 面积的比值
    return inter_area / (area + eps)


def box_iou(box1, box2, eps=1e-7):
    """
    计算边界框的交并比 (IoU)。两个边界框的输入应为 (x1, y1, x2, y2) 格式。
    基于 https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py。

    参数:
        box1 (torch.Tensor): 形状为 (N, 4) 的张量，表示 N 个边界框。
        box2 (torch.Tensor): 形状为 (M, 4) 的张量，表示 M 个边界框。
        eps (float, optional): 用于避免除零的小值，默认值为 1e-7。

    返回:
        (torch.Tensor): 形状为 (N, M) 的张量，表示 box1 和 box2 中每一对元素的 IoU。
    """
    # 注意：需要 .float() 来获取准确的 IoU 值
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    计算边界框之间的交并比 (IoU)。

    该函数支持 `box1` 和 `box2` 的多种形状，只要最后一维是 4 即可。
    例如，可以传递形状为 (4,)、(N, 4)、(B, N, 4) 或 (B, N, 1, 4) 的张量。
    内部代码会将最后一维分为 (x, y, w, h) 格式，如果 `xywh=True`，
    或 (x1, y1, x2, y2) 格式，如果 `xywh=False`。

    参数:
        box1 (torch.Tensor): 形状为 (N, 4) 或其他形状的张量，表示边界框。
        box2 (torch.Tensor): 形状为 (M, 4) 或其他形状的张量，表示边界框。
        xywh (bool, optional): 如果为 True，则输入框为 (x, y, w, h) 格式。如果为 False，则输入框为
                               (x1, y1, x2, y2) 格式。默认值为 True。
        GIoU (bool, optional): 如果为 True，则计算广义 IoU。默认值为 False。
        DIoU (bool, optional): 如果为 True，则计算距离 IoU。默认值为 False。
        CIoU (bool, optional): 如果为 True，则计算完全 IoU。默认值为 False。
        eps (float, optional): 用于避免除零的小值，默认值为 1e-7。

    返回:
        (torch.Tensor): 根据指定的标志返回 IoU、GIoU、DIoU 或 CIoU 的值。
    """
    # 获取边界框的坐标
    if xywh:  # 从 xywh 转换到 xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # 交集面积
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # 并集面积
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # 最小外接框宽度
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # 最小外接框高度
        if CIoU or DIoU:  # 距离或完全 IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # 外接对角线的平方
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # 中心距离的平方
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # 外接框面积
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def mask_iou(mask1, mask2, eps=1e-7):
    """
    计算 mask 的交并比 (IoU)。

    参数:
        mask1 (torch.Tensor): 形状为 (N, n) 的张量，其中 N 是真实目标的数量，n 是图像宽度和高度的乘积。
        mask2 (torch.Tensor): 形状为 (M, n) 的张量，其中 M 是预测目标的数量，n 是图像宽度和高度的乘积。
        eps (float, optional): 用于避免除零的小值，默认值为 1e-7。

    返回:
        (torch.Tensor): 形状为 (N, M) 的张量，表示 mask 的交并比。
    """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def kpt_iou(kpt1, kpt2, area, sigma, eps=1e-7):
    """
    计算物体关键点相似度（OKS）。

    参数：
        kpt1 (torch.Tensor): 形状为 (N, 17, 3) 的张量，表示真实的关键点。
        kpt2 (torch.Tensor): 形状为 (M, 17, 3) 的张量，表示预测的关键点。
        area (torch.Tensor): 形状为 (N,) 的张量，表示真实边界框的面积。
        sigma (list): 一个包含 17 个值的列表，表示关键点的尺度。
        eps (float, 可选): 一个小值，用于避免除以零的错误，默认为 1e-7。

    返回：
        (torch.Tensor): 形状为 (N, M) 的张量，表示关键点相似度。
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)  # (N, M, 17)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  # (17, )
    kpt_mask = kpt1[..., 2] != 0  # (N, 17)
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # 来自 cocoeval
    # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # 来自公式
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)


def _get_covariance_matrix(boxes):
    """
    从旋转边界框生成协方差矩阵。

    参数：
        boxes (torch.Tensor): 形状为 (N, 5) 的张量，表示旋转边界框，xywhr 格式。

    返回：
        (torch.Tensor): 与原始旋转边界框对应的协方差矩阵。
    """
    # 高斯边界框，忽略中心点（前两列），因为在这里不需要
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    计算定向边界框之间的概率 IoU。

    实现算法来自：https://arxiv.org/pdf/2106.06072v1.pdf。

    参数：
        obb1 (torch.Tensor): 真实的 OBB，形状为 (N, 5)，格式为 xywhr。
        obb2 (torch.Tensor): 预测的 OBB，形状为 (N, 5)，格式为 xywhr。
        CIoU (bool, 可选): 如果为 True，则计算 CIoU。默认为 False。
        eps (float, 可选): 一个小值，用于避免除以零的错误，默认为 1e-7。

    返回：
        (torch.Tensor): OBB 相似度，形状为 (N,)。

    注意：
        OBB 格式：[center_x, center_y, width, height, rotation_angle]。
        如果 CIoU 为 True，则返回 CIoU 而不是 IoU。
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:  # 仅包括宽高比部分
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou


def batch_probiou(obb1, obb2, eps=1e-7):
    """
    计算批量定向边界框之间的概率 IoU，https://arxiv.org/pdf/2106.06072v1.pdf。

    参数：
        obb1 (torch.Tensor | np.ndarray): 形状为 (N, 5) 的张量，表示真实的 OBB，格式为 xywhr。
        obb2 (torch.Tensor | np.ndarray): 形状为 (M, 5) 的张量，表示预测的 OBB，格式为 xywhr。
        eps (float, 可选): 一个小值，用于避免除以零的错误，默认为 1e-7。

    返回：
        (torch.Tensor): 形状为 (N, M) 的张量，表示 OBB 相似度。
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def smooth_bce(eps=0.1):
    """
    计算平滑的正负二元交叉熵目标。

    该函数根据给定的 epsilon 值计算平滑的正负标签交叉熵目标。
    实现细节参考：https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441。

    参数：
        eps (float, 可选): 用于标签平滑的 epsilon 值，默认为 0.1。

    返回：
        (tuple): 返回一个包含正负标签平滑交叉熵目标的元组。
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class ConfusionMatrix:
    """
    用于计算和更新目标检测和分类任务的混淆矩阵的类。

    属性：
        task (str): 任务类型，'detect' 或 'classify'。
        matrix (np.ndarray): 混淆矩阵，维度取决于任务类型。
        nc (int): 类别数量。
        conf (float): 检测的置信度阈值。
        iou_thres (float): 交并比（IoU）阈值。
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45, task="detect"):
        """初始化 YOLO 模型的属性。"""
        self.task = task
        self.matrix = np.zeros((nc + 1, nc + 1)) if self.task == "detect" else np.zeros((nc, nc))
        self.nc = nc  # 类别数量
        self.conf = 0.25 if conf in {None, 0.001} else conf  # 如果传入的是默认值 conf，则应用 0.25
        self.iou_thres = iou_thres

    def process_cls_preds(self, preds, targets):
        """
        更新分类任务的混淆矩阵。

        参数：
            preds (Array[N, min(nc,5)]): 预测的类别标签。
            targets (Array[N, 1]): 真实的类别标签。
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        更新目标检测任务的混淆矩阵。

        参数：
            detections (Array[N, 6] | Array[N, 7]): 检测到的边界框及其相关信息。
                                      每一行应包含 (x1, y1, x2, y2, conf, class)
                                      或在有额外的 `angle` 时是 obb 格式。
            gt_bboxes (Array[M, 4]| Array[N, 5]): 真实的边界框，xyxy/xyxyr 格式。
            gt_cls (Array[M]): 类别标签。
        """
        if gt_cls.shape[0] == 0:  # 检查标签是否为空
            if detections is not None:
                detections = detections[detections[:, 4] > self.conf]
                detection_classes = detections[:, 5].int()
                for dc in detection_classes:
                    self.matrix[dc, self.nc] += 1  # 错误的正例（假阳性）
            return
        if detections is None:
            gt_classes = gt_cls.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # 背景漏检（假阴性）
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = gt_cls.int()
        detection_classes = detections[:, 5].int()
        is_obb = detections.shape[1] == 7 and gt_bboxes.shape[1] == 5  # 有额外的 `angle` 维度
        iou = (
            batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
            if is_obb
            else box_iou(gt_bboxes, detections[:, :4])
        )

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # 正确的
            else:
                self.matrix[self.nc, gc] += 1  # 真背景

        for i, dc in enumerate(detection_classes):
            if not any(m1 == i):
                self.matrix[dc, self.nc] += 1  # 预测的背景

    def matrix(self):
        """返回混淆矩阵。"""
        return self.matrix

    def tp_fp(self):
        """返回真正例和假正例。"""
        tp = self.matrix.diagonal()  # 真正例
        fp = self.matrix.sum(1) - tp  # 假正例
        # fn = self.matrix.sum(0) - tp  # 假阴性（漏检）
        return (tp[:-1], fp[:-1]) if self.task == "detect" else (tp, fp)  # 如果任务是 detect，则去掉背景类

    @TryExcept("WARNING ⚠️ 混淆矩阵绘图失败")
    @plt_settings()
    def plot(self, normalize=True, save_dir="", names=(), on_plot=None):
        """
        使用 seaborn 绘制混淆矩阵并将其保存到文件。

        参数：
            normalize (bool): 是否对混淆矩阵进行归一化。
            save_dir (str): 绘图文件保存的目录。
            names (tuple): 类别名称，用作绘图时的标签。
            on_plot (func): 可选回调函数，在渲染图形时传递图形路径和数据。
        """
        import seaborn  # 为了更快地 'import ultralytics'

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # 归一化列
        array[array < 0.005] = np.nan  # 不注释（避免显示为 0.00）

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # 类别数量，名称
        seaborn.set_theme(font_scale=1.0 if nc < 50 else 0.8)  # 调整标签大小
        labels = (0 < nn < 99) and (nn == nc)  # 如果名称数量与类别数量匹配，则应用名称到刻度标签
        ticklabels = (list(names) + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 抑制空矩阵的运行时警告：遇到全部为 NaN 的切片
            seaborn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f" if normalize else ".0f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.set_title(title)
        plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)

    def print(self):
        """将混淆矩阵打印到控制台。"""
        for i in range(self.nc + 1):
            LOGGER.info(" ".join(map(str, self.matrix[i])))


def smooth(y, f=0.05):
    """对数据应用箱型滤波器，过滤器的分数为 f."""
    nf = round(len(y) * f * 2) // 2 + 1  # 滤波器元素的数量（必须是奇数）
    p = np.ones(nf // 2)  # 填充用的1
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # 填充 y
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # 对 y 进行平滑处理


@plt_settings()
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names={}, on_plot=None):
    """绘制精度-召回曲线."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # 如果类别少于21个，则显示每个类别的图例
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # 绘制(召回率, 精度)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # 绘制(召回率, 精度)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"All classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


@plt_settings()
def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names={}, xlabel="Confidence", ylabel="Metric", on_plot=None):
    """绘制度量-置信度曲线."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # 如果类别少于21个，则显示每个类别的图例
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # 绘制(置信度, 度量)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # 绘制(置信度, 度量)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"All Classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


def compute_ap(recall, precision):
    """
    根据召回率和精度曲线计算平均精度（AP）。

    参数：
        recall (list): 召回率曲线。
        precision (list): 精度曲线。

    返回：
        (float): 平均精度。
        (np.ndarray): 精度包络曲线。
        (np.ndarray): 修改后的召回率曲线，开始和结束处添加了哨兵值。
    """
    # 在开始和结束处添加哨兵值
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # 计算精度包络
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 计算曲线下的面积
    method = "interp"  # 方法: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101点插值（COCO）
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # 积分
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # 召回率变化的点
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # 曲线下的面积

    return ap, mpre, mrec


def ap_per_class(
    tp, conf, pred_cls, target_cls, plot=False, on_plot=None, save_dir=Path(), names={}, eps=1e-16, prefix=""
):
    """
    计算每个类别的平均精度（AP）用于目标检测评估。

    参数:
        tp (np.ndarray): 二进制数组，表示检测是否正确（True）或不正确（False）。
        conf (np.ndarray): 检测的置信度得分数组。
        pred_cls (np.ndarray): 检测的预测类别数组。
        target_cls (np.ndarray): 检测的真实类别数组。
        plot (bool, 可选): 是否绘制PR曲线。默认为False。
        on_plot (func, 可选): 绘制图表时传递图表路径和数据的回调函数。默认为None。
        save_dir (Path, 可选): 保存PR曲线的目录。默认为空路径。
        names (dict, 可选): 用于绘制PR曲线的类别名称字典。默认为空字典。
        eps (float, 可选): 一个小值，用于避免除以零。默认为1e-16。
        prefix (str, 可选): 保存图表文件时使用的前缀字符串。默认为空字符串。

    返回:
        tp (np.ndarray): 每个类别在最大F1度量下的真阳性计数，形状为(nc,)。
        fp (np.ndarray): 每个类别在最大F1度量下的假阳性计数，形状为(nc,)。
        p (np.ndarray): 每个类别在最大F1度量下的精度值，形状为(nc,)。
        r (np.ndarray): 每个类别在最大F1度量下的召回率值，形状为(nc,)。
        f1 (np.ndarray): 每个类别在最大F1度量下的F1得分，形状为(nc,)。
        ap (np.ndarray): 每个类别在不同IoU阈值下的平均精度，形状为(nc, 10)。
        unique_classes (np.ndarray): 包含数据的唯一类别数组，形状为(nc,)。
        p_curve (np.ndarray): 每个类别的精度曲线，形状为(nc, 1000)。
        r_curve (np.ndarray): 每个类别的召回曲线，形状为(nc, 1000)。
        f1_curve (np.ndarray): 每个类别的F1得分曲线，形状为(nc, 1000)。
        x (np.ndarray): 曲线的X轴值，形状为(1000,)。
        prec_values (np.ndarray): 每个类别在mAP@0.5下的精度值，形状为(nc, 1000)。
    """
    # 按对象性排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 查找唯一类别
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # 类别数，检测数

    # 创建精度-召回曲线并计算每个类别的AP
    x, prec_values = np.linspace(0, 1, 1000), []

    # 平均精度、精度和召回曲线
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # 标签数
        n_p = i.sum()  # 预测数
        if n_p == 0 or n_l == 0:
            continue

        # 累积FP和TP
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # 召回率
        recall = tpc / (n_l + eps)  # 召回曲线
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # 负x，xp因为xp是递减的

        # 精度
        precision = tpc / (tpc + fpc)  # 精度曲线
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # pr_score下的p

        # 从召回-精度曲线计算AP
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # mAP@0.5下的精度

    prec_values = np.array(prec_values)  # (nc, 1000)

    # 计算F1（精度和召回的调和平均）
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # 列表：只有包含数据的类别
    names = dict(enumerate(names))  # 转换为字典
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # 最大F1索引
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # 最大F1的精度、召回率和F1值
    tp = (r * nt).round()  # 真阳性
    fp = (tp / (p + eps) - tp).round()  # 假阳性
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values


class Metric(SimpleClass):
    """
    用于计算YOLOv8模型评估指标的类。

    属性:
        p (list): 每个类别的精度，形状为(nc,)。
        r (list): 每个类别的召回率，形状为(nc,)。
        f1 (list): 每个类别的F1得分，形状为(nc,)。
        all_ap (list): 所有类别和所有IoU阈值下的AP分数，形状为(nc, 10)。
        ap_class_index (list): 每个AP分数对应的类别索引，形状为(nc,)。
        nc (int): 类别数。

    方法:
        ap50(): IoU阈值为0.5时所有类别的AP，返回：AP分数列表，形状为(nc, )或[]。
        ap(): IoU阈值从0.5到0.95时所有类别的AP，返回：AP分数列表，形状为(nc, )或[]。
        mp(): 所有类别的平均精度，返回：浮动值。
        mr(): 所有类别的平均召回率，返回：浮动值。
        map50(): IoU阈值为0.5时所有类别的平均AP，返回：浮动值。
        map75(): IoU阈值为0.75时所有类别的平均AP，返回：浮动值。
        map(): IoU阈值从0.5到0.95时所有类别的平均AP，返回：浮动值。
        mean_results(): 结果的均值，返回mp、mr、map50、map。
        class_result(i): 类别感知结果，返回p[i]、r[i]、ap50[i]、ap[i]。
        maps(): 每个类别的mAP，返回：mAP分数数组，形状为(nc,)。
        fitness(): 模型适应度作为加权组合的指标，返回：浮动值。
        update(results): 使用新的评估结果更新指标属性。
    """

    def __init__(self) -> None:
        """初始化Metric实例，用于计算YOLOv8模型的评估指标。"""
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    @property
    def ap50(self):
        """
        返回所有类别在IoU阈值为0.5时的平均精度（AP）。

        返回:
            (np.ndarray, list): 形状为(nc,)的AP50值数组，或如果没有则返回空列表。
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """
        返回所有类别在IoU阈值从0.5到0.95时的平均精度（AP）。

        返回:
            (np.ndarray, list): 形状为(nc,)的AP50-95值数组，或如果没有则返回空列表。
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """
        返回所有类别的平均精度。

        返回:
            (float): 所有类别的平均精度。
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """
        返回所有类别的平均召回率。

        返回:
            (float): 所有类别的平均召回率。
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """
        返回在 IoU 阈值为 0.5 时的平均精度 (mAP)。

        返回:
            (float): 在 IoU 阈值为 0.5 时的 mAP。
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self):
        """
        返回在 IoU 阈值为 0.75 时的平均精度 (mAP)。

        返回:
            (float): 在 IoU 阈值为 0.75 时的 mAP。
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """
        返回在 IoU 阈值从 0.5 到 0.95，步长为 0.05 时的平均精度 (mAP)。

        返回:
            (float): 在 IoU 阈值从 0.5 到 0.95，步长为 0.05 时的 mAP。
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """返回各项指标的平均值，包括 mp, mr, map50, map。"""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i):
        """返回指定类别的结果，包括 p[i], r[i], ap50[i], ap[i]。"""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        """返回每个类别的 mAP 值。"""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self):
        """计算模型的适应度，作为多个指标的加权组合。"""
        w = [0.0, 0.0, 0.1, 0.9]  # 权重：[P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.array(self.mean_results()) * w).sum()

    def update(self, results):
        """
        使用一组新的结果更新模型的评估指标。

        参数:
            results (tuple): 一个包含以下评估指标的元组：
                - p (list): 每个类别的精度。形状: (nc,)。
                - r (list): 每个类别的召回率。形状: (nc,)。
                - f1 (list): 每个类别的 F1 分数。形状: (nc,)。
                - all_ap (list): 所有类别在所有 IoU 阈值下的 AP 分数。形状: (nc, 10)。
                - ap_class_index (list): 每个 AP 分数对应的类别索引。形状: (nc,)。

        副作用:
            根据提供的结果元组中的值更新类属性 `self.p`、`self.r`、`self.f1`、`self.all_ap` 和 `self.ap_class_index`。
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    @property
    def curves(self):
        """返回用于访问特定指标曲线的曲线列表。"""
        return []

    @property
    def curves_results(self):
        """返回用于访问特定指标曲线的曲线列表。"""
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]


class DetMetrics(SimpleClass):
    """
    用于计算目标检测模型的检测指标，如精度、召回率和平均精度 (mAP) 的实用类。

    参数:
        save_dir (Path): 输出图表保存的目录路径。默认为当前目录。
        plot (bool): 一个标志，指示是否为每个类别绘制精度-召回曲线。默认为 False。
        on_plot (func): 可选的回调函数，在渲染图表时传递图表路径和数据。默认为 None。
        names (dict of str): 一个表示类别名称的字符串字典。默认为空字典。

    属性:
        save_dir (Path): 输出图表保存的目录路径。
        plot (bool): 一个标志，指示是否为每个类别绘制精度-召回曲线。
        on_plot (func): 可选的回调函数，在渲染图表时传递图表路径和数据。
        names (dict of str): 一个表示类别名称的字符串字典。
        box (Metric): 存储检测指标结果的 Metric 类实例。
        speed (dict): 用于存储目标检测过程各部分执行时间的字典。

    方法:
        process(tp, conf, pred_cls, target_cls): 使用最新的预测批次更新指标结果。
        keys: 返回一个列表，用于访问已计算的检测指标。
        mean_results: 返回已计算的检测指标的平均值列表。
        class_result(i): 返回指定类别的检测指标结果。
        maps: 返回不同 IoU 阈值下的平均精度 (mAP) 值字典。
        fitness: 基于已计算的检测指标计算适应度评分。
        ap_class_index: 返回按平均精度 (AP) 排序的类别索引列表。
        results_dict: 返回一个字典，将检测指标键映射到已计算的值。
        curves: TODO
        curves_results: TODO
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names={}) -> None:
        """初始化 DetMetrics 实例，设置保存目录、绘图标志、回调函数和类名称。"""
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "detect"

    def process(self, tp, conf, pred_cls, target_cls):
        """处理预测结果并更新指标，适用于物体检测任务。"""
        results = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            save_dir=self.save_dir,
            names=self.names,
            on_plot=self.on_plot,
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self):
        """返回用于访问特定指标的键的列表。"""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def mean_results(self):
        """计算并返回检测物体的精度、召回率、mAP50 和 mAP50-95 的平均值。"""
        return self.box.mean_results()

    def class_result(self, i):
        """返回评估物体检测模型在特定类别上表现的结果。"""
        return self.box.class_result(i)

    @property
    def maps(self):
        """返回每个类别的平均精度（mAP）得分。"""
        return self.box.maps

    @property
    def fitness(self):
        """返回边界框对象的适应度得分。"""
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """返回每个类别的平均精度索引。"""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """返回已计算的性能指标和统计数据的字典。"""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """返回用于访问特定指标曲线的列表。"""
        return ["Precision-Recall(B)", "F1-Confidence(B)", "Precision-Confidence(B)", "Recall-Confidence(B)"]

    @property
    def curves_results(self):
        """返回已计算的性能指标和统计数据的字典。"""
        return self.box.curves_results


class SegmentMetrics(SimpleClass):
    """
    计算并汇总给定类别集上的检测和分割指标。

    参数：
        save_dir (Path): 输出绘图应保存的目录路径。默认为当前目录。
        plot (bool): 是否保存检测和分割绘图。默认为 False。
        on_plot (func): 一个可选的回调函数，在绘图渲染时传递绘图路径和数据。默认为 None。
        names (list): 类别名称列表。默认为空列表。

    属性：
        save_dir (Path): 输出绘图应保存的目录路径。
        plot (bool): 是否保存检测和分割绘图。
        on_plot (func): 一个可选的回调函数，在绘图渲染时传递绘图路径和数据。
        names (list): 类别名称列表。
        box (Metric): 用于计算边界框检测指标的 Metric 类实例。
        seg (Metric): 用于计算分割掩膜指标的 Metric 类实例。
        speed (dict): 存储推理各个阶段所花时间的字典。

    方法：
        process(tp_m, tp_b, conf, pred_cls, target_cls): 处理给定预测集上的指标。
        mean_results(): 返回所有类别上检测和分割指标的平均值。
        class_result(i): 返回类别 `i` 的检测和分割指标。
        maps: 返回 mAP 得分，适用于 IoU 阈值从 0.50 到 0.95 的范围。
        fitness: 返回适应度得分，这是一个加权的指标组合。
        ap_class_index: 返回用于计算平均精度（AP）的类别索引列表。
        results_dict: 返回包含所有检测和分割指标以及适应度得分的字典。
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """初始化 SegmentMetrics 实例，设置保存目录、绘图标志、回调函数和类名称。"""
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.seg = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "segment"

    def process(self, tp, tp_m, conf, pred_cls, target_cls):
        """
        处理给定预测集上的检测和分割指标。

        参数：
            tp (list): 真实正例边界框列表。
            tp_m (list): 真实正例掩膜列表。
            conf (list): 信心分数列表。
            pred_cls (list): 预测类别列表。
            target_cls (list): 目标类别列表。
        """
        results_mask = ap_per_class(
            tp_m,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Mask",
        )[2:]
        self.seg.nc = len(self.names)
        self.seg.update(results_mask)
        results_box = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Box",
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results_box)

    @property
    def keys(self):
        """返回用于访问指标的键列表。"""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
        ]

    def mean_results(self):
        """返回边界框和分割结果的平均指标。"""
        return self.box.mean_results() + self.seg.mean_results()

    def class_result(self, i):
        """返回指定类别索引的分类结果。"""
        return self.box.class_result(i) + self.seg.class_result(i)

    @property
    def maps(self):
        """返回目标检测和语义分割模型的 mAP 分数。"""
        return self.box.maps + self.seg.maps

    @property
    def fitness(self):
        """获取边界框和分割模型的适应度分数。"""
        return self.seg.fitness() + self.box.fitness()

    @property
    def ap_class_index(self):
        """边界框和掩模使用相同的 ap_class_index。"""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """返回用于评估的目标检测模型结果字典。"""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """返回用于访问特定指标曲线的曲线列表。"""
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
            "Precision-Recall(M)",
            "F1-Confidence(M)",
            "Precision-Confidence(M)",
            "Recall-Confidence(M)",
        ]

    @property
    def curves_results(self):
        """返回计算的性能指标和统计信息字典。"""
        return self.box.curves_results + self.seg.curves_results


class PoseMetrics(SegmentMetrics):
    """
    计算并聚合给定类别集上的检测和姿态指标。

    参数：
        save_dir (Path): 输出图像保存的目录路径，默认为当前目录。
        plot (bool): 是否保存检测和分割图像，默认为 False。
        on_plot (func): 一个可选的回调函数，用于在图像渲染时传递图像路径和数据，默认为 None。
        names (list): 类别名称的列表，默认为空列表。

    属性：
        save_dir (Path): 输出图像保存的目录路径。
        plot (bool): 是否保存检测和分割图像。
        on_plot (func): 一个可选的回调函数，用于在图像渲染时传递图像路径和数据。
        names (list): 类别名称的列表。
        box (Metric): 用于计算边界框检测指标的 Metric 类实例。
        pose (Metric): 用于计算掩模分割指标的 Metric 类实例。
        speed (dict): 存储推理过程各阶段所用时间的字典。

    方法：
        process(tp_m, tp_b, conf, pred_cls, target_cls): 处理给定预测集的指标。
        mean_results(): 返回所有类别上的检测和分割指标的平均值。
        class_result(i): 返回类别 `i` 的检测和分割指标。
        maps: 返回 IoU 阈值从 0.50 到 0.95 范围的平均精度（mAP）分数。
        fitness: 返回适应度分数，它是指标的加权组合。
        ap_class_index: 返回用于计算平均精度（AP）的类别索引列表。
        results_dict: 返回包含所有检测和分割指标及适应度分数的字典。
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """初始化 PoseMetrics 类，设置目录路径、类别名称和绘图选项。"""
        super().__init__(save_dir, plot, names)
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.pose = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "pose"

    def process(self, tp, tp_p, conf, pred_cls, target_cls):
        """
        处理给定预测集的检测和姿态指标。

        参数：
            tp (list): 真正例边界框的列表。
            tp_p (list): 真正例关键点的列表。
            conf (list): 置信度分数的列表。
            pred_cls (list): 预测类别的列表。
            target_cls (list): 目标类别的列表。
        """
        results_pose = ap_per_class(
            tp_p,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Pose",
        )[2:]
        self.pose.nc = len(self.names)
        self.pose.update(results_pose)
        results_box = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Box",
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results_box)

    @property
    def keys(self):
        """返回评估指标键的列表。"""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(P)",
            "metrics/recall(P)",
            "metrics/mAP50(P)",
            "metrics/mAP50-95(P)",
        ]

    def mean_results(self):
        """返回框和姿态的平均结果。"""
        return self.box.mean_results() + self.pose.mean_results()

    def class_result(self, i):
        """返回特定类别 i 的类别检测结果。"""
        return self.box.class_result(i) + self.pose.class_result(i)

    @property
    def maps(self):
        """返回框和姿态检测的每个类别的平均精度（mAP）。"""
        return self.box.maps + self.pose.maps

    @property
    def fitness(self):
        """使用 `targets` 和 `pred` 输入计算分类指标和速度。"""
        return self.pose.fitness() + self.box.fitness()

    @property
    def curves(self):
        """返回一个列表，用于访问特定指标的曲线。"""
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
            "Precision-Recall(P)",
            "F1-Confidence(P)",
            "Precision-Confidence(P)",
            "Recall-Confidence(P)",
        ]

    @property
    def curves_results(self):
        """返回计算后的性能指标和统计数据的字典。"""
        return self.box.curves_results + self.pose.curves_results


class ClassifyMetrics(SimpleClass):
    """
    计算分类指标的类，包括 top-1 和 top-5 准确率。

    属性：
        top1 (float): top-1 准确率。
        top5 (float): top-5 准确率。
        speed (Dict[str, float]): 包含每个步骤时间的字典（例如预处理、推理等）。
        fitness (float): 模型的适应度，等于 top-5 准确率。
        results_dict (Dict[str, Union[float, str]]): 包含分类指标和适应度的字典。
        keys (List[str]): results_dict 的键列表。

    方法：
        process(targets, pred): 处理目标和预测，以计算分类指标。
    """

    def __init__(self) -> None:
        """初始化一个 ClassifyMetrics 实例。"""
        self.top1 = 0
        self.top5 = 0
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "classify"

    def process(self, targets, pred):
        """目标类和预测类."""
        pred, targets = torch.cat(pred), torch.cat(targets)
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) 准确率
        self.top1, self.top5 = acc.mean(0).tolist()

    @property
    def fitness(self):
        """返回 top-1 和 top-5 准确率的平均值作为适应度评分。"""
        return (self.top1 + self.top5) / 2

    @property
    def results_dict(self):
        """返回一个字典，包含模型的性能指标和适应度评分。"""
        return dict(zip(self.keys + ["fitness"], [self.top1, self.top5, self.fitness]))

    @property
    def keys(self):
        """返回 results_dict 属性的键列表。"""
        return ["metrics/accuracy_top1", "metrics/accuracy_top5"]

    @property
    def curves(self):
        """返回一个列表，用于访问特定的指标曲线。"""
        return []

    @property
    def curves_results(self):
        """返回一个列表，用于访问特定的指标曲线。"""
        return []


class OBBMetrics(SimpleClass):
    """评估定向边界框（OBB）检测的指标，参见 https://arxiv.org/pdf/2106.06072.pdf。"""

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """初始化 OBBMetrics 实例，包括目录、绘图、回调和类名。"""
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

    def process(self, tp, conf, pred_cls, target_cls):
        """处理目标检测的预测结果并更新指标。"""
        results = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            save_dir=self.save_dir,
            names=self.names,
            on_plot=self.on_plot,
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self):
        """返回一个用于访问特定指标的键的列表。"""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def mean_results(self):
        """计算检测到的物体的均值，并返回精度、召回率、mAP50 和 mAP50-95。"""
        return self.box.mean_results()

    def class_result(self, i):
        """返回在特定类别上评估目标检测模型性能的结果。"""
        return self.box.class_result(i)

    @property
    def maps(self):
        """返回每个类别的平均精度（mAP）得分。"""
        return self.box.maps

    @property
    def fitness(self):
        """返回box对象的适应度。"""
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """返回每个类别的平均精度索引。"""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """返回已计算的性能指标和统计信息的字典。"""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """返回用于访问特定指标曲线的曲线列表。"""
        return []

    @property
    def curves_results(self):
        """返回用于访问特定指标曲线的曲线列表。"""
        return []
