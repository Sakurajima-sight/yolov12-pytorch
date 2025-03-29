# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Zhang等人提出的变焦损失（Varifocal Loss）。

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """初始化VarifocalLoss类。"""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """计算变焦损失（Varifocal Loss）。"""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """将焦点损失（Focal Loss）封装在现有的损失函数（loss_fcn）周围，示例：criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)。"""

    def __init__(self):
        """初始化FocalLoss类，无需参数。"""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """计算并更新目标检测/分类任务的混淆矩阵。"""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # 为了梯度稳定性，非零次方

        # TensorFlow实现 https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # 从logits计算概率
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """用于计算训练过程中分布焦点损失（DFL）的标准类。"""

    def __init__(self, reg_max=16) -> None:
        """初始化DFL模块。"""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        返回左右DFL损失的总和。

        分布焦点损失（DFL）提出于《广义焦点损失》一文中
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # 左侧目标
        tr = tl + 1  # 右侧目标
        wl = tr - target  # 左侧权重
        wr = 1 - wl  # 右侧权重
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """用于计算训练过程中边界框损失的标准类。"""

    def __init__(self, reg_max=16):
        """初始化BboxLoss模块，带有正则化最大值和DFL设置。"""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """计算IoU损失。"""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL损失
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """用于计算旋转边界框损失的标准类。"""

    def __init__(self, reg_max):
        """初始化BboxLoss模块，带有正则化最大值和DFL设置。"""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """计算IoU损失。"""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL损失
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """用于计算训练损失的标准类。"""

    def __init__(self, sigmas) -> None:
        """初始化 KeypointLoss 类。"""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """计算预测关键点与实际关键点的关键点损失因子和欧几里得距离损失。"""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # 来自公式
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # 来自 cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """用于计算训练损失的标准类。"""

    def __init__(self, model, tal_topk=10):  # 模型必须是去并行化的
        """初始化 v8DetectionLoss 类，定义与模型相关的属性和 BCE 损失函数。"""
        device = next(model.parameters()).device  # 获取模型的设备
        h = model.args  # 超参数

        m = model.model[-1]  # Detect() 模块
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # 模型的步幅
        self.nc = m.nc  # 类别数
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """预处理目标计数并与输入批大小匹配，以输出张量。"""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # 图像索引
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """根据锚点和分布解码预测的物体边界框坐标。"""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # 批次，锚点，通道数
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """计算边界框、分类和 dfl 的损失和，乘以批大小。"""
        loss = torch.zeros(3, device=self.device)  # 边界框、分类、dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # 图像尺寸（高，宽）
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # 目标
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # 分类，xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # 预测边界框
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # 分类损失
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL 方法
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # 边界框损失
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # 边界框增益
        loss[1] *= self.hyp.cls  # 分类增益
        loss[2] *= self.hyp.dfl  # dfl 增益

        return loss.sum() * batch_size, loss.detach()  # 损失（边界框，分类，dfl）


class v8SegmentationLoss(v8DetectionLoss):
    """用于计算训练损失的标准类。"""

    def __init__(self, model):  # 模型必须是去并行化的
        """初始化 v8SegmentationLoss 类，传入去并行化的模型作为参数。"""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """计算并返回 YOLO 模型的损失值。"""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, mask 数量, mask 高度, mask 宽度
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # 图像尺寸 (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # 类别，xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ 数据集格式不正确或不是分割数据集。\n"
                "该错误可能在错误地使用 'detect' 数据集训练 'segment' 模型时发生，"
                "例如 'yolo train model=yolov8n-seg.pt data=coco8.yaml'。\n请验证你的数据集是一个"
                "正确格式的 'segment' 数据集，使用 'data=coco8-seg.yaml' 作为示例。\n更多帮助请访问 "
                "https://docs.ultralytics.com/datasets/segment/。"
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # 类别损失
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL 方法
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # 边界框损失
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Mask 损失
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # 下采样
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # 警告：以下代码行防止多 GPU DDP '未使用梯度' PyTorch 错误，请勿删除
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf 求和可能导致 nan 损失

        loss[0] *= self.hyp.box  # 边界框增益
        loss[1] *= self.hyp.box  # seg 增益
        loss[2] *= self.hyp.cls  # 类别增益
        loss[3] *= self.hyp.dfl  # dfl 增益

        return loss.sum() * batch_size, loss.detach()  # 损失（box，cls，dfl）

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        计算单张图片的实例分割损失。

        参数:
            gt_mask (torch.Tensor): 真实的目标 mask，形状为 (n, H, W)，其中 n 是目标的数量。
            pred (torch.Tensor): 预测的 mask 系数，形状为 (n, 32)。
            proto (torch.Tensor): 原型 mask，形状为 (32, H, W)。
            xyxy (torch.Tensor): 真实的边界框，采用 xyxy 格式，归一化到 [0, 1]，形状为 (n, 4)。
            area (torch.Tensor): 每个真实边界框的面积，形状为 (n,)。

        返回:
            (torch.Tensor): 计算出的单张图片的 mask 损失。

        注意:
            该函数使用公式 pred_mask = torch.einsum('in,nhw->ihw', pred, proto) 来通过原型 mask 和预测的 mask 系数
            生成预测的 mask。
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        计算实例分割的损失。

        参数:
            fg_mask (torch.Tensor): 形状为 (BS, N_anchors) 的二进制张量，表示哪些锚点是正样本。
            masks (torch.Tensor): 真实的目标 masks，形状为 (BS, H, W)，如果 `overlap` 为 False， 否则形状为 (BS, ?, H, W)。
            target_gt_idx (torch.Tensor): 每个锚点的真实目标索引，形状为 (BS, N_anchors)。
            target_bboxes (torch.Tensor): 每个锚点的真实边界框，形状为 (BS, N_anchors, 4)。
            batch_idx (torch.Tensor): 批次索引，形状为 (N_labels_in_batch, 1)。
            proto (torch.Tensor): 原型 masks，形状为 (BS, 32, H, W)。
            pred_masks (torch.Tensor): 每个锚点的预测 masks，形状为 (BS, N_anchors, 32)。
            imgsz (torch.Tensor): 输入图像的大小，形状为 (2)，即 (H, W)。
            overlap (bool): 如果 masks 中的 mask 重叠，则为 True。

        返回:
            (torch.Tensor): 计算出的实例分割损失。

        注意:
            可以通过批量计算来提高速度，牺牲一些内存使用。
            例如，预测的 mask 可以通过以下方式计算：
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # 归一化到 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # 目标边界框的面积
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # 归一化到 mask 大小
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # 警告：以下代码行防止多 GPU DDP '未使用梯度' PyTorch 错误，请勿删除
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf 求和可能导致 nan 损失

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """用于计算训练损失的准则类。"""

    def __init__(self, model):  # 模型必须是去并行化的
        """初始化 v8PoseLoss 类，设置关键点变量并声明一个关键点损失实例。"""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # 关键点的数量
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """计算总损失并将其分离。"""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # 图像大小 (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # 目标
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # 类别损失
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL 方式
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # 边界框损失
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # 边界框增益
        loss[1] *= self.hyp.pose  # 姿态增益
        loss[2] *= self.hyp.kobj  # 关键点对象增益
        loss[3] *= self.hyp.cls  # 类别增益
        loss[4] *= self.hyp.dfl  # DFL 增益

        return loss.sum() * batch_size, loss.detach()  # 损失（box, cls, dfl）

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """将预测的关键点解码到图像坐标。"""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        计算模型的关键点损失。

        该函数计算给定批次的关键点损失和关键点对象损失。关键点损失基于预测关键点与真实关键点之间的差异。关键点对象损失是一个二分类损失，分类关键点是否存在。

        参数：
            masks (torch.Tensor): 二值掩膜张量，指示物体是否存在，形状为 (BS, N_anchors)。
            target_gt_idx (torch.Tensor): 将锚点映射到真实物体的索引张量，形状为 (BS, N_anchors)。
            keypoints (torch.Tensor): 真实关键点，形状为 (N_kpts_in_batch, N_kpts_per_object, kpts_dim)。
            batch_idx (torch.Tensor): 关键点的批次索引张量，形状为 (N_kpts_in_batch, 1)。
            stride_tensor (torch.Tensor): 锚点的步幅张量，形状为 (N_anchors, 1)。
            target_bboxes (torch.Tensor): 真实边界框，格式为 (x1, y1, x2, y2)，形状为 (BS, N_anchors, 4)。
            pred_kpts (torch.Tensor): 预测的关键点，形状为 (BS, N_anchors, N_kpts_per_object, kpts_dim)。

        返回：
            kpts_loss (torch.Tensor): 关键点损失。
            kpts_obj_loss (torch.Tensor): 关键点对象损失。
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # 找到单个图像中最多的关键点数量
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # 创建一个张量来保存批次的关键点
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: 有没有办法对这个过程进行向量化？
        # 根据 batch_idx 填充 batched_keypoints
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # 扩展 target_gt_idx 维度，以匹配 batched_keypoints 的形状
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # 使用 target_gt_idx_expanded 从 batched_keypoints 中选择关键点
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # 将坐标除以步幅
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # 姿态损失

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # 关键点对象损失

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """用于计算训练损失的标准类."""

    def __call__(self, preds, batch):
        """计算预测与真实标签之间的分类损失."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """计算旋转YOLO模型中的目标检测、分类和框分布的损失."""

    def __init__(self, model):
        """初始化 v8OBBLoss，包括模型、分配器和旋转边界框损失；请注意，模型必须是去并行化的."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """预处理目标计数和与输入批次大小匹配，输出一个张量."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # 图像索引
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """计算并返回YOLO模型的损失."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # 批次大小，掩码数量，掩码高度，掩码宽度
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # 图像大小 (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # 过滤掉尺寸过小的旋转框以稳定训练
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # 类别标签，xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "错误 ❌ OBB 数据集格式不正确或不是 OBB 数据集。\n"
                "该错误可能发生在将 'OBB' 模型错误地用于 'detect' 数据集时，"
                "例如 'yolo train model=yolov8n-obb.pt data=dota8.yaml'。\n请验证你的数据集是否是 "
                "正确格式化的 'OBB' 数据集，使用 'data=dota8.yaml' 作为示例。\n详见 https://docs.ultralytics.com/datasets/obb/ 获取帮助。"
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # 仅需要对前四个元素进行缩放
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # 类别损失
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL方式
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # 边界框损失
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # 边界框增益
        loss[1] *= self.hyp.cls  # 类别增益
        loss[2] *= self.hyp.dfl  # dfl增益

        return loss.sum() * batch_size, loss.detach()  # 损失(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        从锚点和分布解码预测的目标边界框坐标。

        参数：
            anchor_points (torch.Tensor): 锚点，(h*w, 2)。
            pred_dist (torch.Tensor): 预测的旋转距离，(bs, h*w, 4)。
            pred_angle (torch.Tensor): 预测的角度，(bs, h*w, 1)。

        返回：
            (torch.Tensor): 带角度的预测旋转边界框，(bs, h*w, 5)。
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # 批次大小，锚点数量，通道数
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """用于计算训练损失的标准类。"""

    def __init__(self, model):
        """使用提供的模型初始化E2EDetectLoss，包含一对多和一对一的检测损失。"""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """计算边界框、类别和DFL的损失总和，并乘以批次大小。"""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
