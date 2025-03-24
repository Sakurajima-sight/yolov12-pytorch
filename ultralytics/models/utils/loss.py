# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import FocalLoss, VarifocalLoss
from ultralytics.utils.metrics import bbox_iou

from .ops import HungarianMatcher


class DETRLoss(nn.Module):
    """
    DETR（DEtection TRansformer）损失函数类。该类用于计算并返回DETR目标检测模型中的各类损失组件。
    它包括分类损失、边界框回归损失、GIoU损失，并支持辅助损失的计算。

    属性:
        nc (int): 类别数量。
        loss_gain (dict): 各类损失组件的权重系数。
        aux_loss (bool): 是否计算辅助损失。
        use_fl (bool): 是否使用FocalLoss。
        use_vfl (bool): 是否使用VarifocalLoss。
        use_uni_match (bool): 是否在辅助分支中使用固定层来分配标签。
        uni_match_ind (int): 如果use_uni_match为True，使用的固定层索引。
        matcher (HungarianMatcher): 用于计算匹配代价与索引的匈牙利匹配器。
        fl (FocalLoss 或 None): 如果use_fl为True则使用FocalLoss，否则为None。
        vfl (VarifocalLoss 或 None): 如果use_vfl为True则使用VarifocalLoss，否则为None。
        device (torch.device): 当前使用的设备（CPU或GPU）。
    """

    def __init__(
        self, nc=80, loss_gain=None, aux_loss=True, use_fl=True, use_vfl=False, use_uni_match=False, uni_match_ind=0
    ):
        """
        初始化DETR损失函数，并支持自定义各个损失组件和权重。

        如果未提供loss_gain，则使用默认值。内部使用预设代价系数初始化匈牙利匹配器。
        同时支持辅助损失、FocalLoss与VarifocalLoss。

        参数:
            nc (int): 类别数量。
            loss_gain (dict): 各损失项的权重系数。
            aux_loss (bool): 是否使用来自每个解码器层的辅助损失。
            use_fl (bool): 是否使用FocalLoss。
            use_vfl (bool): 是否使用VarifocalLoss。
            use_uni_match (bool): 是否在辅助分支中使用固定层分配标签。
            uni_match_ind (int): 当使用固定分配层时，该层的索引。
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss() if use_fl else None
        self.vfl = VarifocalLoss() if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=""):
        """
        计算分类损失，根据预测值、目标标签和真实得分。

        参数:
            pred_scores (Tensor): 预测得分，形状为 [batch_size, query_num, num_classes]
            targets (Tensor): 标签，形状为 list[[n, 1]]
            gt_scores (Tensor): ground truth得分（来自匈牙利匹配的结果）
            num_gts (int): 当前批次中目标数量
            postfix (str): 后缀名，用于区分主损失或辅助损失

        返回:
            dict: 包含加权后的分类损失项
        """
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]

        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # 方式一，注释掉了
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]  # 去除背景类别维度
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq  # 对目标数进行归一化
        else:
            # 若未启用FocalLoss，使用BCE作为YOLO的分类损失
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=""):
        """
        计算预测框与真实框之间的边界框损失（L1）与GIoU损失。

        参数:
            pred_bboxes (Tensor): 预测框，形状为 [batch, query, 4]
            gt_bboxes (Tensor): 真实框，形状为 list[[n, 4]]
            postfix (str): 后缀名，用于区分主损失或辅助损失

        返回:
            dict: 包含 bbox 和 giou 的损失项
        """
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    # 以下函数用于未来RT-DETR分割模型中的Mask分支
    # def _get_loss_mask(self, masks, gt_mask, match_indices, postfix=''):
    #     # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
    #     name_mask = f'loss_mask{postfix}'
    #     name_dice = f'loss_dice{postfix}'
    #
    #     loss = {}
    #     if sum(len(a) for a in gt_mask) == 0:
    #         loss[name_mask] = torch.tensor(0., device=self.device)
    #         loss[name_dice] = torch.tensor(0., device=self.device)
    #         return loss
    #
    #     num_gts = len(gt_mask)
    #     src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices)
    #     src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
    #     # 注意：当前版本的torch中没有内置sigmoid_focal_loss，但因为我们暂时不使用mask分支，所以不紧急。
    #     loss[name_mask] = self.loss_gain['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
    #                                                                     torch.tensor([num_gts], dtype=torch.float32))
    #     loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
    #     return loss

    # 以下函数用于未来RT-DETR分割模型中的Dice损失
    # @staticmethod
    # def _dice_loss(inputs, targets, num_gts):
    #     inputs = F.sigmoid(inputs).flatten(1)
    #     targets = targets.flatten(1)
    #     numerator = 2 * (inputs * targets).sum(1)
    #     denominator = inputs.sum(-1) + targets.sum(-1)
    #     loss = 1 - (numerator + 1) / (denominator + 1)
    #     return loss.sum() / num_gts

    def _get_loss_aux(
        self,
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        match_indices=None,
        postfix="",
        masks=None,
        gt_mask=None,
    ):
        """计算辅助损失（auxiliary losses）"""
        # 注意：包括分类损失、bbox损失、GIoU损失、mask损失、dice损失
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            # 如果使用mask损失
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']

        loss = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }
        # 如果使用mask损失，则添加mask和dice损失
        # if masks is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    @staticmethod
    def _get_index(match_indices):
        """从匹配索引中返回批次索引、源索引和目标索引"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(self, pred_bboxes, gt_bboxes, match_indices):
        """根据匹配索引，将预测框分配给对应的真实框（GT）"""
        pred_assigned = torch.cat(
            [
                t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (i, _) in zip(pred_bboxes, match_indices)
            ]
        )
        gt_assigned = torch.cat(
            [
                t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (_, j) in zip(gt_bboxes, match_indices)
            ]
        )
        return pred_assigned, gt_assigned

    def _get_loss(
        self,
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        masks=None,
        gt_mask=None,
        postfix="",
        match_indices=None,
    ):
        """主损失计算逻辑"""
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        return {
            **self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix),
            **self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix),
            # 如果启用mask损失，则解开注释
            # **(self._get_loss_mask(masks, gt_mask, match_indices, postfix) if masks is not None and gt_mask is not None else {})
        }

    def forward(self, pred_bboxes, pred_scores, batch, postfix="", **kwargs):
        """
        计算预测框与分类得分的损失。

        参数:
            pred_bboxes (torch.Tensor): 预测的边界框，形状为 [l, b, query, 4]。
            pred_scores (torch.Tensor): 预测的分类得分，形状为 [l, b, query, num_classes]。
            batch (dict): 包含以下字段的batch信息：
                cls (torch.Tensor): 真实的类别标签，形状为 [num_gts]。
                bboxes (torch.Tensor): 真实的边界框，形状为 [num_gts, 4]。
                gt_groups (List[int]): 每张图像中真实框的数量。
            postfix (str): 用于标注损失名称的后缀。
            **kwargs (Any): 其他附加参数，可能包含'match_indices'。

        返回:
            (dict): 返回一个包含主损失和辅助损失的字典（如果启用了辅助损失）。

        注意:
            主损失使用最后一层的预测框和得分；
            如果启用了self.aux_loss，则使用前面几层计算辅助损失。
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get("match_indices", None)
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]

        # 主损失（使用最后一层预测结果）
        total_loss = self._get_loss(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, postfix=postfix, match_indices=match_indices
        )

        # 辅助损失（使用前几层预测结果）
        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices, postfix
                )
            )

        return total_loss


class RTDETRDetectionLoss(DETRLoss):
    """
    实时目标检测器（RT-DETR）检测损失类，继承自DETRLoss。

    该类用于计算RT-DETR模型的检测损失，包括标准检测损失，以及在提供去噪元数据时的去噪训练损失。
    """

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        """
        前向传播计算检测损失。

        参数：
            preds (tuple): 模型预测的边界框和分类分数。
            batch (dict): 包含真实标签信息的批次数据。
            dn_bboxes (torch.Tensor, optional): 去噪预测的边界框，默认值为 None。
            dn_scores (torch.Tensor, optional): 去噪预测的分类分数，默认值为 None。
            dn_meta (dict, optional): 去噪相关的元数据信息，默认值为 None。

        返回：
            (dict): 一个包含总损失的字典，如果提供了去噪信息，也会包含去噪损失。
        """
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # 如果提供了去噪元数据，则计算去噪训练损失
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # 获取去噪的匹配索引
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            # 计算去噪训练损失
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # 如果没有提供去噪信息，则所有去噪损失设为0
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(dn_pos_idx, dn_num_group, gt_groups):
        """
        获取用于去噪的匹配索引。

        参数：
            dn_pos_idx (List[torch.Tensor]): 包含每张图像的正样本去噪索引的张量列表。
            dn_num_group (int): 去噪组的数量。
            gt_groups (List[int]): 每张图像的真实目标数量列表。

        返回：
            (List[tuple]): 每张图像的去噪匹配索引元组列表，每个元组包含(预测索引, 真实标签索引)。
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), "期望长度一致，" \
                    f"但得到的分别是 {len(dn_pos_idx[i])} 和 {len(gt_idx)}。"
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices
