# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh


class HungarianMatcher(nn.Module):
    """
    实现 HungarianMatcher 的模块，这是一个可微分模块，用于以端到端的方式解决匹配问题。

    HungarianMatcher 使用一个成本函数，在预测框与真实框之间执行最优匹配，该成本函数考虑了分类分数、
    边界框坐标，以及可选的 mask 预测。

    属性：
        cost_gain (dict): 各种成本项的权重系数字典，包括 'class', 'bbox', 'giou', 'mask', 和 'dice'。
        use_fl (bool): 是否在计算分类成本时使用 Focal Loss。
        with_mask (bool): 模型是否进行了掩码预测。
        num_sample_points (int): 在计算 mask 成本时所使用的采样点数量。
        alpha (float): Focal Loss 中的 α 系数。
        gamma (float): Focal Loss 中的 γ 系数。

    方法：
        forward(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None): 
            针对一个 batch 的图像，计算预测与真实之间的匹配关系。
        _cost_mask(bs, num_gts, masks=None, gt_mask=None): 
            如果模型预测了掩码，则计算 mask 成本和 dice 成本。
    """

    def __init__(self, cost_gain=None, use_fl=True, with_mask=False, num_sample_points=12544, alpha=0.25, gamma=2.0):
        """初始化 HungarianMatcher 模块，用于预测边界框与真实框之间的最优匹配。"""
        super().__init__()
        if cost_gain is None:
            cost_gain = {"class": 1, "bbox": 5, "giou": 2, "mask": 1, "dice": 1}
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None):
        """
        HungarianMatcher 的前向传播。该函数根据预测值和真实标签计算匹配成本（包括分类成本、
        边界框的 L1 距离成本、GIoU 成本），并基于这些成本找到最优的匹配关系。

        参数：
            pred_bboxes (Tensor): 预测边界框，形状为 [batch_size, num_queries, 4]。
            pred_scores (Tensor): 预测类别分数，形状为 [batch_size, num_queries, num_classes]。
            gt_cls (torch.Tensor): 真实标签类别，形状为 [num_gts, ]。
            gt_bboxes (torch.Tensor): 真实边界框，形状为 [num_gts, 4]。
            gt_groups (List[int]): 一个列表，长度为 batch_size，包含每张图像的 ground truth 数量。
            masks (Tensor, optional): 预测的掩码张量，形状为 [batch_size, num_queries, height, width]。
            gt_mask (List[Tensor], optional): 每个元素是 [num_masks, Height, Width] 的真实掩码张量列表。

        返回：
            List[Tuple[Tensor, Tensor]]: 返回一个列表，长度等于 batch 大小。每个元素是一个二元组 (index_i, index_j)，表示：
                - index_i 是预测框中被选中的索引（按顺序排列）
                - index_j 是对应的 ground truth 索引（按顺序排列）
                对于每一个 batch 的样本，有：
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, nq, nc = pred_scores.shape

        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        # 展平用于批量计算成本矩阵
        # [batch_size * num_queries, num_classes]
        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        # [batch_size * num_queries, 4]
        pred_bboxes = pred_bboxes.detach().view(-1, 4)

        # 计算分类成本
        pred_scores = pred_scores[:, gt_cls]
        if self.use_fl:
            neg_cost_class = (1 - self.alpha) * (pred_scores**self.gamma) * (-(1 - pred_scores + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - pred_scores) ** self.gamma) * (-(pred_scores + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores

        # 计算边界框的 L1 距离成本
        cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)  # (bs*num_queries, num_gt)

        # 计算边界框的 GIoU 成本，形状为 (bs*num_queries, num_gt)
        cost_giou = 1.0 - bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0), xywh=True, GIoU=True).squeeze(-1)

        # 最终的成本矩阵
        C = (
            self.cost_gain["class"] * cost_class
            + self.cost_gain["bbox"] * cost_bbox
            + self.cost_gain["giou"] * cost_giou
        )

        # 计算 mask 成本和 dice 成本（如果使用了掩码）
        if self.with_mask:
            C += self._cost_mask(bs, gt_groups, masks, gt_mask)

        # 将无效数值（NaN 和 inf）置为 0，避免 ValueError: matrix contains invalid numeric entries
        C[C.isnan() | C.isinf()] = 0.0

        C = C.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_groups, -1))]
        gt_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)  # 查询和 gt 的偏移索引
        return [
            (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_groups[k])
            for k, (i, j) in enumerate(indices)
        ]

    # 此函数用于未来支持 RT-DETR Segment 模型
    # def _cost_mask(self, bs, num_gts, masks=None, gt_mask=None):
    #     assert masks is not None and gt_mask is not None, '确保输入中包含 `mask` 和 `gt_mask`'
    #     # 所有 mask 共享同一批采样点以提高匹配效率
    #     sample_points = torch.rand([bs, 1, self.num_sample_points, 2])
    #     sample_points = 2.0 * sample_points - 1.0
    #
    #     out_mask = F.grid_sample(masks.detach(), sample_points, align_corners=False).squeeze(-2)
    #     out_mask = out_mask.flatten(0, 1)
    #
    #     tgt_mask = torch.cat(gt_mask).unsqueeze(1)
    #     sample_points = torch.cat([a.repeat(b, 1, 1, 1) for a, b in zip(sample_points, num_gts) if b > 0])
    #     tgt_mask = F.grid_sample(tgt_mask, sample_points, align_corners=False).squeeze([1, 2])
    #
    #     with torch.amp.autocast("cuda", enabled=False):
    #         # 二值交叉熵成本
    #         pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.ones_like(out_mask), reduction='none')
    #         neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.zeros_like(out_mask), reduction='none')
    #         cost_mask = torch.matmul(pos_cost_mask, tgt_mask.T) + torch.matmul(neg_cost_mask, 1 - tgt_mask.T)
    #         cost_mask /= self.num_sample_points
    #
    #         # Dice 成本
    #         out_mask = F.sigmoid(out_mask)
    #         numerator = 2 * torch.matmul(out_mask, tgt_mask.T)
    #         denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
    #         cost_dice = 1 - (numerator + 1) / (denominator + 1)
    #
    #         C = self.cost_gain['mask'] * cost_mask + self.cost_gain['dice'] * cost_dice
    #     return C


def get_cdn_group(
    batch, num_classes, num_queries, class_embed, num_dn=100, cls_noise_ratio=0.5, box_noise_scale=1.0, training=False
):
    """
    获取对比去噪训练组。本函数从ground truth中创建带正负样本的对比去噪训练组，对类别标签和边界框坐标添加噪声，
    并返回修改后的标签、边界框、注意力掩码及元信息。

    参数：
        batch (dict): 包含以下内容的字典：
            - 'gt_cls'：类别标签 (形状为[num_gts,])；
            - 'gt_bboxes'：边界框坐标 (形状为[num_gts, 4])；
            - 'gt_groups'：一个长度为batch size的列表，表示每张图像中gt的数量。
        num_classes (int): 类别总数。
        num_queries (int): 查询的总数。
        class_embed (torch.Tensor): 将类别标签映射到嵌入空间的权重。
        num_dn (int, optional): 去噪样本总数。默认值为100。
        cls_noise_ratio (float, optional): 类别标签的噪声比例。默认值为0.5。
        box_noise_scale (float, optional): 边界框坐标的噪声缩放比例。默认值为1.0。
        training (bool, optional): 是否处于训练模式。默认值为False。

    返回：
        (Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): 
        返回处理后的类别嵌入、边界框、注意力掩码以及元信息。
        若不在训练模式或num_dn <= 0，则所有返回项均为None。
    """
    if (not training) or num_dn <= 0:
        return None, None, None, None

    gt_groups = batch["gt_groups"]  # 每张图像的gt数量
    total_num = sum(gt_groups)      # 所有图像中gt总数
    max_nums = max(gt_groups)       # 当前batch中最大gt数
    if max_nums == 0:
        return None, None, None, None

    num_group = num_dn // max_nums  # 每组最多包含max_nums个目标
    num_group = 1 if num_group == 0 else num_group  # 至少为1组

    # 将gt填充到batch中最大gt数量
    bs = len(gt_groups)
    gt_cls = batch["cls"]  # 所有gt的类别标签 (bs*num, )
    gt_bbox = batch["bboxes"]  # 所有gt的边界框坐标 (bs*num, 4)
    b_idx = batch["batch_idx"]  # 每个gt所属的图像索引

    # 每组包含正负样本
    dn_cls = gt_cls.repeat(2 * num_group)        # 重复2倍数量：正负样本
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)   # 同样重复边界框
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)  # 扁平化图像索引

    # 负样本的索引（第二段为负样本）
    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num

    # 为类别添加噪声
    if cls_noise_ratio > 0:
        # 半数类别标签加噪声
        mask = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask).squeeze(-1)
        # 用随机类别替换
        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device)
        dn_cls[idx] = new_label

    # 为边界框添加噪声
    if box_noise_scale > 0:
        known_bbox = xywh2xyxy(dn_bbox)

        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale  # 噪声比例，基于宽高的一半

        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0  # -1 或 1 的符号
        rand_part = torch.rand_like(dn_bbox)                      # 随机扰动部分
        rand_part[neg_idx] += 1.0                                 # 负样本扰动加大
        rand_part *= rand_sign                                    # 应用符号
        known_bbox += rand_part * diff                            # 应用扰动
        known_bbox.clip_(min=0.0, max=1.0)                         # 限制在[0, 1]范围内
        dn_bbox = xyxy2xywh(known_bbox)
        dn_bbox = torch.logit(dn_bbox, eps=1e-6)                  # 应用逆sigmoid以用于后续输入网络

    # 计算最终的去噪查询数
    num_dn = int(max_nums * 2 * num_group)

    # 类别嵌入查表
    dn_cls_embed = class_embed[dn_cls]  # 查询所有类别嵌入 (bs*num*2*num_group, 256)

    # 初始化填充
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
    padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)

    # 正样本的索引映射，用于meta信息记录
    map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups])
    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)

    # 全部样本（正负）对应位置的映射索引
    map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])
    padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed
    padding_bbox[(dn_b_idx, map_indices)] = dn_bbox

    # 构建注意力掩码
    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    # 匹配查询不能看到重建部分
    attn_mask[num_dn:, :num_dn] = True
    # 重建部分不能看到彼此
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * i * 2] = True
        else:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * 2 * i] = True

    # 构建元信息
    dn_meta = {
        "dn_pos_idx": [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
        "dn_num_group": num_group,
        "dn_num_split": [num_dn, num_queries],
    }

    return (
        padding_cls.to(class_embed.device),
        padding_bbox.to(class_embed.device),
        attn_mask.to(class_embed.device),
        dn_meta,
    )
