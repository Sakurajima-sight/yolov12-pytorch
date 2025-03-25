# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect"


class Detect(nn.Module):
    """YOLO 检测头部模块，适用于检测任务。"""

    dynamic = False  # 强制重建 anchor 网格（动态网格）
    export = False  # 是否为模型导出模式
    format = None  # 导出格式（例如 tflite、onnx）
    end2end = False  # 是否为端到端模式（YOLOv10）
    max_det = 300  # 每张图像最多检测框数
    shape = None  # 输入张量的空间形状（H, W）
    anchors = torch.empty(0)  # 初始化 anchor
    strides = torch.empty(0)  # 初始化步长
    legacy = False  # 向后兼容 v3/v5/v8/v9 模型结构

    def __init__(self, nc=80, ch=()):
        """初始化 YOLO 检测层，指定类别数和每个检测层的输入通道数。"""
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数量
        self.reg_max = 16  # DFL（Distribution Focal Loss）通道数（用于回归）
        self.no = nc + self.reg_max * 4  # 每个 anchor 的输出数量
        self.stride = torch.zeros(self.nl)  # 步长在 build 时自动计算

        c2 = max((16, ch[0] // 4, self.reg_max * 4))  # 回归分支的中间通道
        c3 = max(ch[0], min(self.nc, 100))  # 分类分支的中间通道

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )  # 回归分支

        self.cv3 = (
            nn.ModuleList(
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
            ) if self.legacy else
            nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                ) for x in ch
            )
        )  # 分类分支（现代或 legacy 分支）

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()  # 回归解码器（DFL）

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """拼接并返回边界框和类别概率预测。"""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)  # 回归 + 分类分支拼接
        if self.training:  # 训练阶段
            return x
        y = self._inference(x)  # 推理阶段
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        执行 YOLOv10 模式的前向传播（端到端）。

        参数:
            x (tensor): 输入特征图。

        返回:
            (dict, tensor): 若非训练模式，返回 one2many 和 one2one 两个检测路径的输出结果；
                            若为训练模式，分别返回两个路径的原始输出。
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1)
            for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # 训练模式下分别返回
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """根据多层特征图解码预测边界框和类别概率。"""
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            # 构建 anchor 网格和步长（仅当形状改变或为动态推理时）
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            # 避免 TensorFlow 导出时生成 FlexSplitV 操作
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # 为提升数值稳定性，预计算归一化因子
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            # 用于 Intel IMX 格式的特殊推理路径
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            # 默认推理路径（常规导出或训练）
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)  # 拼接框与类别概率

    def bias_init(self):
        """初始化 Detect() 模块中的偏置项（注意：需要先完成 stride 构建）。"""
        m = self  # m 指向 Detect() 模块本身

        # 遍历每个层的回归分支（a）和分类分支（b）进行偏置初始化
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0  # 回归头偏置设为常数
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # 分类头偏置，假设 0.01 的目标概率

        # 如果启用了 end2end，还要为 one2one 检测路径初始化偏置
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):
                a[-1].bias.data[:] = 1.0
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """解码边界框（bounding boxes）。"""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        对 YOLO 模型预测结果进行后处理。

        参数:
            preds (torch.Tensor): 原始预测张量，形状为 (batch_size, num_anchors, 4 + nc)，
                最后一维格式为 [x, y, w, h, class_probs]。
            max_det (int): 每张图像的最大检测数。
            nc (int, 可选): 类别数量。默认值为 80。

        返回:
            (torch.Tensor): 处理后的预测结果，形状为 (batch_size, min(max_det, num_anchors), 6)，
                最后一维格式为 [x, y, w, h, max_class_prob, class_index]。
        """
        batch_size, anchors, _ = preds.shape  # 即 (16, 8400, 84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch 索引
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class Segment(Detect):
    """YOLO 分割模型的 Segment 头部模块。"""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """初始化 YOLO 分割模型所需的掩膜数量、原型数量、卷积层等属性。"""
        super().__init__(nc, ch)
        self.nm = nm  # 掩膜数量（mask channels）
        self.npr = npr  # 原型数量（prototype channels）
        self.proto = Proto(ch[0], self.npr, self.nm)  # 掩膜原型模块

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(
            Conv(x, c4, 3),
            Conv(c4, c4, 3),
            nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """训练时返回预测结果和掩膜系数，推理时返回拼接后的结果和原型图。"""
        p = self.proto(x[0])  # 掩膜原型特征
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # 掩膜系数
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLO 用于旋转框检测（OBB：Oriented Bounding Box）的头部模块。"""

    def __init__(self, nc=80, ne=1, ch=()):
        """初始化 OBB 模块，设置类别数量 nc 和每层通道 ch。"""
        super().__init__(nc, ch)
        self.ne = ne  # 额外参数数量（如角度）

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(
            Conv(x, c4, 3),
            Conv(c4, c4, 3),
            nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """拼接并返回预测框和类别概率。"""
        bs = x[0].shape[0]  # batch 大小
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB 角度预测

        # 注意：设置 angle 为类属性，便于后续 decode_bboxes 使用
        angle = (angle.sigmoid() - 0.25) * math.pi  # 映射到 [-π/4, 3π/4]
        # angle = angle.sigmoid() * math.pi / 2  # 映射到 [0, π/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """解码旋转边界框。"""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLO 用于关键点检测（Pose Estimation）的头部模块。"""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """初始化 YOLO 关键点模型，包括关键点数量、维度与卷积结构。"""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # (关键点数，每个关键点的维度，例如 2 或 3)
        self.nk = kpt_shape[0] * kpt_shape[1]  # 总的关键点数

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(
            Conv(x, c4, 3),
            Conv(c4, c4, 3),
            nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """执行前向传播，返回检测框与关键点预测结果。"""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # 关键点预测 (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """解码关键点坐标。"""
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {"tflite", "edgetpu"}:
                # 预计算归一化因子以提升数值稳定性，适配 TFLite
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                # 兼容 NCNN 导出
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # 对可见性通道做 sigmoid（注意 Apple MPS 的 bug）
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLO 分类头模块，即将输入张量从 (b,c1,20,20) 转换为 (b,c2)。"""

    export = False  # 导出模式

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """初始化 YOLO 分类头，将输入张量从 (b,c1,20,20) 转换为 (b,c2) 形状。"""
        super().__init__()
        c_ = 1280  # efficientnet_b0 的通道数
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # 输出为 x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # 输出为 x(b,c2)

    def forward(self, x):
        """对输入图像数据执行 YOLO 模型的前向传播。"""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # 获取最终输出
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """集成了文本嵌入语义理解的 YOLO 检测模型头部模块。"""

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """初始化 YOLO 检测层，设置类别数量 nc 以及特征图通道列表 ch。"""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """拼接并返回预测边界框与类别概率。"""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        # 推理路径
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # 避免 TF FlexSplitV 运算
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # 预计算归一化因子以提高数值稳定性
            # 参考：https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """初始化 Detect() 层的偏置项，注意：依赖 stride 的可用性。"""
        m = self  # self.model[-1]  # Detect() 模块
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # 针对每个检测层
            a[-1].bias.data[:] = 1.0  # 初始化 box 偏置
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # 初始化 cls 偏置 (.01 目标概率，80 类，640 图像)


class RTDETRDecoder(nn.Module):
    """
    RTDETR 解码器：基于实时可变形 Transformer 的目标检测解码模块。

    该模块结合 Transformer 架构与可变形卷积，用于预测图像中目标的边界框与类别标签。
    它融合多个层级的特征，通过多层解码器实现最终预测。
    """

    export = False  # 导出模式

    def __init__(
        self,
        nc=80,                  # 类别数量
        ch=(512, 1024, 2048),   # 主干输出通道
        hd=256,                 # 隐藏维度
        nq=300,                 # 查询数量
        ndp=4,                  # 解码器采样点数
        nh=8,                   # 多头注意力头数
        ndl=6,                  # 解码器层数
        d_ffn=1024,             # FFN 网络隐藏维度
        dropout=0.0,            # dropout 概率
        act=nn.ReLU(),          # 激活函数
        eval_idx=-1,            # 用于评估的层索引
        # 以下为训练参数
        nd=100,                 # 噪声查询数
        label_noise_ratio=0.5,  # 标签噪声比例
        box_noise_scale=1.0,    # 边界框噪声缩放
        learnt_init_query=False # 是否学习初始化查询
    ):
        """
        初始化 RTDETR 解码模块，包含特征融合、解码器结构、分类与框回归头。

        参数说明：
            nc (int): 类别数量，默认 80。
            ch (tuple): 主干特征图的通道数，默认 (512, 1024, 2048)。
            hd (int): 隐藏层维度，默认 256。
            nq (int): 查询点数量，默认 300。
            ndp (int): 每层解码器使用的采样点数，默认 4。
            nh (int): 多头注意力中的头数，默认 8。
            ndl (int): 解码器层数，默认 6。
            d_ffn (int): 前馈网络维度，默认 1024。
            dropout (float): dropout 比例，默认 0。
            act (nn.Module): 激活函数，默认 nn.ReLU。
            eval_idx (int): 用于评估的层索引，默认 -1。
            nd (int): 去噪查询数量，默认 100。
            label_noise_ratio (float): 标签噪声比例，默认 0.5。
            box_noise_scale (float): 框噪声比例，默认 1.0。
            learnt_init_query (bool): 是否学习初始化查询嵌入，默认 False。
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # 特征层数
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # 主干特征投影
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)

        # Transformer 解码器模块
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # 去噪处理部分
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # 解码器查询嵌入
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # 编码器头部
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # 解码器头部
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """
        模块的前向传播过程。

        参数：
            x (Tensor): 输入特征。
            batch (dict, optional): 训练时使用的批次数据。

        返回：
            训练阶段返回解码器输出和中间结果，
            推理阶段返回最终预测结果或（预测，全部中间输出）。
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # 编码器输入投影和特征嵌入
        feats, shapes = self._get_encoder_input(x)

        # 准备去噪训练（Denoising Training）
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        # 获取解码器输入
        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # 解码器前向传播
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)：将边框和类别得分拼接
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """
        基于输入形状生成 anchor 框，并进行归一化和合法性掩码处理。

        参数：
            shapes (List[Tuple[int, int]]): 每层特征图的高宽。
            grid_size (float): 每个 anchor 的初始尺寸。
            dtype: 张量数据类型。
            device: 所在设备。
            eps: 边界阈值，防止 anchor 靠近 0 或 1。

        返回：
            anchors (Tensor): 归一化后的 anchor。
            valid_mask (Tensor): 有效 anchor 掩码。
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """
        获取编码器输入。

        通过输入投影提取特征，并将特征图拉平成序列形式。

        返回：
            feats (Tensor): 形状为 (b, ∑(hw), c) 的特征序列。
            shapes (List[Tuple[int, int]]): 每层特征图的高度和宽度。
        """
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            feats.append(feat.flatten(2).permute(0, 2, 1))  # [b, h*w, c]
            shapes.append([h, w])

        feats = torch.cat(feats, 1)  # 合并所有特征图序列
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """
        构造解码器的输入，包括参考框、嵌入、编码器输出分数等。

        参数：
            feats (Tensor): 编码器特征。
            shapes (List[Tuple[int, int]]): 特征图尺寸。
            dn_embed (Tensor): 去噪嵌入向量。
            dn_bbox (Tensor): 去噪边框。

        返回：
            embeddings, refer_bbox, enc_bboxes, enc_scores
        """
        bs = feats.shape[0]
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # 分类得分 (bs, h*w, nc)

        # TopK 查询选择
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors
        enc_bboxes = refer_bbox.sigmoid()

        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """
        初始化或重置模型中各模块的参数，包括分类头、边框头、位置嵌入等。
        """
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """
    v10 检测头模块，来源：https://arxiv.org/pdf/2405.14458。

    参数：
        nc (int): 类别数。
        ch (tuple): 每个输入特征图的通道数。

    属性：
        max_det (int): 最大检测数量。

    方法：
        __init__: 初始化检测头。
        forward: 前向传播。
        bias_init: 初始化偏置项。
    """

    end2end = True

    def __init__(self, nc=80, ch=()):
        """使用指定类别数和输入通道初始化 v10Detect 检测头。"""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # 通道数限制

        # 轻量级分类头
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)
