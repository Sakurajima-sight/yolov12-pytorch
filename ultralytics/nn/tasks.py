# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import thop
import torch
import torch.nn as nn

from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    v10Detect,
    A2C2f,
)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)


class BaseModel(nn.Module):
    """BaseModel 类作为所有 Ultralytics YOLO 系列模型的基类。"""

    def forward(self, x, *args, **kwargs):
        """
        执行模型的前向传播过程，可以用于训练或推理。

        如果 x 是字典，则计算并返回训练的损失。否则，返回推理的预测结果。

        参数：
            x (torch.Tensor | dict): 输入张量用于推理，或者包含图像张量和标签的字典用于训练。
            *args (Any): 可变长度的位置参数。
            **kwargs (Any): 任意的关键字参数。

        返回：
            (torch.Tensor): 如果 x 是字典（训练），则返回损失；否则返回网络的预测结果（推理）。
        """
        if isinstance(x, dict):  # 用于训练和验证的情况
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        执行一次前向传播通过网络。

        参数：
            x (torch.Tensor): 输入的张量。
            profile (bool): 如果为 True，打印每一层的计算时间，默认为 False。
            visualize (bool): 如果为 True，保存模型的特征图，默认为 False。
            augment (bool): 在推理过程中是否对图像进行增强，默认为 False。
            embed (list, 可选): 一个特征向量/嵌入的列表来返回。

        返回：
            (torch.Tensor): 模型的最后输出。
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        执行一次前向传播通过网络。

        参数：
            x (torch.Tensor): 输入的张量。
            profile (bool): 如果为 True，打印每一层的计算时间，默认为 False。
            visualize (bool): 如果为 True，保存模型的特征图，默认为 False。
            embed (list, 可选): 一个特征向量/嵌入的列表来返回。

        返回：
            (torch.Tensor): 模型的最后输出。
        """
        y, dt, embeddings = [], [], []  # 输出
        for m in self.model:
            if m.f != -1:  # 如果不是来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 来自之前的层
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 执行
            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 扁平化
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """对输入图像 x 执行增强操作并返回增强后的推理结果。"""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} 不支持 'augment=True' 推理。"
            f"回退到单尺度推理。"
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        对模型的单个层进行计算时间和 FLOPs（每秒浮点运算数）的分析，并将结果添加到提供的列表中。

        参数：
            m (nn.Module): 需要分析的层。
            x (torch.Tensor): 输入数据。
            dt (list): 用于存储该层计算时间的列表。

        返回：
            None
        """
        c = m == self.model[-1] and isinstance(x, list)  # 是最后一层且输入是列表，复制输入以修正 in-place
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        将模型中的 `Conv2d()` 和 `BatchNorm2d()` 层融合为一个层，以提高计算效率。

        返回：
            (nn.Module): 返回融合后的模型。
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新卷积层
                    delattr(m, "bn")  # 移除 batchnorm
                    m.forward = m.forward_fuse  # 更新前向传播
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # 移除 batchnorm
                    m.forward = m.forward_fuse  # 更新前向传播
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # 更新前向传播
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        检查模型中的 BatchNorm 层是否少于给定的阈值。

        参数：
            thresh (int, 可选): BatchNorm 层的阈值，默认是 10。

        返回：
            (bool): 如果模型中的 BatchNorm 层数少于阈值，则返回 True，否则返回 False。
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # 正则化层，例如 BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # 如果 BatchNorm 层数少于 'thresh'，则返回 True

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        打印模型信息。

        参数：
            detailed (bool): 如果为 True，打印模型的详细信息。默认是 False。
            verbose (bool): 如果为 True，打印模型信息。默认是 False。
            imgsz (int): 模型训练时使用的图像尺寸。默认是 640。
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        将一个函数应用到模型中所有非参数或注册缓冲区的张量上。

        参数：
            fn (function): 要应用到模型的函数。

        返回：
            (BaseModel): 更新后的 BaseModel 对象。
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # 包括所有 Detect 子类，如 Segment、Pose、OBB、WorldDetect
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        加载预训练权重到模型中。

        参数:
            weights (dict | torch.nn.Module): 需要加载的预训练权重。
            verbose (bool, optional): 是否打印加载进度。默认为True。
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision模型不是字典类型
        csd = model.float().state_dict()  # 以FP32格式获取检查点的state_dict
        csd = intersect_dicts(csd, self.state_dict())  # 进行字典交集
        self.load_state_dict(csd, strict=False)  # 加载权重
        if verbose:
            LOGGER.info(f"已从预训练权重中转移 {len(csd)}/{len(self.model.state_dict())} 项")

    def loss(self, batch, preds=None):
        """
        计算损失。

        参数:
            batch (dict): 用于计算损失的批次数据
            preds (torch.Tensor | List[torch.Tensor]): 预测结果。
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """初始化BaseModel的损失函数。"""
        raise NotImplementedError("compute_loss() 需要在任务头部实现")


class DetectionModel(BaseModel):
    """YOLOv8检测模型。"""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  # 模型，输入通道数，类别数
        """使用给定的配置和参数初始化YOLOv8检测模型。"""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # 配置字典
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "警告 ⚠️ YOLOv9 `Silence` 模块已被弃用，改为使用 nn.Identity。"
                "请删除本地 *.pt 文件并重新下载最新的模型检查点。"
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # 定义模型
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # 输入通道数
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"将模型配置中的 nc={self.yaml['nc']} 覆盖为 nc={nc}")
            self.yaml["nc"] = nc  # 覆盖YAML配置中的类别数
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # 模型和保存列表
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # 默认的类别名称字典
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # 构建步幅
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # 包括所有Detect子类，如Segment、Pose、OBB、WorldDetect等
            s = 256  # 2倍最小步幅
            m.inplace = self.inplace

            def _forward(x):
                """执行模型的前向传播，处理不同Detect子类类型。"""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # 前向传播
            self.stride = m.stride
            m.bias_init()  # 只执行一次
        else:
            self.stride = torch.Tensor([32])  # 默认步幅，例如对于RTDETR

        # 初始化权重和偏置
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """对输入图像x进行增强并返回增强后的推理和训练输出。"""
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("警告 ⚠️ 模型不支持 'augment=True'，回退为单尺度预测。")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # 高度，宽度
        s = [1, 0.83, 0.67]  # 缩放因子
        f = [None, 3, None]  # 翻转（2-上下翻转，3-左右翻转）
        y = []  # 输出
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # 前向传播
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # 裁剪增强后的尾部
        return torch.cat(y, -1), None  # 增强后的推理，训练

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """对增强推理后的预测结果进行反缩放操作。"""
        p[:, :4] /= scale  # 反缩放
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # 反向上下翻转
        elif flips == 3:
            x = img_size[1] - x  # 反向左右翻转
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """裁剪YOLO增强推理的尾部部分。"""
        nl = self.model[-1].nl  # 检测层的数量（P3-P5）
        g = sum(4**x for x in range(nl))  # 网格点
        e = 1  # 排除层数
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # 索引
        y[0] = y[0][..., :-i]  # 大尺度
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # 索引
        y[-1] = y[-1][..., i:]  # 小尺度
        return y

    def init_criterion(self):
        """初始化DetectionModel的损失函数。"""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class OBBModel(DetectionModel):
    """YOLOv8定向边界框（OBB）模型。"""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        """初始化YOLOv8 OBB模型，使用给定的配置和参数。"""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """初始化模型的损失函数。"""
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLOv8 分割模型。"""

    def __init__(self, cfg="yolov8n-seg.yaml", ch=3, nc=None, verbose=True):
        """初始化 YOLOv8 分割模型，使用给定的配置和参数。"""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """初始化分割模型的损失函数。"""
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLOv8 姿态模型。"""

    def __init__(self, cfg="yolov8n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """初始化 YOLOv8 姿态模型。"""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # 加载模型的 YAML 配置
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"将 model.yaml 中的 kpt_shape={cfg['kpt_shape']} 替换为 kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """初始化姿态模型的损失函数。"""
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLOv8 分类模型。"""

    def __init__(self, cfg="yolov8n-cls.yaml", ch=3, nc=None, verbose=True):
        """初始化分类模型，传入 YAML 配置、通道数、类别数和是否打印详细信息的标志。"""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """设置 YOLOv8 模型的配置并定义模型架构。"""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # 获取配置字典

        # 定义模型
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # 输入通道数
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"将 model.yaml 中的 nc={self.yaml['nc']} 替换为 nc={nc}")
            self.yaml["nc"] = nc  # 覆盖 YAML 配置中的值
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("未指定 nc。必须在 model.yaml 或函数参数中指定 nc。")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # 获取模型和保存列表
        self.stride = torch.Tensor([1])  # 没有步幅限制
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # 默认的类别名称字典
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """根据需要更新 TorchVision 分类模型的类别数 'n'。"""
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # 获取最后一个模块
        if isinstance(m, Classify):  # YOLO 分类头
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(nn.Linear)  # 获取最后一个 nn.Linear 的索引
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(nn.Conv2d)  # 获取最后一个 nn.Conv2d 的索引
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """初始化分类模型的损失函数。"""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR（基于变压器的实时目标检测和跟踪）检测模型类。

    此类负责构建 RTDETR 架构，定义损失函数，并支持训练和推理过程。RTDETR 是一个目标检测和跟踪模型，继承自 DetectionModel 基类。

    属性：
        cfg (str)：配置文件路径或预设字符串。默认为 'rtdetr-l.yaml'。
        ch (int)：输入通道数。默认为 3（RGB）。
        nc (int, 可选)：目标检测的类别数。默认为 None。
        verbose (bool)：指定是否在初始化时显示详细信息。默认为 True。

    方法：
        init_criterion：初始化用于损失计算的标准。
        loss：计算并返回训练过程中的损失。
        predict：执行前向传播并返回输出。
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        初始化 RTDETRDetectionModel。

        参数：
            cfg (str)：配置文件名称或路径。
            ch (int)：输入通道数。
            nc (int, 可选)：类别数。默认为 None。
            verbose (bool, 可选)：在初始化时打印附加信息。默认为 True。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """初始化 RTDETRDetectionModel 的损失函数。"""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """
        计算给定数据批次的损失。

        参数：
            batch (dict): 包含图像和标签数据的字典。
            preds (torch.Tensor, optional): 预计算的模型预测结果。默认为 None。

        返回：
            (tuple): 一个元组，包含总损失和主损失的三个部分（在张量中）。
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # 注：将 gt_bbox 和 gt_labels 预处理为列表。
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # 注：RTDETR 中有大约 12 个损失，反向传播时使用所有损失，但仅显示主要的三个损失。
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        在模型中执行前向传播。

        参数：
            x (torch.Tensor): 输入张量。
            profile (bool, optional): 如果为 True，则对每层计算时间进行性能分析。默认为 False。
            visualize (bool, optional): 如果为 True，则保存特征图以供可视化。默认为 False。
            batch (dict, optional): 用于评估的地面真实数据。默认为 None。
            augment (bool, optional): 如果为 True，则在推理期间执行数据增强。默认为 False。
            embed (list, optional): 一个特征向量/嵌入的列表，用于返回。

        返回：
            (torch.Tensor): 模型的输出张量。
        """
        y, dt, embeddings = [], [], []  # 输出
        for m in self.model[:-1]:  # 除了头部部分
            if m.f != -1:  # 如果不是来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 来自更早的层
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 执行
            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 展平
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # 头部推理
        return x


class WorldModel(DetectionModel):
    """YOLOv8 世界模型。"""

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """使用给定的配置和参数初始化 YOLOv8 世界模型。"""
        self.txt_feats = torch.randn(1, nc or 80, 512)  # 特征占位符
        self.clip_model = None  # CLIP 模型占位符
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """提前设置类，以便模型可以进行离线推理而不需要 CLIP 模型。"""
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        if (
            not getattr(self, "clip_model", None) and cache_clip_model
        ):  # 为了兼容没有 clip_model 属性的模型
            self.clip_model = clip.load("ViT-B/32")[0]
        model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(text).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.model[-1].nc = len(text)

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        在模型中执行前向传播。

        参数：
            x (torch.Tensor): 输入张量。
            profile (bool, optional): 如果为 True，则对每层计算时间进行性能分析。默认为 False。
            visualize (bool, optional): 如果为 True，则保存特征图以供可视化。默认为 False。
            txt_feats (torch.Tensor): 文本特征，如果提供则使用它。默认为 None。
            augment (bool, optional): 如果为 True，则在推理期间执行数据增强。默认为 False。
            embed (list, optional): 一个特征向量/嵌入的列表，用于返回。

        返回：
            (torch.Tensor): 模型的输出张量。
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # 输出
        for m in self.model:  # 除了头部部分
            if m.f != -1:  # 如果不是来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 来自更早的层
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # 执行

            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 展平
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        计算损失。

        参数：
            batch (dict): 用于计算损失的数据批次。
            preds (torch.Tensor | List[torch.Tensor]): 模型预测结果。
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class Ensemble(nn.ModuleList):
    """模型集成。"""

    def __init__(self):
        """初始化模型集成。"""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """生成 YOLO 网络的最终输出层。"""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # 最大值集成
        # y = torch.stack(y).mean(0)  # 平均值集成
        y = torch.cat(y, 2)  # NMS 集成，y 的形状为 (B, HW, C)
        return y, None  # 推理和训练输出


# 函数 ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    用于临时添加或修改 Python 模块缓存（`sys.modules`）中的模块的上下文管理器。

    该函数可用于在运行时更改模块路径。当你重构代码时很有用，如果你将某个模块从一个位置移动到另一个位置，
    但仍希望支持旧的导入路径以保持向后兼容性。

    参数：
        modules (dict, optional): 一个字典，将旧的模块路径映射到新的模块路径。
        attributes (dict, optional): 一个字典，将旧模块的属性映射到新的模块属性。

    示例：
        ```python
        with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
            import old.module  # 现在将导入 new.module
            from old.module import attribute  # 现在将导入 new.module.attribute
        ```

    注意：
        更改仅在上下文管理器内生效，并且在上下文管理器退出后会撤销。请注意，直接操作 `sys.modules` 可能会导致不可预测的结果，特别是在大型应用程序或库中。在使用此函数时要小心。
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # 设置 sys.modules 中的旧名称的属性
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # 在 sys.modules 中设置旧名称的模块
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # 删除临时的模块路径
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """一个占位类，用于在反序列化时替换未知类。"""

    def __init__(self, *args, **kwargs):
        """初始化 SafeClass 实例，忽略所有参数。"""
        pass

    def __call__(self, *args, **kwargs):
        """运行 SafeClass 实例，忽略所有参数。"""
        pass


class SafeUnpickler(pickle.Unpickler):
    """自定义的 Unpickler，用于将未知类替换为 SafeClass。"""

    def find_class(self, module, name):
        """尝试查找类，如果不在安全模块中则返回 SafeClass。"""
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # 添加其他被认为是安全的模块
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """
    尝试使用 torch.load() 函数加载 PyTorch 模型。如果引发 ModuleNotFoundError 错误，它会捕获该错误，记录警告信息，
    并尝试通过 check_requirements() 函数安装缺少的模块。安装完成后，函数会再次尝试使用 torch.load() 加载模型。

    参数：
        weight (str): PyTorch 模型的文件路径。
        safe_only (bool): 如果为 True，在加载过程中替换未知的类为 SafeClass。

    示例：
    ```python
    from ultralytics.nn.tasks import torch_safe_load

    ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    ```

    返回：
        ckpt (dict): 加载的模型检查点。
        file (str): 加载的文件名
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # 如果本地缺失则在线查找
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # 通过自定义的 pickle 模块加载
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch.load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name 是缺失的模块名
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} 看起来是一个 Ultralytics YOLOv5 模型，最初是在 "
                    f"https://github.com/ultralytics/yolov5 上训练的。\n该模型与 YOLOv8 不兼容。"
                    f"\n推荐的解决方案是使用最新的 'ultralytics' 包重新训练一个模型，或者使用一个官方 Ultralytics 模型运行命令，例如 'yolo predict model=yolov8n.pt'"
                )
            ) from e
        LOGGER.warning(
            f"警告 ⚠️ {weight} 似乎需要 '{e.name}'，但该模块不在 Ultralytics 的要求中。"
            f"\nAutoInstall 将现在为 '{e.name}' 运行安装，但此功能将来会被移除。"
            f"\n推荐的解决方案是使用最新的 'ultralytics' 包重新训练一个模型，或者使用一个官方 Ultralytics 模型运行命令，例如 'yolo predict model=yolov8n.pt'"
        )
        check_requirements(e.name)  # 安装缺失的模块
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # 文件可能是用 torch.save(model, "saved_model.pt") 保存的 YOLO 实例
        LOGGER.warning(
            f"警告 ⚠️ 文件 '{weight}' 似乎没有正确保存或格式不正确。"
            f"为了获得最佳效果，请使用 model.save('filename.pt') 正确保存 YOLO 模型。"
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """加载一个模型权重的集合 weights=[a,b,c] 或单个模型权重 weights=[a] 或 weights=a。"""
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # 加载检查点
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # 合并参数
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 模型

        # 模型兼容性更新
        model.args = args  # 将参数附加到模型
        model.pt_path = w  # 将 *.pt 文件路径附加到模型
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # 添加到模型集合
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # 将模型设置为评估模式

    # 模块更新
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # 兼容 torch 1.11.0

    # 返回模型
    if len(ensemble) == 1:
        return ensemble[-1]

    # 返回模型集合
    LOGGER.info(f"已创建包含以下权重的模型集合：{weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"模型类别数不同 {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """加载单个模型的权重。"""
    ckpt, weight = torch_safe_load(weight)  # 加载检查点
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # 合并模型参数和默认参数，优先使用模型参数
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 模型

    # 模型兼容性更新
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # 将参数附加到模型
    model.pt_path = weight  # 将 *.pt 文件路径附加到模型
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # 将模型设置为评估模式

    # 模块更新
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # 兼容 torch 1.11.0

    # 返回模型和检查点
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """将 YOLO 模型的 yaml 字典解析为 PyTorch 模型。"""
    import ast

    # 参数
    legacy = True  # 向后兼容 v3/v5/v8/v9 模型
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"警告 ⚠️ 没有传递模型缩放参数。假设 scale='{scale}'。")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # 重新定义默认激活函数，例如 Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('激活函数:')} {act}")  # 打印激活函数

    if verbose:
        LOGGER.info(f"\n{'':>3}{'来源':>20}{'数量':>3}{'参数':>10}  {'模块':<45}{'参数':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # 层、保存列表、输出通道
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # 来源，数量，模块，参数
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # 获取模块
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # 深度增益
        if m in {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # 如果 c2 不等于类别数（即 Classify() 输出）
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # 嵌入通道数
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # 头数

            args = [c1, c2, *args[1:]]
            if m in {
                BottleneckCSP,
                C1,
                C2,
                C2f,
                C3k2,
                C2fAttn,
                C3,
                C3TR,
                C3Ghost,
                C3x,
                RepC3,
                C2fPSA,
                C2fCIB,
                C2PSA,
                A2C2f,
            }:
                args.insert(2, n)  # 重复次数
                n = 1
            if m is C3k2:  # 对 M/L/X 尺寸
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:  # 对 M/L/X 尺寸
                legacy = False
                if scale in "mlx":
                    args[3] = True
                if scale in "lx":
                    args.append(True)
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # 重复次数
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}:
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:  # 特殊情况，通道参数必须传递给索引 1
            args.insert(1, [ch[x] for x in f])
        elif m in {CBLinear, TorchVision, Index}:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # 模块
        t = str(m)[8:-2].replace("__main__.", "")  # 模块类型
        m_.np = sum(x.numel() for x in m_.parameters())  # 参数数量
        m_.i, m_.f, m_.type = i, f, t  # 附加索引，'from' 索引，类型
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # 打印信息
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 添加到保存列表
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """从 YAML 文件加载 YOLOv8 模型。"""
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ Ultralytics YOLO P6 模型现在使用 -p6 后缀。正在将 {path.stem} 重命名为 {new_stem}。")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # 即 yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # 模型字典
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    输入 YOLO 模型的 YAML 文件路径，提取模型规模的字符。该函数使用正则表达式匹配模型规模的模式，
    模型规模通过 n、s、m、l 或 x 表示。函数返回模型规模的字符作为字符串。

    参数：
        model_path (str | Path): YOLO 模型的 YAML 文件路径。

    返回：
        (str): 模型规模的字符，可以是 n、s、m、l 或 x。
    """
    try:
        return re.search(r"yolo[v]?\d+([nslmx])", Path(model_path).stem).group(1)  # noqa，返回 n、s、m、l 或 x
    except AttributeError:
        return ""


def guess_model_task(model):
    """
    从 PyTorch 模型的架构或配置中推测模型的任务。

    参数：
        model (nn.Module | dict): PyTorch 模型或 YAML 格式的模型配置。

    返回：
        (str): 模型的任务（'detect'，'segment'，'classify'，'pose'）。

    异常：
        SyntaxError: 如果无法确定模型的任务，则引发该异常。
    """

    def cfg2task(cfg):
        """从 YAML 字典中推测任务。"""
        m = cfg["head"][-1][-2].lower()  # 输出模块名称
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if m == "segment":
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # 从模型配置中推测任务
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # 从 PyTorch 模型中推测任务
    if isinstance(model, nn.Module):  # PyTorch 模型
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, v10Detect)):
                return "detect"

    # 从模型文件名推测任务
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # 无法从模型中确定任务
    LOGGER.warning(
        "WARNING ⚠️ 无法自动推测模型任务，假设任务为 'task=detect'。"
        "请显式定义模型的任务，如 'task=detect'，'segment'，'classify'，'pose' 或 'obb'。"
    )
    return "detect"  # 假设任务为 detect
