# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class DetectionValidator(BaseValidator):
    """
    用于基于目标检测模型的验证类，继承自 BaseValidator。

    示例：
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model="yolo11n.pt", data="coco8.yaml")
        validator = DetectionValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """初始化目标检测验证器，并设置必要的变量与配置。"""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # mAP@0.5:0.95的IoU向量
        self.niou = self.iouv.numel()
        self.lb = []  # 用于自动标注的标签
        if self.args.save_hybrid:
            LOGGER.warning(
                "⚠️ 警告：'save_hybrid=True' 会将真实标签附加到预测结果中用于自动标注。\n"
                "⚠️ 警告：'save_hybrid=True' 会导致 mAP 计算不准确。\n"
            )

    def preprocess(self, batch):
        """对YOLO模型训练前的一批图像数据进行预处理。"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        return batch

    def init_metrics(self, model):
        """初始化用于YOLO模型评估的各项指标。"""
        val = self.data.get(self.args.split, "")  # 验证集路径
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # 是否是COCO数据集
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # 是否是LVIS数据集
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # 是否执行最终验证
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        """返回一条格式化字符串，用于描述YOLO模型的类别级评估指标。"""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        """对模型预测结果应用非极大值抑制（NMS）处理。"""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
        )

    def _prepare_batch(self, si, batch):
        """准备单张图像及其标注信息用于验证。"""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # 转换为xyxy格式
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # 将预测框映射回原始图像尺寸
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """准备预测结果以适应原始图像尺寸。"""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # 将预测框映射回原始图像空间
        return predn

    def update_metrics(self, preds, batch):
        """更新评估指标。"""
        for si, pred in enumerate(preds):
            self.seen += 1  # 增加已处理图像数
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),  # 预测置信度
                pred_cls=torch.zeros(0, device=self.device),  # 预测类别
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),  # 正确预测标志（针对不同 IoU 阈值）
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)  # 当前图像中真实标签的数量
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()  # 当前图像中出现的类别集合
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        # 没有预测，但有标签，需要更新混淆矩阵
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # 预测结果处理
            if self.args.single_cls:
                pred[:, 5] = 0  # 如果是单类别任务，所有类别置为 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # 匹配预测和真实框，计算 TP（True Positive）
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # 保存 JSON 预测结果（用于 COCO 评估）
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            # 保存预测为文本格式
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def finalize_metrics(self, *args, **kwargs):
        """设置最终评估结果，包括速度和混淆矩阵。"""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """返回评估指标的统计信息和结果字典。"""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # 转换为 NumPy 数组
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)  # 每个类别的目标数量
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)  # 每张图像中目标类别分布
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict

    def print_results(self):
        """打印训练/验证集的评估结果，包含每个类别的指标。"""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # 打印格式字符串
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"⚠️ 警告：在 {self.args.task} 数据集中未找到标签，无法计算评估指标")

        # 按类别逐一打印指标结果
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )

        # 绘制混淆矩阵图
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        计算每个预测是否为正确预测（用于计算 TP）。

        参数：
            detections (torch.Tensor): 预测框张量，形状为 (N, 6)，每行表示一个预测 (x1, y1, x2, y2, conf, class)。
            gt_bboxes (torch.Tensor): 真实框张量，形状为 (M, 4)，格式为 (x1, y1, x2, y2)。
            gt_cls (torch.Tensor): 真实类别标签，形状为 (M,)。

        返回：
            (torch.Tensor): 布尔类型张量，形状为 (N, 10)，表示每个预测在 10 个 IoU 阈值下是否为 TP。

        说明：
            该函数返回的是一个中间标志矩阵，用于后续评估匹配，而非最终指标。
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        构建 YOLO 所需的数据集。

        参数：
            img_path (str): 图像所在的文件夹路径。
            mode (str): 数据集模式，可选 "train" 或 "val"，支持为不同模式配置不同的数据增强。
            batch (int, optional): 批处理大小，用于长宽比排列（rect）模式。默认值为 None。
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """构建并返回数据加载器（DataLoader）。"""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)

    def plot_val_samples(self, batch, ni):
        """绘制验证集图像样本（带真实标签）。"""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """绘制预测结果（含预测框）在图像上，并保存可视化图像。"""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # 预测结果可视化

    def save_one_txt(self, predn, save_conf, shape, file):
        """将YOLO的检测结果以归一化坐标的特定格式保存为txt文件。"""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),  # 使用空白图像作为占位
            path=None,
            names=self.names,
            boxes=predn[:, :6],  # 前6列包含xyxy坐标、置信度、类别
        ).save_txt(file, save_conf=save_conf)  # 保存为txt格式，是否保留置信度取决于save_conf


    def pred_to_json(self, predn, filename):
        """将YOLO预测结果序列化为COCO格式的JSON结构。"""
        stem = Path(filename).stem  # 提取文件名主干（不含扩展名）
        image_id = int(stem) if stem.isnumeric() else stem  # 若主干为数字则作为图像ID，否则使用字符串
        box = ops.xyxy2xywh(predn[:, :4])  # 将边界框从xyxy格式转换为xywh格式
        box[:, :2] -= box[:, 2:] / 2  # 将xy中心点转换为左上角坐标

        # 遍历每个预测结果和对应的转换后的边界框，填入JSON结构
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,  # 图像ID
                    "category_id": self.class_map[int(p[5])],  # 类别ID映射
                    "bbox": [round(x, 3) for x in b],  # 边界框坐标保留三位小数
                    "score": round(p[4], 5),  # 置信度保留五位小数
                }
            )


    def eval_json(self, stats):
        """评估以JSON格式保存的YOLO输出，并返回评估统计指标。"""
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # 预测结果保存路径
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # 评估所需的标注文件路径
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\n正在使用 {pred_json} 和 {anno_json} 评估 {pkg} 的 mAP...")

            try:
                # 确保预测文件和标注文件存在
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} 文件未找到"
                # 检查依赖项
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")

                if self.is_coco:
                    # COCO格式的加载与评估
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # 加载标注文件
                    pred = anno.loadRes(str(pred_json))  # 加载预测结果（注意要传字符串路径）
                    val = COCOeval(anno, pred, "bbox")  # 创建评估器实例
                else:
                    # LVIS格式的加载与评估
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  # 加载标注
                    pred = anno._load_json(str(pred_json))  # 加载预测结果
                    val = LVISEval(anno, pred, "bbox")  # 创建评估器

                # 设置要评估的图像ID列表
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]

                # 执行评估流程
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # 对于LVIS，显示更详细的结果

                # 提取并更新指标：mAP50-95 和 mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )

            except Exception as e:
                LOGGER.warning(f"{pkg} 无法运行评估: {e}")
        return stats
