# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import copy

import numpy as np
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class DetectionTrainer(BaseTrainer):
    """
    一个用于基于检测模型进行训练的类，继承自 BaseTrainer。

    示例：
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        构建 YOLO 数据集。

        参数：
            img_path (str): 包含图像的文件夹路径。
            mode (str): 模式，"train" 或 "val"，用户可以自定义每种模式的数据增强。
            batch (int, optional): 批次大小，用于启用 `rect` 模式时。默认值为 None。
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """构建并返回 dataloader。"""
        assert mode in {"train", "val"}, f"模式必须为 'train' 或 'val'，当前为 {mode}。"
        with torch_distributed_zero_first(rank):  # 如果使用 DDP，则只初始化一次 .cache 文件
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("⚠️ 警告：当 rect=True 时与 Dataloader 的 shuffle 不兼容，自动设置 shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # 返回 dataloader

    def preprocess_batch(self, batch):
        """对图像批次进行预处理：缩放并转换为 float。"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # 新的输入尺寸
            sf = sz / max(imgs.shape[2:])  # 缩放因子
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # 新形状（扩展为 stride 的倍数）
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """设置模型的基本属性，如类别数量、类别名称、训练参数等。"""
        # Nl = de_parallel(self.model).model[-1].nl  # 检测头数量（用于缩放超参）
        # self.args.box *= 3 / nl  # 根据层数缩放 box 损失
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # 根据类别数和层数缩放 cls 损失
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # 根据图像尺寸和层数缩放 cls 损失
        self.model.nc = self.data["nc"]  # 将类别数附加到模型上
        self.model.names = self.data["names"]  # 将类别名称附加到模型上
        self.model.args = self.args  # 将训练参数附加到模型上
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回一个 YOLO 检测模型。"""
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """返回 YOLO 模型的 DetectionValidator 用于验证评估。"""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        返回一个带标签的训练损失项字典。

        分类任务不需要，但检测和分割任务需要。
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # 将 tensor 转换为保留五位小数的 float
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """返回一个格式化的训练进度字符串，包含 epoch、GPU 使用、loss、目标数、图像尺寸等信息。"""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """绘制训练样本及其标注信息。"""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """根据 CSV 文件绘制训练指标图。"""
        plot_results(file=self.csv, on_plot=self.on_plot)  # 保存结果图 results.png

    def plot_training_labels(self):
        """创建一个带有标注信息的训练标签可视化图，用于展示 YOLO 模型训练数据分布。"""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """根据模型显存占用自动估算最优批量大小。"""
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        # 使用 mosaic 数据增强时需要乘以 4
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        return super().auto_batch(max_num_obj)
