# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr

from .val import RTDETRDataset, RTDETRValidator


class RTDETRTrainer(DetectionTrainer):
    """
    RT-DETR模型的训练器类，由百度开发，用于实时物体检测。扩展了YOLO的DetectionTrainer类，适应RT-DETR的特定特性和架构。
    该模型利用视觉变换器（Vision Transformers），并具备像IoU感知查询选择和可调推理速度等能力。

    注意：
        - RT-DETR中使用的F.grid_sample不支持`deterministic=True`参数。
        - AMP训练可能导致NaN输出，并可能在二分图匹配期间产生错误。

    示例：
        ```python
        from ultralytics.models.rtdetr.train import RTDETRTrainer

        args = dict(model="rtdetr-l.yaml", data="coco8.yaml", imgsz=640, epochs=3)
        trainer = RTDETRTrainer(overrides=args)
        trainer.train()
        ```
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        初始化并返回用于物体检测任务的RT-DETR模型。

        参数：
            cfg (dict, optional): 模型配置。默认为None。
            weights (str, optional): 预训练模型权重的路径。默认为None。
            verbose (bool): 如果为True，则显示详细日志。默认为True。

        返回：
            (RTDETRDetectionModel): 初始化后的模型。
        """
        model = RTDETRDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        构建并返回用于训练或验证的RT-DETR数据集。

        参数：
            img_path (str): 包含图像的文件夹路径。
            mode (str): 数据集模式，'train'或'val'。
            batch (int, optional): 矩形训练的批次大小。默认为None。

        返回：
            (RTDETRDataset): 特定模式的数据集对象。
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_validator(self):
        """
        返回适用于RT-DETR模型验证的DetectionValidator。

        返回：
            (RTDETRValidator): 用于模型验证的验证器对象。
        """
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def preprocess_batch(self, batch):
        """
        预处理一批图像。缩放并将图像转换为浮动格式。

        参数：
            batch (dict): 包含一批图像、边界框和标签的字典。

        返回：
            (dict): 预处理后的批次。
        """
        batch = super().preprocess_batch(batch)
        bs = len(batch["img"])
        batch_idx = batch["batch_idx"]
        gt_bbox, gt_class = [], []
        for i in range(bs):
            gt_bbox.append(batch["bboxes"][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch["cls"][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
        return batch
