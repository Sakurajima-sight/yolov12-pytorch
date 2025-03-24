# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

import torch

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.utils.torch_utils import is_parallel, strip_optimizer, torch_distributed_zero_first


class ClassificationTrainer(BaseTrainer):
    """
    分类模型训练器类，继承自 BaseTrainer。

    说明：
        - torchvision 的分类模型同样可以作为 'model' 参数传入，如 model='resnet18'。

    示例：
        ```python
        from ultralytics.models.yolo.classify import ClassificationTrainer

        args = dict(model="yolov8n-cls.pt", data="imagenet10", epochs=3)
        trainer = ClassificationTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化 ClassificationTrainer 对象，支持传入配置覆盖项与回调函数。"""
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        super().__init__(cfg, overrides, _callbacks)

    def set_model_attributes(self):
        """从已加载的数据集中设置 YOLO 模型的类别名称。"""
        self.model.names = self.data["names"]

    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回一个用于训练 YOLO 的分类模型实例（PyTorch 模型）。"""
        model = ClassificationModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # 设置 dropout
        for p in model.parameters():
            p.requires_grad = True  # 启用梯度计算用于训练
        return model

    def setup_model(self):
        """加载、创建或下载适用于任何任务的模型。"""
        import torchvision  # 将import放在本地以加快 ultralytics 的导入速度

        if str(self.model) in torchvision.models.__dict__:
            self.model = torchvision.models.__dict__[self.model](
                weights="IMAGENET1K_V1" if self.args.pretrained else None
            )
            ckpt = None
        else:
            ckpt = super().setup_model()
        ClassificationModel.reshape_outputs(self.model, self.data["nc"])
        return ckpt

    def build_dataset(self, img_path, mode="train", batch=None):
        """根据图像路径与模式（如train/test）创建 ClassificationDataset 实例。"""
        return ClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """返回一个 PyTorch DataLoader，并自动应用图像预处理变换用于推理。"""
        with torch_distributed_zero_first(rank):  # 如果使用 DDP，只在主进程初始化数据集 *.cache
            dataset = self.build_dataset(dataset_path, mode)

        loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank)
        # 附加推理使用的图像预处理变换
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def preprocess_batch(self, batch):
        """对一批图像和标签进行预处理。"""
        batch["img"] = batch["img"].to(self.device)
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def progress_string(self):
        """返回格式化的训练进度字符串。"""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):
        """返回 ClassificationValidator 实例用于验证。"""
        self.loss_names = ["loss"]
        return yolo.classify.ClassificationValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        返回一个带标签的训练损失字典。

        对于分类任务不是必须，但在分割与检测任务中是必要的。
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def plot_metrics(self):
        """从 CSV 文件中绘制评估指标图像。"""
        plot_results(file=self.csv, classify=True, on_plot=self.on_plot)  # 保存结果图 results.png

    def final_eval(self):
        """评估训练后的模型并保存验证结果。"""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # 移除优化器部分用于保存推理模型
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.data = self.args.data
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def plot_training_samples(self, batch, ni):
        """绘制训练样本及其标签。"""
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=batch["cls"].view(-1),  # 注意：分类模型需使用 .view() 而不是 .squeeze()
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
