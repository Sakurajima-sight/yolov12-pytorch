# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images


class ClassificationValidator(BaseValidator):
    """
    这是一个继承自 BaseValidator 的分类模型验证器类，用于对分类模型进行评估验证。

    注意：
        - 你可以将 torchvision 中的分类模型传入 model 参数，例如 model='resnet18'。

    示例：
        ```python
        from ultralytics.models.yolo.classify import ClassificationValidator

        args = dict(model="yolov8n-cls.pt", data="imagenet10")
        validator = ClassificationValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """初始化 ClassificationValidator 实例，传入 dataloader、保存目录、进度条、参数和回调函数。"""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task = "classify"
        self.metrics = ClassifyMetrics()

    def get_desc(self):
        """返回格式化的字符串，描述分类任务中的各项指标（如 top1 与 top5 准确率）。"""
        return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")

    def init_metrics(self, model):
        """初始化混淆矩阵、类别名称，以及 Top-1 和 Top-5 的准确率统计。"""
        self.names = model.names
        self.nc = len(model.names)
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf, task="classify")
        self.pred = []
        self.targets = []

    def preprocess(self, batch):
        """对输入的 batch 进行预处理并返回处理结果。"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        """根据模型的预测结果和当前 batch 的真实标签，更新运行中的评估指标。"""
        n5 = min(len(self.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu())
        self.targets.append(batch["cls"].type(torch.int32).cpu())

    def finalize_metrics(self, *args, **kwargs):
        """最终计算并整理模型评估指标，如混淆矩阵和评估时间等。"""
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

    def postprocess(self, preds):
        """对分类模型的预测结果进行后处理。"""
        return preds[0] if isinstance(preds, (list, tuple)) else preds

    def get_stats(self):
        """处理预测值与目标标签，返回一个包含分类结果的指标字典。"""
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def build_dataset(self, img_path):
        """根据图像路径及预处理参数创建并返回一个 ClassificationDataset 实例。"""
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)

    def get_dataloader(self, dataset_path, batch_size):
        """为分类任务构建并返回一个数据加载器（DataLoader）。"""
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self):
        """打印 YOLO 分类模型的评估结果指标。"""
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)  # 打印格式
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch, ni):
        """绘制验证集图像样本。"""
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=batch["cls"].view(-1),  # 注意：分类模型中使用 .view()，而非 .squeeze()
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """在输入图像上绘制预测类别，并保存预测可视化结果。"""
        plot_images(
            batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=torch.argmax(preds, dim=1),
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred
