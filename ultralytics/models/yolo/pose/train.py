# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.plotting import plot_images, plot_results


class PoseTrainer(yolo.detect.DetectionTrainer):
    """
    基于姿态估计模型的训练器类，继承自 DetectionTrainer。

    示例：
        ```python
        from ultralytics.models.yolo.pose import PoseTrainer

        args = dict(model="yolov8n-pose.pt", data="coco8-pose.yaml", epochs=3)
        trainer = PoseTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """使用指定的配置和参数覆盖项初始化一个 PoseTrainer 对象。"""
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

        # 针对 Apple MPS 后端的已知姿态估计兼容性问题给出警告
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "⚠️ 警告：Apple MPS 存在已知姿态估计 bug，建议将 device 设置为 'cpu'。"
                "详见：https://github.com/ultralytics/ultralytics/issues/4031"
            )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """使用指定配置和权重加载姿态估计模型。"""
        model = PoseModel(cfg, ch=3, nc=self.data["nc"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose)
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """设置 PoseModel 的关键点结构属性。"""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """返回用于验证的 PoseValidator 实例。"""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return yolo.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """绘制带有类别标签、边界框和关键点注释的一批训练样本。"""
        images = batch["img"]
        kpts = batch["keypoints"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        plot_images(
            images,
            batch_idx,
            cls,
            bboxes,
            kpts=kpts,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """绘制训练/验证阶段的指标图表。"""
        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # 保存为 results.png
