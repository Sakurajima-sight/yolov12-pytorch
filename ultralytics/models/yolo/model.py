# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from ultralytics.utils import ROOT, yaml_load


class YOLO(Model):
    """YOLO（You Only Look Once）目标检测模型。"""

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """初始化 YOLO 模型，如果模型文件名包含 '-world'，则切换到 YOLOWorld 模型。"""
        path = Path(model)
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # 如果是 YOLOWorld PyTorch 模型
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # 继续使用默认的 YOLO 初始化
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """将头部映射到模型、训练器、验证器和预测器类。"""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    """YOLO-World 目标检测模型。"""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        使用预训练的 YOLOv8-World 模型文件初始化 YOLOv8-World 模型。

        加载用于目标检测的 YOLOv8-World 模型。如果没有提供自定义类别名称，则默认分配 COCO 类别名称。

        参数：
            model (str | Path): 预训练模型文件的路径。支持 *.pt 和 *.yaml 格式。
            verbose (bool): 如果为 True，则在初始化过程中打印附加信息。
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # 如果没有自定义类别名称，则分配默认的 COCO 类别名称
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """将头部映射到模型、验证器和预测器类。"""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        """
        设置类别。

        参数：
            classes (List(str)): 类别名称的列表，例如 ["person"]。
        """
        self.model.set_classes(classes)
        # 如果给定了背景类别，则将其移除
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # 重置方法的类别名称
        # self.predictor = None  # 重置预测器，否则旧的类别名称会残留
        if self.predictor:
            self.predictor.model.names = classes
