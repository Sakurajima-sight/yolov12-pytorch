# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO-NAS模型接口。

示例：
    ```python
    from ultralytics import NAS

    model = NAS("yolo_nas_s")
    results = model.predict("ultralytics/assets/bus.jpg")
    ```
"""

from pathlib import Path

import torch

from ultralytics.engine.model import Model
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.torch_utils import model_info

from .predict import NASPredictor
from .val import NASValidator


class NAS(Model):
    """
    YOLO-NAS模型用于目标检测。

    该类提供了YOLO-NAS模型的接口，并扩展了Ultralytics引擎中的`Model`类。
    它旨在通过预训练或自定义训练的YOLO-NAS模型，简化目标检测任务。

    示例：
        ```python
        from ultralytics import NAS

        model = NAS("yolo_nas_s")
        results = model.predict("ultralytics/assets/bus.jpg")
        ```

    属性：
        model (str): 预训练模型或模型名称的路径。默认为'yolo_nas_s.pt'。

    注意：
        YOLO-NAS模型仅支持预训练模型。请不要提供YAML配置文件。
    """

    def __init__(self, model="yolo_nas_s.pt") -> None:
        """初始化NAS模型，使用提供的或默认的'yolo_nas_s.pt'模型。"""
        assert Path(model).suffix not in {".yaml", ".yml"}, "YOLO-NAS模型仅支持预训练模型。"
        super().__init__(model, task="detect")

    def _load(self, weights: str, task=None) -> None:
        """加载现有的NAS模型权重，或者如果没有提供，使用预训练权重创建一个新的NAS模型。"""
        import super_gradients

        suffix = Path(weights).suffix
        if suffix == ".pt":
            self.model = torch.load(attempt_download_asset(weights))

        elif suffix == "":
            self.model = super_gradients.training.models.get(weights, pretrained_weights="coco")

        # 重写forward方法以忽略额外的参数
        def new_forward(x, *args, **kwargs):
            """忽略额外的__call__参数。"""
            return self.model._original_forward(x)

        self.model._original_forward = self.model.forward
        self.model.forward = new_forward

        # 标准化模型
        self.model.fuse = lambda verbose=True: self.model
        self.model.stride = torch.tensor([32])
        self.model.names = dict(enumerate(self.model._class_names))
        self.model.is_fused = lambda: False  # 用于info()
        self.model.yaml = {}  # 用于info()
        self.model.pt_path = weights  # 用于导出()
        self.model.task = "detect"  # 用于导出()

    def info(self, detailed=False, verbose=True):
        """
        记录模型信息。

        参数：
            detailed (bool): 显示关于模型的详细信息。
            verbose (bool): 控制输出的详细程度。
        """
        return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640)

    @property
    def task_map(self):
        """返回一个字典，将任务映射到相应的预测器和验证器类。"""
        return {"detect": {"predictor": NASPredictor, "validator": NASValidator}}
