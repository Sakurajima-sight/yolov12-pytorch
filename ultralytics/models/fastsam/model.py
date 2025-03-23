# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.engine.model import Model

from .predict import FastSAMPredictor
from .val import FastSAMValidator


class FastSAM(Model):
    """
    FastSAM模型接口。

    示例：
        ```python
        from ultralytics import FastSAM

        model = FastSAM("last.pt")
        results = model.predict("ultralytics/assets/bus.jpg")
        ```
    """

    def __init__(self, model="FastSAM-x.pt"):
        """调用父类（YOLO）的__init__方法，并更新默认模型。"""
        if str(model) == "FastSAM.pt":
            model = "FastSAM-x.pt"
        assert Path(model).suffix not in {".yaml", ".yml"}, "FastSAM模型仅支持预训练模型。"
        super().__init__(model=model, task="segment")

    def predict(self, source, stream=False, bboxes=None, points=None, labels=None, texts=None, **kwargs):
        """
        对图像或视频源进行分割预测。

        支持通过边界框、点、标签和文本进行提示分割。

        参数：
            source (str | PIL.Image | numpy.ndarray): 输入源。
            stream (bool): 启用实时流处理。
            bboxes (list): 提示分割的边界框坐标。
            points (list): 提示分割的点。
            labels (list): 提示分割的标签。
            texts (list): 提示分割的文本。
            **kwargs (Any): 其他关键字参数。

        返回：
            (list): 模型预测结果。
        """
        prompts = dict(bboxes=bboxes, points=points, labels=labels, texts=texts)
        return super().predict(source, stream, prompts=prompts, **kwargs)

    @property
    def task_map(self):
        """返回一个字典，将分割任务映射到相应的预测器和验证器类。"""
        return {"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}
