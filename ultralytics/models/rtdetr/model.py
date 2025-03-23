# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
百度RT-DETR模型接口，基于视觉变换器（Vision Transformer）实现的实时物体检测器。RT-DETR提供实时性能和高精度，特别擅长加速后端如CUDA和TensorRT。它具有高效的混合编码器和IoU感知查询选择，提升了检测精度。

有关RT-DETR的更多信息，请访问：https://arxiv.org/pdf/2304.08069.pdf
"""

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel

from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):
    """
    百度RT-DETR模型接口。该基于视觉变换器的物体检测器提供高精度的实时性能。支持高效的混合编码、IoU感知查询选择，以及可调的推理速度。

    属性：
        model (str): 预训练模型的路径。默认为'rtdetr-l.pt'。
    """

    def __init__(self, model="rtdetr-l.pt") -> None:
        """
        使用给定的预训练模型文件初始化RT-DETR模型。支持.pt和.yaml格式。

        参数：
            model (str): 预训练模型的路径。默认为'rtdetr-l.pt'。

        异常：
            NotImplementedError: 如果模型文件扩展名不是'pt'、'yaml'或'yml'。
        """
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        """
        返回RT-DETR的任务映射，将任务与对应的Ultralytics类关联。

        返回：
            dict: 一个字典，将任务名称映射到RT-DETR模型的Ultralytics任务类。
        """
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }
