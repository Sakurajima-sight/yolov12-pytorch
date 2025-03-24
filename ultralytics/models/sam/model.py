# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
SAM模型接口。

本模块提供了Ultralytics的Segment Anything Model（SAM）的接口，旨在实现实时图像分割任务。SAM模型支持可提示的分割，在图像分析中具有无与伦比的多功能性，并且已在SA-1B数据集上进行训练。它具备零-shot性能，能够在没有事先知识的情况下适应新的图像分布和任务。

主要特点：
    - 可提示的分割
    - 实时性能
    - 零-shot迁移能力
    - 在SA-1B数据集上训练
"""

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.utils.torch_utils import model_info

from .build import build_sam
from .predict import Predictor, SAM2Predictor


class SAM(Model):
    """
    SAM（Segment Anything Model）接口类，用于实时图像分割任务。

    此类提供了Ultralytics的Segment Anything Model（SAM）的接口，旨在实现可提示的分割，并在图像分析中具有多功能性。它支持多种提示方式，如边界框、点或标签，并具备零-shot性能。

    属性：
        model (torch.nn.Module): 加载的SAM模型。
        is_sam2 (bool): 指示模型是否为SAM2变体。
        task (str): 任务类型，设置为"segment"表示SAM模型。

    方法：
        predict: 对给定的图像或视频源进行分割预测。
        info: 记录有关SAM模型的信息。

    示例：
        >>> sam = SAM("sam_b.pt")
        >>> results = sam.predict("image.jpg", points=[[500, 375]])
        >>> for r in results:
        >>>     print(f"检测到 {len(r.masks)} 个掩膜")
    """

    def __init__(self, model="sam_b.pt") -> None:
        """
        初始化SAM（Segment Anything Model）实例。

        参数：
            model (str): 预训练SAM模型文件的路径。文件应该具有.pt或.pth扩展名。

        异常：
            NotImplementedError: 如果模型文件的扩展名不是.pt或.pth。

        示例：
            >>> sam = SAM("sam_b.pt")
            >>> print(sam.is_sam2)
        """
        if model and Path(model).suffix not in {".pt", ".pth"}:
            raise NotImplementedError("SAM预测需要预训练的*.pt或*.pth模型。")
        self.is_sam2 = "sam2" in Path(model).stem
        super().__init__(model=model, task="segment")

    def _load(self, weights: str, task=None):
        """
        将指定的权重加载到SAM模型中。

        该方法使用提供的权重文件初始化SAM模型，设置模型架构并加载预训练参数。

        参数：
            weights (str): 权重文件的路径。应为.pt或.pth文件，包含模型参数。
            task (str | None): 任务名称。如果提供，它指定模型加载时的特定任务。

        示例：
            >>> sam = SAM("sam_b.pt")
            >>> sam._load("path/to/custom_weights.pt")
        """
        self.model = build_sam(weights)

    def predict(self, source, stream=False, bboxes=None, points=None, labels=None, **kwargs):
        """
        对给定的图像或视频源进行分割预测。

        参数：
            source (str | PIL.Image | numpy.ndarray): 图像或视频文件的路径，或PIL.Image对象，或numpy.ndarray对象。
            stream (bool): 如果为True，则启用实时流式处理。
            bboxes (List[List[float]] | None): 用于提示分割的边界框坐标列表。
            points (List[List[float]] | None): 用于提示分割的点的列表。
            labels (List[int] | None): 用于提示分割的标签列表。
            **kwargs (Any): 预测的其他关键字参数。

        返回：
            (List): 模型的预测结果。

        示例：
            >>> sam = SAM("sam_b.pt")
            >>> results = sam.predict("image.jpg", points=[[500, 375]])
            >>> for r in results:
            ...     print(f"检测到 {len(r.masks)} 个掩膜")
        """
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024)
        kwargs = {**overrides, **kwargs}
        prompts = dict(bboxes=bboxes, points=points, labels=labels)
        return super().predict(source, stream, prompts=prompts, **kwargs)

    def __call__(self, source=None, stream=False, bboxes=None, points=None, labels=None, **kwargs):
        """
        对给定的图像或视频源进行分割预测。

        该方法是 'predict' 方法的别名，提供了一种方便的方式调用 SAM 模型进行分割任务。

        参数:
            source (str | PIL.Image | numpy.ndarray | None): 图像或视频文件的路径，或者 PIL.Image 对象，或者 numpy.ndarray 对象。
            stream (bool): 如果为 True，启用实时流式处理。
            bboxes (List[List[float]] | None): 用于提示分割的边界框坐标列表。
            points (List[List[float]] | None): 用于提示分割的点的列表。
            labels (List[int] | None): 用于提示分割的标签列表。
            **kwargs (Any): 传递给 predict 方法的其他关键字参数。

        返回:
            (List): 模型的预测结果，通常包含分割掩码和其他相关信息。

        示例:
            >>> sam = SAM("sam_b.pt")
            >>> results = sam("image.jpg", points=[[500, 375]])
            >>> print(f"检测到 {len(results[0].masks)} 个掩码")
        """
        return self.predict(source, stream, bboxes, points, labels, **kwargs)

    def info(self, detailed=False, verbose=True):
        """
        记录关于 SAM 模型的信息。

        该方法提供有关 Segment Anything Model (SAM) 的详细信息，包括其架构、参数和计算需求。

        参数:
            detailed (bool): 如果为 True，显示关于模型层和操作的详细信息。
            verbose (bool): 如果为 True，打印信息到控制台。

        返回:
            (tuple): 一个元组，包含模型的信息（模型的字符串表示）。

        示例:
            >>> sam = SAM("sam_b.pt")
            >>> info = sam.info()
            >>> print(info[0])  # 打印摘要信息
        """
        return model_info(self.model, detailed=detailed, verbose=verbose)

    @property
    def task_map(self):
        """
        提供从 'segment' 任务到其对应 'Predictor' 的映射。

        返回:
            (Dict[str, Type[Predictor]]): 一个字典，将 'segment' 任务映射到对应的 Predictor 类。对于 SAM2 模型，它映射到 SAM2Predictor，否则映射到标准的 Predictor。

        示例:
            >>> sam = SAM("sam_b.pt")
            >>> task_map = sam.task_map
            >>> print(task_map)
            {'segment': <class 'ultralytics.models.sam.predict.Predictor'>}
        """
        return {"segment": {"predictor": SAM2Predictor if self.is_sam2 else Predictor}}
