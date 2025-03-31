# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import inspect
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image

from huggingface_hub import PyTorchModelHubMixin

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.engine.results import Results
from ultralytics.hub import HUB_WEB_ROOT, HUBTrainingSession
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    SETTINGS,
    callbacks,
    checks,
    emojis,
    yaml_load,
)


class Model(nn.Module, PyTorchModelHubMixin, repo_url="https://github.com/ultralytics/ultralytics", pipeline_tag="object-detection", license="agpl-3.0"):
    """
    YOLO模型的基类，实现不同模型类型的API统一。

    本类提供了与YOLO模型相关的各种操作的公共接口，如训练、验证、预测、导出和基准测试。它支持不同类型的模型，包括从本地文件、Ultralytics HUB或Triton Server加载的模型。

    属性:
        callbacks (Dict): 一个字典，包含各种事件的回调函数。
        predictor (BasePredictor): 用于进行预测的预测器对象。
        model (nn.Module): 基础的PyTorch模型。
        trainer (BaseTrainer): 用于训练模型的训练器对象。
        ckpt (Dict): 如果从*.pt文件加载模型，则为检查点数据。
        cfg (str): 如果从*.yaml文件加载模型，则为模型配置。
        ckpt_path (str): 检查点文件的路径。
        overrides (Dict): 用于模型配置的覆盖字典。
        metrics (Dict): 最新的训练/验证指标。
        session (HUBTrainingSession): 如果适用，则为Ultralytics HUB会话。
        task (str): 模型的任务类型。
        model_name (str): 模型的名称。

    方法:
        __call__: 预测方法的别名，使得模型实例可以被调用。
        _new: 基于配置文件初始化新的模型。
        _load: 从检查点文件加载模型。
        _check_is_pytorch_model: 确保模型是PyTorch模型。
        reset_weights: 重置模型的权重为初始状态。
        load: 从指定文件加载模型权重。
        save: 将当前模型状态保存到文件。
        info: 记录或返回关于模型的信息。
        fuse: 融合Conv2d和BatchNorm2d层以优化推理。
        predict: 执行目标检测预测。
        track: 执行目标跟踪。
        val: 在数据集上验证模型。
        benchmark: 在各种导出格式下对模型进行基准测试。
        export: 将模型导出到不同的格式。
        train: 在数据集上训练模型。
        tune: 执行超参数调优。
        _apply: 将函数应用于模型的张量。
        add_callback: 为事件添加回调函数。
        clear_callback: 清除事件的所有回调函数。
        reset_callbacks: 重置所有回调函数为默认函数。

    示例:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> results = model.predict("image.jpg")
        >>> model.train(data="coco8.yaml", epochs=3)
        >>> metrics = model.val()
        >>> model.export(format="onnx")
    """

    def __init__(
        self,
        model: Union[str, Path] = "yolo11n.pt",
        task: str = None,
        verbose: bool = False,
    ) -> None:
        """
        初始化YOLO模型类的实例。

        该构造函数根据提供的模型路径或名称设置模型。它处理多种类型的模型源，包括本地文件、Ultralytics HUB模型和Triton Server模型。该方法初始化模型的多个重要属性，并准备好执行诸如训练、预测或导出等操作。

        参数:
            model (Union[str, Path]): 要加载或创建的模型的路径或名称。可以是本地文件路径、Ultralytics HUB上的模型名称或Triton Server模型。
            task (str | None): 与YOLO模型相关的任务类型，指定其应用领域。
            verbose (bool): 如果为True，在模型初始化和后续操作期间启用详细输出。

        异常:
            FileNotFoundError: 如果指定的模型文件不存在或无法访问。
            ValueError: 如果模型文件或配置无效或不受支持。
            ImportError: 如果未安装特定模型类型所需的依赖项（如HUB SDK）。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> model = Model("path/to/model.yaml", task="detect")
            >>> model = Model("hub_model", verbose=True)
        """
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # 重用预测器
        self.model = None  # 模型对象
        self.trainer = None  # 训练器对象
        self.ckpt = {}  # 如果从*.pt文件加载
        self.cfg = None  # 如果从*.yaml文件加载
        self.ckpt_path = None
        self.overrides = {}  # 用于训练器对象的覆盖
        self.metrics = None  # 训练/验证指标
        self.session = None  # HUB会话
        self.task = task  # 任务类型
        model = str(model).strip()

        # 检查是否是来自Ultralytics HUB的模型 https://hub.ultralytics.com
        if self.is_hub_model(model):
            # 从HUB获取模型
            checks.check_requirements("hub-sdk>=0.0.12")
            session = HUBTrainingSession.create_session(model)
            model = session.model_file
            if session.train_args:  # 如果是从HUB发送的训练
                self.session = session

        # 检查是否是Triton Server模型
        elif self.is_triton_model(model):
            self.model_name = self.model = model
            self.overrides["task"] = task or "detect"  # 如果未显式设置，则设置为`task=detect`
            return

        # 加载或创建新的YOLO模型
        if Path(model).suffix in {".yaml", ".yml"}:
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task)

        # 删除super().training，以便访问self.model.training
        del self.training

    def __call__(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """
        预测方法的别名，使得模型实例可以被调用来进行预测。

        该方法通过允许直接调用模型实例并传入所需的参数来简化预测过程。

        参数:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple): 要进行预测的图像来源。可以是文件路径、URL、PIL图像、numpy数组、PyTorch张量或这些类型的列表/元组。
            stream (bool): 如果为True，视输入源为连续流进行预测。
            **kwargs: 配置预测过程的附加关键字参数。

        返回:
            (List[ultralytics.engine.results.Results]): 预测结果的列表，每个结果都封装在一个`Results`对象中。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model("https://ultralytics.com/images/bus.jpg")
            >>> for r in results:
            ...     print(f"检测到 {len(r)} 个物体")
        """
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def is_triton_model(model: str) -> bool:
        """
        检查给定的模型字符串是否为 Triton Server 的 URL。

        这个静态方法通过使用 urllib.parse.urlsplit() 解析组件，判断提供的模型字符串是否表示有效的 Triton Server URL。

        参数:
            model (str): 要检查的模型字符串。

        返回:
            (bool): 如果模型字符串是有效的 Triton Server URL，则返回 True，否则返回 False。

        示例:
            >>> Model.is_triton_model("http://localhost:8000/v2/models/yolov8n")
            True
            >>> Model.is_triton_model("yolo11n.pt")
            False
        """
        from urllib.parse import urlsplit

        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    @staticmethod
    def is_hub_model(model: str) -> bool:
        """
        检查提供的模型是否为 Ultralytics HUB 模型。

        这个静态方法判断给定的模型字符串是否表示有效的 Ultralytics HUB 模型标识符。

        参数:
            model (str): 要检查的模型字符串。

        返回:
            (bool): 如果模型是有效的 Ultralytics HUB 模型，则返回 True，否则返回 False。

        示例:
            >>> Model.is_hub_model("https://hub.ultralytics.com/models/MODEL")
            True
            >>> Model.is_hub_model("yolo11n.pt")
            False
        """
        return model.startswith(f"{HUB_WEB_ROOT}/models/")

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """
        初始化一个新的模型并从模型定义中推断任务类型。

        该方法基于提供的配置文件创建一个新的模型实例。如果未指定任务类型，将从配置中推断任务类型，并使用任务映射中的适当类来初始化模型。

        参数:
            cfg (str): 模型配置文件的路径（YAML 格式）。
            task (str | None): 模型的特定任务。如果为 None，则会从配置中推断。
            model (torch.nn.Module | None): 自定义模型实例。如果提供，将使用该实例，而不是创建一个新的模型。
            verbose (bool): 如果为 True，在加载模型时显示模型信息。

        异常:
            ValueError: 如果配置文件无效或无法推断任务类型。
            ImportError: 如果指定任务所需的依赖项未安装。

        示例:
            >>> model = Model()
            >>> model._new("yolov8n.yaml", task="detect", verbose=True)
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)  # 构建模型
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # 以下添加的代码允许从 YAML 文件中导出
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # 合并默认和模型参数（优先使用模型参数）
        self.model.task = self.task
        self.model_name = cfg

    def _load(self, weights: str, task=None) -> None:
        """
        从检查点文件加载模型或从权重文件初始化模型。

        该方法处理从 .pt 检查点文件或其他权重文件格式加载模型。它根据加载的权重设置模型、任务和相关属性。

        参数:
            weights (str): 要加载的模型权重文件的路径。
            task (str | None): 与模型相关的任务。如果为 None，将从模型中推断任务。

        异常:
            FileNotFoundError: 如果指定的权重文件不存在或无法访问。
            ValueError: 如果权重文件格式不受支持或无效。

        示例:
            >>> model = Model()
            >>> model._load("yolo11n.pt")
            >>> model._load("path/to/weights.pth", task="detect")
        """
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights, download_dir=SETTINGS["weights_dir"])  # 下载并返回本地文件
        weights = checks.check_model_file_from_stem(weights)  # 添加后缀，即 yolov8n -> yolov8n.pt

        if Path(weights).suffix == ".pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)  # 在所有情况下运行，未与上述调用重复
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def _check_is_pytorch_model(self) -> None:
        """
        检查模型是否为 PyTorch 模型，如果不是则抛出 TypeError。

        该方法验证模型是否为 PyTorch 模块或 .pt 文件。用于确保某些需要 PyTorch 模型的操作仅在兼容的模型类型上执行。

        异常:
            TypeError: 如果模型不是 PyTorch 模块或 .pt 文件，错误消息会提供有关支持的模型格式和操作的详细信息。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> model._check_is_pytorch_model()  # 不会抛出错误
            >>> model = Model("yolov8n.onnx")
            >>> model._check_is_pytorch_model()  # 抛出 TypeError
        """
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt"
        pt_module = isinstance(self.model, nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' 应该是一个 *.pt PyTorch 模型才能运行此方法，但它是其他格式的模型。"
                f"PyTorch 模型可以进行训练、验证、预测和导出，i.e. 'model.train(data=...)'，但是导出的格式 "
                f"如 ONNX、TensorRT 等仅支持 'predict' 和 'val' 模式，"
                f"i.e. 'yolo predict model=yolov8n.onnx'。\n要运行 CUDA 或 MPS 推理，请直接在推理命令中传递设备 "
                f"参数，i.e. 'model.predict(source=..., device=0)'"
            )

    def reset_weights(self) -> "Model":
        """
        重置模型的权重到初始状态。

        该方法遍历模型中的所有模块，并在这些模块具有 'reset_parameters' 方法时重置它们的参数。它还确保所有参数的 'requires_grad' 设置为 True，
        使其在训练期间可以更新。

        返回:
            (Model): 带有重置权重的类实例。

        异常:
            AssertionError: 如果模型不是 PyTorch 模型。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> model.reset_weights()
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: Union[str, Path] = "yolo11n.pt") -> "Model":
        """
        从指定的权重文件加载参数到模型中。

        该方法支持从文件或直接从权重对象加载权重。它根据名称和形状匹配参数，并将其传递给模型。

        参数:
            weights (Union[str, Path]): 权重文件的路径或权重对象。

        返回:
            (Model): 加载了权重的类实例。

        异常:
            AssertionError: 如果模型不是PyTorch模型。

        示例:
            >>> model = Model()
            >>> model.load("yolo11n.pt")
            >>> model.load(Path("path/to/weights.pt"))
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            self.overrides["pretrained"] = weights  # 记住用于DDP训练的权重
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def save(self, filename: Union[str, Path] = "saved_model.pt") -> None:
        """
        将当前模型状态保存到文件。

        该方法将模型的检查点（ckpt）导出到指定的文件名。它包含元数据，如日期、Ultralytics版本、许可信息和文档链接。

        参数:
            filename (Union[str, Path]): 要保存模型的文件名。

        异常:
            AssertionError: 如果模型不是PyTorch模型。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> model.save("my_model.pt")
        """
        self._check_is_pytorch_model()
        from copy import deepcopy
        from datetime import datetime

        from ultralytics import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, filename)

    def info(self, detailed: bool = False, verbose: bool = True):
        """
        记录或返回模型信息。

        该方法根据传入的参数提供关于模型的概述或详细信息。它可以控制输出的详细程度，并将信息作为列表返回。

        参数:
            detailed (bool): 如果为True，显示关于模型层和参数的详细信息。
            verbose (bool): 如果为True，打印信息。如果为False，返回信息列表。

        返回:
            (List[str]): 包含关于模型的各种信息的字符串列表，包括模型摘要、层的详细信息和参数计数。如果verbose为True，返回为空。

        异常:
            TypeError: 如果模型不是PyTorch模型。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> model.info()  # 打印模型摘要
            >>> info_list = model.info(detailed=True, verbose=False)  # 作为列表返回详细信息
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        """
        融合模型中的 Conv2d 和 BatchNorm2d 层以优化推理。

        该方法遍历模型的模块，并将连续的 Conv2d 和 BatchNorm2d 层融合为一个单独的层。这种融合可以通过减少前向传播过程中所需的操作和内存访问数量，显著提高推理速度。

        融合过程通常涉及将 BatchNorm2d 的参数（均值、方差、权重和偏置）折叠到前一个 Conv2d 层的权重和偏置中。这样就生成了一个同时执行卷积和归一化的单一 Conv2d 层。

        异常:
            TypeError: 如果模型不是 PyTorch 的 nn.Module。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> model.fuse()
            >>> # 模型现在已经融合，可以用于优化后的推理
        """
        self._check_is_pytorch_model()
        self.model.fuse()

    def embed(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """
        基于提供的源生成图像嵌入。

        该方法是对 'predict()' 方法的封装，专注于从图像源生成嵌入。它允许通过各种关键字参数定制嵌入过程。

        参数:
            source (str | Path | int | List | Tuple | np.ndarray | torch.Tensor): 用于生成嵌入的图像源。可以是文件路径、URL、PIL 图像、numpy 数组等。
            stream (bool): 如果为 True，则进行流式预测。
            **kwargs: 配置嵌入过程的其他关键字参数。

        返回:
            (List[torch.Tensor]): 包含图像嵌入的列表。

        异常:
            AssertionError: 如果模型不是 PyTorch 模型。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> image = "https://ultralytics.com/images/bus.jpg"
            >>> embeddings = model.embed(image)
            >>> print(embeddings[0].shape)
        """
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]  # 如果没有传入索引，则嵌入倒数第二层
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        predictor=None,
        **kwargs: Any,
    ) -> List[Results]:
        """
        使用 YOLO 模型对给定的图像源进行预测。

        该方法简化了预测过程，通过关键字参数允许各种配置。它支持使用自定义预测器或默认预测器进行预测。该方法可以处理不同类型的图像源，并可以在流式模式下工作。

        参数:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple): 进行预测的图像源。支持多种类型，包括文件路径、URL、PIL 图像、numpy 数组和 torch 张量。
            stream (bool): 如果为 True，则将输入源视为连续流进行预测。
            predictor (BasePredictor | None): 用于进行预测的自定义预测器类实例。如果为 None，则使用默认预测器。
            **kwargs: 配置预测过程的其他关键字参数。

        返回:
            (List[ultralytics.engine.results.Results]): 预测结果的列表，每个结果都封装在一个 Results 对象中。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.predict(source="path/to/image.jpg", conf=0.25)
            >>> for r in results:
            ...     print(r.boxes.data)  # 打印检测的边界框

        注意:
            - 如果未提供 'source'，默认使用 ASSETS 常量并发出警告。
            - 如果预测器尚未设置，该方法将设置一个新的预测器，并在每次调用时更新其参数。
            - 对于 SAM 类型的模型，可以通过关键字参数传递 'prompts'。
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(f"警告 ⚠️ 'source' 参数缺失。使用默认 'source={source}'。")

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict"}  # 方法默认配置
        args = {**self.overrides, **custom, **kwargs}  # 右侧参数优先
        prompts = args.pop("prompts", None)  # 对于 SAM 类型模型

        if not self.predictor:
            self.predictor = (predictor or self._smart_load("predictor"))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # 只有在预测器已设置时才更新参数
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # 对于 SAM 类型模型
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs: Any,
    ) -> List[Results]:
        """
        对指定的输入源进行目标跟踪，使用已注册的跟踪器。

        该方法使用模型的预测器以及可选的已注册跟踪器执行目标跟踪。它处理各种输入源，例如文件路径或视频流，并支持通过关键字参数进行定制。
        如果跟踪器尚未注册，方法会进行注册，并且可以在不同调用之间保持跟踪器的状态。

        参数:
            source (Union[str, Path, int, List, Tuple, np.ndarray, torch.Tensor], 可选): 目标跟踪的输入源。
                可以是文件路径、URL 或视频流。
            stream (bool): 如果为 True，则将输入源视为连续的视频流。默认为 False。
            persist (bool): 如果为 True，则在不同的调用之间保持跟踪器的状态。默认为 False。
            **kwargs: 配置跟踪过程的其他关键字参数。

        返回:
            (List[ultralytics.engine.results.Results]): 跟踪结果的列表，每个结果是一个 Results 对象。

        异常:
            AttributeError: 如果预测器没有注册的跟踪器。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.track(source="path/to/video.mp4", show=True)
            >>> for r in results:
            ...     print(r.boxes.id)  # 打印跟踪 ID

        注意:
            - 该方法为基于 ByteTrack 的跟踪设置了默认的置信度阈值 0.1。
            - 跟踪模式在关键字参数中显式设置。
            - 视频跟踪时批量大小设置为 1。
        """
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # 基于 ByteTrack 的方法需要低置信度预测作为输入
        kwargs["batch"] = kwargs.get("batch") or 1  # 视频跟踪时批量大小为 1
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def val(
        self,
        validator=None,
        **kwargs: Any,
    ):
        """
        使用指定的数据集和验证配置对模型进行验证。

        该方法简化了模型验证过程，允许通过各种设置进行定制。它支持使用自定义验证器或默认验证方法进行验证。该方法将默认配置、方法特定的默认值和
        用户提供的参数结合起来，配置验证过程。

        参数:
            validator (ultralytics.engine.validator.BaseValidator | None): 用于验证模型的自定义验证器类实例。
            **kwargs: 用于定制验证过程的任意关键字参数。

        返回:
            (ultralytics.utils.metrics.DetMetrics): 从验证过程中获得的验证指标。

        异常:
            AssertionError: 如果模型不是 PyTorch 模型。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.val(data="coco8.yaml", imgsz=640)
            >>> print(results.box.map)  # 打印 mAP50-95
        """
        custom = {"rect": True}  # 方法默认配置
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # 右侧参数优先

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(
        self,
        **kwargs: Any,
    ):
        """
        在各种导出格式下对模型进行基准测试，以评估性能。

        该方法评估模型在不同导出格式（如 ONNX、TorchScript 等）下的性能。
        它使用 ultralytics.utils.benchmarks 模块中的 'benchmark' 函数进行基准测试。基准测试的配置
        结合了默认配置值、模型特定的参数、方法特定的默认值和用户提供的其他关键字参数。

        参数:
            **kwargs: 用于定制基准测试过程的任意关键字参数。这些参数与默认配置、模型特定的参数以及
                方法默认值进行合并。常见选项包括：
                - data (str): 用于基准测试的数据集路径。
                - imgsz (int | List[int]): 用于基准测试的图像大小。
                - half (bool): 是否使用半精度（FP16）模式。
                - int8 (bool): 是否使用 int8 精度模式。
                - device (str): 用于运行基准测试的设备（例如，'cpu'、'cuda'）。
                - verbose (bool): 是否打印详细的基准测试信息。

        返回:
            (Dict): 包含基准测试结果的字典，包括不同导出格式的指标。

        异常:
            AssertionError: 如果模型不是 PyTorch 模型。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.benchmark(data="coco8.yaml", imgsz=640, half=True)
            >>> print(results)
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        custom = {"verbose": False}  # 方法默认配置
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        return benchmark(
            model=self,
            data=kwargs.get("data"),  # 如果没有传递 'data' 参数，则默认为 None
            imgsz=args["imgsz"],
            half=args["half"],
            int8=args["int8"],
            device=args["device"],
            verbose=kwargs.get("verbose"),
        )

    def export(
        self,
        **kwargs: Any,
    ) -> str:
        """
        将模型导出为适合部署的不同格式。

        该方法便于将模型导出为多种格式（例如，ONNX、TorchScript）以便部署。它使用“Exporter”类进行导出过程，结合模型特定的覆盖、方法默认值和提供的任何附加参数。

        参数:
            **kwargs: 任意关键字参数，用于自定义导出过程。这些参数将与模型的覆盖和方法默认值结合。常见的参数包括：
                format (str): 导出格式（例如，'onnx'、'engine'、'coreml'）。
                half (bool): 以半精度导出模型。
                int8 (bool): 以int8精度导出模型。
                device (str): 用于导出的设备。
                workspace (int): TensorRT引擎的最大内存工作空间大小。
                nms (bool): 向模型添加非最大抑制（NMS）模块。
                simplify (bool): 简化ONNX模型。

        返回:
            (str): 导出模型文件的路径。

        异常:
            AssertionError: 如果模型不是PyTorch模型。
            ValueError: 如果指定了不支持的导出格式。
            RuntimeError: 如果导出过程由于错误失败。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> model.export(format="onnx", dynamic=True, simplify=True)
            'path/to/exported/model.onnx'
        """
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,  # 重置以避免多GPU错误
            "verbose": False,
        }  # 方法默认值
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # 优先级最高的参数在右边
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(
        self,
        trainer=None,
        **kwargs: Any,
    ):
        """
        使用指定的数据集和训练配置训练模型。

        该方法支持使用一系列可自定义的设置来训练模型。它支持使用自定义训练器或默认训练方法。该方法处理诸如从检查点恢复训练、与Ultralytics HUB集成以及训练后更新模型和配置等场景。

        使用Ultralytics HUB时，如果会话加载了模型，方法会优先使用HUB的训练参数，并在提供本地参数时发出警告。它会检查pip更新，并将默认配置、方法特定的默认值和用户提供的参数结合起来配置训练过程。

        参数:
            trainer (BaseTrainer | None): 自定义训练器实例。如果为None，则使用默认训练器。
            **kwargs: 任意关键字参数，用于训练配置。常见选项包括：
                data (str): 数据集配置文件的路径。
                epochs (int): 训练的轮数。
                batch_size (int): 训练的批量大小。
                imgsz (int): 输入图像的大小。
                device (str): 用于训练的设备（例如，'cuda'、'cpu'）。
                workers (int): 数据加载的工作线程数。
                optimizer (str): 用于训练的优化器。
                lr0 (float): 初始学习率。
                patience (int): 如果在多个epoch内没有明显改进，则提前停止训练。

        返回:
            (Dict | None): 如果训练成功并且有训练指标，则返回训练指标；否则返回None。

        异常:
            AssertionError: 如果模型不是PyTorch模型。
            PermissionError: 如果在HUB会话中出现权限问题。
            ModuleNotFoundError: 如果未安装HUB SDK。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.train(data="coco8.yaml", epochs=3)
        """
        self._check_is_pytorch_model()
        if hasattr(self.session, "model") and self.session.model.id:  # 如果是Ultralytics HUB会话并且加载了模型
            if any(kwargs):
                LOGGER.warning("警告 ⚠️ 使用HUB训练参数，忽略本地训练参数。")
            kwargs = self.session.train_args  # 重写kwargs

        checks.check_pip_update_available()

        overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {
            # 注意：处理'cfg'包含'data'的情况。
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }  # 方法默认值
        args = {**overrides, **custom, **kwargs, "mode": "train"}  # 优先级最高的参数在右边
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
        if not args.get("resume"):  # 如果不是恢复训练，则手动设置模型
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

        self.trainer.hub_session = self.session  # 附加可选的HUB会话
        self.trainer.train()
        # 训练后更新模型和配置
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, self.ckpt = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: DDP没有返回指标
        return self.metrics

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args: Any,
        **kwargs: Any,
    ):
        """
        对模型进行超参数调优，并提供使用Ray Tune的选项。

        该方法支持两种超参数调优模式：使用Ray Tune或自定义调优方法。当启用Ray Tune时，它会利用来自ultralytics.utils.tuner模块的“run_ray_tune”函数。否则，它使用内部的'Tuner'类进行调优。该方法将默认值、覆盖值和自定义参数结合起来配置调优过程。

        参数:
            use_ray (bool): 如果为True，则使用Ray Tune进行超参数调优。默认值为False。
            iterations (int): 要执行的调优迭代次数。默认值为10。
            *args: 可变长度的附加参数列表。
            **kwargs: 任意关键字参数。这些参数将与模型的覆盖和默认值结合使用。

        返回:
            (Dict): 包含超参数搜索结果的字典。

        异常:
            AssertionError: 如果模型不是PyTorch模型。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.tune(use_ray=True, iterations=20)
            >>> print(results)
        """
        self._check_is_pytorch_model()
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune

            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}  # 方法默认值
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}  # 优先级最高的参数在右边
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    def _apply(self, fn) -> "Model":
        """
        将一个函数应用于模型的张量（不包括参数或已注册的缓冲区）。

        该方法扩展了父类 _apply 方法的功能，额外地重置预测器并更新模型覆盖项中的设备。它通常用于诸如
        将模型移动到不同设备或更改其精度之类的操作。

        参数:
            fn (Callable): 要应用于模型张量的函数。通常是像 to()、cpu()、cuda()、half() 或 float() 这样的函数。

        返回:
            (Model): 应用该函数并更新属性后的模型实例。

        异常:
            AssertionError: 如果模型不是 PyTorch 模型。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> model = model._apply(lambda t: t.cuda())  # 将模型移到 GPU
        """
        self._check_is_pytorch_model()
        self = super()._apply(fn)  # noqa
        self.predictor = None  # 重置预测器，因为设备可能已经改变
        self.overrides["device"] = self.device  # 将设备从 str(self.device) 形式更新为 'cuda:0'
        return self

    @property
    def names(self) -> Dict[int, str]:
        """
        获取与加载的模型关联的类名。

        该属性返回类名（如果在模型中定义了类名）。它通过 ultralytics.nn.autobackend 模块中的 'check_class_names' 函数
        来验证类名的有效性。如果预测器未初始化，则在获取类名之前进行设置。

        返回:
            (Dict[int, str]): 与模型关联的类名字典。

        异常:
            AttributeError: 如果模型或预测器没有 'names' 属性。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.names)
            {0: 'person', 1: 'bicycle', 2: 'car', ...}
        """
        from ultralytics.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        if not self.predictor:  # 导出格式在调用 predict() 之前不会定义预测器
            self.predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=False)
        return self.predictor.model.names

    @property
    def device(self) -> torch.device:
        """
        获取模型参数所在的设备。

        该属性确定模型参数当前存储的设备（CPU 或 GPU）。仅适用于 nn.Module 实例的模型。

        返回:
            (torch.device): 模型的设备（CPU/GPU）。

        异常:
            AttributeError: 如果模型不是 PyTorch nn.Module 实例。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.device)
            device(type='cuda', index=0)  # 如果可用，返回 CUDA 设备
            >>> model = model.to("cpu")
            >>> print(model.device)
            device(type='cpu')
        """
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        """
        获取已加载模型的输入数据所应用的转换（数据预处理）操作。

        该属性返回模型中定义的转换（如果有）。这些转换通常包括在将输入数据传递到模型之前，进行的预处理步骤，如调整大小、归一化和数据增强等。

        返回:
            (object | None): 如果模型中有定义转换，则返回转换对象，否则返回 None。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> transforms = model.transforms
            >>> if transforms:
            ...     print(f"模型转换操作: {transforms}")
            ... else:
            ...     print("此模型未定义转换操作。")
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        """
        为指定事件添加回调函数。

        该方法允许注册自定义回调函数，这些回调函数会在模型操作中的特定事件（如训练或推理）发生时被触发。回调函数提供了一种在模型生命周期的各个阶段扩展和定制其行为的方式。

        参数:
            event (str): 要附加回调的事件名称。必须是 Ultralytics 框架识别的有效事件名称。
            func (Callable): 要注册的回调函数。该函数将在指定事件发生时被调用。

        异常:
            ValueError: 如果事件名称无法识别或无效。

        示例:
            >>> def on_train_start(trainer):
            ...     print("训练开始！")
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", on_train_start)
            >>> model.train(data="coco8.yaml", epochs=1)
        """
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """
        清除为指定事件注册的所有回调函数。

        该方法移除与给定事件相关的所有自定义和默认回调函数。它将指定事件的回调列表重置为空列表，从而有效地移除该事件的所有已注册回调函数。

        参数:
            event (str): 要清除回调的事件名称。该名称应是 Ultralytics 回调系统识别的有效事件名称。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", lambda: print("训练开始"))
            >>> model.clear_callback("on_train_start")
            >>> # 现在所有 'on_train_start' 的回调都已移除

        注意:
            - 此方法会影响用户添加的自定义回调以及 Ultralytics 框架提供的默认回调。
            - 调用此方法后，在为指定事件添加新的回调之前，该事件将不会执行任何回调。
            - 使用时需谨慎，因为它会移除所有回调，包括一些可能对某些操作的正常运行至关重要的回调。
        """
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        """
        重置所有回调函数为默认函数。

        该方法将所有事件的回调函数恢复为默认函数，移除之前添加的任何自定义回调。它会遍历所有默认回调事件，并将当前回调替换为默认回调。

        默认回调函数在 'callbacks.default_callbacks' 字典中定义，其中包含了模型生命周期中各种事件的预定义函数，例如 on_train_start、on_epoch_end 等。

        该方法在你希望在做了自定义修改后恢复到原始回调集时非常有用，确保在不同运行或实验中的一致性行为。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", custom_function)
            >>> model.reset_callbacks()
            # 所有回调函数现在已重置为默认函数
        """
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        """
        在加载 PyTorch 模型检查点时重置特定参数。

        该静态方法过滤输入的参数字典，仅保留一组被认为对模型加载重要的键。它用于确保从检查点加载模型时，只保留相关的参数，丢弃任何不必要或可能冲突的设置。

        参数:
            args (dict): 包含各种模型参数和设置的字典。

        返回:
            (dict): 仅包含从输入参数中保留的特定键的新字典。

        示例:
            >>> original_args = {"imgsz": 640, "data": "coco.yaml", "task": "detect", "batch": 16, "epochs": 100}
            >>> reset_args = Model._reset_ckpt_args(original_args)
            >>> print(reset_args)
            {'imgsz': 640, 'data': 'coco.yaml', 'task': 'detect'}
        """
        include = {"imgsz", "data", "task", "single_cls"}  # 仅在加载 PyTorch 模型时记住这些参数
        return {k: v for k, v in args.items() if k in include}

    # def __getattr__(self, attr):
    #    """当对象没有请求的属性时引发错误。"""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' 对象没有属性 '{attr}'。请参阅下面的有效属性。\n{self.__doc__}")

    def _smart_load(self, key: str):
        """
        根据模型任务加载适当的模块。

        该方法动态选择并返回正确的模块（模型、训练器、验证器或预测器），根据模型当前的任务和提供的键进行选择。它使用 task_map 属性来确定加载正确的模块。

        参数:
            key (str): 要加载的模块类型。必须是 'model'、'trainer'、'validator' 或 'predictor' 之一。

        返回:
            (object): 与指定键和当前任务对应的加载模块。

        异常:
            NotImplementedError: 如果指定的键对于当前任务不被支持。

        示例:
            >>> model = Model(task="detect")
            >>> predictor = model._smart_load("predictor")
            >>> trainer = model._smart_load("trainer")

        备注:
            - 该方法通常由 Model 类的其他方法内部使用。
            - task_map 属性应正确初始化，并为每个任务提供正确的映射。
        """
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # 获取函数名。
            raise NotImplementedError(
                emojis(f"警告 ⚠️ '{name}' 模型尚不支持 '{self.task}' 任务的 '{mode}' 模式。")
            ) from e

    @property
    def task_map(self) -> dict:
        """
        提供从模型任务到不同模式对应类的映射。

        此属性方法返回一个字典，该字典将每个支持的任务（例如，检测、分割、分类）映射到一个嵌套字典。该嵌套字典包含不同操作模式（模型、训练器、验证器、预测器）到其各自类实现的映射。

        该映射允许根据模型的任务和所需的操作模式动态加载适当的类。这为处理Ultralytics框架内各种任务和模式提供了灵活和可扩展的架构。

        返回:
            (Dict[str, Dict[str, Any]]): 一个字典，其中键是任务名称（str），值是嵌套字典。每个嵌套字典包含键 'model'、'trainer'、'validator' 和 'predictor'，这些键映射到各自的类实现。

        示例:
            >>> model = Model()
            >>> task_map = model.task_map
            >>> detect_class_map = task_map["detect"]
            >>> segment_class_map = task_map["segment"]

        注意:
            此方法的实际实现可能会根据Ultralytics框架所支持的具体任务和类有所不同。此文档字符串提供了预期行为和结构的通用描述。
        """
        raise NotImplementedError("请为您的模型提供任务映射！")

    def eval(self):
        """
        将模型设置为评估模式。

        该方法将模型的模式更改为评估模式，这会影响像dropout和batch normalization等层，在训练和评估期间表现不同。

        返回:
            (Model): 设置为评估模式的模型实例。

        示例:
            >> model = YOLO("yolo11n.pt")
            >> model.eval()
        """
        self.model.eval()
        return self

    def __getattr__(self, name):
        """
        通过Model类直接访问模型属性。

        该方法提供了一种通过Model类实例直接访问底层模型属性的方式。它首先检查请求的属性是否是'model'，如果是，则返回模块字典中的模型。否则，它将属性查找委托给底层模型。

        参数:
            name (str): 要检索的属性名称。

        返回:
            (Any): 请求的属性值。

        异常:
            AttributeError: 如果请求的属性在模型中不存在。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.stride)
            >>> print(model.task)
        """
        return self._modules["model"] if name == "model" else getattr(self.model, name)
