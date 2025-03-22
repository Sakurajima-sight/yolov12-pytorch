# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics 结果、框和掩码类，用于处理推理结果。

用法：请参阅 https://docs.ultralytics.com/modes/predict/
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER, SimpleClass, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode


class BaseTensor(SimpleClass):
    """
    基础张量类，提供附加方法以便于操作和设备处理。

    属性:
        data (torch.Tensor | np.ndarray): 预测数据，如边界框、掩码或关键点。
        orig_shape (Tuple[int, int]): 图像的原始形状，通常为 (高度, 宽度) 格式。

    方法:
        cpu: 返回存储在CPU内存中的张量副本。
        numpy: 将张量返回为numpy数组副本。
        cuda: 将张量移动到GPU内存中，如果需要则返回新的实例。
        to: 返回具有指定设备和数据类型的张量副本。

    示例:
        >>> import torch
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> orig_shape = (720, 1280)
        >>> base_tensor = BaseTensor(data, orig_shape)
        >>> cpu_tensor = base_tensor.cpu()
        >>> numpy_array = base_tensor.numpy()
        >>> gpu_tensor = base_tensor.cuda()
    """

    def __init__(self, data, orig_shape) -> None:
        """
        使用预测数据和图像的原始形状初始化BaseTensor。

        参数:
            data (torch.Tensor | np.ndarray): 预测数据，如边界框、掩码或关键点。
            orig_shape (Tuple[int, int]): 图像的原始形状，格式为 (高度, 宽度)。

        示例:
            >>> import torch
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
        """
        assert isinstance(data, (torch.Tensor, np.ndarray)), "data 必须是 torch.Tensor 或 np.ndarray 类型"
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        """
        返回底层数据张量的形状。

        返回:
            (Tuple[int, ...]): 数据张量的形状。

        示例:
            >>> data = torch.rand(100, 4)
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> print(base_tensor.shape)
            (100, 4)
        """
        return self.data.shape

    def cpu(self):
        """
        返回存储在CPU内存中的张量副本。

        返回:
            (BaseTensor): 一个新的BaseTensor对象，其中的数据张量被移动到CPU内存中。

        示例:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> cpu_tensor = base_tensor.cpu()
            >>> isinstance(cpu_tensor, BaseTensor)
            True
            >>> cpu_tensor.data.device
            device(type='cpu')
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """
        返回张量的numpy数组副本。

        返回:
            (np.ndarray): 一个包含与原张量相同数据的numpy数组。

        示例:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
            >>> numpy_array = base_tensor.numpy()
            >>> print(type(numpy_array))
            <class 'numpy.ndarray'>
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """
        将张量移动到GPU内存中。

        返回:
            (BaseTensor): 一个新的BaseTensor实例，其中的数据被移动到GPU内存中，如果数据已经是numpy数组，则返回自身。

        示例:
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> gpu_tensor = base_tensor.cuda()
            >>> print(gpu_tensor.data.device)
            cuda:0
        """
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """
        返回具有指定设备和数据类型的张量副本。

        参数:
            *args (Any): 传递给 torch.Tensor.to() 的可变长度参数。
            **kwargs (Any): 传递给 torch.Tensor.to() 的任意关键字参数。

        返回:
            (BaseTensor): 一个新的BaseTensor实例，数据被移动到指定的设备和/或数据类型。

        示例:
            >>> base_tensor = BaseTensor(torch.randn(3, 4), orig_shape=(480, 640))
            >>> cuda_tensor = base_tensor.to("cuda")
            >>> float16_tensor = base_tensor.to(dtype=torch.float16)
        """
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self):  # 重写len(results)
        """
        返回底层数据张量的长度。

        返回:
            (int): 数据张量在第一个维度上的元素数量。

        示例:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> len(base_tensor)
            2
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回包含指定索引元素的新BaseTensor实例。

        参数:
            idx (int | List[int] | torch.Tensor): 用于选择数据张量中的元素的索引或索引列表。

        返回:
            (BaseTensor): 一个新的BaseTensor实例，包含索引后的数据。

        示例:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> result = base_tensor[0]  # 选择第一行
            >>> print(result.data)
            tensor([1, 2, 3])
        """
        return self.__class__(self.data[idx], self.orig_shape)


class Results(SimpleClass):
    """
    用于存储和操作推理结果的类。

    该类封装了处理 YOLO 模型的检测、分割、姿态估计和分类结果的功能。

    属性:
        orig_img (numpy.ndarray): 作为 numpy 数组的原始图像。
        orig_shape (Tuple[int, int]): 图像的原始形状，以 (高度, 宽度) 格式表示。
        boxes (Boxes | None): 包含检测边界框的对象。
        masks (Masks | None): 包含检测掩模的对象。
        probs (Probs | None): 包含分类任务中每个类的概率的对象。
        keypoints (Keypoints | None): 包含每个对象检测到的关键点的对象。
        obb (OBB | None): 包含面向目标的边界框的对象。
        speed (Dict[str, float | None]): 包含预处理、推理和后处理速度的字典。
        names (Dict[int, str]): 类 ID 与类名的映射字典。
        path (str): 图像文件的路径。
        _keys (Tuple[str, ...]): 内部使用的属性名称元组。

    方法:
        update: 使用新的检测结果更新对象属性。
        cpu: 返回一个副本，其中所有张量都存储在 CPU 内存中。
        numpy: 返回一个副本，其中所有张量都转换为 numpy 数组。
        cuda: 返回一个副本，其中所有张量都存储在 GPU 内存中。
        to: 返回一个副本，其中张量存储在指定的设备和数据类型中。
        new: 返回一个新的 Results 对象，具有相同的图像、路径和名称。
        plot: 在输入图像上绘制检测结果，并返回带注释的图像。
        show: 在屏幕上显示带注释的结果。
        save: 将带注释的结果保存到文件。
        verbose: 返回每个任务的日志字符串，详细描述检测和分类结果。
        save_txt: 将检测结果保存到文本文件。
        save_crop: 保存裁剪后的检测图像。
        tojson: 将检测结果转换为 JSON 格式。

    示例:
        >>> results = model("path/to/image.jpg")
        >>> for result in results:
        ...     print(result.boxes)  # 打印检测框
        ...     result.show()  # 显示带注释的图像
        ...     result.save(filename="result.jpg")  # 保存带注释的图像
    """

    def __init__(
        self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None, speed=None
    ) -> None:
        """
        初始化 Results 类，用于存储和操作推理结果。

        参数:
            orig_img (numpy.ndarray): 作为 numpy 数组的原始图像。
            path (str): 图像文件的路径。
            names (Dict): 类名字典。
            boxes (torch.Tensor | None): 包含每个检测的边界框坐标的二维张量。
            masks (torch.Tensor | None): 包含检测掩模的三维张量，每个掩模是一个二值图像。
            probs (torch.Tensor | None): 包含每个类的概率的张量，用于分类任务。
            keypoints (torch.Tensor | None): 包含每个检测的关键点坐标的二维张量。
            obb (torch.Tensor | None): 包含每个检测的面向目标的边界框坐标的二维张量。
            speed (Dict | None): 包含预处理、推理和后处理速度（ms/图像）的字典。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> result = results[0]  # 获取第一个结果
            >>> boxes = result.boxes  # 获取第一个结果的边界框
            >>> masks = result.masks  # 获取第一个结果的掩模

        注意:
            对于默认的姿态模型，人体姿态估计的关键点索引为：
            0: 鼻子, 1: 左眼, 2: 右眼, 3: 左耳, 4: 右耳
            5: 左肩, 6: 右肩, 7: 左肘, 8: 右肘
            9: 左腕, 10: 右腕, 11: 左臀, 12: 右臀
            13: 左膝, 14: 右膝, 15: 左踝, 16: 右踝
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # 原始大小的边界框
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # 原始大小或图像大小的掩模
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.speed = speed if speed is not None else {"preprocess": None, "inference": None, "postprocess": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"

    def __getitem__(self, idx):
        """
        返回推理结果中特定索引的 Results 对象。

        参数:
            idx (int | slice): 索引或切片，用于从 Results 对象中检索。

        返回:
            (Results): 一个新的 Results 对象，包含推理结果的指定子集。

        示例:
            >>> results = model("path/to/image.jpg")  # 执行推理
            >>> single_result = results[0]  # 获取第一个结果
            >>> subset_results = results[1:4]  # 获取一个结果的切片
        """
        return self._apply("__getitem__", idx)

    def __len__(self):
        """
        返回 Results 对象中的检测数量。

        返回:
            (int): 检测的数量，取决于第一个非空属性（boxes、masks、probs、keypoints 或 obb）的长度。

        示例:
            >>> results = Results(orig_img, path, names, boxes=torch.rand(5, 4))
            >>> len(results)
            5
        """
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes=None, masks=None, probs=None, obb=None):
        """
        使用新的检测数据更新 Results 对象。

        该方法允许更新 Results 对象的边界框、掩模、概率和面向目标的边界框（OBB）。它确保边界框被裁剪到原始图像形状。

        参数:
            boxes (torch.Tensor | None): 形状为 (N, 6) 的张量，包含边界框坐标和置信度。格式为 (x1, y1, x2, y2, conf, class)。
            masks (torch.Tensor | None): 形状为 (N, H, W) 的张量，包含分割掩模。
            probs (torch.Tensor | None): 形状为 (num_classes,) 的张量，包含每个类的概率。
            obb (torch.Tensor | None): 形状为 (N, 5) 的张量，包含面向目标的边界框坐标。

        示例:
            >>> results = model("image.jpg")
            >>> new_boxes = torch.tensor([[100, 100, 200, 200, 0.9, 0]])
            >>> results[0].update(boxes=new_boxes)
        """
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs
        if obb is not None:
            self.obb = OBB(obb, self.orig_shape)

    def _apply(self, fn, *args, **kwargs):
        """
        将函数应用于所有非空属性，并返回一个新的 Results 对象，其中包含修改后的属性。

        该方法通常由 .to()、.cuda()、.cpu() 等方法内部调用。

        参数:
            fn (str): 要应用的函数的名称。
            *args (Any): 要传递给函数的可变长度参数。
            **kwargs (Any): 要传递给函数的任意关键字参数。

        返回:
            (Results): 一个新的 Results 对象，其中包含被应用函数修改过的属性。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result_cuda = result.cuda()
            ...     result_cpu = result.cpu()
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        """
        返回一个副本，其中所有张量都被移动到 CPU 内存中。

        该方法创建一个新的 Results 对象，其中所有张量属性（边界框、掩模、概率、关键点、面向目标的边界框）都被移动到 CPU 内存中。
        它对于将数据从 GPU 转移到 CPU 进行进一步处理或保存非常有用。

        返回:
            (Results): 一个新的 Results 对象，其中所有张量属性都在 CPU 内存中。

        示例:
            >>> results = model("path/to/image.jpg")  # 执行推理
            >>> cpu_result = results[0].cpu()  # 将第一个结果移到 CPU
            >>> print(cpu_result.boxes.device)  # 输出: cpu
        """
        return self._apply("cpu")

    def numpy(self):
        """
        将 Results 对象中的所有张量转换为 numpy 数组。

        返回:
            (Results): 一个新的 Results 对象，其中所有张量都被转换为 numpy 数组。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> numpy_result = results[0].numpy()
            >>> type(numpy_result.boxes.data)
            <class 'numpy.ndarray'>

        注意:
            该方法创建一个新的 Results 对象，原始对象保持不变。它对于与基于 numpy 的库进行交互，或者当需要 CPU 上的操作时非常有用。
        """
        return self._apply("numpy")

    def cuda(self):
        """
        将 Results 对象中的所有张量移动到 GPU 内存中。

        返回:
            (Results): 一个新的 Results 对象，其中所有张量都被移动到 CUDA 设备上。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> cuda_results = results[0].cuda()  # 将第一个结果移到 GPU
            >>> for result in results:
            ...     result_cuda = result.cuda()  # 将每个结果移到 GPU
        """
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        """
        将 Results 对象中的所有张量移动到指定的设备和数据类型中。

        参数:
            *args (Any): 要传递给 torch.Tensor.to() 的可变长度参数。
            **kwargs (Any): 要传递给 torch.Tensor.to() 的任意关键字参数。

        返回:
            (Results): 一个新的 Results 对象，其中所有张量都被移动到指定的设备和数据类型中。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> result_cuda = results[0].to("cuda")  # 将第一个结果移到 GPU
            >>> result_cpu = results[0].to("cpu")  # 将第一个结果移到 CPU
            >>> result_half = results[0].to(dtype=torch.float16)  # 将第一个结果转换为半精度
        """
        return self._apply("to", *args, **kwargs)

    def new(self):
        """
        创建一个新的 Results 对象，具有相同的图像、路径、名称和速度属性。

        返回:
            (Results): 一个新的 Results 对象，复制了原始实例的属性。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> new_result = results[0].new()
        """
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
        color_mode="class",
    ):
        """
        在输入的 RGB 图像上绘制检测结果。

        参数:
            conf (bool): 是否绘制检测的置信度分数。
            line_width (float | None): 边界框的线宽。如果为 None，则根据图像大小进行缩放。
            font_size (float | None): 文本的字体大小。如果为 None，则根据图像大小进行缩放。
            font (str): 用于文本的字体。
            pil (bool): 是否返回 PIL 图像。
            img (np.ndarray | None): 要绘制的图像。如果为 None，则使用原始图像。
            im_gpu (torch.Tensor | None): GPU 上的归一化图像，用于更快速地绘制掩模。
            kpt_radius (int): 绘制的关键点半径。
            kpt_line (bool): 是否绘制连接关键点的线。
            labels (bool): 是否绘制边界框的标签。
            boxes (bool): 是否绘制边界框。
            masks (bool): 是否绘制掩模。
            probs (bool): 是否绘制分类概率。
            show (bool): 是否显示带注释的图像。
            save (bool): 是否保存带注释的图像。
            filename (str | None): 保存图像的文件名，如果 save 为 True。
            color_mode (bool): 指定颜色模式，例如 'instance' 或 'class'。默认为 'class'。

        返回:
            (np.ndarray): 带注释的图像作为 numpy 数组。

        示例:
            >>> results = model("image.jpg")
            >>> for result in results:
            ...     im = result.plot()
            ...     im.show()
        """
        assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # 分类任务默认使用 pil=True
            example=names,
        )

        # 绘制分割结果
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
            idx = (
                pred_boxes.id
                if pred_boxes.id is not None and color_mode == "instance"
                else pred_boxes.cls
                if pred_boxes and color_mode == "class"
                else reversed(range(len(pred_masks)))
            )
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        # 绘制检测结果
        if pred_boxes is not None and show_boxes:
            for i, d in enumerate(reversed(pred_boxes)):
                c, d_conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name} {d_conf:.2f}" if conf else name) if labels else None
                box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(
                    box,
                    label,
                    color=colors(
                        c
                        if color_mode == "class"
                        else id
                        if id is not None
                        else i
                        if color_mode == "instance"
                        else None,
                        True,
                    ),
                    rotated=is_obb,
                )

        # 绘制分类结果
        if pred_probs is not None and show_probs:
            text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: 允许设置颜色

        # 绘制姿态结果
        if self.keypoints is not None:
            for i, k in enumerate(reversed(self.keypoints.data)):
                annotator.kpts(
                    k,
                    self.orig_shape,
                    radius=kpt_radius,
                    kpt_line=kpt_line,
                    kpt_color=colors(i, True) if color_mode == "instance" else None,
                )

        # 显示结果
        if show:
            annotator.show(self.path)

        # 保存结果
        if save:
            annotator.save(filename)

        return annotator.result()

    def show(self, *args, **kwargs):
        """
        显示带注释的推理结果图像。

        该方法在原始图像上绘制检测结果并显示它。它是直接可视化模型预测的方便方法。

        参数:
            *args (Any): 可变长度的参数列表，将传递给 `plot()` 方法。
            **kwargs (Any): 可变关键字参数，将传递给 `plot()` 方法。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> results[0].show()  # 显示第一个结果
            >>> for result in results:
            ...     result.show()  # 显示所有结果
        """
        self.plot(show=True, *args, **kwargs)

    def save(self, filename=None, *args, **kwargs):
        """
        将带注释的推理结果图像保存到文件。

        该方法在原始图像上绘制检测结果并将带注释的图像保存到文件。它利用 `plot` 方法生成带注释的图像，然后保存到指定的文件名。

        参数:
            filename (str | Path | None): 保存带注释图像的文件名。如果为 None，则根据原始图像路径生成默认文件名。
            *args (Any): 可变长度的参数列表，将传递给 `plot` 方法。
            **kwargs (Any): 可变关键字参数，将传递给 `plot` 方法。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save("annotated_image.jpg")
            >>> # 或者使用自定义绘图参数
            >>> for result in results:
            ...     result.save("annotated_image.jpg", conf=False, line_width=2)
        """
        if not filename:
            filename = f"results_{Path(self.path).name}"
        self.plot(save=True, filename=filename, *args, **kwargs)
        return filename

    def verbose(self):
        """
        返回每个任务的日志字符串，详细描述检测和分类结果。

        该方法生成一个可读性强的字符串，总结检测和分类结果。它包括每个类的检测数量和分类任务的前五个概率。

        返回:
            (str): 一个格式化的字符串，包含结果摘要。对于检测任务，它包括每个类的检测数量。对于分类任务，它包含前五个类的概率。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     print(result.verbose())
            2 persons, 1 car, 3 traffic lights,
            dog 0.92, cat 0.78, horse 0.64,

        注意:
            - 如果没有检测到任何结果，则该方法返回 "(no detections), "。
            - 对于分类任务，返回前五个类的概率和对应的类名。
            - 返回的字符串按逗号分隔，并以逗号和空格结尾。
        """
        log_string = ""
        probs = self.probs
        if len(self) == 0:
            return log_string if probs is not None else f"{log_string}(no detections), "
        if probs is not None:
            log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
        if boxes := self.boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()  # 每个类的检测数量
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return log_string

    def save_txt(self, txt_file, save_conf=False):
        """
        将检测结果保存到文本文件中。

        参数:
            txt_file (str | Path): 输出文本文件的路径。
            save_conf (bool): 是否将置信度分数包含在输出中。

        返回:
            (str): 保存的文本文件的路径。

        示例:
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo11n.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_txt("output.txt")

        注意:
            - 文件中将包含每个检测或分类的一行，结构如下：
              - 对于检测：`class confidence x_center y_center width height`
              - 对于分类：`confidence class_name`
              - 对于掩模和关键点，格式会有所不同。
            - 该函数会在文件不存在时创建输出目录。
            - 如果 `save_conf` 为 False，则置信度分数将不包含在输出中。
            - 文件的现有内容不会被覆盖；新结果会附加到文件末尾。
        """
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # 分类
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            # 检测/分割/姿态
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # 反向掩模.xyn，(n,2) 转为 (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # 创建目录
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)

    def save_crop(self, save_dir, file_name=Path("im.jpg")):
        """
        将裁剪后的检测图像保存到指定目录。

        该方法将检测到的物体裁剪图像保存到指定目录。每个裁剪图像保存到以物体类命名的子目录中，文件名基于输入的 `file_name`。

        参数:
            save_dir (str | Path): 要保存裁剪图像的目录路径。
            file_name (str | Path): 保存裁剪图像的基文件名。默认值为 Path("im.jpg")。

        注意:
            - 该方法不支持分类任务或面向目标的边界框（OBB）任务。
            - 裁剪图像将保存为 'save_dir/class_name/file_name.jpg'。
            - 如果子目录不存在，将会自动创建。
            - 原始图像在裁剪前被复制，以避免修改原始图像。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_crop(save_dir="path/to/crops", file_name="detection")
        """
        if self.probs is not None:
            LOGGER.warning("WARNING ⚠️ 分类任务不支持 `save_crop`。")
            return
        if self.obb is not None:
            LOGGER.warning("WARNING ⚠️ OBB 任务不支持 `save_crop`。")
            return
        for d in self.boxes:
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir) / self.names[int(d.cls)] / Path(file_name).with_suffix(".jpg"),
                BGR=True,
            )

    def summary(self, normalize=False, decimals=5):
        """
        将推理结果转换为汇总字典，并可选地对边界框坐标进行归一化。

        该方法创建一个包含每个检测或分类结果的字典列表。对于分类任务，它返回 top 类和对应的置信度。对于检测任务，它包含类信息、边界框坐标，并可选地包括掩模段和关键点。

        参数:
            normalize (bool): 是否根据图像尺寸归一化边界框坐标。默认为 False。
            decimals (int): 输出值的小数位数。默认为 5。

        返回:
            (List[Dict]): 一个包含每个检测或分类结果的字典列表。每个字典的结构根据任务类型（分类或检测）和可用的信息（边界框、掩模、关键点）有所不同。

        示例:
            >>> results = model("image.jpg")
            >>> summary = results[0].summary()
            >>> print(summary)
        """
        # 创建检测结果的字典列表
        results = []
        if self.probs is not None:
            class_id = self.probs.top1
            results.append(
                {
                    "name": self.names[class_id],
                    "class": class_id,
                    "confidence": round(self.probs.top1conf.item(), decimals),
                }
            )
            return results

        is_obb = self.obb is not None
        data = self.obb if is_obb else self.boxes
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):  # xyxy, track_id 如果有跟踪，conf, class_id
            class_id, conf = int(row.cls), round(row.conf.item(), decimals)
            box = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()
            xy = {}
            for j, b in enumerate(box):
                xy[f"x{j + 1}"] = round(b[0] / w, decimals)
                xy[f"y{j + 1}"] = round(b[1] / h, decimals)
            result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": xy}
            if data.is_track:
                result["track_id"] = int(row.id.item())  # 跟踪 ID
            if self.masks:
                result["segments"] = {
                    "x": (self.masks.xy[i][:, 0] / w).round(decimals).tolist(),
                    "y": (self.masks.xy[i][:, 1] / h).round(decimals).tolist(),
                }
            if self.keypoints is not None:
                x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                result["keypoints"] = {
                    "x": (x / w).numpy().round(decimals).tolist(),  # decimals 参数需要命名
                    "y": (y / h).numpy().round(decimals).tolist(),
                    "visible": visible.numpy().round(decimals).tolist(),
                }
            results.append(result)

        return results

    def to_df(self, normalize=False, decimals=5):
        """
        将检测结果转换为 Pandas Dataframe 格式。

        该方法将检测结果转换为 Pandas Dataframe 格式。它包含有关检测到的物体的信息，例如边界框、类名、置信度分数，并可选地包括分割掩模和关键点。

        参数:
            normalize (bool): 是否将边界框坐标归一化到图像尺寸。如果为 True，坐标将返回为 0 到 1 之间的浮动值。默认为 False。
            decimals (int): 输出值的小数位数。默认为 5。

        返回:
            (DataFrame): 一个 Pandas Dataframe，其中包含所有结果的信息，并按组织的方式排列。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> df_result = results[0].to_df()
            >>> print(df_result)
        """
        import pandas as pd  # 加载以优化性能

        return pd.DataFrame(self.summary(normalize=normalize, decimals=decimals))

    def to_csv(self, normalize=False, decimals=5, *args, **kwargs):
        """
        将检测结果转换为 CSV 格式。

        该方法将检测结果序列化为 CSV 格式。它包含有关检测到的物体的信息，例如边界框、类名、置信度分数，并可选地包括分割掩模和关键点。

        参数:
            normalize (bool): 是否将边界框坐标归一化到图像尺寸。如果为 True，坐标将返回为 0 到 1 之间的浮动值。默认为 False。
            decimals (int): 输出值的小数位数。默认为 5。
            *args (Any): 可变长度的参数列表，将传递给 pandas.DataFrame.to_csv()。
            **kwargs (Any): 可变关键字参数，将传递给 pandas.DataFrame.to_csv()。

        返回:
            (str): 包含所有结果的 CSV 文件，以组织良好的方式存储信息。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> csv_result = results[0].to_csv()
            >>> print(csv_result)
        """
        return self.to_df(normalize=normalize, decimals=decimals).to_csv(*args, **kwargs)

    def to_xml(self, normalize=False, decimals=5, *args, **kwargs):
        """
        将检测结果转换为 XML 格式。

        该方法将检测结果序列化为 XML 格式。它包含有关检测到的物体的信息，例如边界框、类名、置信度分数，并可选地包括分割掩模和关键点。

        参数:
            normalize (bool): 是否将边界框坐标归一化到图像尺寸。如果为 True，坐标将返回为 0 到 1 之间的浮动值。默认为 False。
            decimals (int): 输出值的小数位数。默认为 5。
            *args (Any): 可变长度的参数列表，将传递给 pandas.DataFrame.to_xml()。
            **kwargs (Any): 可变关键字参数，将传递给 pandas.DataFrame.to_xml()。

        返回:
            (str): 包含所有结果的 XML 字符串，以组织良好的方式存储信息。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> xml_result = results[0].to_xml()
            >>> print(xml_result)
        """
        check_requirements("lxml")
        df = self.to_df(normalize=normalize, decimals=decimals)
        return '<?xml version="1.0" encoding="utf-8"?>\n<root></root>' if df.empty else df.to_xml(*args, **kwargs)

    def tojson(self, normalize=False, decimals=5):
        """已弃用的 to_json() 版本。"""
        LOGGER.warning("WARNING ⚠️ 'result.tojson()' 已弃用，请使用 'result.to_json()'。")
        return self.to_json(normalize, decimals)

    def to_json(self, normalize=False, decimals=5):
        """
        将检测结果转换为 JSON 格式。

        该方法将检测结果序列化为 JSON 格式。它包含有关检测到的物体的信息，例如边界框、类名、置信度分数，并可选地包括分割掩模和关键点。

        参数:
            normalize (bool): 是否将边界框坐标归一化到图像尺寸。如果为 True，坐标将返回为 0 到 1 之间的浮动值。默认为 False。
            decimals (int): 输出值的小数位数。默认为 5。

        返回:
            (str): 一个 JSON 字符串，包含序列化后的检测结果。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> json_result = results[0].to_json()
            >>> print(json_result)

        注意:
            - 对于分类任务，JSON 将包含类概率，而不是边界框。
            - 对于目标检测任务，JSON 将包括边界框坐标、类名和置信度分数。
            - 如果可用，JSON 输出还将包含分割掩模和关键点。
            - 该方法内部使用 `summary` 方法生成数据结构，然后将其转换为 JSON。
        """
        import json

        return json.dumps(self.summary(normalize=normalize, decimals=decimals), indent=2)


class Boxes(BaseTensor):
    """
    管理和操作检测框的类。

    该类提供了处理检测框的功能，包括其坐标、置信度分数、类别标签和可选的跟踪ID。它支持多种框格式，并提供方法方便地在不同坐标系统之间进行操作和转换。

    属性:
        data (torch.Tensor | numpy.ndarray): 包含检测框及相关数据的原始张量。
        orig_shape (Tuple[int, int]): 图像的原始尺寸 (高度, 宽度)。
        is_track (bool): 指示框数据中是否包含跟踪ID。
        xyxy (torch.Tensor | numpy.ndarray): 以 [x1, y1, x2, y2] 格式的框。
        conf (torch.Tensor | numpy.ndarray): 每个框的置信度分数。
        cls (torch.Tensor | numpy.ndarray): 每个框的类别标签。
        id (torch.Tensor | numpy.ndarray): 每个框的跟踪ID（如果有）。
        xywh (torch.Tensor | numpy.ndarray): 以 [x, y, width, height] 格式的框。
        xyxyn (torch.Tensor | numpy.ndarray): 相对于原始尺寸的归一化 [x1, y1, x2, y2] 框。
        xywhn (torch.Tensor | numpy.ndarray): 相对于原始尺寸的归一化 [x, y, width, height] 框。

    方法:
        cpu(): 返回一个将所有张量存储在CPU内存中的对象副本。
        numpy(): 返回一个将所有张量转换为numpy数组的对象副本。
        cuda(): 返回一个将所有张量存储在GPU内存中的对象副本。
        to(*args, **kwargs): 返回一个将张量存储在指定设备和数据类型中的对象副本。

    示例:
        >>> import torch
        >>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        >>> orig_shape = (480, 640)  # 高度, 宽度
        >>> boxes = Boxes(boxes_data, orig_shape)
        >>> print(boxes.xyxy)
        >>> print(boxes.conf)
        >>> print(boxes.cls)
        >>> print(boxes.xywhn)
    """

    def __init__(self, boxes, orig_shape) -> None:
        """
        使用检测框数据和原始图像形状初始化Boxes类。

        该类管理检测框，提供便捷的访问和操作框坐标、置信度分数、类别标识符和可选的跟踪ID。它支持多种框坐标格式，包括绝对坐标和归一化格式。

        参数:
            boxes (torch.Tensor | np.ndarray): 一个形状为 (num_boxes, 6) 或 (num_boxes, 7) 的张量或numpy数组。
                列应包含 [x1, y1, x2, y2, confidence, class, (可选) track_id]。
            orig_shape (Tuple[int, int]): 原始图像的尺寸 (高度, 宽度)，用于归一化。

        属性:
            data (torch.Tensor): 包含检测框及其相关数据的原始张量。
            orig_shape (Tuple[int, int]): 用于归一化的原始图像尺寸。
            is_track (bool): 指示框数据中是否包含跟踪ID。

        示例:
            >>> import torch
            >>> boxes = torch.tensor([[100, 50, 150, 100, 0.9, 0]])
            >>> orig_shape = (480, 640)
            >>> detection_boxes = Boxes(boxes, orig_shape)
            >>> print(detection_boxes.xyxy)
            tensor([[100.,  50., 150., 100.]])
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        """
        返回以 [x1, y1, x2, y2] 格式的边界框。

        返回:
            (torch.Tensor | numpy.ndarray): 形状为 (n, 4) 的张量或numpy数组，其中包含以 [x1, y1, x2, y2] 格式表示的边界框坐标，n 为框的数量。

        示例:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> xyxy = boxes.xyxy
            >>> print(xyxy)
        """
        return self.data[:, :4]

    @property
    def conf(self):
        """
        返回每个检测框的置信度分数。

        返回:
            (torch.Tensor | numpy.ndarray): 一个一维张量或数组，包含每个检测框的置信度分数，形状为 (N,)，其中 N 为检测框的数量。

        示例:
            >>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 0]]), orig_shape=(100, 100))
            >>> conf_scores = boxes.conf
            >>> print(conf_scores)
            tensor([0.9000])
        """
        return self.data[:, -2]

    @property
    def cls(self):
        """
        返回表示每个边界框类别预测的类别ID张量。

        返回:
            (torch.Tensor | numpy.ndarray): 一个包含每个检测框的类别ID的张量或numpy数组，形状为 (N,)，其中 N 为框的数量。

        示例:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> class_ids = boxes.cls
            >>> print(class_ids)  # tensor([0., 2., 1.])
        """
        return self.data[:, -1]

    @property
    def id(self):
        """
        如果可用，返回每个检测框的跟踪ID。

        返回:
            (torch.Tensor | None): 如果启用了跟踪，则返回包含每个框的跟踪ID的张量；否则返回None。形状为 (N,)，其中 N 为框的数量。

        示例:
            >>> results = model.track("path/to/video.mp4")
            >>> for result in results:
            ...     boxes = result.boxes
            ...     if boxes.is_track:
            ...         track_ids = boxes.id
            ...         print(f"Tracking IDs: {track_ids}")
            ...     else:
            ...         print("这些框没有启用跟踪。")

        备注:
            - 只有在启用跟踪时（即`is_track`为True）该属性才可用。
            - 跟踪ID通常用于视频分析中的多帧目标关联。
        """
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        """
        将边界框从 [x1, y1, x2, y2] 格式转换为 [x, y, width, height] 格式。

        返回:
            (torch.Tensor | numpy.ndarray): 以 [x_center, y_center, width, height] 格式表示的框，其中 x_center 和 y_center 是边界框的中心点坐标，width 和 height 是边界框的宽度和高度，返回的张量形状为 (N, 4)，其中 N 为框的数量。

        示例:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100], [200, 150, 300, 250]]), orig_shape=(480, 640))
            >>> xywh = boxes.xywh
            >>> print(xywh)
            tensor([[100.0000,  50.0000,  50.0000,  50.0000],
                    [200.0000, 150.0000, 100.0000, 100.0000]])
        """
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        """
        返回相对于原始图像大小的归一化边界框坐标。

        该属性计算并返回以 [x1, y1, x2, y2] 格式表示的边界框坐标，坐标值根据原始图像尺寸归一化到 [0, 1] 范围。

        返回:
            (torch.Tensor | numpy.ndarray): 归一化的边界框坐标，形状为 (N, 4)，其中 N 为框的数量。每一行包含归一化的 [x1, y1, x2, y2] 值。

        示例:
            >>> boxes = Boxes(torch.tensor([[100, 50, 300, 400, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xyxyn
            >>> print(normalized)
            tensor([[0.1562, 0.1042, 0.4688, 0.8333]])
        """
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        """
        返回归一化的 [x, y, width, height] 格式的边界框。

        该属性计算并返回以 [x_center, y_center, width, height] 格式表示的归一化边界框坐标，所有值相对于原始图像尺寸。

        返回:
            (torch.Tensor | numpy.ndarray): 归一化的边界框，形状为 (N, 4)，其中 N 为框的数量。每一行包含归一化的 [x_center, y_center, width, height] 值。

        示例:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xywhn
            >>> print(normalized)
            tensor([[0.1953, 0.1562, 0.0781, 0.1042]])
        """
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


class Masks(BaseTensor):
    """
    用于存储和操作检测掩码的类。

    该类扩展了 BaseTensor 并提供了处理分割掩码的功能，包括在像素坐标和归一化坐标之间转换的方法。

    属性:
        data (torch.Tensor | numpy.ndarray): 包含掩码数据的原始张量或数组。
        orig_shape (tuple): 原始图像的形状，格式为 (高度, 宽度)。
        xy (List[numpy.ndarray]): 像素坐标下的分段列表。
        xyn (List[numpy.ndarray]): 归一化坐标下的分段列表。

    方法:
        cpu(): 返回一个掩码张量存储在 CPU 内存中的 Masks 对象副本。
        numpy(): 返回一个掩码张量作为 numpy 数组存储的 Masks 对象副本。
        cuda(): 返回一个掩码张量存储在 GPU 内存中的 Masks 对象副本。
        to(*args, **kwargs): 返回一个掩码张量存储在指定设备和数据类型中的 Masks 对象副本。

    示例:
        >>> masks_data = torch.rand(1, 160, 160)
        >>> orig_shape = (720, 1280)
        >>> masks = Masks(masks_data, orig_shape)
        >>> pixel_coords = masks.xy
        >>> normalized_coords = masks.xyn
    """

    def __init__(self, masks, orig_shape) -> None:
        """
        使用检测掩码数据和原始图像形状初始化 Masks 类。

        参数:
            masks (torch.Tensor | np.ndarray): 检测掩码，形状为 (num_masks, height, width)。
            orig_shape (tuple): 原始图像形状，格式为 (高度, 宽度)，用于归一化。

        示例:
            >>> import torch
            >>> from ultralytics.engine.results import Masks
            >>> masks = torch.rand(10, 160, 160)  # 10 个 160x160 分辨率的掩码
            >>> orig_shape = (720, 1280)  # 原始图像形状
            >>> mask_obj = Masks(masks, orig_shape)
        """
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """
        返回分割掩码的归一化 xy 坐标。

        该属性计算并缓存分割掩码的归一化 xy 坐标。坐标是相对于原始图像形状进行归一化的。

        返回:
            (List[numpy.ndarray]): 一个 numpy 数组的列表，每个数组包含一个分割掩码的归一化 xy 坐标。
                每个数组的形状为 (N, 2)，其中 N 是掩码轮廓中的点数。

        示例:
            >>> results = model("image.jpg")
            >>> masks = results[0].masks
            >>> normalized_coords = masks.xyn
            >>> print(normalized_coords[0])  # 第一个掩码的归一化坐标
        """
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)
        ]

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """
        返回掩码张量中每个分段的 [x, y] 像素坐标。

        该属性计算并返回 Masks 对象中每个分割掩码的像素坐标列表。坐标被缩放以匹配原始图像的尺寸。

        返回:
            (List[numpy.ndarray]): 一个 numpy 数组的列表，每个数组包含一个分割掩码的 [x, y] 像素坐标。
                每个数组的形状为 (N, 2)，其中 N 是分段中的点数。

        示例:
            >>> results = model("image.jpg")
            >>> masks = results[0].masks
            >>> xy_coords = masks.xy
            >>> print(len(xy_coords))  # 掩码的数量
            >>> print(xy_coords[0].shape)  # 第一个掩码坐标的形状
        """
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks2segments(self.data)
        ]


class Keypoints(BaseTensor):
    """
    存储和操作检测关键点的类。

    该类封装了处理关键点数据的功能，包括坐标操作、归一化和置信度值。

    属性:
        data (torch.Tensor): 包含关键点数据的原始张量。
        orig_shape (Tuple[int, int]): 图像的原始尺寸 (高度, 宽度)。
        has_visible (bool): 指示关键点是否包含可见性信息。
        xy (torch.Tensor): 关键点坐标，以 [x, y] 格式表示。
        xyn (torch.Tensor): 相对于原始图像尺寸的归一化关键点坐标，以 [x, y] 格式表示。
        conf (torch.Tensor): 每个关键点的置信度值（如果可用）。

    方法:
        cpu(): 返回一个将关键点张量存储在CPU内存中的副本。
        numpy(): 返回一个将关键点张量转换为numpy数组的副本。
        cuda(): 返回一个将关键点张量存储在GPU内存中的副本。
        to(*args, **kwargs): 返回一个将关键点张量存储在指定设备和数据类型中的副本。

    示例:
        >>> import torch
        >>> from ultralytics.engine.results import Keypoints
        >>> keypoints_data = torch.rand(1, 17, 3)  # 1 个检测，17 个关键点，(x, y, conf)
        >>> orig_shape = (480, 640)  # 原始图像尺寸 (高度, 宽度)
        >>> keypoints = Keypoints(keypoints_data, orig_shape)
        >>> print(keypoints.xy.shape)  # 访问 xy 坐标
        >>> print(keypoints.conf)  # 访问置信度值
        >>> keypoints_cpu = keypoints.cpu()  # 将关键点移动到 CPU
    """

    @smart_inference_mode()  # 避免 keypoints < conf 发生原地错误
    def __init__(self, keypoints, orig_shape) -> None:
        """
        使用检测关键点和原始图像尺寸初始化 Keypoints 对象。

        该方法处理输入的关键点张量，支持 2D 和 3D 格式。对于 3D 张量 (x, y, confidence)，
        它会通过将其坐标设为零来屏蔽掉低置信度的关键点。

        参数:
            keypoints (torch.Tensor): 包含关键点数据的张量。形状可以是：
                - (num_objects, num_keypoints, 2)，表示只有 x, y 坐标
                - (num_objects, num_keypoints, 3)，表示 x, y 坐标和置信度分数
            orig_shape (Tuple[int, int]): 原始图像尺寸 (高度, 宽度)。

        示例:
            >>> kpts = torch.rand(1, 17, 3)  # 1 个对象，17 个关键点（COCO 格式），x,y,conf
            >>> orig_shape = (720, 1280)  # 原始图像高度，宽度
            >>> keypoints = Keypoints(kpts, orig_shape)
        """
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        if keypoints.shape[2] == 3:  # x, y, conf
            mask = keypoints[..., 2] < 0.5  # 置信度小于 0.5 的点（不可见）
            keypoints[..., :2][mask] = 0
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """
        返回关键点的 x, y 坐标。

        返回:
            (torch.Tensor): 一个张量，包含关键点的 x, y 坐标，形状为 (N, K, 2)，
                其中 N 是检测的数量，K 是每个检测的关键点数量。

        示例:
            >>> results = model("image.jpg")
            >>> keypoints = results[0].keypoints
            >>> xy = keypoints.xy
            >>> print(xy.shape)  # (N, K, 2)
            >>> print(xy[0])  # 第一个检测的关键点 x, y 坐标

        备注:
            - 返回的坐标是相对于原始图像尺寸的像素单位。
            - 如果关键点初始化时包含置信度值，只有置信度 >= 0.5 的关键点才会被返回。
            - 该属性使用 LRU 缓存来提高重复访问时的性能。
        """
        return self.data[..., :2]

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """
        返回相对于原始图像尺寸的归一化关键点坐标 (x, y)。

        返回:
            (torch.Tensor | numpy.ndarray): 一个张量或数组，形状为 (N, K, 2)，包含归一化的关键点坐标，
                其中 N 是实例的数量，K 是关键点的数量，最后一维包含归一化后的 [x, y] 值，范围为 [0, 1]。

        示例:
            >>> keypoints = Keypoints(torch.rand(1, 17, 2), orig_shape=(480, 640))
            >>> normalized_kpts = keypoints.xyn
            >>> print(normalized_kpts.shape)
            torch.Size([1, 17, 2])
        """
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] /= self.orig_shape[1]
        xy[..., 1] /= self.orig_shape[0]
        return xy

    @property
    @lru_cache(maxsize=1)
    def conf(self):
        """
        返回每个关键点的置信度值。

        返回:
            (torch.Tensor | None): 如果可用，返回包含每个关键点置信度分数的张量，
                否则返回 None。形状为 (num_detections, num_keypoints) 对于批处理数据，
                或 (num_keypoints,) 对于单个检测。

        示例:
            >>> keypoints = Keypoints(torch.rand(1, 17, 3), orig_shape=(640, 640))  # 1 个检测，17 个关键点
            >>> conf = keypoints.conf
            >>> print(conf.shape)  # torch.Size([1, 17])
        """
        return self.data[..., 2] if self.has_visible else None


class Probs(BaseTensor):
    """
    一个用于存储和操作分类概率的类。

    该类扩展了 BaseTensor，并提供了访问和操作分类概率的方法，包括 top-1 和 top-5 的预测。

    属性:
        data (torch.Tensor | numpy.ndarray): 包含分类概率的原始张量或数组。
        orig_shape (tuple | None): 原始图像的形状，格式为 (高度, 宽度)。在此类中未使用，但为了与其他结果类的一致性而保留。
        top1 (int): 具有最高概率的类别的索引。
        top5 (List[int]): 按概率排序的前 5 个类别的索引。
        top1conf (torch.Tensor | numpy.ndarray): top 1 类别的置信度分数。
        top5conf (torch.Tensor | numpy.ndarray): top 5 类别的置信度分数。

    方法:
        cpu(): 返回一个将所有张量移至 CPU 内存的概率张量副本。
        numpy(): 返回一个将所有张量转换为 numpy 数组的概率张量副本。
        cuda(): 返回一个将所有张量移至 GPU 内存的概率张量副本。
        to(*args, **kwargs): 返回一个将张量移至指定设备和数据类型的概率张量副本。

    示例:
        >>> probs = torch.tensor([0.1, 0.3, 0.6])
        >>> p = Probs(probs)
        >>> print(p.top1)
        2
        >>> print(p.top5)
        [2, 1, 0]
        >>> print(p.top1conf)
        tensor(0.6000)
        >>> print(p.top5conf)
        tensor([0.6000, 0.3000, 0.1000])
    """

    def __init__(self, probs, orig_shape=None) -> None:
        """
        使用分类概率初始化 Probs 类。

        该类存储和管理分类概率，提供便捷的访问方式来查看 top 预测和它们的置信度。

        参数:
            probs (torch.Tensor | np.ndarray): 一个 1D 的张量或数组，表示分类概率。
            orig_shape (tuple | None): 原始图像的形状，格式为 (高度, 宽度)。在此类中未使用，但为了与其他结果类的一致性而保留。

        属性:
            data (torch.Tensor | np.ndarray): 包含分类概率的原始张量或数组。
            top1 (int): top 1 类别的索引。
            top5 (List[int]): top 5 类别的索引。
            top1conf (torch.Tensor | np.ndarray): top 1 类别的置信度。
            top5conf (torch.Tensor | np.ndarray): top 5 类别的置信度。

        示例:
            >>> import torch
            >>> probs = torch.tensor([0.1, 0.3, 0.2, 0.4])
            >>> p = Probs(probs)
            >>> print(p.top1)
            3
            >>> print(p.top1conf)
            tensor(0.4000)
            >>> print(p.top5)
            [3, 1, 2, 0]
        """
        super().__init__(probs, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def top1(self):
        """
        返回具有最高概率的类别的索引。

        返回:
            (int): 具有最高概率的类别的索引。

        示例:
            >>> probs = Probs(torch.tensor([0.1, 0.3, 0.6]))
            >>> probs.top1
            2
        """
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize=1)
    def top5(self):
        """
        返回 top 5 类别的概率的索引。

        返回:
            (List[int]): 一个包含按概率降序排列的 top 5 类别的索引列表。

        示例:
            >>> probs = Probs(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
            >>> print(probs.top5)
            [4, 3, 2, 1, 0]
        """
        return (-self.data).argsort(0)[:5].tolist()  # 这种方式适用于 torch 和 numpy。

    @property
    @lru_cache(maxsize=1)
    def top1conf(self):
        """
        返回最高概率类别的置信度分数。

        该属性获取分类结果中具有最高预测概率的类别的置信度分数（概率）。

        返回:
            (torch.Tensor | numpy.ndarray): 包含 top 1 类别置信度分数的张量。

        示例:
            >>> results = model("image.jpg")  # 对图像进行分类
            >>> probs = results[0].probs  # 获取分类概率
            >>> top1_confidence = probs.top1conf  # 获取 top 1 类别的置信度
            >>> print(f"Top 1 类别置信度: {top1_confidence.item():.4f}")
        """
        return self.data[self.top1]

    @property
    @lru_cache(maxsize=1)
    def top5conf(self):
        """
        返回 top 5 分类预测的置信度分数。

        该属性获取模型预测的 top 5 类别的置信度分数。它提供了一种快速访问最有可能的类别预测
        及其相关置信度的方法。

        返回:
            (torch.Tensor | numpy.ndarray): 一个包含 top 5 预测类别置信度分数的张量或数组，
                按概率降序排列。

        示例:
            >>> results = model("image.jpg")
            >>> probs = results[0].probs
            >>> top5_conf = probs.top5conf
            >>> print(top5_conf)  # 打印 top 5 类别的置信度
        """
        return self.data[self.top5]


class OBB(BaseTensor):
    """
    用于存储和操作定向边界框（OBB）的类。

    该类提供了处理定向边界框的功能，包括在不同格式之间的转换、归一化以及访问框的各种属性。

    属性:
        data (torch.Tensor): 包含边界框坐标和相关数据的原始 OBB 张量。
        orig_shape (tuple): 原始图像大小，格式为 (高度, 宽度)。
        is_track (bool): 指示框数据中是否包含跟踪 ID。
        xywhr (torch.Tensor | numpy.ndarray): 以 [x_center, y_center, width, height, rotation] 格式表示的框。
        conf (torch.Tensor | numpy.ndarray): 每个框的置信度分数。
        cls (torch.Tensor | numpy.ndarray): 每个框的类别标签。
        id (torch.Tensor | numpy.ndarray): 每个框的跟踪 ID（如果有）。
        xyxyxyxy (torch.Tensor | numpy.ndarray): 以 8 点 [x1, y1, x2, y2, x3, y3, x4, y4] 格式表示的框。
        xyxyxyxyn (torch.Tensor | numpy.ndarray): 相对于原始图像尺寸的归一化 8 点坐标。
        xyxy (torch.Tensor | numpy.ndarray): 以 [x1, y1, x2, y2] 格式表示的轴对齐边界框。

    方法:
        cpu(): 返回一个将所有张量移至 CPU 内存的 OBB 对象副本。
        numpy(): 返回一个将所有张量转换为 numpy 数组的 OBB 对象副本。
        cuda(): 返回一个将所有张量移至 GPU 内存的 OBB 对象副本。
        to(*args, **kwargs): 返回一个将张量移至指定设备和数据类型的 OBB 对象副本。

    示例:
        >>> boxes = torch.tensor([[100, 50, 150, 100, 30, 0.9, 0]])  # xywhr, conf, cls
        >>> obb = OBB(boxes, orig_shape=(480, 640))
        >>> print(obb.xyxyxyxy)
        >>> print(obb.conf)
        >>> print(obb.cls)
    """

    def __init__(self, boxes, orig_shape) -> None:
        """
        使用定向边界框数据和原始图像形状初始化 OBB（定向边界框）实例。

        该类用于存储和操作用于目标检测任务的定向边界框（OBB）。它提供了多种属性和方法来访问和转换 OBB 数据。

        参数:
            boxes (torch.Tensor | numpy.ndarray): 包含检测框的张量或 numpy 数组，形状为 (num_boxes, 7) 或 (num_boxes, 8)。
                最后两列包含置信度和类别值。如果存在，倒数第三列包含跟踪 ID，第五列包含旋转角度。
            orig_shape (Tuple[int, int]): 原始图像大小，格式为 (高度, 宽度)。

        属性:
            data (torch.Tensor | numpy.ndarray): 原始 OBB 张量。
            orig_shape (Tuple[int, int]): 原始图像形状。
            is_track (bool): 是否包含跟踪 ID。

        异常:
            AssertionError: 如果每个框的值数量不是 7 或 8。

        示例:
            >>> import torch
            >>> boxes = torch.rand(3, 7)  # 3 个框，每个框有 7 个值
            >>> orig_shape = (640, 480)
            >>> obb = OBB(boxes, orig_shape)
            >>> print(obb.xywhr)  # 访问 xywhr 格式的框
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {7, 8}, f"期望每个框有 7 或 8 个值，但得到的是 {n}"  # xywh, rotation, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 8
        self.orig_shape = orig_shape

    @property
    def xywhr(self):
        """
        返回 [x_center, y_center, width, height, rotation] 格式的框。

        返回:
            (torch.Tensor | numpy.ndarray): 一个包含定向边界框的张量或 numpy 数组，格式为 [x_center, y_center, width, height, rotation]。
                形状为 (N, 5)，其中 N 是框的数量。

        示例:
            >>> results = model("image.jpg")
            >>> obb = results[0].obb
            >>> xywhr = obb.xywhr
            >>> print(xywhr.shape)
            torch.Size([3, 5])
        """
        return self.data[:, :5]

    @property
    def conf(self):
        """
        返回定向边界框（OBB）的置信度分数。

        该属性获取与每个 OBB 检测相关的置信度值。置信度分数表示模型对检测的可信度。

        返回:
            (torch.Tensor | numpy.ndarray): 形状为 (N,) 的张量或 numpy 数组，包含 N 个检测的置信度分数，
                每个分数的范围为 [0, 1]。

        示例:
            >>> results = model("image.jpg")
            >>> obb_result = results[0].obb
            >>> confidence_scores = obb_result.conf
            >>> print(confidence_scores)
        """
        return self.data[:, -2]

    @property
    def cls(self):
        """
        返回定向边界框的类别值。

        返回:
            (torch.Tensor | numpy.ndarray): 一个包含每个定向边界框类别值的张量或 numpy 数组，形状为 (N,)，
                其中 N 是框的数量。

        示例:
            >>> results = model("image.jpg")
            >>> result = results[0]
            >>> obb = result.obb
            >>> class_values = obb.cls
            >>> print(class_values)
        """
        return self.data[:, -1]

    @property
    def id(self):
        """
        返回定向边界框的跟踪 ID（如果可用）。

        返回:
            (torch.Tensor | numpy.ndarray | None): 一个包含每个定向边界框的跟踪 ID 的张量或 numpy 数组。
                如果没有跟踪 ID，则返回 None。

        示例:
            >>> results = model("image.jpg", tracker=True)  # 使用跟踪运行推理
            >>> for result in results:
            ...     if result.obb is not None:
            ...         track_ids = result.obb.id
            ...         if track_ids is not None:
            ...             print(f"跟踪 ID: {track_ids}")
        """
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxy(self):
        """
        将 OBB 格式转换为 8 点（xyxyxyxy）坐标格式，用于旋转的边界框。

        返回:
            (torch.Tensor | numpy.ndarray): 以 xyxyxyxy 格式表示的旋转边界框，形状为 (N, 4, 2)，其中 N 是框的数量。
                每个框由 4 个点 (x, y) 表示，从左上角开始，按顺时针方向排列。

        示例:
            >>> obb = OBB(torch.tensor([[100, 100, 50, 30, 0.5, 0.9, 0]]), orig_shape=(640, 640))
            >>> xyxyxyxy = obb.xyxyxyxy
            >>> print(xyxyxyxy.shape)
            torch.Size([1, 4, 2])
        """
        return ops.xywhr2xyxyxyxy(self.xywhr)

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxyn(self):
        """
        将旋转的边界框转换为归一化的 xyxyxyxy 格式。

        返回:
            (torch.Tensor | numpy.ndarray): 以 xyxyxyxy 格式表示的归一化旋转边界框，形状为 (N, 4, 2)，
                其中 N 是框的数量。每个框由 4 个点 (x, y) 表示，相对于原始图像尺寸归一化。

        示例:
            >>> obb = OBB(torch.rand(10, 7), orig_shape=(640, 480))  # 10 个随机 OBB
            >>> normalized_boxes = obb.xyxyxyxyn
            >>> print(normalized_boxes.shape)
            torch.Size([10, 4, 2])
        """
        xyxyxyxyn = self.xyxyxyxy.clone() if isinstance(self.xyxyxyxy, torch.Tensor) else np.copy(self.xyxyxyxy)
        xyxyxyxyn[..., 0] /= self.orig_shape[1]
        xyxyxyxyn[..., 1] /= self.orig_shape[0]
        return xyxyxyxyn

    @property
    @lru_cache(maxsize=2)
    def xyxy(self):
        """
        将定向边界框（OBB）转换为轴对齐的边界框，返回 xyxy 格式。

        该属性计算每个定向边界框的最小包围矩形，并将其返回为 xyxy 格式（x1, y1, x2, y2）。
        这是需要与非旋转框计算 IoU 等操作时非常有用的。

        返回:
            (torch.Tensor | numpy.ndarray): 以 xyxy 格式表示的轴对齐边界框，形状为 (N, 4)，其中 N 是框的数量。
                每行包含 [x1, y1, x2, y2] 坐标。

        示例:
            >>> import torch
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolov8n-obb.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     obb = result.obb
            ...     if obb is not None:
            ...         xyxy_boxes = obb.xyxy
            ...         print(xyxy_boxes.shape)  # (N, 4)

        注意:
            - 该方法通过最小包围矩形近似 OBB。
            - 返回的格式兼容标准目标检测指标和可视化工具。
            - 该属性使用缓存来提高重复访问的性能。
        """
        x = self.xyxyxyxy[..., 0]
        y = self.xyxyxyxy[..., 1]
        return (
            torch.stack([x.amin(1), y.amin(1), x.amax(1), y.amax(1)], -1)
            if isinstance(x, torch.Tensor)
            else np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], -1)
        )
