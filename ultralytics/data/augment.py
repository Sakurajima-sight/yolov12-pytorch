# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import deepcopy
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.ops import segment2box, xyxyxyxy2xywhr
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)
DEFAULT_CROP_FRACTION = 1.0


class BaseTransform:
    """
    Ultralytics库中图像变换的基类。

    该类作为实现各种图像处理操作的基础，旨在兼容分类任务和语义分割任务。

    方法：
        apply_image: 对标签应用图像变换。
        apply_instances: 对标签中的物体实例应用变换。
        apply_semantic: 对图像应用语义分割。
        __call__: 对图像、实例和语义掩码应用所有标签变换。

    示例：
        >>> transform = BaseTransform()
        >>> labels = {"image": np.array(...), "instances": [...], "semantic": np.array(...)}
        >>> transformed_labels = transform(labels)
    """

    def __init__(self) -> None:
        """
        初始化BaseTransform对象。

        该构造函数设置基础变换对象，可以扩展用于特定的图像处理任务。设计时兼容分类和语义分割任务。

        示例：
            >>> transform = BaseTransform()
        """
        pass

    def apply_image(self, labels):
        """
        对标签应用图像变换。

        该方法预计由子类重写，以实现特定的图像变换逻辑。在基类中，返回的标签与输入标签相同。

        参数：
            labels (Any): 要进行变换的输入标签。具体的类型和结构可能会根据实现的不同而有所变化。

        返回：
            (Any): 变换后的标签。在基类实现中，这与输入标签相同。

        示例：
            >>> transform = BaseTransform()
            >>> original_labels = [1, 2, 3]
            >>> transformed_labels = transform.apply_image(original_labels)
            >>> print(transformed_labels)
            [1, 2, 3]
        """
        pass

    def apply_instances(self, labels):
        """
        对标签中的物体实例应用变换。

        该方法负责对给定标签中的物体实例应用各种变换。设计时预计由子类重写，以实现特定的实例变换逻辑。

        参数：
            labels (Dict): 包含标签信息的字典，包括物体实例。

        返回：
            (Dict): 经过变换后的标签字典，包含修改后的物体实例。

        示例：
            >>> transform = BaseTransform()
            >>> labels = {"instances": Instances(xyxy=torch.rand(5, 4), cls=torch.randint(0, 80, (5,)))}
            >>> transformed_labels = transform.apply_instances(labels)
        """
        pass

    def apply_semantic(self, labels):
        """
        对图像应用语义分割变换。

        该方法预计由子类重写，以实现特定的语义分割变换。在基类中，不执行任何操作。

        参数：
            labels (Any): 要进行变换的输入标签或语义分割掩码。

        返回：
            (Any): 变换后的语义分割掩码或标签。

        示例：
            >>> transform = BaseTransform()
            >>> semantic_mask = np.zeros((100, 100), dtype=np.uint8)
            >>> transformed_mask = transform.apply_semantic(semantic_mask)
        """
        pass

    def __call__(self, labels):
        """
        对图像、实例和语义掩码应用所有标签变换。

        该方法协调了对输入标签应用BaseTransform类中定义的各种变换，按顺序调用apply_image和apply_instances方法来处理图像和物体实例。

        参数：
            labels (Dict): 包含图像数据和注释的字典。预期的键包括'img'（图像数据）和'instances'（物体实例）。

        返回：
            (Dict): 经过变换后的输入标签字典，包含变换后的图像和实例。

        示例：
            >>> transform = BaseTransform()
            >>> labels = {"img": np.random.rand(640, 640, 3), "instances": []}
            >>> transformed_labels = transform(labels)
        """
        self.apply_image(labels)
        self.apply_instances(labels)
        self.apply_semantic(labels)


class Compose:
    """
    用于组合多个图像变换的类。

    属性：
        transforms (List[Callable]): 要依次应用的变换函数列表。

    方法：
        __call__: 对输入数据应用一系列变换。
        append: 将一个新的变换追加到现有的变换列表中。
        insert: 在指定的索引位置插入一个新的变换。
        __getitem__: 使用索引检索一个特定的变换或一组变换。
        __setitem__: 使用索引设置一个特定的变换或一组变换。
        tolist: 将变换列表转换为标准的Python列表。

    示例：
        >>> transforms = [RandomFlip(), RandomPerspective(30)]
        >>> compose = Compose(transforms)
        >>> transformed_data = compose(data)
        >>> compose.append(CenterCrop((224, 224)))
        >>> compose.insert(0, RandomFlip())
    """

    def __init__(self, transforms):
        """
        使用变换列表初始化Compose对象。

        参数：
            transforms (List[Callable]): 要依次应用的可调用变换对象列表。

        示例：
            >>> from ultralytics.data.augment import Compose, RandomHSV, RandomFlip
            >>> transforms = [RandomHSV(), RandomFlip()]
            >>> compose = Compose(transforms)
        """
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """
        对输入数据应用一系列变换。该方法依次对Compose对象的变换列表中的每个变换应用到输入数据上。

        参数：
            data (Any): 要变换的输入数据。数据类型可以根据变换列表中的变换而不同。

        返回：
            (Any): 应用所有变换后的数据。

        示例：
            >>> transforms = [Transform1(), Transform2(), Transform3()]
            >>> compose = Compose(transforms)
            >>> transformed_data = compose(input_data)
        """
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """
        将一个新的变换追加到现有的变换列表中。

        参数：
            transform (BaseTransform): 要添加到组合中的变换。

        示例：
            >>> compose = Compose([RandomFlip(), RandomPerspective()])
            >>> compose.append(RandomHSV())
        """
        self.transforms.append(transform)

    def insert(self, index, transform):
        """
        在现有的变换列表中，在指定索引位置插入一个新的变换。

        参数：
            index (int): 要插入新变换的索引位置。
            transform (BaseTransform): 要插入的变换对象。

        示例：
            >>> compose = Compose([Transform1(), Transform2()])
            >>> compose.insert(1, Transform3())
            >>> len(compose.transforms)
            3
        """
        self.transforms.insert(index, transform)

    def __getitem__(self, index: Union[list, int]) -> "Compose":
        """
        使用索引检索一个特定的变换或一组变换。

        参数：
            index (int | List[int]): 要检索的变换索引或索引列表。

        返回：
            (Compose): 一个新的Compose对象，包含所选的变换。

        异常：
            AssertionError: 如果索引类型不是int或list。

        示例：
            >>> transforms = [RandomFlip(), RandomPerspective(10), RandomHSV(0.5, 0.5, 0.5)]
            >>> compose = Compose(transforms)
            >>> single_transform = compose[1]  # 返回一个只包含RandomPerspective的Compose对象
            >>> multiple_transforms = compose[0:2]  # 返回一个包含RandomFlip和RandomPerspective的Compose对象
        """
        assert isinstance(index, (int, list)), f"索引应为list或int类型，但获得了{type(index)}"
        index = [index] if isinstance(index, int) else index
        return Compose([self.transforms[i] for i in index])

    def __setitem__(self, index: Union[list, int], value: Union[list, int]) -> None:
        """
        使用索引设置一个或多个变换。

        参数：
            index (int | List[int]): 要设置变换的索引或索引列表。
            value (Any | List[Any]): 要设置的变换或变换列表。

        异常：
            AssertionError: 如果索引类型无效，或者值类型与索引类型不匹配，或者索引超出范围。

        示例：
            >>> compose = Compose([Transform1(), Transform2(), Transform3()])
            >>> compose[1] = NewTransform()  # 替换第二个变换
            >>> compose[0:2] = [NewTransform1(), NewTransform2()]  # 替换前两个变换
        """
        assert isinstance(index, (int, list)), f"索引应为list或int类型，但获得了{type(index)}"
        if isinstance(index, list):
            assert isinstance(value, list), (
                f"索引应与值类型相同，但获得了{type(index)}和{type(value)}"
            )
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f"列表索引{i}超出了范围 {len(self.transforms)}。"
            self.transforms[i] = v

    def tolist(self):
        """
        将变换列表转换为标准的Python列表。

        返回：
            (List): 包含Compose实例中所有变换对象的列表。

        示例：
            >>> transforms = [RandomFlip(), RandomPerspective(10), CenterCrop()]
            >>> compose = Compose(transforms)
            >>> transform_list = compose.tolist()
            >>> print(len(transform_list))
            3
        """
        return self.transforms

    def __repr__(self):
        """
        返回Compose对象的字符串表示。

        返回：
            (str): Compose对象的字符串表示，包括变换列表。

        示例：
            >>> transforms = [RandomFlip(), RandomPerspective(degrees=10, translate=0.1, scale=0.1)]
            >>> compose = Compose(transforms)
            >>> print(compose)
            Compose([
                RandomFlip(),
                RandomPerspective(degrees=10, translate=0.1, scale=0.1)
            ])
        """
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"


class BaseMixTransform:
    """
    混合变换的基类，如MixUp和Mosaic。

    该类为在数据集上实现混合变换提供了基础。它处理基于概率的变换应用，并管理多个图像和标签的混合。

    属性：
        dataset (Any): 包含图像和标签的数据集对象。
        pre_transform (Callable | None): 可选的在混合之前应用的变换。
        p (float): 应用混合变换的概率。

    方法：
        __call__: 对输入标签应用混合变换。
        _mix_transform: 抽象方法，由子类实现具体的混合操作。
        get_indexes: 抽象方法，用于获取需要混合的图像索引。
        _update_label_text: 更新混合图像的标签文本。

    示例：
        >>> class CustomMixTransform(BaseMixTransform):
        ...     def _mix_transform(self, labels):
        ...         # 在这里实现自定义的混合逻辑
        ...         return labels
        ...
        ...     def get_indexes(self):
        ...         return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        >>> dataset = YourDataset()
        >>> transform = CustomMixTransform(dataset, p=0.5)
        >>> mixed_labels = transform(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """
        为混合变换（如MixUp和Mosaic）初始化BaseMixTransform对象。

        该类作为在图像处理管道中实现混合变换的基础。

        参数：
            dataset (Any): 包含图像和标签的数据集对象，用于混合。
            pre_transform (Callable | None): 可选的在混合之前应用的变换。
            p (float): 应用混合变换的概率，应该在[0.0, 1.0]范围内。

        示例：
            >>> dataset = YOLODataset("path/to/data")
            >>> pre_transform = Compose([RandomFlip(), RandomPerspective()])
            >>> mix_transform = BaseMixTransform(dataset, pre_transform, p=0.5)
        """
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels):
        """
        对标签数据应用预处理变换和MixUp/Mosaic变换。

        该方法根据概率因子决定是否应用混合变换。如果应用，它会选择额外的图像，应用预变换（如果指定了），然后执行混合变换。

        参数：
            labels (Dict): 包含图像标签数据的字典。

        返回：
            (Dict): 经过变换的标签字典，可能包含来自其他图像的混合数据。

        示例：
            >>> transform = BaseMixTransform(dataset, pre_transform=None, p=0.5)
            >>> result = transform({"image": img, "bboxes": boxes, "cls": classes})
        """
        if random.uniform(0, 1) > self.p:
            return labels

        # 获取一个或三个其他图像的索引
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # 获取将用于Mosaic或MixUp的图像信息
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        # 更新类和文本
        labels = self._update_label_text(labels)
        # Mosaic或MixUp
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def _mix_transform(self, labels):
        """
        对标签字典应用MixUp或Mosaic增强。

        该方法应该由子类实现，用于执行具体的混合变换，如MixUp或Mosaic。它会就地修改输入的标签字典，添加增强后的数据。

        参数：
            labels (Dict): 包含图像和标签数据的字典，预计会有一个'mix_labels'键，包含用于混合的额外图像和标签数据。

        返回：
            (Dict): 经过混合变换增强后的标签字典。

        示例：
            >>> transform = BaseMixTransform(dataset)
            >>> labels = {"image": img, "bboxes": boxes, "mix_labels": [{"image": img2, "bboxes": boxes2}]}
            >>> augmented_labels = transform._mix_transform(labels)
        """
        raise NotImplementedError

    def get_indexes(self):
        """
        获取用于马赛克增强的打乱后的索引列表。

        返回：
            (List[int]): 从数据集中随机打乱的索引列表。

        示例：
            >>> transform = BaseMixTransform(dataset)
            >>> indexes = transform.get_indexes()
            >>> print(indexes)  # [3, 18, 7, 2]
        """
        raise NotImplementedError

    @staticmethod
    def _update_label_text(labels):
        """
        更新图像增强中的混合标签文本和类ID。

        该方法处理输入标签字典的'texts'和'cls'字段，以及任何混合标签，创建统一的文本标签集并相应更新类ID。

        参数：
            labels (Dict): 包含标签信息的字典，包括'texts'和'cls'字段，可能还会有一个'mix_labels'字段，包含额外的标签字典。

        返回：
            (Dict): 更新后的标签字典，包含统一的文本标签和更新后的类ID。

        示例：
            >>> labels = {
            ...     "texts": [["cat"], ["dog"]],
            ...     "cls": torch.tensor([[0], [1]]),
            ...     "mix_labels": [{"texts": [["bird"], ["fish"]], "cls": torch.tensor([[0], [1]])}],
            ... }
            >>> updated_labels = self._update_label_text(labels)
            >>> print(updated_labels["texts"])
            [['cat'], ['dog'], ['bird'], ['fish']]
            >>> print(updated_labels["cls"])
            tensor([[0],
                    [1]])
            >>> print(updated_labels["mix_labels"][0]["cls"])
            tensor([[2],
                    [3]])
        """
        if "texts" not in labels:
            return labels

        mix_texts = sum([labels["texts"]] + [x["texts"] for x in labels["mix_labels"]], [])
        mix_texts = list({tuple(x) for x in mix_texts})
        text2id = {text: i for i, text in enumerate(mix_texts)}

        for label in [labels] + labels["mix_labels"]:
            for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
                text = label["texts"][int(cls)]
                label["cls"][i] = text2id[tuple(text)]
            label["texts"] = mix_texts
        return labels


class Mosaic(BaseMixTransform):
    """
    图像数据集的马赛克增强（Mosaic）

    该类通过将多张（4或9张）图像合成到一张马赛克图像中来执行马赛克增强。
    该增强会以给定的概率应用到数据集上。

    属性：
        dataset: 应用马赛克增强的数据集。
        imgsz (int): 单张图像经过马赛克处理后的图像大小（高度和宽度）。
        p (float): 应用马赛克增强的概率，必须在0到1之间。
        n (int): 网格大小，4表示2x2网格，9表示3x3网格。
        border (Tuple[int, int]): 边框大小（宽度和高度）。

    方法：
        get_indexes: 返回从数据集中随机选择的索引列表。
        _mix_transform: 对输入图像和标签应用混合变换。
        _mosaic3: 创建1x3的图像马赛克。
        _mosaic4: 创建2x2的图像马赛克。
        _mosaic9: 创建3x3的图像马赛克。
        _update_labels: 更新带有填充的标签。
        _cat_labels: 合并标签并裁剪马赛克边界实例。

    示例：
        >>> from ultralytics.data.augment import Mosaic
        >>> dataset = YourDataset(...)  # 你的图像数据集
        >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        >>> augmented_labels = mosaic_aug(original_labels)
    """

    def __init__(self, dataset, imgsz=640, p=1.0, n=4):
        """
        初始化马赛克增强对象。

        该类通过将多张（4或9张）图像合成到一张马赛克图像中来执行马赛克增强。
        该增强会以给定的概率应用到数据集上。

        参数：
            dataset (Any): 应用马赛克增强的数据集。
            imgsz (int): 单张图像经过马赛克处理后的图像大小（高度和宽度）。
            p (float): 应用马赛克增强的概率，必须在0到1之间。
            n (int): 网格大小，4表示2x2网格，9表示3x3网格。

        示例：
            >>> from ultralytics.data.augment import Mosaic
            >>> dataset = YourDataset(...)
            >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        """
        assert 0 <= p <= 1.0, f"概率应在[0, 1]范围内，但得到{p}."
        assert n in {4, 9}, "网格必须是4或9."
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # 宽度，高度
        self.n = n

    def get_indexes(self, buffer=True):
        """
        返回用于马赛克增强的随机索引列表。

        该方法根据'buffer'参数从缓冲区或整个数据集中随机选择图像索引。
        它用于选择图像以创建马赛克增强。

        参数：
            buffer (bool): 如果为True，从数据集缓冲区中选择图像。如果为False，则从整个数据集中选择。

        返回：
            (List[int]): 随机图像索引的列表，列表长度为n-1，n是用于马赛克的图像数量（如果n为4，则为3，如果n为9，则为8）。

        示例：
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> indexes = mosaic.get_indexes()
            >>> print(len(indexes))  # 输出: 3
        """
        if buffer:  # 从缓冲区选择图像
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # 从整个数据集中选择图像
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self, labels):
        """
        对输入图像和标签应用马赛克增强。

        该方法将多张图像（3张、4张或9张）合成到一张马赛克图像中，基于'n'属性。
        它确保没有矩形注释，并且有其他图像可用于马赛克增强。

        参数：
            labels (Dict): 包含图像数据和注释的字典，预计有以下键：
                - 'rect_shape': 应为None，因为矩形和马赛克是互斥的。
                - 'mix_labels': 一个包含数据的字典列表，用于其他图像进行马赛克。

        返回：
            (Dict): 一个包含马赛克增强图像和更新后的注释的字典。

        异常：
            AssertionError: 如果'rect_shape'不为None，或'mix_labels'为空。

        示例：
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> augmented_data = mosaic._mix_transform(labels)
        """
        assert labels.get("rect_shape", None) is None, "矩形和马赛克是互斥的."
        assert len(labels.get("mix_labels", [])), "没有其他图像可用于马赛克增强."
        return (
            self._mosaic3(labels) if self.n == 3 else self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)
        )  # 此代码已为mosaic3方法进行了修改。

    def _mosaic3(self, labels):
        """
        创建一个1x3的图像马赛克，将三张图像组合在一起。

        该方法将三张图像水平排列，主图像放在中间，另外两张图像分别放在两侧。
        它是用于目标检测的马赛克增强技术的一部分。

        参数：
            labels (Dict): 包含主图像（中心图像）图像和标签信息的字典。
                必须包括'img'键（图像数组），并且'mix_labels'键包含两个字典，
                其中包含侧边图像的相关信息。

        返回：
            (Dict): 一个字典，包含马赛克图像和更新后的标签。键包括：
                - 'img' (np.ndarray): 马赛克图像数组，形状为(H, W, C)。
                - 其他输入标签的键，更新为反映新图像尺寸的值。

        示例：
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=3)
            >>> labels = {
            ...     "img": np.random.rand(480, 640, 3),
            ...     "mix_labels": [{"img": np.random.rand(480, 640, 3)} for _ in range(2)],
            ... }
            >>> result = mosaic._mosaic3(labels)
            >>> print(result["img"].shape)
            (640, 640, 3)
        """
        mosaic_labels = []
        s = self.imgsz
        for i in range(3):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # 加载图像
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # 放置图像到img3
            if i == 0:  # 中心
                img3 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # 以3块图像为基础的图像
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax（基础）坐标
            elif i == 1:  # 右侧
                c = s + w0, s, s + w0 + w, s + h
            elif i == 2:  # 左侧
                c = s - w, s + h0 - h, s, s + h0

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # 分配坐标

            img3[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]  # img3[ymin:ymax, xmin:xmax]
            # hp, wp = h, w  # 为下次迭代准备前一个高度和宽度

            # 更新标签，假设马赛克大小为imgsz*2
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels["img"] = img3[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return final_labels

    def _mosaic4(self, labels):
        """
        创建一个2x2的图像马赛克，由四张图像合成。

        该方法将四张图像合成到一张马赛克图像中，通过将它们排列在2x2的网格中。
        它还更新每个图像在马赛克中的标签。

        参数：
            labels (Dict): 包含基本图像（索引为0）和三张附加图像（索引为1-3）的图像数据和标签，
                这些图像的信息在'mix_labels'键中。

        返回：
            (Dict): 一个字典，包含马赛克图像和更新后的标签。'img'键包含马赛克图像，
                其他键包含所有四张图像的合并和调整后的标签。

        示例：
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> labels = {
            ...     "img": np.random.rand(480, 640, 3),
            ...     "mix_labels": [{"img": np.random.rand(480, 640, 3)} for _ in range(3)],
            ... }
            >>> result = mosaic._mosaic4(labels)
            >>> assert result["img"].shape == (1280, 1280, 3)
        """
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # 马赛克中心x，y
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # 加载图像
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # 放置图像到img4
            if i == 0:  # 左上
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # 以4块图像为基础的图像
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax（大图像）
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax（小图像）
            elif i == 1:  # 右上
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # 左下
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # 右下
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        return final_labels


class MixUp(BaseMixTransform):
    """
    对图像数据集应用MixUp增强。

    该类实现了论文“mixup: Beyond Empirical Risk Minimization”（https://arxiv.org/abs/1710.09412）中描述的MixUp增强技术。
    MixUp通过使用一个随机权重将两张图像及其标签结合在一起。

    属性：
        dataset (Any): 应用MixUp增强的数据集。
        pre_transform (Callable | None): 可选的在MixUp之前应用的变换。
        p (float): 应用MixUp增强的概率。

    方法：
        get_indexes: 返回数据集中的随机索引。
        _mix_transform: 对输入标签应用MixUp增强。

    示例：
        >>> from ultralytics.data.augment import MixUp
        >>> dataset = YourDataset(...)  # 你的图像数据集
        >>> mixup = MixUp(dataset, p=0.5)
        >>> augmented_labels = mixup(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """
        初始化MixUp增强对象。

        MixUp是一种图像增强技术，通过对两张图像的像素值和标签进行加权求和，将两张图像合成一张。
        本实现专为Ultralytics YOLO框架设计。

        参数：
            dataset (Any): 应用MixUp增强的数据集。
            pre_transform (Callable | None): 可选的在MixUp之前应用的变换。
            p (float): 应用MixUp增强的概率，必须在[0, 1]范围内。

        示例：
            >>> from ultralytics.data.dataset import YOLODataset
            >>> dataset = YOLODataset("path/to/data.yaml")
            >>> mixup = MixUp(dataset, pre_transform=None, p=0.5)
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_indexes(self):
        """
        从数据集中获取一个随机索引。

        该方法返回数据集中的单个随机索引，用于选择进行MixUp增强的图像。

        返回：
            (int): 数据集长度范围内的一个随机整数索引。

        示例：
            >>> mixup = MixUp(dataset)
            >>> index = mixup.get_indexes()
            >>> print(index)
            42
        """
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        """
        对输入标签应用MixUp增强。

        该方法实现了论文“mixup: Beyond Empirical Risk Minimization”（https://arxiv.org/abs/1710.09412）中描述的MixUp增强技术。

        参数：
            labels (Dict): 包含原始图像和标签信息的字典。

        返回：
            (Dict): 包含混合图像和合并标签信息的字典。

        示例：
            >>> mixer = MixUp(dataset)
            >>> mixed_labels = mixer._mix_transform(labels)
        """
        r = np.random.beta(32.0, 32.0)  # mixup比例，alpha=beta=32.0
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        return labels


class RandomPerspective:
    """
    实现图像及其对应注释的随机透视和仿射变换。

    该类对图像及其对应的边界框、分割、关键点应用随机旋转、平移、缩放、剪切和透视变换。
    它可以作为目标检测和实例分割任务的数据增强管道的一部分使用。

    属性：
        degrees (float): 随机旋转的最大绝对角度范围。
        translate (float): 最大平移量，占图像大小的比例。
        scale (float): 缩放因子范围，例如，scale=0.1表示0.9到1.1之间。
        shear (float): 最大剪切角度（度）。
        perspective (float): 透视畸变因子。
        border (Tuple[int, int]): 马赛克边框大小（x，y）。
        pre_transform (Callable | None): 可选的在随机透视变换之前应用的变换。

    方法：
        affine_transform: 对输入图像应用仿射变换。
        apply_bboxes: 使用仿射矩阵变换边界框。
        apply_segments: 变换分割并生成新的边界框。
        apply_keypoints: 使用仿射矩阵变换关键点。
        __call__: 对图像和注释应用随机透视变换。
        box_candidates: 根据大小和长宽比过滤变换后的边界框。

    示例：
        >>> transform = RandomPerspective(degrees=10, translate=0.1, scale=0.1, shear=10)
        >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> labels = {"img": image, "cls": np.array([0, 1]), "instances": Instances(...)}
        >>> result = transform(labels)
        >>> transformed_image = result["img"]
        >>> transformed_instances = result["instances"]
    """

    def __init__(
        self, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, border=(0, 0), pre_transform=None
    ):
        """
        初始化RandomPerspective对象，并设置变换参数。

        该类实现图像及其对应的边界框、分割和关键点的随机透视和仿射变换。
        变换包括旋转、平移、缩放和剪切。

        参数：
            degrees (float): 随机旋转的角度范围。
            translate (float): 随机平移的宽度和高度的比例。
            scale (float): 缩放因子的区间，例如，scale=0.5表示缩放因子在50%到150%之间。
            shear (float): 剪切强度（角度）。
            perspective (float): 透视畸变因子。
            border (Tuple[int, int]): 指定马赛克边框（顶部/底部，左/右）的元组。
            pre_transform (Callable | None): 在应用随机变换之前对图像应用的函数或变换。

        示例：
            >>> transform = RandomPerspective(degrees=10.0, translate=0.1, scale=0.5, shear=5.0)
            >>> result = transform(labels)  # 对标签应用随机透视变换
        """
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border  # 马赛克边框
        self.pre_transform = pre_transform

    def affine_transform(self, img, border):
        """
        对图像应用一系列以图像中心为中心的仿射变换。

        该函数对输入图像执行一系列几何变换，包括平移、透视变化、旋转、缩放和剪切。
        变换按特定顺序应用，以保持一致性。

        参数：
            img (np.ndarray): 需要变换的输入图像。
            border (Tuple[int, int]): 变换后图像的边框尺寸。

        返回：
            (Tuple[np.ndarray, np.ndarray, float]): 返回一个元组，包含：
                - np.ndarray: 变换后的图像。
                - np.ndarray: 3x3的变换矩阵。
                - float: 变换过程中应用的缩放因子。

        示例：
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> border = (10, 10)
            >>> transformed_img, matrix, scale = affine_transform(img, border)
        """
        # 中心
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x平移（像素）
        C[1, 2] = -img.shape[0] / 2  # y平移（像素）

        # 透视
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x透视（围绕y轴）
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y透视（围绕x轴）

        # 旋转和缩放
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # 剪切
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x剪切（度）
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y剪切（度）

        # 平移
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x平移（像素）
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y平移（像素）

        # 合并的旋转矩阵
        M = T @ S @ R @ P @ C  # 操作顺序（从右到左）很重要
        # 仿射图像
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # 图像发生了变化
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:  # 仿射
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, M, s

    def apply_bboxes(self, bboxes, M):
        """
        对边界框应用仿射变换。

        该函数使用提供的变换矩阵对一组边界框应用仿射变换。

        参数：
            bboxes (torch.Tensor): 以xyxy格式表示的边界框，形状为(N, 4)，其中N是边界框的数量。
            M (torch.Tensor): 形状为(3, 3)的仿射变换矩阵。

        返回：
            (torch.Tensor): 以xyxy格式表示的变换后的边界框，形状为(N, 4)。

        示例：
            >>> bboxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])
            >>> M = torch.eye(3)
            >>> transformed_bboxes = apply_bboxes(bboxes, M)
        """
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # 变换
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # 透视重缩放或仿射

        # 创建新的框
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def apply_segments(self, segments, M):
        """
        对分割应用仿射变换并生成新的边界框。

        该函数对输入的分割应用仿射变换，并根据变换后的分割生成新的边界框。
        它将变换后的分割裁剪到新的边界框内。

        参数：
            segments (np.ndarray): 输入的分割，形状为(N, M, 2)，其中N是分割的数量，M是每个分割中的点的数量。
            M (np.ndarray): 形状为(3, 3)的仿射变换矩阵。

        返回：
            (Tuple[np.ndarray, np.ndarray]): 返回一个元组，包含：
                - 新的边界框，形状为(N, 4)的xyxy格式。
                - 变换后的分割，形状为(N, M, 2)。

        示例：
            >>> segments = np.random.rand(10, 500, 2)  # 10个分割，每个有500个点
            >>> M = np.eye(3)  # 单位变换矩阵
            >>> new_bboxes, new_segments = apply_segments(segments, M)
        """
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments

        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T  # 变换
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack([segment2box(xy, self.size[0], self.size[1]) for xy in segments], 0)
        segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
        segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
        return bboxes, segments

    def apply_keypoints(self, keypoints, M):
        """
        对关键点应用仿射变换。

        该方法使用提供的仿射变换矩阵变换输入关键点。如果需要，它还会处理透视重缩放，
        并更新那些变换后超出图像边界的关键点的可见性。

        参数：
            keypoints (np.ndarray): 关键点数组，形状为(N, 17, 3)，其中N是实例的数量，17是每个实例的关键点数量，
                3表示(x, y, 可见性)。
            M (np.ndarray): 3x3仿射变换矩阵。

        返回：
            (np.ndarray): 变换后的关键点数组，形状与输入相同(N, 17, 3)。

        示例：
            >>> random_perspective = RandomPerspective()
            >>> keypoints = np.random.rand(5, 17, 3)  # 5个实例，每个实例有17个关键点
            >>> M = np.eye(3)  # 单位变换矩阵
            >>> transformed_keypoints = random_perspective.apply_keypoints(keypoints, M)
        """
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T  # 变换
        xy = xy[:, :2] / xy[:, 2:3]  # 透视重缩放或仿射
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > self.size[0]) | (xy[:, 1] > self.size[1])
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def __call__(self, labels):
        """
        对图像及其对应的标签应用随机透视和仿射变换。

        该方法对输入图像执行一系列变换，包括旋转、平移、缩放、剪切和透视畸变，
        并相应地调整对应的边界框、分割和关键点。

        参数：
            labels (Dict): 包含图像数据和注释的字典。
                必须包括：
                    'img' (ndarray): 输入图像。
                    'cls' (ndarray): 类别标签。
                    'instances' (Instances): 包含边界框、分割和关键点的实例。
                可选地包括：
                    'mosaic_border' (Tuple[int, int]): 马赛克增强的边框大小。

        返回：
            (Dict): 变换后的标签字典，包含：
                - 'img' (np.ndarray): 变换后的图像。
                - 'cls' (np.ndarray): 更新后的类别标签。
                - 'instances' (Instances): 更新后的目标实例。
                - 'resized_shape' (Tuple[int, int]): 变换后的新图像大小。

        示例：
            >>> transform = RandomPerspective()
            >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> labels = {
            ...     "img": image,
            ...     "cls": np.array([0, 1, 2]),
            ...     "instances": Instances(bboxes=np.array([[10, 10, 50, 50], [100, 100, 150, 150]])),
            ... }
            >>> result = transform(labels)
            >>> assert result["img"].shape[:2] == result["resized_shape"]
        """
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)  # 不需要比例填充

        img = labels["img"]
        cls = labels["cls"]
        instances = labels.pop("instances")
        # 确保坐标格式正确
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M是仿射矩阵
        # 用于函数:`box_candidates`的缩放
        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        # 如果有分割，更新边界框
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        # 裁剪
        new_instances.clip(*self.size)

        # 过滤实例
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # 使边界框与新边界框具有相同的缩放
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    @staticmethod
    def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        """
        根据大小和长宽比标准计算候选框，用于进一步处理。

        该方法比较变换前后的边界框，以确定它们是否符合指定的宽度、高度、长宽比和面积标准。
        它用于过滤那些经过变换后被过度扭曲或缩小的边界框。

        参数：
            box1 (numpy.ndarray): 变换前的边界框，形状为(4, N)，其中N是边界框的数量。格式为[x1, y1, x2, y2]（绝对坐标）。
            box2 (numpy.ndarray): 变换后的边界框，形状为(4, N)，格式为[x1, y1, x2, y2]（绝对坐标）。
            wh_thr (float): 宽度和高度阈值（像素）。任何尺寸小于该值的框将被拒绝。
            ar_thr (float): 长宽比阈值。任何长宽比大于该值的框将被拒绝。
            area_thr (float): 面积比阈值。新旧框的面积比（新框面积/旧框面积）小于该值的框将被拒绝。
            eps (float): 防止除零错误的小常数。

        返回：
            (numpy.ndarray): 形状为(n)的布尔数组，指示哪些框是候选框。
                True值对应于符合所有标准的框。

        示例：
            >>> random_perspective = RandomPerspective()
            >>> box1 = np.array([[0, 0, 100, 100], [0, 0, 50, 50]]).T
            >>> box2 = np.array([[10, 10, 90, 90], [5, 5, 45, 45]]).T
            >>> candidates = random_perspective.box_candidates(box1, box2)
            >>> print(candidates)
            [True True]
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # 长宽比
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # 候选框


class RandomHSV:
    """
    随机调整图像的色调（Hue）、饱和度（Saturation）和明度（Value）通道。

    该类对图像应用随机的HSV增强，变化范围由hgain、sgain和vgain设定的预定义限制控制。

    属性：
        hgain (float): 色调的最大变化范围，通常在[0, 1]之间。
        sgain (float): 饱和度的最大变化范围，通常在[0, 1]之间。
        vgain (float): 明度的最大变化范围，通常在[0, 1]之间。

    方法：
        __call__: 对图像应用随机HSV增强。

    示例：
        >>> import numpy as np
        >>> from ultralytics.data.augment import RandomHSV
        >>> augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> labels = {"img": image}
        >>> augmenter(labels)
        >>> augmented_image = augmented_labels["img"]
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        """
        初始化RandomHSV对象，用于随机HSV（色调、饱和度、明度）增强。

        该类在指定的限制范围内对图像的HSV通道进行随机调整。

        参数：
            hgain (float): 色调的最大变化范围，应在[0, 1]之间。
            sgain (float): 饱和度的最大变化范围，应在[0, 1]之间。
            vgain (float): 明度的最大变化范围，应在[0, 1]之间。

        示例：
            >>> hsv_aug = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> hsv_aug(image)
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels):
        """
        在预定义的限制范围内对图像应用随机HSV增强。

        该方法通过随机调整图像的色调、饱和度和明度（HSV）通道来修改输入图像。
        调整范围由初始化时设置的hgain、sgain和vgain控制。

        参数：
            labels (Dict): 包含图像数据和元数据的字典。必须包括一个'img'键，值为图像的numpy数组。

        返回：
            (None): 该函数就地修改输入的'labels'字典，更新'img'键为HSV增强后的图像。

        示例：
            >>> hsv_augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> labels = {"img": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)}
            >>> hsv_augmenter(labels)
            >>> augmented_img = labels["img"]
        """
        img = labels["img"]
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # 随机增益
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # 不需要返回
        return labels


class RandomFlip:
    """
    对图像应用随机水平或垂直翻转，具有给定的概率。

    该类执行随机图像翻转并更新相应的实例注释，如边界框和关键点。

    属性：
        p (float): 应用翻转的概率，必须在0到1之间。
        direction (str): 翻转方向，可以是'horizontal'或'vertical'。
        flip_idx (array-like): 翻转关键点的索引映射（如果适用）。

    方法：
        __call__: 对图像及其注释应用随机翻转变换。

    示例：
        >>> transform = RandomFlip(p=0.5, direction="horizontal")
        >>> result = transform({"img": image, "instances": instances})
        >>> flipped_image = result["img"]
        >>> flipped_instances = result["instances"]
    """

    def __init__(self, p=0.5, direction="horizontal", flip_idx=None) -> None:
        """
        使用概率和方向初始化RandomFlip类。

        该类对图像应用随机水平或垂直翻转，具有给定的概率。
        它还会相应地更新任何实例（如边界框、关键点等）。

        参数：
            p (float): 应用翻转的概率，必须在0到1之间。
            direction (str): 翻转的方向，必须是'horizontal'或'vertical'。
            flip_idx (List[int] | None): 翻转关键点的索引映射（如果有）。

        异常：
            AssertionError: 如果方向不是'horizontal'或'vertical'，或者p不在0到1之间。

        示例：
            >>> flip = RandomFlip(p=0.5, direction="horizontal")
            >>> flip_with_idx = RandomFlip(p=0.7, direction="vertical", flip_idx=[1, 0, 3, 2, 5, 4])
        """
        assert direction in {"horizontal", "vertical"}, f"支持方向 `horizontal` 或 `vertical`，但得到 {direction}"
        assert 0 <= p <= 1.0, f"概率应在[0, 1]范围内，但得到 {p}."

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, labels):
        """
        对图像应用随机翻转，并相应更新实例（如边界框或关键点）。

        该方法根据初始化时的概率和方向随机翻转输入图像，并更新相应的实例（边界框、关键点），
        以匹配翻转后的图像。

        参数：
            labels (Dict): 包含以下键的字典：
                'img' (numpy.ndarray): 需要翻转的图像。
                'instances' (ultralytics.utils.instance.Instances): 包含边界框并可选地包含关键点的实例对象。

        返回：
            (Dict): 相同的字典，其中包含翻转后的图像和更新后的实例：
                'img' (numpy.ndarray): 翻转后的图像。
                'instances' (ultralytics.utils.instance.Instances): 更新后的实例，匹配翻转后的图像。

        示例：
            >>> labels = {"img": np.random.rand(640, 640, 3), "instances": Instances(...)}
            >>> random_flip = RandomFlip(p=0.5, direction="horizontal")
            >>> flipped_labels = random_flip(labels)
        """
        img = labels["img"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        # 上下翻转
        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            instances.flipud(h)
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            instances.fliplr(w)
            # 对关键点进行翻转
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        labels["img"] = np.ascontiguousarray(img)
        labels["instances"] = instances
        return labels


class LetterBox:
    """
    图像的缩放和填充，适用于目标检测、实例分割和姿态估计任务。

    该类将图像调整为指定的大小，并添加填充，同时保持图像的宽高比。它还会更新相应的标签和边界框。

    属性：
        new_shape (tuple): 目标大小（高度，宽度）用于缩放。
        auto (bool): 是否使用最小矩形。
        scaleFill (bool): 是否将图像拉伸到新的大小而不进行填充。
        scaleup (bool): 是否允许放大。如果为False，仅进行缩小。
        stride (int): 用于调整填充的步长。
        center (bool): 是否将图像居中，或对齐到左上角。

    方法：
        __call__: 缩放并填充图像，更新标签和边界框。

    示例：
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result["img"]
        >>> updated_instances = result["instances"]
    """

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """
        初始化LetterBox对象，用于缩放和填充图像。

        该类旨在为目标检测、实例分割和姿态估计任务缩放和填充图像。
        它支持各种缩放模式，包括自动缩放、填充缩放和信箱填充（letterboxing）。

        参数：
            new_shape (Tuple[int, int]): 缩放后的目标大小（高度，宽度）。
            auto (bool): 如果为True，使用最小矩形进行缩放。如果为False，直接使用new_shape。
            scaleFill (bool): 如果为True，图像会被拉伸到new_shape而不进行填充。
            scaleup (bool): 如果为True，允许放大。如果为False，仅进行缩小。
            center (bool): 如果为True，居中显示图像。如果为False，将图像放置在左上角。
            stride (int): 模型的步长（例如，YOLOv5的步长为32）。

        属性：
            new_shape (Tuple[int, int]): 缩放后的目标大小。
            auto (bool): 使用最小矩形缩放的标志。
            scaleFill (bool): 是否拉伸图像而不进行填充。
            scaleup (bool): 是否允许放大。
            stride (int): 确保图像大小可被步长整除的步长值。

        示例：
            >>> letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32)
            >>> resized_img = letterbox(original_img)
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # 是否将图像居中或放置在左上角

    def __call__(self, labels=None, image=None):
        """
        为目标检测、实例分割或姿态估计任务缩放和填充图像。

        该方法对输入图像应用信箱填充，即在保持图像宽高比的同时缩放图像并添加填充，
        使其符合新的形状。它还会相应地更新所有关联的标签。

        参数：
            labels (Dict | None): 包含图像数据和关联标签的字典，如果为None则为空字典。
            image (np.ndarray | None): 输入图像的numpy数组。如果为None，则从'labels'中获取图像。

        返回：
            (Dict | Tuple): 如果提供了'labels'，则返回更新后的字典，包括缩放和填充后的图像，
                更新后的标签，以及其他元数据。如果'labels'为空，则返回一个包含缩放和填充后图像的元组，
                以及比例和填充大小的元组（ratio，(left_pad, top_pad)）。

        示例：
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> result = letterbox(labels={"img": np.zeros((480, 640, 3)), "instances": Instances(...)})
            >>> resized_img = result["img"]
            >>> updated_instances = result["instances"]
        """
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # 当前形状 [高度, 宽度]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 缩放比例（新 / 旧）
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # 仅缩小，不放大（为提高验证集mAP）
            r = min(r, 1.0)

        # 计算填充
        ratio = r, r  # 宽度，高度比例
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽高填充
        if self.auto:  # 最小矩形
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # 宽高填充
        elif self.scaleFill:  # 拉伸
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度，高度比例

        if self.center:
            dw /= 2  # 将填充分配到两边
            dh /= 2

        if shape[::-1] != new_unpad:  # 缩放
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # 添加边框
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # 用于评估

        if len(labels):
            labels = self._update_labels(labels, ratio, left, top)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    @staticmethod
    def _update_labels(labels, ratio, padw, padh):
        """
        在对图像应用信箱填充后更新标签。

        该方法修改标签中的实例的边界框坐标，以考虑信箱填充过程中的缩放和填充。

        参数：
            labels (Dict): 包含图像标签和实例的字典。
            ratio (Tuple[float, float]): 应用到图像的缩放比例（宽度，高度）。
            padw (float): 图像添加的填充宽度。
            padh (float): 图像添加的填充高度。

        返回：
            (Dict): 更新后的标签字典，包含修改后的实例坐标。

        示例：
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> labels = {"instances": Instances(...)}
            >>> ratio = (0.5, 0.5)
            >>> padw, padh = 10, 20
            >>> updated_labels = letterbox._update_labels(labels, ratio, padw, padh)
        """
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class CopyPaste(BaseMixTransform):
    """
    Copy-Paste类，用于对图像数据集应用Copy-Paste增强。

    该类实现了论文“Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation”
    （https://arxiv.org/abs/2012.07177）中描述的Copy-Paste增强技术。它将来自不同图像的物体组合，
    创建新的训练样本。

    属性：
        dataset (Any): 应用Copy-Paste增强的数据集。
        pre_transform (Callable | None): 可选的在Copy-Paste之前应用的变换。
        p (float): 应用Copy-Paste增强的概率。

    方法：
        get_indexes: 返回数据集中的随机索引。
        _mix_transform: 对输入标签应用Copy-Paste增强。
        __call__: 对图像和注释应用Copy-Paste变换。

    示例：
        >>> from ultralytics.data.augment import CopyPaste
        >>> dataset = YourDataset(...)  # 你的图像数据集
        >>> copypaste = CopyPaste(dataset, p=0.5)
        >>> augmented_labels = copypaste(original_labels)
    """

    def __init__(self, dataset=None, pre_transform=None, p=0.5, mode="flip") -> None:
        """初始化CopyPaste对象，包含数据集、预变换和应用MixUp的概率。"""
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        assert mode in {"flip", "mixup"}, f"期望`mode`为`flip`或`mixup`，但得到 {mode}."
        self.mode = mode

    def get_indexes(self):
        """返回数据集中的随机索引，用于CopyPaste增强。"""
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        """应用Copy-Paste增强，将另一个图像的物体组合到当前图像中。"""
        labels2 = labels["mix_labels"][0]
        return self._transform(labels, labels2)

    def __call__(self, labels):
        """对图像及其标签应用Copy-Paste增强。"""
        if len(labels["instances"].segments) == 0 or self.p == 0:
            return labels
        if self.mode == "flip":
            return self._transform(labels)

        # 获取一个或多个其他图像的索引
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # 获取用于Mosaic或MixUp的图像信息
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        # 更新类别和文本
        labels = self._update_label_text(labels)
        # Mosaic或MixUp
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def _transform(self, labels1, labels2={}):
        """应用Copy-Paste增强，将另一个图像的物体组合到当前图像中。"""
        im = labels1["img"]
        cls = labels1["cls"]
        h, w = im.shape[:2]
        instances = labels1.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)

        im_new = np.zeros(im.shape, np.uint8)
        instances2 = labels2.pop("instances", None)
        if instances2 is None:
            instances2 = deepcopy(instances)
            instances2.fliplr(w)
        ioa = bbox_ioa(instances2.bboxes, instances.bboxes)  # 计算交集面积，(N, M)
        indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
        n = len(indexes)
        sorted_idx = np.argsort(ioa.max(1)[indexes])
        indexes = indexes[sorted_idx]
        for j in indexes[: round(self.p * n)]:
            cls = np.concatenate((cls, labels2.get("cls", cls)[[j]]), axis=0)
            instances = Instances.concatenate((instances, instances2[[j]]), axis=0)
            cv2.drawContours(im_new, instances2.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)

        result = labels2.get("img", cv2.flip(im, 1))  # 增强分割
        i = im_new.astype(bool)
        im[i] = result[i]

        labels1["img"] = im
        labels1["cls"] = cls
        labels1["instances"] = instances
        return labels1


class Albumentations:
    """
    用于图像增强的Albumentations变换。

    该类使用Albumentations库应用各种图像变换，包括模糊、均值模糊、转换为灰度图、对比度限制自适应直方图均衡化（CLAHE）、随机亮度和对比度变化、随机伽马调整，以及通过压缩降低图像质量等操作。

    属性：
        p (float): 应用变换的概率。
        transform (albumentations.Compose): 组成的Albumentations变换。
        contains_spatial (bool): 指示变换是否包含空间操作。

    方法：
        __call__: 对输入标签应用Albumentations变换。

    示例：
        >>> transform = Albumentations(p=0.5)
        >>> augmented_labels = transform(labels)
    """

    def __init__(self, p=1.0):
        """
        初始化Albumentations变换对象，用于YOLO边界框格式的参数。

        该类使用Albumentations库应用各种图像增强，包括模糊、均值模糊、转换为灰度图、对比度限制自适应直方图均衡化、随机亮度和对比度变化、随机伽马调整，以及通过压缩降低图像质量。

        参数：
            p (float): 应用增强的概率，必须在0到1之间。

        属性：
            p (float): 应用增强的概率。
            transform (albumentations.Compose): 组成的Albumentations变换。
            contains_spatial (bool): 指示变换是否包含空间变换。

        异常：
            ImportError: 如果未安装Albumentations包。
            Exception: 初始化过程中遇到任何其他错误。

        示例：
            >>> transform = Albumentations(p=0.5)
            >>> augmented = transform(image=image, bboxes=bboxes, class_labels=classes)
            >>> augmented_image = augmented["image"]
            >>> augmented_bboxes = augmented["bboxes"]

        备注：
            - 需要安装Albumentations版本1.0.3或更高版本。
            - 空间变换需要特别处理，以确保与边界框的兼容性。
            - 一些变换默认以非常低的概率（0.01）应用。
        """
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")

        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # 版本要求

            # 空间变换的可能列表
            spatial_transforms = {
                "Affine",
                "BBoxSafeRandomCrop",
                "CenterCrop",
                "CoarseDropout",
                "Crop",
                "CropAndPad",
                "CropNonEmptyMaskIfExists",
                "D4",
                "ElasticTransform",
                "Flip",
                "GridDistortion",
                "GridDropout",
                "HorizontalFlip",
                "Lambda",
                "LongestMaxSize",
                "MaskDropout",
                "MixUp",
                "Morphological",
                "NoOp",
                "OpticalDistortion",
                "PadIfNeeded",
                "Perspective",
                "PiecewiseAffine",
                "PixelDropout",
                "RandomCrop",
                "RandomCropFromBorders",
                "RandomGridShuffle",
                "RandomResizedCrop",
                "RandomRotate90",
                "RandomScale",
                "RandomSizedBBoxSafeCrop",
                "RandomSizedCrop",
                "Resize",
                "Rotate",
                "SafeRotate",
                "ShiftScaleRotate",
                "SmallestMaxSize",
                "Transpose",
                "VerticalFlip",
                "XYMasking",
            }  # 来自 https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms

            # 变换列表
            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]

            # 组合变换
            self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
            self.transform = (
                A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
                if self.contains_spatial
                else A.Compose(T)
            )
            if hasattr(self.transform, "set_random_seed"):
                # 需要在albumentations>=1.4.21版本中进行确定性变换
                self.transform.set_random_seed(torch.initial_seed())
            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # 未安装包，跳过
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels):
        """
        对输入标签应用Albumentations变换。

        该方法使用Albumentations库应用一系列图像增强。它可以对输入图像及其相应的标签执行空间和非空间变换。

        参数：
            labels (Dict): 包含图像数据和注释的字典。预期的键有：
                - 'img': 表示图像的numpy.ndarray
                - 'cls': 类标签的numpy.ndarray
                - 'instances': 包含边界框和其他实例信息的对象

        返回：
            (Dict): 包含增强后的图像和更新后的注释的输入字典。

        示例：
            >>> transform = Albumentations(p=0.5)
            >>> labels = {
            ...     "img": np.random.rand(640, 640, 3),
            ...     "cls": np.array([0, 1]),
            ...     "instances": Instances(bboxes=np.array([[0, 0, 1, 1], [0.5, 0.5, 0.8, 0.8]])),
            ... }
            >>> augmented = transform(labels)
            >>> assert augmented["img"].shape == (640, 640, 3)

        备注：
            - 此方法以self.p的概率应用变换。
            - 空间变换更新边界框，而非空间变换仅修改图像。
            - 需要安装Albumentations库。
        """
        if self.transform is None or random.random() > self.p:
            return labels

        if self.contains_spatial:
            cls = labels["cls"]
            if len(cls):
                im = labels["img"]
                labels["instances"].convert_bbox("xywh")
                labels["instances"].normalize(*im.shape[:2][::-1])
                bboxes = labels["instances"].bboxes
                # TODO: 添加对分割和关键点的支持
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # 变换后的图像
                if len(new["class_labels"]) > 0:  # 如果新图像中没有边界框，则跳过更新
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"])
                    bboxes = np.array(new["bboxes"], dtype=np.float32)
                labels["instances"].update(bboxes=bboxes)
        else:
            labels["img"] = self.transform(image=labels["img"])["image"]  # 变换后的图像

        return labels


class Format:
    """
    用于目标检测、实例分割和姿态估计任务的图像注释格式化类。

    该类将图像和实例注释标准化，以便在PyTorch DataLoader的`collate_fn`中使用。

    属性：
        bbox_format (str): 边界框格式。可选值为'xywh'或'xyxy'。
        normalize (bool): 是否对边界框进行归一化。
        return_mask (bool): 是否返回实例掩码用于分割任务。
        return_keypoint (bool): 是否返回关键点用于姿态估计任务。
        return_obb (bool): 是否返回定向边界框。
        mask_ratio (int): 掩码的下采样比例。
        mask_overlap (bool): 是否允许掩码重叠。
        batch_idx (bool): 是否保留批次索引。
        bgr (float): 是否返回BGR图像的概率。

    方法：
        __call__: 格式化包含图像、类别、边界框的标签字典，并可选择性地返回掩码和关键点。
        _format_img: 将图像从Numpy数组转换为PyTorch张量。
        _format_segments: 将多边形点转换为位图掩码。

    示例：
        >>> formatter = Format(bbox_format="xywh", normalize=True, return_mask=True)
        >>> formatted_labels = formatter(labels)
        >>> img = formatted_labels["img"]
        >>> bboxes = formatted_labels["bboxes"]
        >>> masks = formatted_labels["masks"]
    """

    def __init__(
        self,
        bbox_format="xywh",
        normalize=True,
        return_mask=False,
        return_keypoint=False,
        return_obb=False,
        mask_ratio=4,
        mask_overlap=True,
        batch_idx=True,
        bgr=0.0,
    ):
        """
        使用给定的参数初始化Format类，用于图像和实例注释的格式化。

        该类将图像和实例注释标准化，以便用于目标检测、实例分割和姿态估计任务，准备好用于PyTorch DataLoader的`collate_fn`。

        参数：
            bbox_format (str): 边界框格式。可选值为'xywh'、'xyxy'等。
            normalize (bool): 是否将边界框归一化到[0,1]。
            return_mask (bool): 如果为True，返回实例掩码，用于分割任务。
            return_keypoint (bool): 如果为True，返回姿态估计任务中的关键点。
            return_obb (bool): 如果为True，返回定向边界框。
            mask_ratio (int): 掩码的下采样比例。
            mask_overlap (bool): 如果为True，允许掩码重叠。
            batch_idx (bool): 如果为True，保留批次索引。
            bgr (float): 是否返回BGR图像的概率。

        属性：
            bbox_format (str): 边界框格式。
            normalize (bool): 是否归一化边界框。
            return_mask (bool): 是否返回实例掩码。
            return_keypoint (bool): 是否返回关键点。
            return_obb (bool): 是否返回定向边界框。
            mask_ratio (int): 掩码的下采样比例。
            mask_overlap (bool): 是否允许掩码重叠。
            batch_idx (bool): 是否保留批次索引。
            bgr (float): 是否返回BGR图像的概率。

        示例：
            >>> format = Format(bbox_format="xyxy", return_mask=True, return_keypoint=False)
            >>> print(format.bbox_format)
            xyxy
        """
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask  # 在仅训练检测任务时设置为False
        self.return_keypoint = return_keypoint
        self.return_obb = return_obb
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx  # 保留批次索引
        self.bgr = bgr

    def __call__(self, labels):
        """
        格式化目标检测、实例分割和姿态估计任务的图像注释。

        该方法标准化输入标签字典中的图像和实例注释，转换为指定格式，并在需要时应用归一化。

        参数：
            labels (Dict): 包含图像和注释数据的字典，预期包含以下键：
                - 'img': 输入图像（Numpy数组）
                - 'cls': 实例的类别标签
                - 'instances': 包含边界框、分割和关键点等信息的实例对象

        返回：
            (Dict): 包含格式化数据的字典，包括：
                - 'img': 格式化后的图像张量。
                - 'cls': 类别标签张量。
                - 'bboxes': 以指定格式存储的边界框张量。
                - 'masks': 实例掩码张量（如果return_mask为True）。
                - 'keypoints': 关键点张量（如果return_keypoint为True）。
                - 'batch_idx': 批次索引张量（如果batch_idx为True）。

        示例：
            >>> formatter = Format(bbox_format="xywh", normalize=True, return_mask=True)
            >>> labels = {"img": np.random.rand(640, 640, 3), "cls": np.array([0, 1]), "instances": Instances(...)}
            >>> formatted_labels = formatter(labels)
            >>> print(formatted_labels.keys())
        """
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)

        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
            else:
                masks = torch.zeros(
                    1 if self.mask_overlap else nl, img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio
                )
            labels["masks"] = masks
        labels["img"] = self._format_img(img)
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        if self.return_keypoint:
            labels["keypoints"] = torch.from_numpy(instances.keypoints)
            if self.normalize:
                labels["keypoints"][..., 0] /= w
                labels["keypoints"][..., 1] /= h
        if self.return_obb:
            labels["bboxes"] = (
                xyxyxyxy2xywhr(torch.from_numpy(instances.segments)) if len(instances.segments) else torch.zeros((0, 5))
            )
        # NOTE: 需要对obb进行归一化，确保宽高一致
        if self.normalize:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
        # 然后我们可以使用collate_fn
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels

    def _format_img(self, img):
        """
        将图像格式化为YOLO格式，从Numpy数组转换为PyTorch张量。

        此函数执行以下操作：
        1. 确保图像具有3个维度（如果需要，添加一个通道维度）。
        2. 将图像从HWC格式转换为CHW格式。
        3. 可选择性地将颜色通道从RGB翻转为BGR。
        4. 将图像转换为连续数组。
        5. 将Numpy数组转换为PyTorch张量。

        参数：
            img (np.ndarray): 输入图像，形状为(H, W, C)或(H, W)。

        返回：
            (torch.Tensor): 格式化后的图像PyTorch张量，形状为(C, H, W)。

        示例：
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> formatted_img = self._format_img(img)
            >>> print(formatted_img.shape)
            torch.Size([3, 100, 100])
        """
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr else img)
        img = torch.from_numpy(img)
        return img

    def _format_segments(self, instances, cls, w, h):
        """
        将多边形分割转换为位图掩码。

        参数：
            instances (Instances): 包含分割信息的对象。
            cls (numpy.ndarray): 每个实例的类别标签。
            w (int): 图像的宽度。
            h (int): 图像的高度。

        返回：
            masks (numpy.ndarray): 形状为(N, H, W)的位图掩码（如果mask_overlap为True，则为(1, H, W)）。
            instances (Instances): 更新后的实例对象（如果mask_overlap为True，则包含排序后的分割）。
            cls (numpy.ndarray): 更新后的类别标签（如果mask_overlap为True，则为排序后的标签）。

        备注：
            - 如果self.mask_overlap为True，则掩码重叠并按面积排序。
            - 如果self.mask_overlap为False，则每个掩码单独表示。
            - 掩码根据self.mask_ratio进行下采样。
        """
        segments = instances.segments
        if self.mask_overlap:
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)
            masks = masks[None]  # (640, 640) -> (1, 640, 640)
            instances = instances[sorted_idx]
            cls = cls[sorted_idx]
        else:
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)

        return masks, instances, cls


class RandomLoadText:
    """
    随机采样正负文本，并相应更新类别索引。

    该类负责从给定的类别文本集合中随机采样文本，包括正样本（图像中存在的文本）和负样本（图像中不存在的文本）。它会更新类别索引以反映采样的文本，并且可以选择性地将文本列表填充到固定长度。

    属性：
        prompt_format (str): 文本提示的格式化字符串。
        neg_samples (Tuple[int, int]): 随机采样负文本的范围。
        max_samples (int): 每个图像中不同文本样本的最大数量。
        padding (bool): 是否将文本填充到max_samples。
        padding_value (str): 当padding为True时用于填充的文本。

    方法：
        __call__: 处理输入标签并返回更新后的类别和文本。

    示例：
        >>> loader = RandomLoadText(prompt_format="Object: {}", neg_samples=(5, 10), max_samples=20)
        >>> labels = {"cls": [0, 1, 2], "texts": [["cat"], ["dog"], ["bird"]], "instances": [...]}
        >>> updated_labels = loader(labels)
        >>> print(updated_labels["texts"])
        ['Object: cat', 'Object: dog', 'Object: bird', 'Object: elephant', 'Object: car']
    """

    def __init__(
        self,
        prompt_format: str = "{}",
        neg_samples: Tuple[int, int] = (80, 80),
        max_samples: int = 80,
        padding: bool = False,
        padding_value: str = "",
    ) -> None:
        """
        初始化RandomLoadText类，用于随机采样正负文本。

        该类用于随机采样正文本和负文本，并根据样本数量相应地更新类别索引。它可用于基于文本的目标检测任务。

        参数：
            prompt_format (str): 提示的格式字符串。默认是'{}'。格式字符串应该包含一对大括号{}，其中将插入文本。
            neg_samples (Tuple[int, int]): 随机采样负文本的范围。第一个整数指定负样本的最小数量，第二个整数指定最大数量。默认是(80, 80)。
            max_samples (int): 每个图像中不同文本样本的最大数量。默认是80。
            padding (bool): 是否将文本填充到max_samples。如果为True，文本数量将始终等于max_samples。默认是False。
            padding_value (str): 当padding为True时用于填充的文本。默认是空字符串。

        属性：
            prompt_format (str): 提示格式字符串。
            neg_samples (Tuple[int, int]): 随机采样负文本的范围。
            max_samples (int): 最大文本样本数量。
            padding (bool): 是否启用填充。
            padding_value (str): 填充时使用的值。

        示例：
            >>> random_load_text = RandomLoadText(prompt_format="Object: {}", neg_samples=(50, 100), max_samples=120)
            >>> random_load_text.prompt_format
            'Object: {}'
            >>> random_load_text.neg_samples
            (50, 100)
            >>> random_load_text.max_samples
            120
        """
        self.prompt_format = prompt_format
        self.neg_samples = neg_samples
        self.max_samples = max_samples
        self.padding = padding
        self.padding_value = padding_value

    def __call__(self, labels: dict) -> dict:
        """
        随机采样正负文本并相应更新类别索引。

        该方法基于图像中现有的类别标签采样正文本，并从剩余类别中随机选择负文本。然后，它更新类别索引以匹配新的采样文本顺序。

        参数：
            labels (Dict): 包含图像标签和元数据的字典。必须包含'texts'和'cls'键。

        返回：
            (Dict): 更新后的标签字典，包含新的'cls'和'texts'条目。

        示例：
            >>> loader = RandomLoadText(prompt_format="A photo of {}", neg_samples=(5, 10), max_samples=20)
            >>> labels = {"cls": np.array([[0], [1], [2]]), "texts": [["dog"], ["cat"], ["bird"]]}
            >>> updated_labels = loader(labels)
        """
        assert "texts" in labels, "标签中未找到文本."
        class_texts = labels["texts"]
        num_classes = len(class_texts)
        cls = np.asarray(labels.pop("cls"), dtype=int)
        pos_labels = np.unique(cls).tolist()

        if len(pos_labels) > self.max_samples:
            pos_labels = random.sample(pos_labels, k=self.max_samples)

        neg_samples = min(min(num_classes, self.max_samples) - len(pos_labels), random.randint(*self.neg_samples))
        neg_labels = [i for i in range(num_classes) if i not in pos_labels]
        neg_labels = random.sample(neg_labels, k=neg_samples)

        sampled_labels = pos_labels + neg_labels
        random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}
        valid_idx = np.zeros(len(labels["instances"]), dtype=bool)
        new_cls = []
        for i, label in enumerate(cls.squeeze(-1).tolist()):
            if label not in label2ids:
                continue
            valid_idx[i] = True
            new_cls.append([label2ids[label]])
        labels["instances"] = labels["instances"][valid_idx]
        labels["cls"] = np.array(new_cls)

        # 随机选择一个提示（如果有多个提示）
        texts = []
        for label in sampled_labels:
            prompts = class_texts[label]
            assert len(prompts) > 0
            prompt = self.prompt_format.format(prompts[random.randrange(len(prompts))])
            texts.append(prompt)

        if self.padding:
            valid_labels = len(pos_labels) + len(neg_labels)
            num_padding = self.max_samples - valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding

        labels["texts"] = texts
        return labels


def v8_transforms(dataset, imgsz, hyp, stretch=False):
    """
    应用一系列图像转换用于训练。

    这个函数创建了一系列图像增强技术的组合，以准备图像用于YOLO训练。包括的操作有马赛克、复制粘贴、随机透视、MixUp以及各种颜色调整。

    参数：
        dataset (Dataset): 包含图像数据和注释的数据集对象。
        imgsz (int): 目标图像大小，用于调整大小。
        hyp (Namespace): 控制各种转换方面的超参数字典。
        stretch (bool): 如果为True，则应用图像拉伸。如果为False，则使用LetterBox调整大小。

    返回：
        (Compose): 一组图像转换，将应用于数据集。

    示例：
        >>> from ultralytics.data.dataset import YOLODataset
        >>> from ultralytics.utils import IterableSimpleNamespace
        >>> dataset = YOLODataset(img_path="path/to/images", imgsz=640)
        >>> hyp = IterableSimpleNamespace(mosaic=1.0, copy_paste=0.5, degrees=10.0, translate=0.2, scale=0.9)
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
        >>> augmented_data = transforms(dataset[0])
    """
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
    )

    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])  # 用于关键点增强
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ 数据文件中未定义 'flip_idx' 数组，已将增强 'fliplr=0.0' 设置")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml中的flip_idx={flip_idx}长度必须与kpt_shape[0]={kpt_shape[0]}相等")

    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # 转换序列


# 分类增强 -----------------------------------------------------------------------------------------
def classify_transforms(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    interpolation="BILINEAR",
    crop_fraction: float = DEFAULT_CROP_FRACTION,
):
    """
    创建一个用于分类任务的图像转换组合。

    这个函数生成一个适用于分类模型的图像预处理变换序列，用于评估或推理过程中。包括调整大小、中心裁剪、转换为张量以及归一化。

    参数：
        size (int | tuple): 转换后的目标图像大小。如果是整数，则表示最短边的大小。如果是元组，则表示(height, width)。
        mean (tuple): 用于归一化的每个RGB通道的均值。
        std (tuple): 用于归一化的每个RGB通道的标准差。
        interpolation (str): 插值方法，选项有 'NEAREST', 'BILINEAR' 或 'BICUBIC'。
        crop_fraction (float): 要裁剪的图像部分的比例。

    返回：
        (torchvision.transforms.Compose): 一个torchvision的图像转换组合。

    示例：
        >>> transforms = classify_transforms(size=224)
        >>> img = Image.open("path/to/image.jpg")
        >>> transformed_img = transforms(img)
    """
    import torchvision.transforms as T  # 在更快的 'import ultralytics' 范围内

    if isinstance(size, (tuple, list)):
        assert len(size) == 2, f"'size'元组必须是长度为2，而不是长度为{len(size)}"
        scale_size = tuple(math.floor(x / crop_fraction) for x in size)
    else:
        scale_size = math.floor(size / crop_fraction)
        scale_size = (scale_size, scale_size)

    # 保持纵横比，裁剪图像的中心，不添加边框，图像会丢失
    if scale_size[0] == scale_size[1]:
        # 简单情况，使用torchvision内建的Resize，采用最短边模式（标量大小参数）
        tfl = [T.Resize(scale_size[0], interpolation=getattr(T.InterpolationMode, interpolation))]
    else:
        # 将最短边调整到匹配目标尺寸的非方形目标
        tfl = [T.Resize(scale_size)]
    tfl.extend(
        [
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    )
    return T.Compose(tfl)


# 分类训练增强 --------------------------------------------------------------------------------
def classify_augmentations(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    auto_augment=None,
    hsv_h=0.015,  # 图像 HSV-色调增强（比例）
    hsv_s=0.4,  # 图像 HSV-饱和度增强（比例）
    hsv_v=0.4,  # 图像 HSV-明度增强（比例）
    force_color_jitter=False,
    erasing=0.0,
    interpolation="BILINEAR",
):
    """
    创建一个用于分类任务的图像增强转换组合。

    这个函数生成一组适用于训练分类模型的图像增强变换。包括调整大小、翻转、颜色抖动、自动增强和随机擦除等选项。

    参数：
        size (int): 转换后图像的目标大小。
        mean (tuple): 每个通道的均值，用于归一化。
        std (tuple): 每个通道的标准差，用于归一化。
        scale (tuple | None): 原始大小裁剪的范围。
        ratio (tuple | None): 原始宽高比裁剪的范围。
        hflip (float): 水平翻转的概率。
        vflip (float): 垂直翻转的概率。
        auto_augment (str | None): 自动增强策略。可以是 'randaugment'、'augmix'、'autoaugment' 或 None。
        hsv_h (float): 图像 HSV-色调增强因子。
        hsv_s (float): 图像 HSV-饱和度增强因子。
        hsv_v (float): 图像 HSV-明度增强因子。
        force_color_jitter (bool): 是否即使启用了自动增强也强制应用颜色抖动。
        erasing (float): 随机擦除的概率。
        interpolation (str): 插值方法，可以是 'NEAREST'、'BILINEAR' 或 'BICUBIC'。

    返回：
        (torchvision.transforms.Compose): 一个图像增强转换的组合。

    示例：
        >>> transforms = classify_augmentations(size=224, auto_augment="randaugment")
        >>> augmented_image = transforms(original_image)
    """
    # 如果没有安装Albumentations，应用的变换
    import torchvision.transforms as T  # 加速 'import ultralytics'

    if not isinstance(size, int):
        raise TypeError(f"classify_transforms() size {size} 必须是整数，而不是（列表、元组）")
    scale = tuple(scale or (0.08, 1.0))  # 默认的imagenet缩放范围
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # 默认的imagenet宽高比范围
    interpolation = getattr(T.InterpolationMode, interpolation)
    primary_tfl = [T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if hflip > 0.0:
        primary_tfl.append(T.RandomHorizontalFlip(p=hflip))
    if vflip > 0.0:
        primary_tfl.append(T.RandomVerticalFlip(p=vflip))

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str), f"提供的参数应为字符串，但得到的类型是 {type(auto_augment)}"
        # 通常如果启用了AA/RA，则禁用颜色抖动，
        # 这允许在不破坏旧配置的情况下进行覆盖
        disable_color_jitter = not force_color_jitter

        if auto_augment == "randaugment":
            if TORCHVISION_0_11:
                secondary_tfl.append(T.RandAugment(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=randaugment" 需要 torchvision >= 0.11.0。禁用它。')

        elif auto_augment == "augmix":
            if TORCHVISION_0_13:
                secondary_tfl.append(T.AugMix(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=augmix" 需要 torchvision >= 0.13.0。禁用它。')

        elif auto_augment == "autoaugment":
            if TORCHVISION_0_10:
                secondary_tfl.append(T.AutoAugment(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=autoaugment" 需要 torchvision >= 0.10.0。禁用它。')

        else:
            raise ValueError(
                f'无效的 auto_augment 策略: {auto_augment}。应为 "randaugment"、'
                f'"augmix"、"autoaugment" 或 None'
            )

    if not disable_color_jitter:
        secondary_tfl.append(T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h))

    final_tfl = [
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        T.RandomErasing(p=erasing, inplace=True),
    ]

    return T.Compose(primary_tfl + secondary_tfl + final_tfl)


# 保持向后兼容的类 ---------------------------------------------------------------------------------
class ClassifyLetterBox:
    """
    用于分类任务的图像调整大小和填充的类。

    这个类设计为图像转换管道的一部分，例如：T.Compose([LetterBox(size), ToTensor()])。
    它调整和填充图像到指定的大小，同时保持原始的纵横比。

    属性：
        h (int): 图像目标高度。
        w (int): 图像目标宽度。
        auto (bool): 如果为True，则自动计算短边使用步幅。
        stride (int): 步幅值，用于 'auto' 为True时。

    方法：
        __call__: 将LetterBox转换应用于输入图像。

    示例：
        >>> transform = ClassifyLetterBox(size=(640, 640), auto=False, stride=32)
        >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> result = transform(img)
        >>> print(result.shape)
        (640, 640, 3)
    """

    def __init__(self, size=(640, 640), auto=False, stride=32):
        """
        初始化用于图像预处理的ClassifyLetterBox对象。

        这个类旨在作为图像分类任务的转换管道的一部分。它调整和填充图像到指定的大小，同时保持原始的纵横比。

        参数：
            size (int | Tuple[int, int]): LetterBox图像的目标大小。如果是整数，则创建一个方形图像（size, size）。如果是元组，则应为（高度，宽度）。
            auto (bool): 如果为True，则自动根据步幅计算短边。默认为False。
            stride (int): 用于自动计算短边的步幅值。默认为32。

        属性：
            h (int): LetterBox图像的目标高度。
            w (int): LetterBox图像的目标宽度。
            auto (bool): 标志，指示是否自动计算短边。
            stride (int): 用于自动计算短边的步幅值。

        示例：
            >>> transform = ClassifyLetterBox(size=224)
            >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> result = transform(img)
            >>> print(result.shape)
            (224, 224, 3)
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # 传递最大尺寸整数，使用步幅自动计算短边
        self.stride = stride  # 使用auto时

    def __call__(self, im):
        """
        使用LetterBox方法调整图像大小和填充。

        这个方法将输入图像调整为适应指定尺寸，同时保持其纵横比，然后填充调整后的图像以匹配目标尺寸。

        参数：
            im (numpy.ndarray): 输入图像，形状为（H, W, C）的numpy数组。

        返回：
            (numpy.ndarray): 调整大小并填充后的图像，形状为（hs, ws, 3），其中hs和ws是目标高度和宽度。

        示例：
            >>> letterbox = ClassifyLetterBox(size=(640, 640))
            >>> image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            >>> resized_image = letterbox(image)
            >>> print(resized_image.shape)
            (640, 640, 3)
        """
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # 新旧尺寸的比例
        h, w = round(imh * r), round(imw * r)  # 调整后的图像尺寸

        # 计算填充尺寸
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else (self.h, self.w)
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)

        # 创建填充后的图像
        im_out = np.full((hs, ws, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


class CenterCrop:
    """
    对图像应用中心裁剪，用于分类任务。

    这个类对输入图像进行中心裁剪，将它们调整为指定的大小，同时保持原始图像的纵横比。
    它旨在作为图像转换管道的一部分，例如：T.Compose([CenterCrop(size), ToTensor()]).

    属性：
        h (int): 裁剪后图像的目标高度。
        w (int): 裁剪后图像的目标宽度。

    方法：
        __call__: 对输入图像应用中心裁剪转换。

    示例：
        >>> transform = CenterCrop(640)
        >>> image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        >>> cropped_image = transform(image)
        >>> print(cropped_image.shape)
        (640, 640, 3)
    """

    def __init__(self, size=640):
        """
        初始化CenterCrop对象，用于图像预处理。

        这个类设计为图像转换管道的一部分，例如：T.Compose([CenterCrop(size), ToTensor()]).
        它对输入图像进行中心裁剪，裁剪到指定的大小。

        参数：
            size (int | Tuple[int, int]): 裁剪后图像的目标大小。如果size是一个整数，则裁剪为方形图像（size, size）。如果size是一个元组（h, w），则用作输出大小。

        返回：
            (None): 这个方法初始化对象，并不返回任何东西。

        示例：
            >>> transform = CenterCrop(224)
            >>> img = np.random.rand(300, 300, 3)
            >>> cropped_img = transform(img)
            >>> print(cropped_img.shape)
            (224, 224, 3)
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):
        """
        对输入图像应用中心裁剪。

        这个方法使用letterbox方法调整图像大小，并裁剪图像的中心部分。它保持原始图像的纵横比，同时将图像适应到指定的尺寸。

        参数：
            im (numpy.ndarray | PIL.Image.Image): 输入图像，形状为（H, W, C）的numpy数组，或一个PIL图像对象。

        返回：
            (numpy.ndarray): 裁剪并调整大小后的图像，形状为(self.h, self.w, C)的numpy数组。

        示例：
            >>> transform = CenterCrop(size=224)
            >>> image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            >>> cropped_image = transform(image)
            >>> assert cropped_image.shape == (224, 224, 3)
        """
        if isinstance(im, Image.Image):  # 如果是PIL图像，则转换为numpy数组
            im = np.asarray(im)
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # 最小尺寸
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ToTensor:
    """
    将图像从numpy数组转换为PyTorch张量。

    这个类旨在作为图像预处理转换管道的一部分，例如：T.Compose([LetterBox(size), ToTensor()]).

    属性：
        half (bool): 如果为True，将图像转换为半精度（float16）。

    方法：
        __call__: 对输入图像应用张量转换。

    示例：
        >>> transform = ToTensor(half=True)
        >>> img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> tensor_img = transform(img)
        >>> print(tensor_img.shape, tensor_img.dtype)
        torch.Size([3, 640, 640]) torch.float16

    注意：
        输入图像预计为BGR格式，形状为(H, W, C)。
        输出张量将为RGB格式，形状为(C, H, W)，并归一化到[0, 1]。
    """

    def __init__(self, half=False):
        """
        初始化ToTensor对象，用于将图像转换为PyTorch张量。

        这个类旨在作为Ultralytics YOLO框架中图像预处理转换管道的一部分。它将numpy数组或PIL图像转换为PyTorch张量，并提供半精度（float16）转换选项。

        参数：
            half (bool): 如果为True，则将张量转换为半精度（float16）。默认值为False。

        示例：
            >>> transform = ToTensor(half=True)
            >>> img = np.random.rand(640, 640, 3)
            >>> tensor_img = transform(img)
            >>> print(tensor_img.dtype)
            torch.float16
        """
        super().__init__()
        self.half = half

    def __call__(self, im):
        """
        将图像从numpy数组转换为PyTorch张量。

        该方法将输入图像从numpy数组转换为PyTorch张量，应用可选的半精度转换和归一化。图像将从HWC格式转置为CHW格式，并且颜色通道从BGR反转为RGB。

        参数：
            im (numpy.ndarray): 输入图像，形状为(H, W, C)，BGR顺序。

        返回：
            (torch.Tensor): 转换后的图像作为PyTorch张量，数据类型为float32或float16，归一化到[0, 1]，形状为(C, H, W)，RGB顺序。

        示例：
            >>> transform = ToTensor(half=True)
            >>> img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> tensor_img = transform(img)
            >>> print(tensor_img.shape, tensor_img.dtype)
            torch.Size([3, 640, 640]) torch.float16
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC转CHW -> BGR转RGB -> 连续内存
        im = torch.from_numpy(im)  # 转换为torch张量
        im = im.half() if self.half else im.float()  # uint8 转为 fp16/32
        im /= 255.0  # 0-255 范围转换到 0.0-1.0
        return im
