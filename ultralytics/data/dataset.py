# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics 数据集 *.cache 版本，>= 1.0.0 用于 YOLOv8
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    用于加载 YOLO 格式的目标检测和/或分割标签的数据集类。

    参数：
        data (dict, 可选)：数据集的 YAML 字典。默认为 None。
        task (str)：显式参数，用于指定当前任务，默认为 'detect'。

    返回：
        (torch.utils.data.Dataset)：一个 PyTorch 数据集对象，可用于训练目标检测模型。
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """初始化 YOLODataset，支持可选的分割和关键点配置。"""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "不能同时使用分割和关键点。"
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        缓存数据集标签，检查图像并读取形状。

        参数：
            path (Path)：保存缓存文件的路径。默认值是 Path("./labels.cache")。

        返回：
            (dict)：标签。
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # 缺失的，找到的，空的，损坏的，消息
        desc = f"{self.prefix}正在扫描 {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' 在 data.yaml 中缺失或不正确。应为一个包含 [关键点数量, 维度数量 (2 表示 x,y 或 3 表示 x,y,可见性)] 的列表，例如 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} 张图片，{nm + ne} 个背景，{nc} 个损坏"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}警告 ⚠️ 在 {path} 中未找到标签。 {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # 警告信息
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """返回 YOLO 训练的标签字典。"""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # 尝试加载 *.cache 文件
            assert cache["version"] == DATASET_CACHE_VERSION  # 确保版本一致
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # 确保哈希一致
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # 执行缓存操作

        # 显示缓存信息
        nf, nm, ne, nc, n = cache.pop("results")  # 找到的，缺失的，空的，损坏的，总数
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"正在扫描 {cache_path}... {nf} 张图片，{nm + ne} 个背景，{nc} 个损坏"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # 显示结果
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # 显示警告信息

        # 读取缓存
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # 移除缓存中的项目
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"警告 ⚠️ 在 {cache_path} 中未找到图片，训练可能无法正常进行。 {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # 更新 im_files

        # 检查数据集是全是框还是全是分割
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"警告 ⚠️ 框和分割数量应该相等，但 len(segments) = {len_segments}, len(boxes) = {len_boxes} 不相等。为了解决此问题，将只使用框，并删除所有分割。"
                "为了避免这种情况，请提供检测或分割数据集，而不是检测-分割混合数据集。"
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"警告 ⚠️ 在 {cache_path} 中未找到标签，训练可能无法正常进行。 {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """构建并附加变换到列表中。"""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # 仅影响训练
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """将 mosaic、copy_paste 和 mixup 选项设置为 0.0，并构建变换。"""
        hyp.mosaic = 0.0  # 设置 mosaic 比例为 0.0
        hyp.copy_paste = 0.0  # 保持与以前 v8 close-mosaic 相同的行为
        hyp.mixup = 0.0  # 保持与以前 v8 close-mosaic 相同的行为
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        在此处自定义您的标签格式。

        注意：
            cls 不再包含 bboxes，现在分类和语义分割需要独立的 cls 标签
            还可以通过添加或删除字典键来支持分类和语义分割。
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # 注意：不要重新采样有定向框的
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # 确保如果原始长度大于 segment_resamples 时，分割正确插值
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """将数据样本合并为批次。"""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # 为 build_targets() 添加目标图像索引
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """
    用于加载YOLO格式的目标检测和/或分割标签的数据集类。

    参数：
        data (dict, 可选): 一个数据集的YAML字典。默认为None。
        task (str): 一个明确的参数来指定当前任务，默认为'detect'。

    返回：
        (torch.utils.data.Dataset): 一个PyTorch数据集对象，可用于训练目标检测模型。
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """初始化一个用于目标检测任务的数据集对象，可以选择性地指定其他参数。"""
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label):
        """为多模态模型训练添加文本信息。"""
        labels = super().update_labels_info(label)
        # 注意：某些类别与其同义词通过“/”进行连接。
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]
        return labels

    def build_transforms(self, hyp=None):
        """增强数据转换，支持多模态训练时的文本数据增强。"""
        transforms = super().build_transforms(hyp)
        if self.augment:
            # 注意：目前硬编码了这些参数。
            transforms.insert(-1, RandomLoadText(max_samples=min(self.data["nc"], 80), padding=True))
        return transforms


class GroundingDataset(YOLODataset):
    """处理目标检测任务，通过从指定的 JSON 文件加载注释，支持 YOLO 格式。"""

    def __init__(self, *args, task="detect", json_file, **kwargs):
        """初始化 GroundingDataset 进行目标检测，从指定的 JSON 文件加载注释。"""
        assert task == "detect", "`GroundingDataset` 目前只支持 `detect` 任务！"
        self.json_file = json_file
        super().__init__(*args, task=task, data={}, **kwargs)

    def get_img_files(self, img_path):
        """图像文件将在 `get_labels` 函数中读取，这里返回空列表。"""
        return []

    def get_labels(self):
        """从 JSON 文件加载注释，过滤并归一化每个图像的边界框。"""
        labels = []
        LOGGER.info("正在加载注释文件...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"正在读取注释 {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]])
                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # 类别
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)
            labels.append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        return labels

    def build_transforms(self, hyp=None):
        """配置训练时的数据增强，包括可选的文本加载；`hyp` 调整增强的强度。"""
        transforms = super().build_transforms(hyp)
        if self.augment:
            # 注意：当前硬编码了参数。
            transforms.insert(-1, RandomLoadText(max_samples=80, padding=True))
        return transforms


class YOLOConcatDataset(ConcatDataset):
    """
    作为多个数据集拼接的数据集类。

    这个类用于组装不同的现有数据集。
    """

    @staticmethod
    def collate_fn(batch):
        """将数据样本整理成批次。"""
        return YOLODataset.collate_fn(batch)


# TODO: 支持语义分割
class SemanticDataset(BaseDataset):
    """
    语义分割数据集。

    这个类负责处理用于语义分割任务的数据集。它继承了BaseDataset类的功能。

    注意：
        目前这个类是一个占位符，需要填充方法和属性以支持语义分割任务。
    """

    def __init__(self):
        """初始化一个SemanticDataset对象。"""
        super().__init__()


class ClassificationDataset:
    """
    扩展了torchvision的ImageFolder，以支持YOLO分类任务，提供图像增强、缓存和验证等功能。
    该类旨在高效处理大规模数据集，用于训练深度学习模型，并可选择性地进行图像变换和缓存机制，以加速训练。

    此类支持使用torchvision和Albumentations库进行增强，并支持将图像缓存到RAM或磁盘上，以减少训练过程中的IO开销。
    此外，它还实现了一个强大的验证过程，以确保数据的完整性和一致性。

    属性：
        cache_ram (bool): 指示是否启用RAM缓存。
        cache_disk (bool): 指示是否启用磁盘缓存。
        samples (list): 一个包含元组的列表，每个元组包含图像路径、其类别索引、其.np文件的路径
                        （如果使用磁盘缓存），以及可选的已加载的图像数组（如果使用RAM缓存）。
        torch_transforms (callable): 应用于图像的PyTorch变换函数。
    """

    def __init__(self, root, args, augment=False, prefix=""):
        """
        使用根目录、图像大小、增强和缓存设置初始化YOLO对象。

        参数：
            root (str): 数据集目录路径，其中图像以类别特定的文件夹结构存储。
            args (Namespace): 包含数据集相关设置的配置，如图像大小、增强参数和缓存设置。它包括属性如`imgsz`（图像大小）、
                `fraction`（使用的数据比例）、`scale`、`fliplr`、`flipud`、`cache`（用于更快训练的磁盘或RAM缓存）、`auto_augment`、
                `hsv_h`、`hsv_s`、`hsv_v`和`crop_fraction`等。
            augment (bool, 可选): 是否对数据集进行增强。默认为False。
            prefix (str, 可选): 日志和缓存文件名的前缀，帮助数据集的识别和调试。默认为空字符串。
        """
        import torchvision  # 用于加速'import ultralytics'

        # 将基类作为属性分配，而不是用作基类，以便在导入torchvision时进行作用域控制
        if TORCHVISION_0_18:  # 'allow_empty'参数首次在torchvision 0.18中引入
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        # 初始化属性
        if augment and args.fraction < 1.0:  # 减少训练数据比例
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # 将图像缓存到RAM中
        if self.cache_ram:
            LOGGER.warning(
                "WARNING ⚠️ 分类训练 `cache_ram` 在 "
                "https://github.com/ultralytics/ultralytics/issues/9824 中有已知内存泄漏问题，正在将 `cache_ram=False`。"
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"  # 将图像缓存到硬盘上，作为未压缩的*.npy文件
        self.samples = self.verify_images()  # 过滤掉坏的图像
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # 文件、索引、npy、图像
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz, crop_fraction=args.crop_fraction)
        )

    def __getitem__(self, i):
        """返回与给定索引对应的数据和目标子集。"""
        f, j, fn, im = self.samples[i]  # 文件名、索引、文件名.with_suffix('.npy')、图像
        if self.cache_ram:
            if im is None:  # 注意：这里需要两个单独的if语句，不能将其与前面的行合并
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # 加载npy文件
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # 读取图像
            im = cv2.imread(f)  # BGR
        # 将NumPy数组转换为PIL图像
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """返回数据集中的样本总数。"""
        return len(self.samples)

    def verify_images(self):
        """验证数据集中的所有图像。"""
        desc = f"{self.prefix}扫描 {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache文件路径

        try:
            cache = load_dataset_cache_file(path)  # 尝试加载*.cache文件
            assert cache["version"] == DATASET_CACHE_VERSION  # 与当前版本匹配
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # 哈希值相同
            nf, nc, n, samples = cache.pop("results")  # 找到、缺失、空白、损坏、总数
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} 图像, {nc} 损坏"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # 显示警告信息
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # 如果无法检索*.cache，则运行扫描
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} 图像, {nc} 损坏"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # 警告信息
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples
