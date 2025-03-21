# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM


class BaseDataset(Dataset):
    """
    基础数据集类，用于加载和处理图像数据。

    参数：
        img_path (str): 图像所在文件夹的路径。
        imgsz (int, optional): 图像大小。默认为640。
        cache (bool, optional): 在训练过程中将图像缓存到内存或磁盘。默认为False。
        augment (bool, optional): 如果为True，则应用数据增强。默认为True。
        hyp (dict, optional): 用于数据增强的超参数。默认为None。
        prefix (str, optional): 用于打印日志信息的前缀。默认为''。
        rect (bool, optional): 如果为True，则使用矩形训练。默认为False。
        batch_size (int, optional): 批量大小。默认为None。
        stride (int, optional): 步幅。默认为32。
        pad (float, optional): 填充。默认为0.0。
        single_cls (bool, optional): 如果为True，则使用单类别训练。默认为False。
        classes (list): 包含的类别列表。默认为None。
        fraction (float): 要使用的数据集的比例。默认为1.0（使用所有数据）。

    属性：
        im_files (list): 图像文件路径列表。
        labels (list): 标签数据字典列表。
        ni (int): 数据集中的图像数量。
        ims (list): 加载的图像列表。
        npy_files (list): numpy文件路径列表。
        transforms (callable): 图像转换函数。
    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
    ):
        """根据给定的配置和选项初始化BaseDataset。"""
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls 和 include_class
        self.ni = len(self.labels)  # 图像数量
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # 为mosaic图像缓冲线程
        self.buffer = []  # 缓冲区大小 = 批量大小
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # 缓存图像（选项：cache = True, False, None, "ram", "disk"）
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "警告 ⚠️ cache='ram' 可能会产生非确定性的训练结果。如果你的磁盘空间允许，考虑使用 cache='disk' 作为确定性替代方案。"
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # 转换
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        """读取图像文件。"""
        try:
            f = []  # 图像文件
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # 操作系统无关的路径处理
                if p.is_dir():  # 目录
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                elif p.is_file():  # 文件
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # 从本地路径转为全局路径
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} 不存在")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            assert im_files, f"{self.prefix}在 {img_path} 中未找到图像。{FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}加载数据时出错 {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # 只保留数据集的一个子集
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """更新标签，只包含这些类别（可选）。"""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i, rect_mode=True):
        """从数据集索引 'i' 加载1张图像，返回 (im, resized hw)。"""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # 如果没有缓存到内存
            if fn.exists():  # 加载.npy文件
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}警告 ⚠️ 移除损坏的 *.npy 图像文件 {fn}，原因：{e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)  # 读取BGR图像
            else:  # 读取图像
                im = cv2.imread(f)  # 读取BGR图像
            if im is None:
                raise FileNotFoundError(f"未找到图像 {f}")

            h0, w0 = im.shape[:2]  # 原始图像大小
            if rect_mode:  # 根据长边调整大小，保持宽高比
                r = self.imgsz / max(h0, w0)  # 比例
                if r != 1:  # 如果图像尺寸不相等
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # 如果不是正方形，则将图像拉伸为正方形
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # 在训练时使用数据增强，则将图像添加到缓冲区
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # 防止缓冲区为空
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self):
        """将图像缓存到内存或磁盘中。"""
        b, gb = 0, 1 << 30  # 缓存图像的字节数，每GB的字节数
        fcn, storage = (self.cache_images_to_disk, "磁盘") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}正在缓存图像 ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i):
        """将图像保存为 *.npy 文件，以加速加载。"""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)

    def check_cache_disk(self, safety_margin=0.5):
        """检查图像缓存需求与可用磁盘空间的关系。"""
        import shutil

        b, gb = 0, 1 << 30  # 缓存图像的字节数，每GB的字节数
        n = min(self.ni, 30)  # 从30个随机图像推算
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = cv2.imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.info(f"{self.prefix}跳过图像缓存到磁盘，因为目录不可写 ⚠️")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # 缓存数据集到磁盘所需的字节数
        total, used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{disk_required / gb:.1f}GB磁盘空间所需，"
                f"安全余量{int(safety_margin * 100)}%，但只剩下{free / gb:.1f}/{total / gb:.1f}GB空闲空间，"
                f"不缓存图像到磁盘 ⚠️"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin=0.5):
        """检查图像缓存需求与可用内存的关系。"""
        b, gb = 0, 1 << 30  # 缓存图像的字节数，每GB的字节数
        n = min(self.ni, 30)  # 从30个随机图像推算
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # 随机选择图像
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # 比例
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)  # 缓存数据集到内存所需的GB
        mem = psutil.virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{mem_required / gb:.1f}GB内存所需缓存图像，"
                f"安全余量{int(safety_margin * 100)}%，但只有{mem.available / gb:.1f}/{mem.total / gb:.1f}GB可用，"
                f"不缓存图像 ⚠️"
            )
            return False
        return True

    def set_rectangle(self):
        """设置YOLO检测的矩形边界框形状。"""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # 批量索引
        nb = bi[-1] + 1  # 批量数量

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # 宽高比
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # 设置训练图像的形状
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # 图像的批量索引

    def __getitem__(self, index):
        """返回给定索引的转换标签信息。"""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """从数据集获取并返回标签信息。"""
        label = deepcopy(self.labels[index])  # 需要深拷贝
        label.pop("shape", None)  # rect用的shape，删除
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # 用于评估
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """返回数据集标签列表的长度。"""
        return len(self.labels)

    def update_labels_info(self, label):
        """自定义标签格式化。"""
        return label

    def build_transforms(self, hyp=None):
        """
        用户可以在这里自定义数据增强。

        示例：
            ```python
            if self.augment:
                # 训练时的转换
                return Compose([])
            else:
                # 验证时的转换
                return Compose([])
            ```
        """
        raise NotImplementedError

    def get_labels(self):
        """
        用户可以在这里自定义标签格式。

        注意：
            确保输出是一个包含以下键的字典：
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # 格式: (height, width)
                cls=cls,
                bboxes=bboxes,  # xywh
                segments=segments,  # xy
                keypoints=keypoints,  # xy
                normalized=True,  # 或 False
                bbox_format="xyxy",  # 或 xywh, ltwh
            )
            ```
        """
        raise NotImplementedError
