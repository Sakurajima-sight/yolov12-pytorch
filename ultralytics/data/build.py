# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed

from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, PIN_MEMORY, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file


class InfiniteDataLoader(dataloader.DataLoader):
    """
    无限数据加载器，重复使用工作线程。

    使用与普通DataLoader相同的语法。
    """

    def __init__(self, *args, **kwargs):
        """初始化一个无限循环回收工作线程的数据加载器，继承自DataLoader。"""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """返回批次采样器的长度。"""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """创建一个可以无限重复的采样器。"""
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        """确保工作线程被终止。"""
        if hasattr(self.iterator, "_workers"):
            for w in self.iterator._workers:  # 强制终止
                if w.is_alive():
                    w.terminate()
            self.iterator._shutdown_workers()  # 清理工作线程

    def reset(self):
        """
        重置迭代器。

        当我们在训练时想要修改数据集设置时，这个方法非常有用。
        """
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """
    一个永远重复的采样器。

    参数：
        sampler (Dataset.sampler): 要重复的采样器。
    """

    def __init__(self, sampler):
        """初始化一个永远重复给定采样器的对象。"""
        self.sampler = sampler

    def __iter__(self):
        """迭代'sampler'并不断输出其内容。"""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):  # noqa
    """设置dataloader工作线程的种子 https://pytorch.org/docs/stable/notes/randomness.html#dataloader。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """构建YOLO数据集。"""
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # 数据增强
        hyp=cfg,  # TODO: 可能需要添加一个从cfg获取超参数的函数
        rect=cfg.rect or rect,  # 是否使用矩形批次
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_grounding(cfg, img_path, json_file, batch, mode="train", rect=False, stride=32):
    """构建YOLO数据集。"""
    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # 数据增强
        hyp=cfg,  # TODO: 可能需要添加一个从cfg获取超参数的函数
        rect=cfg.rect or rect,  # 矩形批次
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """返回用于训练或验证集的InfiniteDataLoader或DataLoader。"""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # CUDA设备数量
    nw = min(os.cpu_count() // max(nd, 1), workers)  # 工作线程数量
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def check_source(source):
    """检查源类型并返回对应的标志值。"""
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # 对于本地USB摄像头，int类型
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
        is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source.lower() == "screen"
        if is_url and is_file:
            source = check_file(source)  # 下载
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # 将所有列表元素转换为PIL或np数组
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("不支持的图像类型。有关支持的类型，请参见 https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(source=None, batch=1, vid_stride=1, buffer=False):
    """
    加载用于目标检测的推理源，并应用必要的转换。

    参数：
        source (str, Path, Tensor, PIL.Image, np.ndarray): 用于推理的输入源。
        batch (int, 可选): 数据加载器的批次大小。默认为1。
        vid_stride (int, 可选): 视频源的帧间隔。默认为1。
        buffer (bool, 可选): 确定是否缓冲流帧。默认为False。

    返回：
        dataset (Dataset): 为指定输入源创建的数据集对象。
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # 数据加载器
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
    elif screenshot:
        dataset = LoadScreenshots(source)
    elif from_img:
        dataset = LoadPilAndNumpy(source)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride)

    # 将源类型附加到数据集
    setattr(dataset, "source_type", source_type)

    return dataset
