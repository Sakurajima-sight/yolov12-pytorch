# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import itertools
from glob import glob
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from ultralytics.data.utils import exif_size, img2label_paths
from ultralytics.utils.checks import check_requirements


def bbox_iof(polygon1, bbox2, eps=1e-6):
    """
    计算多边形和边界框之间的前景交集比例（IoF）。

    参数：
        polygon1 (np.ndarray): 多边形坐标，形状为 (n, 8)。
        bbox2 (np.ndarray): 边界框，形状为 (n, 4)。
        eps (float, 可选): 防止除以零的小值。默认为 1e-6。

    返回：
        (np.ndarray): IoF 分数，形状为 (n, 1) 或 (n, m)，如果 bbox2 是 (m, 4)。

    注意：
        多边形格式：[x1, y1, x2, y2, x3, y3, x4, y4]。
        边界框格式：[x_min, y_min, x_max, y_max]。
    """
    check_requirements("shapely")
    from shapely.geometry import Polygon

    polygon1 = polygon1.reshape(-1, 4, 2)
    lt_point = np.min(polygon1, axis=-2)  # 左上角
    rb_point = np.max(polygon1, axis=-2)  # 右下角
    bbox1 = np.concatenate([lt_point, rb_point], axis=-1)

    lt = np.maximum(bbox1[:, None, :2], bbox2[..., :2])
    rb = np.minimum(bbox1[:, None, 2:], bbox2[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf)
    h_overlaps = wh[..., 0] * wh[..., 1]

    left, top, right, bottom = (bbox2[..., i] for i in range(4))
    polygon2 = np.stack([left, top, right, top, right, bottom, left, bottom], axis=-1).reshape(-1, 4, 2)

    sg_polys1 = [Polygon(p) for p in polygon1]
    sg_polys2 = [Polygon(p) for p in polygon2]
    overlaps = np.zeros(h_overlaps.shape)
    for p in zip(*np.nonzero(h_overlaps)):
        overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area
    unions = np.array([p.area for p in sg_polys1], dtype=np.float32)
    unions = unions[..., None]

    unions = np.clip(unions, eps, np.inf)
    outputs = overlaps / unions
    if outputs.ndim == 1:
        outputs = outputs[..., None]
    return outputs


def load_yolo_dota(data_root, split="train"):
    """
    加载 DOTA 数据集。

    参数：
        data_root (str): 数据根目录。
        split (str): 数据集的拆分，可以是 `train` 或 `val`。

    注意：
        假设 DOTA 数据集的目录结构如下：
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    assert split in {"train", "val"}, f"拆分必须是 'train' 或 'val'，而不是 {split}。"
    im_dir = Path(data_root) / "images" / split
    assert im_dir.exists(), f"找不到 {im_dir}，请检查数据根目录。"
    im_files = glob(str(Path(data_root) / "images" / split / "*"))
    lb_files = img2label_paths(im_files)
    annos = []
    for im_file, lb_file in zip(im_files, lb_files):
        w, h = exif_size(Image.open(im_file))
        with open(lb_file) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            lb = np.array(lb, dtype=np.float32)
        annos.append(dict(ori_size=(h, w), label=lb, filepath=im_file))
    return annos


def get_windows(im_size, crop_sizes=(1024,), gaps=(200,), im_rate_thr=0.6, eps=0.01):
    """
    获取窗口的坐标。

    参数：
        im_size (tuple): 原始图像的大小，(h, w)。
        crop_sizes (List(int)): 窗口的裁剪大小。
        gaps (List(int)): 裁剪之间的间隔。
        im_rate_thr (float): 窗口面积与图像面积之比的阈值。
        eps (float): 数学运算中的小值 epsilon。
    """
    h, w = im_size
    windows = []
    for crop_size, gap in zip(crop_sizes, gaps):
        assert crop_size > gap, f"无效的裁剪大小与间隔组合 [{crop_size} {gap}]"
        step = crop_size - gap

        xn = 1 if w <= crop_size else ceil((w - crop_size) / step + 1)
        xs = [step * i for i in range(xn)]
        if len(xs) > 1 and xs[-1] + crop_size > w:
            xs[-1] = w - crop_size

        yn = 1 if h <= crop_size else ceil((h - crop_size) / step + 1)
        ys = [step * i for i in range(yn)]
        if len(ys) > 1 and ys[-1] + crop_size > h:
            ys[-1] = h - crop_size

        start = np.array(list(itertools.product(xs, ys)), dtype=np.int64)
        stop = start + crop_size
        windows.append(np.concatenate([start, stop], axis=1))
    windows = np.concatenate(windows, axis=0)

    im_in_wins = windows.copy()
    im_in_wins[:, 0::2] = np.clip(im_in_wins[:, 0::2], 0, w)
    im_in_wins[:, 1::2] = np.clip(im_in_wins[:, 1::2], 0, h)
    im_areas = (im_in_wins[:, 2] - im_in_wins[:, 0]) * (im_in_wins[:, 3] - im_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * (windows[:, 3] - windows[:, 1])
    im_rates = im_areas / win_areas
    if not (im_rates > im_rate_thr).any():
        max_rate = im_rates.max()
        im_rates[abs(im_rates - max_rate) < eps] = 1
    return windows[im_rates > im_rate_thr]


def get_window_obj(anno, windows, iof_thr=0.7):
    """获取每个窗口的物体。"""
    h, w = anno["ori_size"]
    label = anno["label"]
    if len(label):
        label[:, 1::2] *= w
        label[:, 2::2] *= h
        iofs = bbox_iof(label[:, 1:], windows)
        # 非归一化且未对齐的坐标
        return [(label[iofs[:, i] >= iof_thr]) for i in range(len(windows))]  # window_anns
    else:
        return [np.zeros((0, 9), dtype=np.float32) for _ in range(len(windows))]  # window_anns


def crop_and_save(anno, windows, window_objs, im_dir, lb_dir, allow_background_images=True):
    """
    裁剪图像并保存新的标签。

    参数：
        anno (dict): 注释字典，包含 `filepath`、`label`、`ori_size` 作为其键。
        windows (list): 一个包含窗口坐标的列表。
        window_objs (list): 一个包含每个窗口内标签的列表。
        im_dir (str): 图像的输出目录路径。
        lb_dir (str): 标签的输出目录路径。
        allow_background_images (bool): 是否允许包括没有标签的背景图像。

    注意：
        假设DOTA数据集的目录结构：
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    im = cv2.imread(anno["filepath"])
    name = Path(anno["filepath"]).stem
    for i, window in enumerate(windows):
        x_start, y_start, x_stop, y_stop = window.tolist()
        new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
        patch_im = im[y_start:y_stop, x_start:x_stop]
        ph, pw = patch_im.shape[:2]

        label = window_objs[i]
        if len(label) or allow_background_images:
            cv2.imwrite(str(Path(im_dir) / f"{new_name}.jpg"), patch_im)
        if len(label):
            label[:, 1::2] -= x_start
            label[:, 2::2] -= y_start
            label[:, 1::2] /= pw
            label[:, 2::2] /= ph

            with open(Path(lb_dir) / f"{new_name}.txt", "w") as f:
                for lb in label:
                    formatted_coords = [f"{coord:.6g}" for coord in lb[1:]]
                    f.write(f"{int(lb[0])} {' '.join(formatted_coords)}\n")


def split_images_and_labels(data_root, save_dir, split="train", crop_sizes=(1024,), gaps=(200,)):
    """
    同时拆分图像和标签。

    注意：
        假设 DOTA 数据集的目录结构如下：
            - data_root
                - images
                    - split
                - labels
                    - split
        输出的目录结构如下：
            - save_dir
                - images
                    - split
                - labels
                    - split
    """
    im_dir = Path(save_dir) / "images" / split
    im_dir.mkdir(parents=True, exist_ok=True)
    lb_dir = Path(save_dir) / "labels" / split
    lb_dir.mkdir(parents=True, exist_ok=True)

    annos = load_yolo_dota(data_root, split=split)
    for anno in tqdm(annos, total=len(annos), desc=split):
        windows = get_windows(anno["ori_size"], crop_sizes, gaps)
        window_objs = get_window_obj(anno, windows)
        crop_and_save(anno, windows, window_objs, str(im_dir), str(lb_dir))


def split_trainval(data_root, save_dir, crop_size=1024, gap=200, rates=(1.0,)):
    """
    拆分 DOTA 数据集的训练集和验证集。

    注意：
        假设 DOTA 数据集的目录结构如下：
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
        输出的目录结构如下：
            - save_dir
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    for split in ["train", "val"]:
        split_images_and_labels(data_root, save_dir, split, crop_sizes, gaps)


def split_test(data_root, save_dir, crop_size=1024, gap=200, rates=(1.0,)):
    """
    拆分 DOTA 的测试集，标签不包含在此集合中。

    注意：
        假设 DOTA 数据集的目录结构如下：
            - data_root
                - images
                    - test
        输出的目录结构如下：
            - save_dir
                - images
                    - test
    """
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    save_dir = Path(save_dir) / "images" / "test"
    save_dir.mkdir(parents=True, exist_ok=True)

    im_dir = Path(data_root) / "images" / "test"
    assert im_dir.exists(), f"找不到 {im_dir}，请检查您的数据根目录。"
    im_files = glob(str(im_dir / "*"))
    for im_file in tqdm(im_files, total=len(im_files), desc="test"):
        w, h = exif_size(Image.open(im_file))
        windows = get_windows((h, w), crop_sizes=crop_sizes, gaps=gaps)
        im = cv2.imread(im_file)
        name = Path(im_file).stem
        for window in windows:
            x_start, y_start, x_stop, y_stop = window.tolist()
            new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
            patch_im = im[y_start:y_stop, x_start:x_stop]
            cv2.imwrite(str(save_dir / f"{new_name}.jpg"), patch_im)


if __name__ == "__main__":
    split_trainval(data_root="DOTAv2", save_dir="DOTAv2-split")
    split_test(data_root="DOTAv2", save_dir="DOTAv2-split")
