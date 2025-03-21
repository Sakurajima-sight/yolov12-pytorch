# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import hashlib
import json
import os
import random
import subprocess
import time
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile

import cv2
import numpy as np
from PIL import Image, ImageOps

from ultralytics.nn.autobackend import check_class_names
from ultralytics.utils import (
    DATASETS_DIR,
    LOGGER,
    NUM_THREADS,
    ROOT,
    SETTINGS_FILE,
    TQDM,
    clean_url,
    colorstr,
    emojis,
    is_dir_writeable,
    yaml_load,
    yaml_save,
)
from ultralytics.utils.checks import check_file, check_font, is_ascii
from ultralytics.utils.downloads import download, safe_download, unzip_file
from ultralytics.utils.ops import segments2boxes

HELP_URL = "请参阅 https://docs.ultralytics.com/datasets 获取数据集格式化指导。"
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # 图像后缀
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # 视频后缀
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # 数据加载器的全局 pin_memory 设置
FORMATS_HELP_MSG = f"支持的格式有：\n图像: {IMG_FORMATS}\n视频: {VID_FORMATS}"


def img2label_paths(img_paths):
    """根据图像路径定义标签路径。"""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/ 和 /labels/ 子字符串
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def get_hash(paths):
    """返回一组路径（文件或目录）的单个哈希值。"""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # 计算文件大小
    h = hashlib.sha256(str(size).encode())  # 对文件大小进行哈希
    h.update("".join(paths).encode())  # 对路径进行哈希
    return h.hexdigest()  # 返回哈希值


def exif_size(img: Image.Image):
    """返回经过 EXIF 修正后的 PIL 图像大小。"""
    s = img.size  # (宽度, 高度)
    if img.format == "JPEG":  # 仅支持 JPEG 图像
        try:
            if exif := img.getexif():
                rotation = exif.get(274, None)  # EXIF 中旋转标签的键是 274
                if rotation in {6, 8}:  # 旋转 270 或 90 度
                    s = s[1], s[0]
        except Exception:
            pass
    return s


def verify_image(args):
    """验证单张图像。"""
    (im_file, cls), prefix = args
    # 数量（已找到，损坏），消息
    nf, nc, msg = 0, 0, ""
    try:
        im = Image.open(im_file)
        im.verify()  # PIL验证
        shape = exif_size(im)  # 图像尺寸
        shape = (shape[1], shape[0])  # 高宽
        assert (shape[0] > 9) & (shape[1] > 9), f"图像尺寸 {shape} 小于10像素"
        assert im.format.lower() in IMG_FORMATS, f"无效的图像格式 {im.format}。 {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # 损坏的JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}警告 ⚠️ {im_file}: 损坏的JPEG已恢复并保存"
        nf = 1
    except Exception as e:
        nc = 1
        msg = f"{prefix}警告 ⚠️ {im_file}: 忽略损坏的图像/标签: {e}"
    return (im_file, cls), nf, nc, msg


def verify_image_label(args):
    """验证单张图像-标签对。"""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
    # 数量（缺失，已找到，空，损坏），消息，分割，关键点
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # 验证图像
        im = Image.open(im_file)
        im.verify()  # PIL验证
        shape = exif_size(im)  # 图像尺寸
        shape = (shape[1], shape[0])  # 高宽
        assert (shape[0] > 9) & (shape[1] > 9), f"图像尺寸 {shape} 小于10像素"
        assert im.format.lower() in IMG_FORMATS, f"无效的图像格式 {im.format}。 {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # 损坏的JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}警告 ⚠️ {im_file}: 损坏的JPEG已恢复并保存"

        # 验证标签
        if os.path.isfile(lb_file):
            nf = 1  # 标签已找到
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):  # 是分割
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (类，xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (类，xywh)
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f"标签需要 {(5 + nkpt * ndim)} 列"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == 5, f"标签需要5列，检测到 {lb.shape[1]} 列"
                    points = lb[:, 1:]
                assert points.max() <= 1, f"非归一化或超出范围的坐标 {points[points > 1]}"
                assert lb.min() >= 0, f"负标签值 {lb[lb < 0]}"

                # 所有标签
                max_cls = lb[:, 0].max()  # 最大标签数
                assert max_cls <= num_cls, (
                    f"标签类别 {int(max_cls)} 超出数据集类别数 {num_cls}。"
                    f"可能的类别标签是 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # 重复行检查
                    lb = lb[i]  # 移除重复项
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}警告 ⚠️ {im_file}: {nl - len(i)} 个重复标签已移除"
            else:
                ne = 1  # 标签为空
                lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
        else:
            nm = 1  # 标签缺失
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}警告 ⚠️ {im_file}: 忽略损坏的图像/标签: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def visualize_image_annotations(image_path, txt_path, label_map):
    """
    在图像上可视化YOLO标注（边界框和类别标签）。

    该函数读取图像及其对应的YOLO格式标注文件，然后
    绘制检测到的物体的边界框，并用相应的类别名称标注它们。
    边界框的颜色根据类别ID分配，文本颜色根据背景色的亮度动态调整，
    以确保可读性。

    参数：
        image_path (str): 要标注的图像文件路径，可以是PIL支持的格式（例如，.jpg，.png）。
        txt_path (str): YOLO格式的标注文件路径，其中每个物体应包含一行，格式为：
                        - class_id (int): 类别索引。
                        - x_center (float): 边界框的X中心（相对于图像宽度）。
                        - y_center (float): 边界框的Y中心（相对于图像高度）。
                        - width (float): 边界框的宽度（相对于图像宽度）。
                        - height (float): 边界框的高度（相对于图像高度）。
        label_map (dict): 一个字典，将类别ID（整数）映射到类别标签（字符串）。

    示例：
        >>> label_map = {0: "cat", 1: "dog", 2: "bird"}  # 应包括所有标注类别的详细信息
        >>> visualize_image_annotations("path/to/image.jpg", "path/to/annotations.txt", label_map)
    """
    import matplotlib.pyplot as plt

    from ultralytics.utils.plotting import colors

    img = np.array(Image.open(image_path))
    img_height, img_width = img.shape[:2]
    annotations = []
    with open(txt_path) as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            x = (x_center - width / 2) * img_width
            y = (y_center - height / 2) * img_height
            w = width * img_width
            h = height * img_height
            annotations.append((x, y, w, h, int(class_id)))
    fig, ax = plt.subplots(1)  # 绘制图像和标注
    for x, y, w, h, label in annotations:
        color = tuple(c / 255 for c in colors(label, True))  # 获取并规范化RGB颜色
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")  # 创建矩形框
        ax.add_patch(rect)
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]  # 计算亮度的公式
        ax.text(x, y - 5, label_map[label], color="white" if luminance < 0.5 else "black", backgroundcolor=color)
    ax.imshow(img)
    plt.show()


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    将一组多边形转换为指定图像大小的二值掩码。

    参数：
        imgsz (tuple): 图像的大小，格式为（高度，宽度）。
        polygons (list[np.ndarray]): 多边形列表。每个多边形是一个形状为 [N, M] 的数组，
                                     其中 N 是多边形的数量，M 是点的数量，且 M % 2 = 0。
        color (int, 可选): 用于填充多边形的颜色值。默认值为1。
        downsample_ratio (int, 可选): 下采样比例因子。默认值为1。

    返回：
        (np.ndarray): 一个二值掩码，大小为指定图像尺寸，且多边形已被填充。
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # 注意：先fillPoly然后再resize，目的是保持与mask-ratio=1时相同的损失计算方法
    return cv2.resize(mask, (nw, nh))


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    将一组多边形转换为指定图像大小的二进制掩膜。

    参数：
        imgsz (tuple): 图像的大小，格式为 (高度, 宽度)。
        polygons (list[np.ndarray]): 多边形列表。每个多边形是一个形状为 [N, M] 的数组，其中
                                     N 是多边形的数量，M 是点的数量，且 M 必须为偶数。
        color (int): 用于填充多边形的颜色值。
        downsample_ratio (int, 可选): 每个掩膜的下采样因子。默认为 1。

    返回：
        (np.ndarray): 一个二进制掩膜集合，图像大小已指定，并且多边形已填充。
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """返回一个 (640, 640) 的重叠掩膜。"""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask.astype(masks.dtype))
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def find_dataset_yaml(path: Path) -> Path:
    """
    查找并返回与 Detect、Segment 或 Pose 数据集关联的 YAML 文件。

    该函数首先在提供的目录的根目录查找 YAML 文件，如果未找到，则进行递归搜索。
    它优先选择与提供的路径具有相同基名的 YAML 文件。如果未找到 YAML 文件或找到多个 YAML 文件，
    将引发 AssertionError。

    参数：
        path (Path): 查找 YAML 文件的目录路径。

    返回：
        (Path): 找到的 YAML 文件的路径。
    """
    files = list(path.glob("*.yaml")) or list(path.rglob("*.yaml"))  # 首先尝试根目录，然后递归查找
    assert files, f"在 '{path.resolve()}' 中未找到 YAML 文件"
    if len(files) > 1:
        files = [f for f in files if f.stem == path.stem]  # 优先选择匹配的 *.yaml 文件
    assert len(files) == 1, f"在 '{path.resolve()}' 中期望找到 1 个 YAML 文件，但找到了 {len(files)} 个。\n{files}"
    return files[0]


def check_det_dataset(dataset, autodownload=True):
    """
    如果数据集在本地未找到，则下载、验证和/或解压数据集。

    该函数检查指定数据集的可用性，如果未找到，则提供下载和解压数据集的选项。
    然后读取并解析附带的 YAML 数据，确保满足关键要求，并解析与数据集相关的路径。

    参数：
        dataset (str): 数据集路径或数据集描述符（如 YAML 文件）。
        autodownload (bool, 可选): 如果未找到数据集，是否自动下载。默认为 True。

    返回：
        (dict): 解析后的数据集信息和路径。
    """
    file = check_file(dataset)

    # 下载（可选）
    extract_dir = ""
    if zipfile.is_zipfile(file) or is_tarfile(file):
        new_dir = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)
        file = find_dataset_yaml(DATASETS_DIR / new_dir)
        extract_dir, autodownload = file.parent, False

    # 读取 YAML
    data = yaml_load(file, append_filename=True)  # 字典形式

    # 检查
    for k in "train", "val":
        if k not in data:
            if k != "val" or "validation" not in data:
                raise SyntaxError(
                    emojis(f"{dataset} 缺少 '{k}:' 键 ❌.\n所有数据 YAML 文件都需要 'train' 和 'val' 键。")
                )
            LOGGER.info("警告 ⚠️ 将数据 YAML 文件中的 'validation' 键重命名为 'val' 以匹配 YOLO 格式。")
            data["val"] = data.pop("validation")  # 用 'val' 键替换 'validation' 键
    if "names" not in data and "nc" not in data:
        raise SyntaxError(emojis(f"{dataset} 缺少 'names' 或 'nc' 键 ❌.\n所有数据 YAML 文件都需要 'names' 或 'nc' 键。"))
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
        raise SyntaxError(emojis(f"{dataset} 'names' 长度 {len(data['names'])} 和 'nc: {data['nc']}' 必须匹配。"))
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    else:
        data["nc"] = len(data["names"])

    data["names"] = check_class_names(data["names"])

    # 解析路径
    path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)  # 数据集根目录
    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()

    # 设置路径
    data["path"] = path  # 下载脚本
    for k in "train", "val", "test", "minival":
        if data.get(k):  # 预先添加路径
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # 解析 YAML
    val, s = (data.get(x) for x in ("val", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val 路径
        if not all(x.exists() for x in val):
            name = clean_url(dataset)  # 去掉 URL 身份验证的 dataset 名称
            m = f"\n数据集 '{name}' 图像未找到 ⚠️，缺少路径 '{[x for x in val if not x.exists()][0]}'"
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\n注意：数据集下载目录为 '{DATASETS_DIR}'。您可以在 '{SETTINGS_FILE}' 中更新此路径"
                raise FileNotFoundError(m)
            t = time.time()
            r = None  # 成功
            if s.startswith("http") and s.endswith(".zip"):  # URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
            elif s.startswith("bash "):  # bash 脚本
                LOGGER.info(f"正在运行 {s} ...")
                r = os.system(s)
            else:  # python 脚本
                exec(s, {"yaml": data})
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"成功 ✅ {dt}, 保存到 {colorstr('bold', DATASETS_DIR)}" if r in {0, None} else f"失败 {dt} ❌"
            LOGGER.info(f"数据集下载 {s}\n")
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf")  # 下载字体

    return data  # 字典


def check_cls_dataset(dataset, split=""):
    """
    检查分类数据集，如 Imagenet。

    该函数接受一个 `dataset` 名称，并尝试检索相应的数据集信息。
    如果数据集在本地未找到，它会尝试从互联网下载并保存到本地。

    参数：
        dataset (str | Path): 数据集的名称。
        split (str, 可选): 数据集的拆分，值可以是 'val'、'test' 或 ''。默认为 ''。

    返回：
        (dict): 一个包含以下键的字典：
            - 'train' (Path): 包含训练集的目录路径。
            - 'val' (Path): 包含验证集的目录路径。
            - 'test' (Path): 包含测试集的目录路径。
            - 'nc' (int): 数据集中的类别数。
            - 'names' (dict): 数据集中的类别名称字典。
    """
    # 下载（如果 dataset=https://file.zip 被直接传递）
    if str(dataset).startswith(("http:/", "https:/")):
        dataset = safe_download(dataset, dir=DATASETS_DIR, unzip=True, delete=False)
    elif Path(dataset).suffix in {".zip", ".tar", ".gz"}:
        file = check_file(dataset)
        dataset = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)

    dataset = Path(dataset)
    data_dir = (dataset if dataset.is_dir() else (DATASETS_DIR / dataset)).resolve()
    if not data_dir.is_dir():
        LOGGER.warning(f"\n未找到数据集 ⚠️，缺少路径 {data_dir}，正在尝试下载...")
        t = time.time()
        if str(dataset) == "imagenet":
            subprocess.run(f"bash {ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)
        else:
            url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{dataset}.zip"
            download(url, dir=data_dir.parent)
        s = f"数据集下载成功 ✅ ({time.time() - t:.1f}s)，保存至 {colorstr('bold', data_dir)}\n"
        LOGGER.info(s)
    train_set = data_dir / "train"
    val_set = (
        data_dir / "val"
        if (data_dir / "val").exists()
        else data_dir / "validation"
        if (data_dir / "validation").exists()
        else None
    )  # data/test 或 data/val
    test_set = data_dir / "test" if (data_dir / "test").exists() else None  # data/val 或 data/test
    if split == "val" and not val_set:
        LOGGER.warning("警告 ⚠️ 未找到数据集 'split=val'，改为使用 'split=test'。")
    elif split == "test" and not test_set:
        LOGGER.warning("警告 ⚠️ 未找到数据集 'split=test'，改为使用 'split=val'。")

    nc = len([x for x in (data_dir / "train").glob("*") if x.is_dir()])  # 类别数量
    names = [x.name for x in (data_dir / "train").iterdir() if x.is_dir()]  # 类别名称列表
    names = dict(enumerate(sorted(names)))

    # 打印到控制台
    for k, v in {"train": train_set, "val": val_set, "test": test_set}.items():
        prefix = f"{colorstr(f'{k}:')} {v}..."
        if v is None:
            LOGGER.info(prefix)
        else:
            files = [path for path in v.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]
            nf = len(files)  # 文件数量
            nd = len({file.parent for file in files})  # 目录数量
            if nf == 0:
                if k == "train":
                    raise FileNotFoundError(emojis(f"{dataset} '{k}:' 没有找到训练图像 ❌ "))
                else:
                    LOGGER.warning(f"{prefix} 在 {nd} 类中找到 {nf} 张图像: 警告 ⚠️ 没有找到图像")
            elif nd != nc:
                LOGGER.warning(f"{prefix} 在 {nd} 类中找到 {nf} 张图像: 错误 ❌️ 需要 {nc} 类，而不是 {nd}")
            else:
                LOGGER.info(f"{prefix} 在 {nd} 类中找到 {nf} 张图像 ✅ ")

    return {"train": train_set, "val": val_set, "test": test_set, "nc": nc, "names": names}


class HUBDatasetStats:
    """
    用于生成 HUB 数据集 JSON 和 `-hub` 数据集目录的类。

    参数：
        path (str): data.yaml 或 data.zip 的路径（data.zip 内包含 data.yaml）。默认为 'coco8.yaml'。
        task (str): 数据集任务。选项有 'detect'、'segment'、'pose'、'classify'。默认为 'detect'。
        autodownload (bool): 如果数据集未在本地找到，是否尝试下载数据集。默认为 False。

    示例：
        从 https://github.com/ultralytics/hub/tree/main/example_datasets 下载 *.zip 文件
            例如 https://github.com/ultralytics/hub/raw/main/example_datasets/coco8.zip 用于 coco8.zip。
        ```python
        from ultralytics.data.utils import HUBDatasetStats

        stats = HUBDatasetStats("path/to/coco8.zip", task="detect")  # detect 数据集
        stats = HUBDatasetStats("path/to/coco8-seg.zip", task="segment")  # segment 数据集
        stats = HUBDatasetStats("path/to/coco8-pose.zip", task="pose")  # pose 数据集
        stats = HUBDatasetStats("path/to/dota8.zip", task="obb")  # OBB 数据集
        stats = HUBDatasetStats("path/to/imagenet10.zip", task="classify")  # 分类数据集

        stats.get_json(save=True)
        stats.process_images()
        ```
    """

    def __init__(self, path="coco8.yaml", task="detect", autodownload=False):
        """初始化类。"""
        path = Path(path).resolve()
        LOGGER.info(f"开始进行 {path} 的 HUB 数据集检查....")

        self.task = task  # detect, segment, pose, classify, obb
        if self.task == "classify":
            unzip_dir = unzip_file(path)
            data = check_cls_dataset(unzip_dir)
            data["path"] = unzip_dir
        else:  # detect, segment, pose, obb
            _, data_dir, yaml_path = self._unzip(Path(path))
            try:
                # 载入带检查的 YAML 文件
                data = yaml_load(yaml_path)
                data["path"] = ""  # 去除路径，因为 YAML 应该位于数据集根目录
                yaml_save(yaml_path, data)
                data = check_det_dataset(yaml_path, autodownload)  # dict
                data["path"] = data_dir  # YAML 路径应设置为空（相对路径）或父路径（绝对路径）
            except Exception as e:
                raise Exception("错误/HUB/dataset_stats/init") from e

        self.hub_dir = Path(f"{data['path']}-hub")
        self.im_dir = self.hub_dir / "images"
        self.stats = {"nc": len(data["names"]), "names": list(data["names"].values())}  # 统计字典
        self.data = data

    @staticmethod
    def _unzip(path):
        """解压 data.zip 文件。"""
        if not str(path).endswith(".zip"):  # 如果路径是 data.yaml
            return False, None, path
        unzip_dir = unzip_file(path, path=path.parent)
        assert unzip_dir.is_dir(), (
            f"解压 {path} 时出错，未找到 {unzip_dir}。path/to/abc.zip 必须解压到 path/to/abc/"
        )
        return True, str(unzip_dir), find_dataset_yaml(unzip_dir)  # 解压后的路径，数据目录，yaml 文件路径

    def _hub_ops(self, f):
        """为 HUB 预览保存压缩后的图像。"""
        compress_one_image(f, self.im_dir / Path(f).name)  # 保存到 dataset-hub

    def get_json(self, save=False, verbose=False):
        """返回用于 Ultralytics HUB 的数据集 JSON。"""

        def _round(labels):
            """更新标签为整数类和四位小数的浮点数。"""
            if self.task == "detect":
                coordinates = labels["bboxes"]
            elif self.task in {"segment", "obb"}:  # 分割和 OBB 使用 segments。OBB segments 是归一化的 xyxyxyxy
                coordinates = [x.flatten() for x in labels["segments"]]
            elif self.task == "pose":
                n, nk, nd = labels["keypoints"].shape
                coordinates = np.concatenate((labels["bboxes"], labels["keypoints"].reshape(n, nk * nd)), 1)
            else:
                raise ValueError(f"未定义的数据集任务={self.task}。")
            zipped = zip(labels["cls"], coordinates)
            return [[int(c[0]), *(round(float(x), 4) for x in points)] for c, points in zipped]

        for split in "train", "val", "test":
            self.stats[split] = None  # 预定义
            path = self.data.get(split)

            # 检查拆分数据集
            if path is None:  # 如果没有该拆分
                continue
            files = [f for f in Path(path).rglob("*.*") if f.suffix[1:].lower() in IMG_FORMATS]  # 获取拆分数据集中的图像文件
            if not files:  # 没有图像文件
                continue

            # 获取数据集统计信息
            if self.task == "classify":
                from torchvision.datasets import ImageFolder  # 用于加速 'import ultralytics'

                dataset = ImageFolder(self.data[split])

                x = np.zeros(len(dataset.classes)).astype(int)
                for im in dataset.imgs:
                    x[im[1]] += 1

                self.stats[split] = {
                    "instance_stats": {"total": len(dataset), "per_class": x.tolist()},
                    "image_stats": {"total": len(dataset), "unlabelled": 0, "per_class": x.tolist()},
                    "labels": [{Path(k).name: v} for k, v in dataset.imgs],
                }
            else:
                from ultralytics.data import YOLODataset

                dataset = YOLODataset(img_path=self.data[split], data=self.data, task=self.task)
                x = np.array(
                    [
                        np.bincount(label["cls"].astype(int).flatten(), minlength=self.data["nc"])
                        for label in TQDM(dataset.labels, total=len(dataset), desc="Statistics")
                    ]
                )  # 形状(128x80)
                self.stats[split] = {
                    "instance_stats": {"total": int(x.sum()), "per_class": x.sum(0).tolist()},
                    "image_stats": {
                        "total": len(dataset),
                        "unlabelled": int(np.all(x == 0, 1).sum()),
                        "per_class": (x > 0).sum(0).tolist(),
                    },
                    "labels": [{Path(k).name: _round(v)} for k, v in zip(dataset.im_files, dataset.labels)],
                }

        # 保存、打印并返回
        if save:
            self.hub_dir.mkdir(parents=True, exist_ok=True)  # 创建 dataset-hub/
            stats_path = self.hub_dir / "stats.json"
            LOGGER.info(f"保存 {stats_path.resolve()}...")
            with open(stats_path, "w") as f:
                json.dump(self.stats, f)  # 保存 stats.json
        if verbose:
            LOGGER.info(json.dumps(self.stats, indent=2, sort_keys=False))
        return self.stats

    def process_images(self):
        """为 Ultralytics HUB 压缩图像。"""
        from ultralytics.data import YOLODataset  # 分类数据集

        self.im_dir.mkdir(parents=True, exist_ok=True)  # 创建 dataset-hub/images/
        for split in "train", "val", "test":
            if self.data.get(split) is None:
                continue
            dataset = YOLODataset(img_path=self.data[split], data=self.data)
            with ThreadPool(NUM_THREADS) as pool:
                for _ in TQDM(pool.imap(self._hub_ops, dataset.im_files), total=len(dataset), desc=f"{split} images"):
                    pass
        LOGGER.info(f"完成。所有图像已保存到 {self.im_dir}")
        return self.im_dir


def compress_one_image(f, f_new=None, max_dim=1920, quality=50):
    """
    压缩单张图像文件，减少文件大小，同时保持其宽高比和质量，使用Python Imaging Library (PIL) 或 OpenCV库。
    如果输入图像小于最大尺寸，则不会调整大小。

    参数：
        f (str): 输入图像文件的路径。
        f_new (str, 可选): 输出图像文件的路径。如果未指定，则会覆盖输入文件。
        max_dim (int, 可选): 输出图像的最大尺寸（宽度或高度）。默认值为1920像素。
        quality (int, 可选): 图像压缩质量的百分比。默认值为50%。

    示例：
        ```python
        from pathlib import Path
        from ultralytics.data.utils import compress_one_image

        for f in Path("path/to/dataset").rglob("*.jpg"):
            compress_one_image(f)
        ```
    """
    try:  # 使用PIL
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)  # 比例
        if r < 1.0:  # 图像太大
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(f_new or f, "JPEG", quality=quality, optimize=True)  # 保存
    except Exception as e:  # 使用OpenCV
        LOGGER.info(f"警告 ⚠️ HUB操作PIL失败 {f}: {e}")
        im = cv2.imread(f)
        im_height, im_width = im.shape[:2]
        r = max_dim / max(im_height, im_width)  # 比例
        if r < 1.0:  # 图像太大
            im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(f_new or f), im)


def autosplit(path=DATASETS_DIR / "coco8/images", weights=(0.9, 0.1, 0.0), annotated_only=False):
    """
    自动将数据集拆分为训练集、验证集和测试集，并将结果保存到autosplit_*.txt文件中。

    参数：
        path (Path, 可选): 图像目录路径。默认值为DATASETS_DIR / 'coco8/images'。
        weights (list | tuple, 可选): 训练集、验证集和测试集的拆分比例。默认值为(0.9, 0.1, 0.0)。
        annotated_only (bool, 可选): 如果为True，则仅使用具有关联txt文件的图像。默认值为False。

    示例：
        ```python
        from ultralytics.data.utils import autosplit

        autosplit()
        ```
    """
    path = Path(path)  # 图像目录
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)  # 仅图像文件
    n = len(files)  # 文件数量
    random.seed(0)  # 设置随机种子，以确保结果可复现
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # 将每个图像分配到一个拆分

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]  # 3个txt文件
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # 删除已存在的文件

    LOGGER.info(f"自动拆分图像来自 {path}" + ", 仅使用 *.txt 标注的图像" * annotated_only)
    for i, img in TQDM(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # 检查标签
            with open(path.parent / txt[i], "a") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")  # 将图像添加到txt文件


def load_dataset_cache_file(path):
    """从路径加载Ultralytics *.cache字典。"""
    import gc

    gc.disable()  # 减少pickle加载时间 https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # 加载字典
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x, version):
    """将Ultralytics数据集*.cache字典x保存到路径。"""
    x["version"] = version  # 添加缓存版本
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # 如果存在，则删除*.cache文件
        np.save(str(path), x)  # 保存缓存以便下次使用
        path.with_suffix(".cache.npy").rename(path)  # 删除 .npy 后缀
        LOGGER.info(f"{prefix}已创建新缓存: {path}")
    else:
        LOGGER.warning(f"{prefix}警告 ⚠️ 缓存目录 {path.parent} 不可写，缓存未保存。")
