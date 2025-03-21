# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ultralytics.utils import DATASETS_DIR, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.downloads import download
from ultralytics.utils.files import increment_path


def coco91_to_coco80_class():
    """
    将91索引的COCO类别ID转换为80索引的COCO类别ID。

    返回：
        (list): 一个包含91个类别ID的列表，其中索引表示80索引类别ID，值为对应的91索引类别ID。
    """
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]


def coco80_to_coco91_class():
    r"""
    将80索引（val2014）转换为91索引（论文中的类别ID）。
    详情请见 https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/。

    示例：
        ```python
        import numpy as np

        a = np.loadtxt("data/coco.names", dtype="str", delimiter="\n")
        b = np.loadtxt("data/coco_paper.names", dtype="str", delimiter="\n")
        x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet转coco
        x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco转darknet
        ```
    """
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]


def convert_coco(
    labels_dir="../coco/annotations/",
    save_dir="coco_converted/",
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
    lvis=False,
):
    """
    将COCO数据集的注释转换为适用于训练YOLO模型的YOLO注释格式。

    参数：
        labels_dir (str, 可选): 包含COCO数据集注释文件的目录路径。
        save_dir (str, 可选): 用于保存结果的目录路径。
        use_segments (bool, 可选): 是否在输出中包含分割掩码。
        use_keypoints (bool, 可选): 是否在输出中包含关键点注释。
        cls91to80 (bool, 可选): 是否将91个COCO类别ID转换为对应的80个COCO类别ID。
        lvis (bool, 可选): 是否以lvis数据集的方式转换数据。

    示例：
        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco("../datasets/coco/annotations/", use_segments=True, use_keypoints=False, cls91to80=False)
        convert_coco(
            "../datasets/lvis/annotations/", use_segments=True, use_keypoints=False, cls91to80=False, lvis=True
        )
        ```

    输出：
        在指定的输出目录中生成输出文件。
    """
    # 创建数据集目录
    save_dir = increment_path(save_dir)  # 如果保存目录已存在，则递增目录名称
    for p in save_dir / "labels", save_dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # 创建目录

    # 转换类别
    coco80 = coco91_to_coco80_class()

    # 导入json
    for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):
        lname = "" if lvis else json_file.stem.replace("instances_", "")
        fn = Path(save_dir) / "labels" / lname  # 文件夹名称
        fn.mkdir(parents=True, exist_ok=True)
        if lvis:
            # NOTE: 提前为train和val创建文件夹，
            # 因为LVIS的val集包含COCO 2017训练集中的图像，以及COCO 2017验证集中的图像。
            (fn / "train2017").mkdir(parents=True, exist_ok=True)
            (fn / "val2017").mkdir(parents=True, exist_ok=True)
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        # 创建图像字典
        images = {f"{x['id']:d}": x for x in data["images"]}
        # 创建图像-注释字典
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        image_txt = []
        # 写入标签文件
        for img_id, anns in TQDM(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:d}"]
            h, w = img["height"], img["width"]
            f = str(Path(img["coco_url"]).relative_to("http://images.cocodataset.org")) if lvis else img["file_name"]
            if lvis:
                image_txt.append(str(Path("./images") / f))

            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue
                # COCO框的格式是 [左上角x, 左上角y, 宽度, 高度]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # 将左上角坐标转换为中心坐标
                box[[0, 2]] /= w  # x坐标归一化
                box[[1, 3]] /= h  # y坐标归一化
                if box[2] <= 0 or box[3] <= 0:  # 如果宽度或高度<=0
                    continue

                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # 类别
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if use_segments and ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append([])
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # 所有分割区域合并
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        s = [cls] + s
                        segments.append(s)
                    if use_keypoints and ann.get("keypoints") is not None:
                        keypoints.append(
                            box + (np.array(ann["keypoints"]).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                        )

            # 写入文件
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = (*(keypoints[i]),)  # 类别，框，关键点
                    else:
                        line = (
                            *(segments[i] if use_segments and len(segments[i]) > 0 else bboxes[i]),  # 类别，框或分割区域
                        )
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

        if lvis:
            with open((Path(save_dir) / json_file.name.replace("lvis_v1_", "").replace(".json", ".txt")), "a") as f:
                f.writelines(f"{line}\n" for line in image_txt)

    LOGGER.info(f"{'LVIS' if lvis else 'COCO'} 数据成功转换。\n结果已保存至 {save_dir.resolve()}")


def convert_segment_masks_to_yolo_seg(masks_dir, output_dir, classes):
    """
    将分割掩膜图像数据集转换为 YOLO 分割格式。

    该函数将包含二进制格式掩膜图像的目录中的图像转换为 YOLO 分割格式。
    转换后的掩膜将保存在指定的输出目录中。

    参数:
        masks_dir (str): 存储所有掩膜图像（png, jpg）的目录路径。
        output_dir (str): 存储转换后的 YOLO 分割掩膜的目录路径。
        classes (int): 数据集中的总类别数，例如 COCO 数据集有 80 个类别。

    示例:
        ```python
        from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

        # 这里的 classes 是数据集中的总类别数，对于 COCO 数据集，类数为 80
        convert_segment_masks_to_yolo_seg("path/to/masks_directory", "path/to/output/directory", classes=80)
        ```

    说明:
        掩膜图像的目录结构如下：

            - masks
                ├─ mask_image_01.png 或 mask_image_01.jpg
                ├─ mask_image_02.png 或 mask_image_02.jpg
                ├─ mask_image_03.png 或 mask_image_03.jpg
                └─ mask_image_04.png 或 mask_image_04.jpg

        执行后，标签将被组织成以下结构：

            - output_dir
                ├─ mask_yolo_01.txt
                ├─ mask_yolo_02.txt
                ├─ mask_yolo_03.txt
                └─ mask_yolo_04.txt
    """
    pixel_to_class_mapping = {i + 1: i for i in range(classes)}
    for mask_path in Path(masks_dir).iterdir():
        if mask_path.suffix in {".png", ".jpg"}:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取掩膜图像
            img_height, img_width = mask.shape  # 获取图像尺寸
            LOGGER.info(f"正在处理 {mask_path} imgsz = {img_height} x {img_width}")

            unique_values = np.unique(mask)  # 获取代表不同类别的唯一像素值
            yolo_format_data = []

            for value in unique_values:
                if value == 0:
                    continue  # 跳过背景
                class_index = pixel_to_class_mapping.get(value, -1)
                if class_index == -1:
                    LOGGER.warning(f"文件 {mask_path} 中像素值 {value} 的类别未知，跳过该值。")
                    continue

                # 为当前类别创建二进制掩膜并找到轮廓
                contours, _ = cv2.findContours(
                    (mask == value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )  # 查找轮廓

                for contour in contours:
                    if len(contour) >= 3:  # YOLO 要求至少有 3 个点才能作为有效的分割
                        contour = contour.squeeze()  # 去除单一维度的条目
                        yolo_format = [class_index]
                        for point in contour:
                            # 标准化坐标
                            yolo_format.append(round(point[0] / img_width, 6))  # 保留 6 位小数
                            yolo_format.append(round(point[1] / img_height, 6))
                        yolo_format_data.append(yolo_format)
            # 将 Ultralytics YOLO 格式的数据保存到文件
            output_path = Path(output_dir) / f"{mask_path.stem}.txt"
            with open(output_path, "w") as file:
                for item in yolo_format_data:
                    line = " ".join(map(str, item))
                    file.write(line + "\n")
            LOGGER.info(f"已处理并存储在 {output_path} imgsz = {img_height} x {img_width}")


def convert_dota_to_yolo_obb(dota_root_path: str):
    """
    将 DOTA 数据集注释转换为 YOLO OBB（定向边界框）格式。

    该函数处理 DOTA 数据集的 'train' 和 'val' 文件夹中的图像。对于每个图像，它读取关联的标签文件并将新的标签以 YOLO OBB 格式写入新的目录。

    参数:
        dota_root_path (str): DOTA 数据集的根目录路径。

    示例:
        ```python
        from ultralytics.data.converter import convert_dota_to_yolo_obb

        convert_dota_to_yolo_obb("path/to/DOTA")
        ```

    说明:
        假设 DOTA 数据集的目录结构如下：

            - DOTA
                ├─ images
                │   ├─ train
                │   └─ val
                └─ labels
                    ├─ train_original
                    └─ val_original

        执行后，标签将被组织成：

            - DOTA
                └─ labels
                    ├─ train
                    └─ val
    """
    dota_root_path = Path(dota_root_path)

    # 类别名称到索引的映射
    class_mapping = {
        "plane": 0,
        "ship": 1,
        "storage-tank": 2,
        "baseball-diamond": 3,
        "tennis-court": 4,
        "basketball-court": 5,
        "ground-track-field": 6,
        "harbor": 7,
        "bridge": 8,
        "large-vehicle": 9,
        "small-vehicle": 10,
        "helicopter": 11,
        "roundabout": 12,
        "soccer-ball-field": 13,
        "swimming-pool": 14,
        "container-crane": 15,
        "airport": 16,
        "helipad": 17,
    }

    def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir):
        """将单个图像的 DOTA 注释转换为 YOLO OBB 格式并保存到指定目录。"""
        orig_label_path = orig_label_dir / f"{image_name}.txt"
        save_path = save_dir / f"{image_name}.txt"

        with orig_label_path.open("r") as f, save_path.open("w") as g:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                class_name = parts[8]
                class_idx = class_mapping[class_name]
                coords = [float(p) for p in parts[:8]]
                normalized_coords = [
                    coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
                ]
                formatted_coords = [f"{coord:.6g}" for coord in normalized_coords]
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")

    for phase in ["train", "val"]:
        image_dir = dota_root_path / "images" / phase
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"
        save_dir = dota_root_path / "labels" / phase

        save_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(image_dir.iterdir())
        for image_path in TQDM(image_paths, desc=f"正在处理 {phase} 图像"):
            if image_path.suffix != ".png":
                continue
            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)


def min_index(arr1, arr2):
    """
    找到两个 2D 点数组之间距离最短的一对索引。

    参数:
        arr1 (np.ndarray): 形状为 (N, 2) 的 NumPy 数组，表示 N 个 2D 点。
        arr2 (np.ndarray): 形状为 (M, 2) 的 NumPy 数组，表示 M 个 2D 点。

    返回:
        (tuple): 包含 arr1 和 arr2 中距离最短的点的索引的元组。
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    将多个分段合并为一个列表，通过连接每个分段之间最小距离的坐标来实现合并。
    此函数通过使用一条细线连接这些坐标，合并所有分段为一个。

    参数：
        segments (List[List]): 来自COCO JSON文件的原始分段。
                               每个元素是一个坐标列表，例如[segmentation1, segmentation2,...]。

    返回：
        s (List[np.ndarray]): 一个表示已连接分段的NumPy数组列表。
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # 记录每个分段之间最小距离的索引
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # 使用两轮连接所有分段
    for k in range(2):
        # 正向连接
        if k == 0:
            for i, idx in enumerate(idx_list):
                # 中间的分段有两个索引，如果中间的分段索引顺序是反的，则将其反转
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # 处理第一个和最后一个分段
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def yolo_bbox2segment(im_dir, save_dir=None, sam_model="sam_b.pt", device=None):
    """
    将现有的目标检测数据集（边界框）转换为分割数据集或有向边界框（OBB）
    以YOLO格式生成分割数据，必要时使用SAM自动标注器生成分割数据。

    参数：
        im_dir (str | Path): 要转换的图像目录路径。
        save_dir (str | Path): 要保存生成的标签的路径。如果save_dir为None，标签将保存到
            与im_dir同级目录中的“labels-segment”文件夹。默认值：None。
        sam_model (str): 用于中间分割数据的分割模型；可选。
        device (int | str): 运行SAM模型的具体设备。默认值：None。

    注意：
        假设输入目录结构如下：

            - im_dir
                ├─ 001.jpg
                ├─ ...
                └─ NNN.jpg
            - labels
                ├─ 001.txt
                ├─ ...
                └─ NNN.txt
    """
    from ultralytics import SAM
    from ultralytics.data import YOLODataset
    from ultralytics.utils import LOGGER
    from ultralytics.utils.ops import xywh2xyxy

    # 注意：添加占位符来通过类索引检查
    dataset = YOLODataset(im_dir, data=dict(names=list(range(1000))))
    if len(dataset.labels[0]["segments"]) > 0:  # 如果是分割数据
        LOGGER.info("检测到分割标签，无需生成新标签！")
        return

    LOGGER.info("检测到检测标签，正在使用SAM模型生成分割标签！")
    sam_model = SAM(sam_model)
    for label in TQDM(dataset.labels, total=len(dataset.labels), desc="生成分割标签"):
        h, w = label["shape"]
        boxes = label["bboxes"]
        if len(boxes) == 0:  # 跳过没有标签的图片
            continue
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        im = cv2.imread(label["im_file"])
        sam_results = sam_model(im, bboxes=xywh2xyxy(boxes), verbose=False, save=False, device=device)
        label["segments"] = sam_results[0].masks.xyn

    save_dir = Path(save_dir) if save_dir else Path(im_dir).parent / "labels-segment"
    save_dir.mkdir(parents=True, exist_ok=True)
    for label in dataset.labels:
        texts = []
        lb_name = Path(label["im_file"]).with_suffix(".txt").name
        txt_file = save_dir / lb_name
        cls = label["cls"]
        for i, s in enumerate(label["segments"]):
            if len(s) == 0:
                continue
            line = (int(cls[i]), *s.reshape(-1))
            texts.append(("%g " * len(line)).rstrip() % line)
        with open(txt_file, "a") as f:
            f.writelines(text + "\n" for text in texts)
    LOGGER.info(f"生成的分割标签已保存到 {save_dir}")


def create_synthetic_coco_dataset():
    """
    创建一个合成的COCO数据集，基于标签列表中的文件名生成随机图像。

    此函数下载COCO标签，读取标签列表文件中的图像文件名，
    为train2017和val2017子集创建合成图像，并将它们组织到COCO数据集结构中。它使用多线程高效生成图像。

    示例：
        >>> from ultralytics.data.converter import create_synthetic_coco_dataset
        >>> create_synthetic_coco_dataset()

    注意：
        - 需要互联网连接下载标签文件。
        - 生成随机RGB图像，尺寸从480x480到640x640像素不等。
        - 删除现有的test2017目录，因为不需要它。
        - 从train2017.txt和val2017.txt文件中读取图像文件名。
    """

    def create_synthetic_image(image_file):
        """生成用于数据集增强或测试目的的合成图像，图像的大小和颜色是随机的。"""
        if not image_file.exists():
            size = (random.randint(480, 640), random.randint(480, 640))
            Image.new(
                "RGB",
                size=size,
                color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            ).save(image_file)

    # 下载标签
    dir = DATASETS_DIR / "coco"
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    label_zip = "coco2017labels-segments.zip"
    download([url + label_zip], dir=dir.parent)

    # 创建合成图像
    shutil.rmtree(dir / "labels" / "test2017", ignore_errors=True)  # 删除不需要的test2017目录
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for subset in ["train2017", "val2017"]:
            subset_dir = dir / "images" / subset
            subset_dir.mkdir(parents=True, exist_ok=True)

            # 从标签列表文件中读取图像文件名
            label_list_file = dir / f"{subset}.txt"
            if label_list_file.exists():
                with open(label_list_file) as f:
                    image_files = [dir / line.strip() for line in f]

                # 提交所有任务
                futures = [executor.submit(create_synthetic_image, image_file) for image_file in image_files]
                for _ in TQDM(as_completed(futures), total=len(futures), desc=f"为{subset}生成图像"):
                    pass  # 实际工作在后台完成
            else:
                print(f"警告：标签文件 {label_list_file} 不存在，跳过{subset}的图像创建。")

    print("合成COCO数据集创建成功。")
