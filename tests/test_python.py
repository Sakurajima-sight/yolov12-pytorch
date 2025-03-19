# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

import contextlib
import csv
import urllib
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from tests import CFG, MODEL, SOURCE, SOURCES_LIST, TMP
from ultralytics import RTDETR, YOLO
from ultralytics.cfg import MODELS, TASK2DATA, TASKS
from ultralytics.data.build import load_inference_source
from ultralytics.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_PATH,
    LOGGER,
    ONLINE,
    ROOT,
    WEIGHTS_DIR,
    WINDOWS,
    checks,
    is_dir_writeable,
    is_github_action_running,
)
from ultralytics.utils.downloads import download
from ultralytics.utils.torch_utils import TORCH_1_9

# 检查临时目录是否可写，必须在测试开始后运行，因为 TMP 目录在初始化时可能不存在
IS_TMP_WRITEABLE = is_dir_writeable(TMP)  


def test_model_forward():
    """测试 YOLO 模型的前向传播过程"""
    model = YOLO(CFG)
    model(source=None, imgsz=32, augment=True)  # 还测试了无 source 和增强模式


def test_model_methods():
    """测试 YOLO 模型的各种方法和属性，确保其正常工作"""
    model = YOLO(MODEL)

    # 测试模型方法
    model.info(verbose=True, detailed=True)
    model = model.reset_weights()
    model = model.load(MODEL)
    model.to("cpu")
    model.fuse()
    model.clear_callback("on_train_start")
    model.reset_callbacks()

    # 测试模型属性
    _ = model.names
    _ = model.device
    _ = model.transforms
    _ = model.task_map


def test_model_profile():
    """测试 YOLO 模型的性能分析（profile=True），评估其性能和资源使用情况"""
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel()  # 构建模型
    im = torch.randn(1, 3, 64, 64)  # 需要最小图像尺寸为 64
    _ = model.predict(im, profile=True)


@pytest.mark.skipif(not IS_TMP_WRITEABLE, reason="目录不可写")
def test_predict_txt():
    """测试 YOLO 预测功能，使用文本文件中列出的文件、目录和模式作为输入源"""
    file = TMP / "sources_multi_row.txt"
    with open(file, "w") as f:
        for src in SOURCES_LIST:
            f.write(f"{src}\n")
    results = YOLO(MODEL)(source=file, imgsz=32)
    assert len(results) == 7  # 1 + 2 + 2 + 2 = 7 张图像


@pytest.mark.skipif(True, reason="测试已禁用")
@pytest.mark.skipif(not IS_TMP_WRITEABLE, reason="目录不可写")
def test_predict_csv_multi_row():
    """测试 YOLO 预测功能，使用 CSV 文件中多行列出的输入源"""
    file = TMP / "sources_multi_row.csv"
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source"])
        writer.writerows([[src] for src in SOURCES_LIST])
    results = YOLO(MODEL)(source=file, imgsz=32)
    assert len(results) == 7  # 1 + 2 + 2 + 2 = 7 张图像


@pytest.mark.skipif(True, reason="测试已禁用")
@pytest.mark.skipif(not IS_TMP_WRITEABLE, reason="目录不可写")
def test_predict_csv_single_row():
    """测试 YOLO 预测功能，使用 CSV 文件中单行列出的输入源"""
    file = TMP / "sources_single_row.csv"
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SOURCES_LIST)
    results = YOLO(MODEL)(source=file, imgsz=32)
    assert len(results) == 7  # 1 + 2 + 2 + 2 = 7 张图像


@pytest.mark.parametrize("model_name", MODELS)
def test_predict_img(model_name):
    """测试 YOLO 模型在不同类型的图像输入源上的预测，包括在线图像"""
    model = YOLO(WEIGHTS_DIR / model_name)
    im = cv2.imread(str(SOURCE))  # uint8 类型的 numpy 数组
    assert len(model(source=Image.open(SOURCE), save=True, verbose=True, imgsz=32)) == 1  # PIL 图像
    assert len(model(source=im, save=True, save_txt=True, imgsz=32)) == 1  # OpenCV 读取的 ndarray 图像
    assert len(model(torch.rand((2, 3, 32, 32)), imgsz=32)) == 2  # 批量大小为 2 的张量（FP32 格式，RGB 0.0-1.0）
    assert len(model(source=[im, im], save=True, save_txt=True, imgsz=32)) == 2  # 批量输入
    assert len(list(model(source=[im, im], save=True, stream=True, imgsz=32))) == 2  # 流式处理
    assert len(model(torch.zeros(320, 640, 3).numpy().astype(np.uint8), imgsz=32)) == 1  # 张量转换为 numpy 数组
    batch = [
        str(SOURCE),  # 文件名
        Path(SOURCE),  # Path 对象
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/zidane.jpg" if ONLINE else SOURCE,  # 在线 URI
        cv2.imread(str(SOURCE)),  # OpenCV 读取的图像
        Image.open(SOURCE),  # PIL 图像
        np.zeros((320, 640, 3), dtype=np.uint8),  # numpy 数组
    ]
    assert len(model(batch, imgsz=32)) == len(batch)  # 测试批量输入不同格式的图像


@pytest.mark.parametrize("model", MODELS)
def test_predict_visualize(model):
    """测试 YOLO 模型预测功能，使用 `visualize=True` 生成并显示预测可视化结果"""
    YOLO(WEIGHTS_DIR / model)(SOURCE, imgsz=32, visualize=True)


def test_predict_grey_and_4ch():
    """测试 YOLO 预测功能，处理灰度图和 4 通道图像，并测试不同文件名格式"""
    im = Image.open(SOURCE)
    directory = TMP / "im4"
    directory.mkdir(parents=True, exist_ok=True)

    source_greyscale = directory / "greyscale.jpg"
    source_rgba = directory / "4ch.png"
    source_non_utf = directory / "non_UTF_测试文件_tést_image.jpg"
    source_spaces = directory / "image with spaces.jpg"

    im.convert("L").save(source_greyscale)  # 灰度图
    im.convert("RGBA").save(source_rgba)  # 4 通道 PNG 图像（带 alpha 通道）
    im.save(source_non_utf)  # 包含非 UTF-8 字符的文件名
    im.save(source_spaces)  # 包含空格的文件名

    # 进行推理
    model = YOLO(MODEL)
    for f in source_rgba, source_greyscale, source_non_utf, source_spaces:
        for source in Image.open(f), cv2.imread(str(f)), f:
            results = model(source, save=True, verbose=True, imgsz=32)
            assert len(results) == 1  # 确保每个输入都成功处理
        f.unlink()  # 清理测试文件


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="环境处于离线状态")
@pytest.mark.skipif(is_github_action_running(), reason="无权限 https://github.com/JuanBindez/pytubefix/issues/166")
def test_youtube():
    """测试 YOLO 模型在 YouTube 视频流上的推理能力，并处理可能的网络错误"""
    model = YOLO(MODEL)
    try:
        model.predict("https://youtu.be/G17sBkb38XQ", imgsz=96, save=True)
    # 处理网络连接错误及 'urllib.error.HTTPError: HTTP Error 429: 请求过多'
    except (urllib.error.HTTPError, ConnectionError) as e:
        LOGGER.warning(f"警告: YouTube 测试错误: {e}")


@pytest.mark.skipif(not ONLINE, reason="环境处于离线状态")
@pytest.mark.skipif(not IS_TMP_WRITEABLE, reason="目录不可写")
def test_track_stream():
    """
    测试 ByteTrack 追踪器和不同的 GMC（全局运动补偿）方法在短视频（10 帧）上的流式跟踪能力。

    备注: 由于追踪需要更高的置信度和更好的匹配，图像尺寸 imgsz=160 是必需的。
    """
    video_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/decelera_portrait_min.mov"
    model = YOLO(MODEL)
    model.track(video_url, imgsz=160, tracker="bytetrack.yaml")
    model.track(video_url, imgsz=160, tracker="botsort.yaml", save_frames=True)  # 还测试保存帧的功能

    # 测试不同的全局运动补偿（GMC）方法
    for gmc in "orb", "sift", "ecc":
        with open(ROOT / "cfg/trackers/botsort.yaml", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        tracker = TMP / f"botsort-{gmc}.yaml"
        data["gmc_method"] = gmc
        with open(tracker, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)
        model.track(video_url, imgsz=160, tracker=tracker)


def test_val():
    """测试 YOLO 模型的验证模式"""
    YOLO(MODEL).val(data="coco8.yaml", imgsz=32, save_hybrid=True)


def test_train_scratch():
    """测试 YOLO 模型从零开始训练的功能"""
    model = YOLO(CFG)
    model.train(data="coco8.yaml", epochs=2, imgsz=32, cache="disk", batch=2, close_mosaic=1, name="model")
    model(SOURCE)


def test_train_pretrained():
    """测试从预训练模型开始训练 YOLO 模型"""
    model = YOLO(WEIGHTS_DIR / "yolo11n-seg.pt")
    model.train(data="coco8-seg.yaml", epochs=1, imgsz=32, cache="ram", copy_paste=0.5, mixup=0.5, name=0)
    model(SOURCE)


def test_all_model_yamls():
    """测试 `cfg/models` 目录下所有 YAML 配置文件的 YOLO 模型创建功能"""
    for m in (ROOT / "cfg" / "models").rglob("*.yaml"):
        if "rtdetr" in m.name:
            if TORCH_1_9:  # 处理 torch<=1.8 版本的 bug - TypeError: __init__() got an unexpected keyword argument 'batch_first'
                _ = RTDETR(m.name)(SOURCE, imgsz=640)  # 图像尺寸必须为 640
        else:
            YOLO(m.name)


@pytest.mark.skipif(WINDOWS, reason="Windows CI 运行缓慢导致导出 bug https://github.com/ultralytics/ultralytics/pull/16003")
def test_workflow():
    """测试完整的 YOLO 工作流，包括训练、验证、预测和模型导出"""
    model = YOLO(MODEL)
    model.train(data="coco8.yaml", epochs=1, imgsz=32, optimizer="SGD")
    model.val(imgsz=32)
    model.predict(SOURCE, imgsz=32)
    model.export(format="torchscript")  # 警告: Windows CI 导出缓慢，可能会导致 bug


def test_predict_callback_and_setup():
    """测试 YOLO 预测设置和执行过程中的回调功能"""

    def on_predict_batch_end(predictor):
        """回调函数，在预测批次结束时执行额外操作"""
        path, im0s, _ = predictor.batch
        im0s = im0s if isinstance(im0s, list) else [im0s]
        bs = [predictor.dataset.bs for _ in range(len(path))]
        predictor.results = zip(predictor.results, im0s, bs)  # results 结构为 List[batch_size]

    model = YOLO(MODEL)
    model.add_callback("on_predict_batch_end", on_predict_batch_end)

    dataset = load_inference_source(source=SOURCE)
    bs = dataset.bs  # 访问 predictor 的 batch size
    results = model.predict(dataset, stream=True, imgsz=160)  # source 预先设置完成
    for r, im0, bs in results:
        print("测试回调函数", im0.shape)
        print("测试回调函数", bs)
        boxes = r.boxes  # 获取边界框对象
        print(boxes)


@pytest.mark.parametrize("model", MODELS)
def test_results(model):
    """确保 YOLO 模型的预测结果可以以多种格式处理和打印"""
    results = YOLO(WEIGHTS_DIR / model)([SOURCE, SOURCE], imgsz=160)
    for r in results:
        r = r.cpu().numpy()
        print(r, len(r), r.path)  # 打印 numpy 属性
        r = r.to(device="cpu", dtype=torch.float32)
        r.save_txt(txt_file=TMP / "runs/tests/label.txt", save_conf=True)
        r.save_crop(save_dir=TMP / "runs/tests/crops/")
        r.to_json(normalize=True)
        r.to_df(decimals=3)
        r.to_csv()
        r.to_xml()
        r.plot(pil=True)
        r.plot(conf=True, boxes=True)
        print(r, len(r), r.path)  # 执行方法后再次打印


def test_labels_and_crops():
    """测试 YOLO 预测参数的输出，确保检测标签和裁剪结果的正确保存"""
    imgs = [SOURCE, ASSETS / "zidane.jpg"]
    results = YOLO(WEIGHTS_DIR / "yolo11n.pt")(imgs, imgsz=160, save_txt=True, save_crop=True)
    save_path = Path(results[0].save_dir)
    for r in results:
        im_name = Path(r.path).stem
        cls_idxs = r.boxes.cls.int().tolist()
        # 检查检测类别是否正确
        assert cls_idxs == ([0, 7, 0, 0] if r.path.endswith("bus.jpg") else [0, 0, 0])  # bus.jpg 和 zidane.jpg 的类别
        # 检查标签文件路径
        labels = save_path / f"labels/{im_name}.txt"
        assert labels.exists()
        # 检查检测数目是否与标签数匹配
        assert len(r.boxes.data) == len([line for line in labels.read_text().splitlines() if line])
        # 检查裁剪图像的路径和文件
        crop_dirs = list((save_path / "crops").iterdir())
        crop_files = [f for p in crop_dirs for f in p.glob("*")]
        # 裁剪目录名称应与检测类别一致
        assert all(r.names.get(c) in {d.name for d in crop_dirs} for c in cls_idxs)
        # 裁剪图像数量应与检测框数量一致
        assert len([f for f in crop_files if im_name in f.name]) == len(r.boxes.data)


@pytest.mark.skipif(not ONLINE, reason="环境处于离线状态")
def test_data_utils():
    """测试 ultralytics/data/utils.py 中的实用函数，包括数据集统计和自动划分"""
    from ultralytics.data.utils import HUBDatasetStats, autosplit
    from ultralytics.utils.downloads import zip_directory

    for task in TASKS:
        file = Path(TASK2DATA[task]).with_suffix(".zip")  # 例如 coco8.zip
        download(f"https://github.com/ultralytics/hub/raw/main/example_datasets/{file}", unzip=False, dir=TMP)
        stats = HUBDatasetStats(TMP / file, task=task)
        stats.get_json(save=True)
        stats.process_images()

    autosplit(TMP / "coco8")
    zip_directory(TMP / "coco8/images/val")  # 压缩数据集


@pytest.mark.skipif(not ONLINE, reason="环境处于离线状态")
def test_data_converter():
    """测试数据集转换功能，将 COCO 格式转换为 YOLO 格式，并进行类别映射"""
    from ultralytics.data.converter import coco80_to_coco91_class, convert_coco

    file = "instances_val2017.json"
    download(f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{file}", dir=TMP)
    convert_coco(labels_dir=TMP, save_dir=TMP / "yolo_labels", use_segments=True, use_keypoints=False, cls91to80=True)
    coco80_to_coco91_class()


def test_data_annotator():
    """使用指定的检测和分割模型自动标注数据"""
    from ultralytics.data.annotator import auto_annotate

    auto_annotate(
        ASSETS,
        det_model=WEIGHTS_DIR / "yolo11n.pt",
        sam_model=WEIGHTS_DIR / "mobile_sam.pt",
        output_dir=TMP / "auto_annotate_labels",
    )


def test_events():
    """测试事件发送功能"""
    from ultralytics.hub.utils import Events

    events = Events()
    events.enabled = True
    cfg = copy(DEFAULT_CFG)  # 不需要深拷贝
    cfg.mode = "test"
    events(cfg)


def test_cfg_init():
    """测试 'ultralytics.cfg' 模块的配置初始化工具"""
    from ultralytics.cfg import check_dict_alignment, copy_default_cfg, smart_value

    with contextlib.suppress(SyntaxError):
        check_dict_alignment({"a": 1}, {"b": 2})
    copy_default_cfg()
    (Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")).unlink(missing_ok=False)
    [smart_value(x) for x in ["none", "true", "false"]]


def test_utils_init():
    """测试 Ultralytics 库中的初始化工具"""
    from ultralytics.utils import get_git_branch, get_git_origin_url, get_ubuntu_version, is_github_action_running

    get_ubuntu_version()
    is_github_action_running()
    get_git_origin_url()
    get_git_branch()


def test_utils_checks():
    """测试各种实用检查，包括文件名、Git 状态、依赖项、图像大小和版本"""
    checks.check_yolov5u_filename("yolov5n.pt")
    checks.git_describe(ROOT)
    checks.check_requirements()  # 检查 requirements.txt
    checks.check_imgsz([600, 600], max_dim=1)
    checks.check_imshow(warn=True)
    checks.check_version("ultralytics", "8.0.0")
    checks.print_args()


@pytest.mark.skipif(WINDOWS, reason="Windows 上的性能分析非常慢（原因未知）")
def test_utils_benchmarks():
    """使用 'ultralytics.utils.benchmarks' 中的 'ProfileModels' 进行模型性能基准测试"""
    from ultralytics.utils.benchmarks import ProfileModels

    ProfileModels(["yolo11n.yaml"], imgsz=32, min_time=1, num_timed_runs=3, num_warmup_runs=1).profile()


def test_utils_torchutils():
    """测试 Torch 相关的实用功能，包括性能分析和 FLOP 计算"""
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.utils.torch_utils import get_flops_with_torch_profiler, profile, time_sync

    x = torch.randn(1, 64, 20, 20)
    m = Conv(64, 64, k=1, s=2)

    profile(x, [m], n=3)
    get_flops_with_torch_profiler(m)
    time_sync()


def test_utils_ops():
    """测试坐标转换和归一化等实用操作函数"""
    from ultralytics.utils.ops import (
        ltwh2xywh,
        ltwh2xyxy,
        make_divisible,
        xywh2ltwh,
        xywh2xyxy,
        xywhn2xyxy,
        xywhr2xyxyxyxy,
        xyxy2ltwh,
        xyxy2xywh,
        xyxy2xywhn,
        xyxyxyxy2xywhr,
    )

    make_divisible(17, torch.tensor([8]))

    boxes = torch.rand(10, 4)  # xywh 格式
    torch.allclose(boxes, xyxy2xywh(xywh2xyxy(boxes)))
    torch.allclose(boxes, xyxy2xywhn(xywhn2xyxy(boxes)))
    torch.allclose(boxes, ltwh2xywh(xywh2ltwh(boxes)))
    torch.allclose(boxes, xyxy2ltwh(ltwh2xyxy(boxes)))

    boxes = torch.rand(10, 5)  # xywhr 格式（用于旋转边界框）
    boxes[:, 4] = torch.randn(10) * 30
    torch.allclose(boxes, xyxyxyxy2xywhr(xywhr2xyxyxyxy(boxes)), rtol=1e-3)


def test_utils_files():
    """测试文件操作工具，包括文件创建时间、最新运行文件、路径空格等"""
    from ultralytics.utils.files import file_age, file_date, get_latest_run, spaces_in_path

    file_age(SOURCE)
    file_date(SOURCE)
    get_latest_run(ROOT / "runs")

    path = TMP / "path/with spaces"
    path.mkdir(parents=True, exist_ok=True)
    with spaces_in_path(path) as new_path:
        print(new_path)


@pytest.mark.slow
def test_utils_patches_torch_save():
    """测试当 `_torch_save` 触发 RuntimeError 时 `torch_save` 的重试机制，确保其健壮性"""
    from unittest.mock import MagicMock, patch

    from ultralytics.utils.patches import torch_save

    mock = MagicMock(side_effect=RuntimeError)

    with patch("ultralytics.utils.patches._torch_save", new=mock):
        with pytest.raises(RuntimeError):
            torch_save(torch.zeros(1), TMP / "test.pt")

    assert mock.call_count == 4, "torch_save 未按照预期次数进行重试"


def test_nn_modules_conv():
    """测试卷积神经网络模块，包括 CBAM、Conv2、ConvTranspose 等"""
    from ultralytics.nn.modules.conv import CBAM, Conv2, ConvTranspose, DWConvTranspose2d, Focus

    c1, c2 = 8, 16  # 输入通道数和输出通道数
    x = torch.zeros(4, c1, 10, 10)  # BCHW 格式的张量

    # 运行未在其他测试中涵盖的所有模块
    DWConvTranspose2d(c1, c2)(x)
    ConvTranspose(c1, c2)(x)
    Focus(c1, c2)(x)
    CBAM(c1)(x)

    # 测试融合操作
    m = Conv2(c1, c2)
    m.fuse_convs()
    m(x)


def test_nn_modules_block():
    """测试神经网络模块中的各种块，包括 C1、C3TR、BottleneckCSP、C3Ghost 和 C3x。"""
    from ultralytics.nn.modules.block import C1, C3TR, BottleneckCSP, C3Ghost, C3x

    c1, c2 = 8, 16  # 输入通道数和输出通道数
    x = torch.zeros(4, c1, 10, 10)  # BCHW 格式的张量

    # 运行所有未在其他测试中覆盖的模块
    C1(c1, c2)(x)
    C3x(c1, c2)(x)
    C3TR(c1, c2)(x)
    C3Ghost(c1, c2)(x)
    BottleneckCSP(c1, c2)(x)


@pytest.mark.skipif(not ONLINE, reason="环境处于离线状态")
def test_hub():
    """测试 Ultralytics HUB 功能（例如导出格式和登出）。"""
    from ultralytics.hub import export_fmts_hub, logout
    from ultralytics.hub.utils import smart_request

    export_fmts_hub()
    logout()
    smart_request("GET", "https://github.com", progress=True)


@pytest.fixture
def image():
    """使用 OpenCV 加载并返回预定义的图像源。"""
    return cv2.imread(str(SOURCE))


@pytest.mark.parametrize(
    "auto_augment, erasing, force_color_jitter",
    [
        (None, 0.0, False),
        ("randaugment", 0.5, True),
        ("augmix", 0.2, False),
        ("autoaugment", 0.0, True),
    ],
)
def test_classify_transforms_train(image, auto_augment, erasing, force_color_jitter):
    """测试分类训练期间的增强变换，以确保功能正确。"""
    from ultralytics.data.augment import classify_augmentations

    transform = classify_augmentations(
        size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        hflip=0.5,
        vflip=0.5,
        auto_augment=auto_augment,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        force_color_jitter=force_color_jitter,
        erasing=erasing,
    )

    transformed_image = transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

    assert transformed_image.shape == (3, 224, 224)
    assert torch.is_tensor(transformed_image)
    assert transformed_image.dtype == torch.float32


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="环境处于离线状态")
def test_model_tune():
    """对 YOLO 模型进行调优以提高性能。"""
    YOLO("yolo11n-pose.pt").tune(data="coco8-pose.yaml", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")
    YOLO("yolo11n-cls.pt").tune(data="imagenet10", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")


def test_model_embeddings():
    """测试 YOLO 模型的嵌入功能。"""
    model_detect = YOLO(MODEL)
    model_segment = YOLO(WEIGHTS_DIR / "yolo11n-seg.pt")

    for batch in [SOURCE], [SOURCE, SOURCE]:  # 测试 batch 大小为 1 和 2
        assert len(model_detect.embed(source=batch, imgsz=32)) == len(batch)
        assert len(model_segment.embed(source=batch, imgsz=32)) == len(batch)


@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="Python 3.12 不支持 YOLOWorld 及 CLIP")
def test_yolo_world():
    """测试支持 CLIP 的 YOLO 世界模型，包括检测和训练场景。"""
    model = YOLO(WEIGHTS_DIR / "yolov8s-world.pt")  # 目前没有 YOLO11n-world 模型
    model.set_classes(["tree", "window"])
    model(SOURCE, conf=0.01)

    model = YOLO(WEIGHTS_DIR / "yolov8s-worldv2.pt")  # 目前没有 YOLO11n-world 模型
    # 从预训练模型开始训练，最终阶段包含评估。
    # 使用 dota8.yaml（类别较少）以减少 CLIP 模型的推理时间。
    model.train(
        data="dota8.yaml",
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
    )

    # 测试 WorldTrainerFromScratch
    from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

    model = YOLO("yolov8s-worldv2.yaml")  # 目前没有 YOLO11n-world 模型
    model.train(
        data={"train": {"yolo_data": ["dota8.yaml"]}, "val": {"yolo_data": ["dota8.yaml"]}},
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
        trainer=WorldTrainerFromScratch,
    )


def test_yolov10():
    """测试 YOLOv10 模型的训练、验证和预测过程，使用最小配置。"""
    model = YOLO("yolov10n.yaml")
    # 训练、验证和预测
    model.train(data="coco8.yaml", epochs=1, imgsz=32, close_mosaic=1, cache="disk")
    model.val(data="coco8.yaml", imgsz=32)
    model.predict(imgsz=32, save_txt=True, save_crop=True, augment=True)
    model(SOURCE)
