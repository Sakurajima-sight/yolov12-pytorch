# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

import subprocess

import pytest
from PIL import Image

from tests import CUDA_DEVICE_COUNT, CUDA_IS_AVAILABLE
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, WEIGHTS_DIR, checks
from ultralytics.utils.torch_utils import TORCH_1_9

# 常量
TASK_MODEL_DATA = [(task, WEIGHTS_DIR / TASK2MODEL[task], TASK2DATA[task]) for task in TASKS]
MODELS = [WEIGHTS_DIR / TASK2MODEL[task] for task in TASKS]


def run(cmd):
    """使用 subprocess 执行 shell 命令。"""
    subprocess.run(cmd.split(), check=True)


def test_special_modes():
    """测试 YOLO 的各种特殊命令行模式。"""
    run("yolo help")
    run("yolo checks")
    run("yolo version")
    run("yolo settings reset")
    run("yolo cfg")


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_train(task, model, data):
    """测试 YOLO 在不同任务、模型和数据集上的训练。"""
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 cache=disk")


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_val(task, model, data):
    """测试 YOLO 的验证过程，使用指定任务、模型和数据集。"""
    run(f"yolo val {task} model={model} data={data} imgsz=32 save_txt save_json")


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_predict(task, model, data):
    """测试 YOLO 预测功能，在提供的示例数据上进行预测。"""
    run(f"yolo predict model={model} source={ASSETS} imgsz=32 save save_crop save_txt")


@pytest.mark.parametrize("model", MODELS)
def test_export(model):
    """测试 YOLO 模型导出为 TorchScript 格式。"""
    run(f"yolo export model={model} format=torchscript imgsz=32")


def test_rtdetr(task="detect", model="yolov8n-rtdetr.yaml", data="coco8.yaml"):
    """测试 Ultralytics 中的 RTDETR 功能，适用于检测任务。"""
    # 注意：必须使用 imgsz=640（同时添加逗号、空格、fraction=0.25 参数以测试单张图片训练）
    run(f"yolo train {task} model={model} data={data} --imgsz=160 epochs=1, cache=disk fraction=0.25")
    run(f"yolo predict {task} model={model} source={ASSETS / 'bus.jpg'} imgsz=160 save save_crop save_txt")
    if TORCH_1_9:
        weights = WEIGHTS_DIR / "rtdetr-l.pt"
        run(f"yolo predict {task} model={weights} source={ASSETS / 'bus.jpg'} imgsz=160 save save_crop save_txt")


@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="MobileSAM 与 CLIP 在 Python 3.12 上不受支持")
def test_fastsam(task="segment", model=WEIGHTS_DIR / "FastSAM-s.pt", data="coco8-seg.yaml"):
    """测试 FastSAM 模型在 Ultralytics 中的分割功能，使用不同的提示进行对象分割。"""
    source = ASSETS / "bus.jpg"

    run(f"yolo segment val {task} model={model} data={data} imgsz=32")
    run(f"yolo segment predict model={model} source={source} imgsz=32 save save_crop save_txt")

    from ultralytics import FastSAM
    from ultralytics.models.sam import Predictor

    # 创建 FastSAM 模型
    sam_model = FastSAM(model)  # 或 FastSAM-x.pt

    # 在图像上运行推理
    for s in (source, Image.open(source)):
        everything_results = sam_model(s, device="cpu", retina_masks=True, imgsz=320, conf=0.4, iou=0.9)

        # 移除小区域
        new_masks, _ = Predictor.remove_small_regions(everything_results[0].masks.data, min_area=20)

        # 通过边界框、点和文本提示同时进行推理
        sam_model(source, bboxes=[439, 437, 524, 709], points=[[200, 200]], labels=[1], texts="a photo of a dog")


def test_mobilesam():
    """测试 MobileSAM 在 Ultralytics 中的点提示分割功能。"""
    from ultralytics import SAM

    # 加载模型
    model = SAM(WEIGHTS_DIR / "mobile_sam.pt")

    # 图像来源
    source = ASSETS / "zidane.jpg"

    # 基于 1D 点提示和 1D 标签进行分割预测
    model.predict(source, points=[900, 370], labels=[1])

    # 基于 3D 点提示和 2D 标签进行分割预测（每个对象多个点）
    model.predict(source, points=[[[900, 370], [1000, 100]]], labels=[[1, 1]])

    # 基于边界框提示进行分割预测
    model.predict(source, bboxes=[439, 437, 524, 709], save=True)

    # 预测所有对象
    # model(source)


# 慢速测试 -----------------------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA 不可用")
@pytest.mark.skipif(CUDA_DEVICE_COUNT < 2, reason="DDP（分布式数据并行）不可用")
def test_train_gpu(task, model, data):
    """测试 YOLO 在 GPU 上的训练，包括单 GPU 和多 GPU 训练。"""
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 device=0")  # 单 GPU
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 device=0,1")  # 多 GPU
