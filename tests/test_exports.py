# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

import shutil
import uuid
from itertools import product
from pathlib import Path

import pytest

from tests import MODEL, SOURCE
from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import (
    IS_RASPBERRYPI,
    LINUX,
    MACOS,
    WINDOWS,
    checks,
)
from ultralytics.utils.torch_utils import TORCH_1_9, TORCH_1_13

import faulthandler
import sys
faulthandler.enable(file=sys.stdout)

def test_export_torchscript():
    """测试 YOLO 模型导出为 TorchScript 格式，以检查其兼容性和正确性"""
    file = YOLO(MODEL).export(format="torchscript", optimize=False, imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)  # 运行导出的模型进行推理


def test_export_onnx():
    """测试 YOLO 模型导出为 ONNX 格式，并支持动态轴"""
    file = YOLO(MODEL).export(format="onnx", dynamic=True, imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)  # 运行导出的模型进行推理


@pytest.mark.skipif(not TORCH_1_13, reason="OpenVINO 需要 torch>=1.13")
def test_export_openvino():
    """测试 YOLO 模型导出为 OpenVINO 格式，以检查推理兼容性"""
    file = YOLO(MODEL).export(format="openvino", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)  # 运行导出的模型进行推理


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_1_13, reason="OpenVINO 需要 torch>=1.13")
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch",
    [  # 生成所有组合，但排除 int8 和 half 同时为 True 的情况
        (task, dynamic, int8, half, batch)
        for task, dynamic, int8, half, batch in product(TASKS, [True, False], [True, False], [True, False], [1, 2])
        if not (int8 and half)  # 排除 int8 和 half 同时为 True 的情况
    ],
)
def test_export_openvino_matrix(task, dynamic, int8, half, batch):
    """在不同的配置矩阵条件下测试 YOLO 模型导出为 OpenVINO"""
    file = YOLO(TASK2MODEL[task]).export(
        format="openvino",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
        data=TASK2DATA[task],
    )
    if WINDOWS:
        # 由于 Windows 文件权限问题，使用唯一的文件名
        # 参考 https://github.com/ultralytics/ultralytics/actions/runs/8957949304/job/24601616830?pr=10423
        file = Path(file)
        file = file.rename(file.with_stem(f"{file.stem}-{uuid.uuid4()}"))
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # 运行导出的模型进行推理
    shutil.rmtree(file, ignore_errors=True)  # 清理文件，防止潜在的多线程文件使用错误


@pytest.mark.slow
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch, simplify", product(TASKS, [True, False], [False], [False], [1, 2], [True, False])
)
def test_export_onnx_matrix(task, dynamic, int8, half, batch, simplify):
    """在不同配置参数下测试 YOLO 模型导出为 ONNX 格式"""
    file = YOLO(TASK2MODEL[task]).export(
        format="onnx",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
        simplify=simplify,
    )
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # 运行导出的模型进行推理
    Path(file).unlink()  # 清理文件


@pytest.mark.slow
@pytest.mark.parametrize("task, dynamic, int8, half, batch", product(TASKS, [False], [False], [False], [1, 2]))
def test_export_torchscript_matrix(task, dynamic, int8, half, batch):
    """在不同配置下测试 YOLO 模型导出为 TorchScript 格式"""
    file = YOLO(TASK2MODEL[task]).export(
        format="torchscript",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
    )
    YOLO(file)([SOURCE] * 3, imgsz=64 if dynamic else 32)  # 运行导出的模型进行推理，批次大小为 3
    Path(file).unlink()  # 清理文件


@pytest.mark.slow
@pytest.mark.skipif(not MACOS, reason="CoreML 推理仅支持 macOS")
@pytest.mark.skipif(not TORCH_1_9, reason="CoreML>=7.2 不支持 PyTorch<=1.8")
@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="CoreML 不支持 Python 3.12")
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch",
    [  # 生成所有组合，但排除 int8 和 half 同时为 True 的情况
        (task, dynamic, int8, half, batch)
        for task, dynamic, int8, half, batch in product(TASKS, [False], [True, False], [True, False], [1])
        if not (int8 and half)  # 排除 int8 和 half 同时为 True 的情况
    ],
)
def test_export_coreml_matrix(task, dynamic, int8, half, batch):
    """在不同参数配置下测试 YOLO 模型导出为 CoreML 格式"""
    file = YOLO(TASK2MODEL[task]).export(
        format="coreml",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
    )
    YOLO(file)([SOURCE] * batch, imgsz=32)  # 运行导出的模型进行推理，批次大小为 3
    shutil.rmtree(file)  # 清理文件


@pytest.mark.slow
@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10, reason="TFLite 导出需要 Python>=3.10")
@pytest.mark.skipif(not LINUX, reason="由于 TensorFlow 在 Windows 和 macOS 上存在安装冲突，此测试被禁用")
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch",
    [  # 生成所有组合，但排除 int8 和 half 同时为 True 的情况
        (task, dynamic, int8, half, batch)
        for task, dynamic, int8, half, batch in product(TASKS, [False], [True, False], [True, False], [1])
        if not (int8 and half)  # 排除 int8 和 half 同时为 True 的情况
    ],
)
def test_export_tflite_matrix(task, dynamic, int8, half, batch):
    """测试 YOLO 模型导出为 TFLite 格式，考虑不同的导出配置"""
    file = YOLO(TASK2MODEL[task]).export(
        format="tflite",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
    )
    YOLO(file)([SOURCE] * batch, imgsz=32)  # 运行导出的模型进行推理，批次大小为 3
    Path(file).unlink()  # 清理文件


@pytest.mark.skipif(not TORCH_1_9, reason="CoreML>=7.2 不支持 PyTorch<=1.8")
@pytest.mark.skipif(WINDOWS, reason="CoreML 不支持 Windows")  # 运行时报错：BlobWriter 未加载
@pytest.mark.skipif(IS_RASPBERRYPI, reason="CoreML 不支持树莓派")
@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="CoreML 不支持 Python 3.12")
def test_export_coreml():
    """测试 YOLO 模型导出为 CoreML 格式，仅在 macOS 上进行优化"""
    if MACOS:
        file = YOLO(MODEL).export(format="coreml", imgsz=32)
        YOLO(file)(SOURCE, imgsz=32)  # 仅在 macOS 上支持 nms=False 的模型进行预测
    else:
        YOLO(MODEL).export(format="coreml", nms=True, imgsz=32)


@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10, reason="TFLite 导出需要 Python>=3.10")
@pytest.mark.skipif(not LINUX, reason="由于 TensorFlow 在 Windows 和 macOS 上存在安装冲突，此测试被禁用")
def test_export_tflite():
    """测试 YOLO 模型导出为 TFLite 格式，需符合特定的操作系统和 Python 版本条件"""
    model = YOLO(MODEL)
    file = model.export(format="tflite", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)


@pytest.mark.skipif(True, reason="测试已禁用")
@pytest.mark.skipif(not LINUX, reason="TensorFlow 在 Windows 和 macOS 上存在安装冲突")
def test_export_pb():
    """测试 YOLO 模型导出为 TensorFlow 的 Protobuf (*.pb) 格式"""
    model = YOLO(MODEL)
    file = model.export(format="pb", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)


@pytest.mark.skipif(True, reason="测试已禁用，因为 Paddle 的 protobuf 和 ONNX 的 protobuf 存在冲突")
def test_export_paddle():
    """测试 YOLO 模型导出为 Paddle 格式，需注意 protobuf 与 ONNX 的冲突"""
    YOLO(MODEL).export(format="paddle", imgsz=32)


@pytest.mark.slow
@pytest.mark.skipif(IS_RASPBERRYPI, reason="MNN 不支持树莓派")
def test_export_mnn():
    """测试 YOLO 模型导出为 MNN 格式（⚠️ 注意：MNN 测试必须在 NCNN 测试之前，否则在 Windows 上 CI 会出错）"""
    file = YOLO(MODEL).export(format="mnn", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)  # 运行导出的模型进行推理


@pytest.mark.slow
def test_export_ncnn():
    """测试 YOLO 模型导出为 NCNN 格式"""
    file = YOLO(MODEL).export(format="ncnn", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)  # 运行导出的模型进行推理


@pytest.mark.skipif(True, reason="测试已禁用，因为 Keras 和 TensorFlow 版本与 TFLite 导出存在冲突")
@pytest.mark.skipif(not LINUX or MACOS, reason="跳过 Windows 和 macOS 上的测试")
def test_export_imx():
    """测试 YOLOv8n 模型导出为 IMX 格式"""
    model = YOLO("yolov8n.pt")
    file = model.export(format="imx", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)
