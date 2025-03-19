# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

from itertools import product
from pathlib import Path

import pytest
import torch

from tests import CUDA_DEVICE_COUNT, CUDA_IS_AVAILABLE, MODEL, SOURCE
from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, WEIGHTS_DIR
from ultralytics.utils.checks import check_amp


def test_checks():
    """验证 CUDA 设置是否与 torch 的 CUDA 相关函数匹配。"""
    assert torch.cuda.is_available() == CUDA_IS_AVAILABLE
    assert torch.cuda.device_count() == CUDA_DEVICE_COUNT


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA 不可用")
def test_amp():
    """测试 AMP（自动混合精度）训练检查。"""
    model = YOLO("yolo11n.pt").model.cuda()
    assert check_amp(model)


@pytest.mark.slow
@pytest.mark.skipif(True, reason="CUDA 导出测试已禁用，等待 Ultralytics 额外的 GPU 服务器资源")
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA 不可用")
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch",
    [  # 生成所有组合，但排除 int8 和 half 均为 True 的情况
        (task, dynamic, int8, half, batch)
        # 由于 GPU 计算资源有限，当前测试数量有所减少
        # for task, dynamic, int8, half, batch in product(TASKS, [True, False], [True, False], [True, False], [1, 2])
        for task, dynamic, int8, half, batch in product(TASKS, [True], [True], [False], [2])
        if not (int8 and half)  # 排除 int8 和 half 同时为 True 的情况
    ],
)
def test_export_engine_matrix(task, dynamic, int8, half, batch):
    """测试 YOLO 模型导出至 TensorRT 格式，并在不同配置下进行推理。"""
    file = YOLO(TASK2MODEL[task]).export(
        format="engine",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
        data=TASK2DATA[task],
        workspace=1,  # 降低 workspace GB 以减少测试期间的资源消耗
        simplify=True,  # 使用 'onnxslim' 进行简化
    )
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # 对导出的模型进行推理
    Path(file).unlink()  # 清理导出文件
    Path(file).with_suffix(".cache").unlink() if int8 else None  # 清理 INT8 缓存


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA 不可用")
def test_train():
    """在可用的 CUDA 设备上测试模型训练（使用最小数据集）。"""
    device = 0 if CUDA_DEVICE_COUNT == 1 else [0, 1]
    YOLO(MODEL).train(data="coco8.yaml", imgsz=64, epochs=1, device=device)  # 需要 imgsz >= 64


@pytest.mark.slow
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA 不可用")
def test_predict_multiple_devices():
    """验证模型在 CPU 和 CUDA 设备上的推理一致性。"""
    model = YOLO("yolo11n.pt")
    model = model.cpu()
    assert str(model.device) == "cpu"
    _ = model(SOURCE)  # 在 CPU 上推理
    assert str(model.device) == "cpu"

    model = model.to("cuda:0")
    assert str(model.device) == "cuda:0"
    _ = model(SOURCE)  # 在 CUDA 上推理
    assert str(model.device) == "cuda:0"

    model = model.cpu()
    assert str(model.device) == "cpu"
    _ = model(SOURCE)  # 在 CPU 上推理
    assert str(model.device) == "cpu"

    model = model.cuda()
    assert str(model.device) == "cuda:0"
    _ = model(SOURCE)  # 在 CUDA 上推理
    assert str(model.device) == "cuda:0"


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA 不可用")
def test_autobatch():
    """使用自动批量计算工具检查 YOLO 模型的最优批量大小。"""
    from ultralytics.utils.autobatch import check_train_batch_size

    check_train_batch_size(YOLO(MODEL).model.cuda(), imgsz=128, amp=True)


@pytest.mark.slow
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA 不可用")
def test_utils_benchmarks():
    """对 YOLO 模型进行性能基准测试。"""
    from ultralytics.utils.benchmarks import ProfileModels

    # 预先导出一个动态的 engine 模型，以便进行动态推理
    YOLO(MODEL).export(format="engine", imgsz=32, dynamic=True, batch=1)
    ProfileModels([MODEL], imgsz=32, half=False, min_time=1, num_timed_runs=3, num_warmup_runs=1).profile()


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA 不可用")
def test_predict_sam():
    """使用不同的提示（包括边界框和点标注）测试 SAM 模型的预测能力。"""
    from ultralytics import SAM
    from ultralytics.models.sam import Predictor as SAMPredictor

    # 加载模型
    model = SAM(WEIGHTS_DIR / "sam2.1_b.pt")

    # 显示模型信息（可选）
    model.info()

    # 进行推理
    model(SOURCE, device=0)

    # 使用边界框进行推理
    model(SOURCE, bboxes=[439, 437, 524, 709], device=0)

    # 进行推理（不带标签）
    model(ASSETS / "zidane.jpg", points=[900, 370], device=0)

    # 使用 1D 点和 1D 标签进行推理
    model(ASSETS / "zidane.jpg", points=[900, 370], labels=[1], device=0)

    # 使用 2D 点和 1D 标签进行推理
    model(ASSETS / "zidane.jpg", points=[[900, 370]], labels=[1], device=0)

    # 使用多个 2D 点和 1D 标签进行推理
    model(ASSETS / "zidane.jpg", points=[[400, 370], [900, 370]], labels=[1, 1], device=0)

    # 使用 3D 点和 2D 标签（每个目标多个点）进行推理
    model(ASSETS / "zidane.jpg", points=[[[900, 370], [1000, 100]]], labels=[[1, 1]], device=0)

    # 创建 SAMPredictor
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model=WEIGHTS_DIR / "mobile_sam.pt")
    predictor = SAMPredictor(overrides=overrides)

    # 设置图片
    predictor.set_image(ASSETS / "zidane.jpg")  # 通过图像文件设置
    # predictor(bboxes=[439, 437, 524, 709])
    # predictor(points=[900, 370], labels=[1])

    # 重置图像
    predictor.reset_image()
