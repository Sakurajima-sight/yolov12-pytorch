# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

import contextlib
import os
import subprocess
import time
from pathlib import Path

import pytest

from tests import MODEL, SOURCE, TMP
from ultralytics import YOLO, download
from ultralytics.utils import DATASETS_DIR, SETTINGS
from ultralytics.utils.checks import check_requirements


@pytest.mark.skipif(not check_requirements("ray", install=False), reason="ray[tune] 未安装")
def test_model_ray_tune():
    """使用 Ray 进行超参数优化以微调 YOLO 模型。"""
    YOLO("yolo11n-cls.yaml").tune(
        use_ray=True, data="imagenet10", grace_period=1, iterations=1, imgsz=32, epochs=1, plots=False, device="cpu"
    )


@pytest.mark.skipif(not check_requirements("mlflow", install=False), reason="mlflow 未安装")
def test_mlflow():
    """测试在训练过程中启用 MLflow 进行跟踪（详情见 https://mlflow.org/）。"""
    SETTINGS["mlflow"] = True
    YOLO("yolo11n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=3, plots=False, device="cpu")
    SETTINGS["mlflow"] = False


@pytest.mark.skipif(True, reason="在 CI 计划任务中测试失败 https://github.com/ultralytics/ultralytics/pull/8868")
@pytest.mark.skipif(not check_requirements("mlflow", install=False), reason="mlflow 未安装")
def test_mlflow_keep_run_active():
    """确保 MLflow 运行状态符合环境变量 MLFLOW_KEEP_RUN_ACTIVE 的设置。"""
    import mlflow

    SETTINGS["mlflow"] = True
    run_name = "测试运行"
    os.environ["MLFLOW_RUN"] = run_name

    # 测试 MLFLOW_KEEP_RUN_ACTIVE=True 时的行为
    os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "True"
    YOLO("yolo11n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
    status = mlflow.active_run().info.status
    assert status == "RUNNING", "当 MLFLOW_KEEP_RUN_ACTIVE=True 时，MLflow 运行状态应为 RUNNING"

    run_id = mlflow.active_run().info.run_id

    # 测试 MLFLOW_KEEP_RUN_ACTIVE=False 时的行为
    os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "False"
    YOLO("yolo11n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
    status = mlflow.get_run(run_id=run_id).info.status
    assert status == "FINISHED", "当 MLFLOW_KEEP_RUN_ACTIVE=False 时，MLflow 运行状态应为 FINISHED"

    # 测试未设置 MLFLOW_KEEP_RUN_ACTIVE 时的行为
    os.environ.pop("MLFLOW_KEEP_RUN_ACTIVE", None)
    YOLO("yolo11n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
    status = mlflow.get_run(run_id=run_id).info.status
    assert status == "FINISHED", "默认情况下，MLflow 运行状态应为 FINISHED"
    SETTINGS["mlflow"] = False


@pytest.mark.skipif(not check_requirements("tritonclient", install=False), reason="tritonclient[all] 未安装")
def test_triton():
    """
    使用 NVIDIA Triton 服务器测试 YOLO 模型。

    详情请见 https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver。
    """
    check_requirements("tritonclient[all]")
    from tritonclient.http import InferenceServerClient  # noqa

    # 定义变量
    model_name = "yolo"
    triton_repo = TMP / "triton_repo"  # Triton 模型仓库路径
    triton_model = triton_repo / model_name  # Triton 模型路径

    # 导出模型为 ONNX 格式
    f = YOLO(MODEL).export(format="onnx", dynamic=True)

    # 准备 Triton 仓库
    (triton_model / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model / "1" / "model.onnx")
    (triton_model / "config.pbtxt").touch()

    # 定义 Docker 镜像（https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver）
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"  # 6.4 GB 大小

    # 拉取 Docker 镜像
    subprocess.call(f"docker pull {tag}", shell=True)

    # 运行 Triton 服务器并捕获容器 ID
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_repo}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )

    # 等待 Triton 服务器启动
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)

    # 等待模型准备就绪
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready(model_name)
            break
        time.sleep(1)

    # 检查 Triton 推理
    YOLO(f"http://localhost:8000/{model_name}", "detect")(SOURCE)  # 进行推理

    # 结束并删除 Docker 容器
    subprocess.call(f"docker kill {container_id}", shell=True)


@pytest.mark.skipif(not check_requirements("pycocotools", install=False), reason="pycocotools 未安装")
def test_pycocotools():
    """使用 pycocotools 验证 YOLO 模型在 COCO 数据集上的预测结果。"""
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.models.yolo.pose import PoseValidator
    from ultralytics.models.yolo.segment import SegmentationValidator

    # 下载 COCO 数据集的标注文件
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"

    # 目标检测验证
    args = {"model": "yolo11n.pt", "data": "coco8.yaml", "save_json": True, "imgsz": 64}
    validator = DetectionValidator(args=args)
    validator()
    validator.is_coco = True
    download(f"{url}instances_val2017.json", dir=DATASETS_DIR / "coco8/annotations")
    _ = validator.eval_json(validator.stats)

    # 语义分割验证
    args = {"model": "yolo11n-seg.pt", "data": "coco8-seg.yaml", "save_json": True, "imgsz": 64}
    validator = SegmentationValidator(args=args)
    validator()
    validator.is_coco = True
    download(f"{url}instances_val2017.json", dir=DATASETS_DIR / "coco8-seg/annotations")
    _ = validator.eval_json(validator.stats)

    # 姿态估计验证
    args = {"model": "yolo11n-pose.pt", "data": "coco8-pose.yaml", "save_json": True, "imgsz": 64}
    validator = PoseValidator(args=args)
    validator()
    validator.is_coco = True
    download(f"{url}person_keypoints_val2017.json", dir=DATASETS_DIR / "coco8-pose/annotations")
    _ = validator.eval_json(validator.stats)
