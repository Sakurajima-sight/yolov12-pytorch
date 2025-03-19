# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

import sys
from unittest import mock

from tests import MODEL
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.models.yolo import classify, detect, segment
from ultralytics.utils import ASSETS, DEFAULT_CFG, WEIGHTS_DIR

import faulthandler
import sys
faulthandler.enable(file=sys.stdout)

def test_func(*args):  # noqa
    """测试回调函数，用于评估 YOLO 模型的性能指标。"""
    print("回调测试通过")


def test_export():
    """测试模型导出功能，通过添加回调并断言其执行情况。"""
    exporter = Exporter()
    exporter.add_callback("on_export_start", test_func)
    assert test_func in exporter.callbacks["on_export_start"], "回调测试失败"
    f = exporter(model=YOLO("yolo11n.yaml").model)
    YOLO(f)(ASSETS)  # 对导出的模型进行推理


def test_detect():
    """测试 YOLO 目标检测的训练、验证和预测功能。"""
    overrides = {"data": "coco8.yaml", "model": "yolo11n.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "coco8.yaml"
    cfg.imgsz = 32

    # 训练器
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "回调测试失败"
    trainer.train()

    # 验证器
    val = detect.DetectionValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "回调测试失败"
    val(model=trainer.best)  # 验证 best.pt

    # 预测器
    pred = detect.DetectionPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "回调测试失败"
    
    # 确保 sys.argv 为空不会引发问题
    with mock.patch.object(sys, "argv", []):
        result = pred(source=ASSETS, model=MODEL)
        assert len(result), "预测器测试失败"

    # 测试恢复训练
    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"捕获到预期的异常: {e}")
        return

    Exception("恢复训练测试失败！")


def test_segment():
    """测试 YOLO 模型的图像分割训练、验证和预测流程。"""
    overrides = {"data": "coco8-seg.yaml", "model": "yolo11n-seg.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "coco8-seg.yaml"
    cfg.imgsz = 32

    # 训练器
    trainer = segment.SegmentationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "回调测试失败"
    trainer.train()

    # 验证器
    val = segment.SegmentationValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "回调测试失败"
    val(model=trainer.best)  # 验证 best.pt

    # 预测器
    pred = segment.SegmentationPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "回调测试失败"
    result = pred(source=ASSETS, model=WEIGHTS_DIR / "yolo11n-seg.pt")
    assert len(result), "预测器测试失败"

    # 测试恢复训练
    overrides["resume"] = trainer.last
    trainer = segment.SegmentationTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"捕获到预期的异常: {e}")
        return

    Exception("恢复训练测试失败！")


def test_classify():
    """测试图像分类，包括训练、验证和预测阶段。"""
    overrides = {"data": "imagenet10", "model": "yolo11n-cls.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "imagenet10"
    cfg.imgsz = 32

    # 训练器
    trainer = classify.ClassificationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "回调测试失败"
    trainer.train()

    # 验证器
    val = classify.ClassificationValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "回调测试失败"
    val(model=trainer.best)

    # 预测器
    pred = classify.ClassificationPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "回调测试失败"
    result = pred(source=ASSETS, model=trainer.best)
    assert len(result), "预测器测试失败"
