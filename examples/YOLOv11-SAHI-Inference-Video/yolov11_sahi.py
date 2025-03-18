# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

import argparse
from pathlib import Path

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.ultralytics import download_yolo11n_model

from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

import multiprocessing
import torch

cv2.setNumThreads(multiprocessing.cpu_count())  # 🚀 让 OpenCV 充分使用 CPU 线程

class SAHIInference:
    """运行 Ultralytics YOLO11 和 SAHI 进行视频目标检测，并提供可视化、保存和跟踪结果的功能。"""

    def __init__(self):
        """初始化 SAHIInference 类，使用 SAHI 和 YOLO11 模型进行切片推理。"""
        self.detection_model = None

    def load_model(self, weights):
        """加载 YOLO11 模型，并使用 SAHI 进行目标检测。"""
        yolo11_model_path = f"models/{weights}"
        download_yolo11n_model(yolo11_model_path)  # 下载 YOLO11 预训练模型

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=yolo11_model_path, device="cpu"
        )
        # 🛠️ 开启 CUDNN 加速（如果 GPU 支持）
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
 
    def inference(
        self,
        weights="yolo11n.pt",
        source="test.mp4",
        view_img=False,
        save_img=False,
        exist_ok=False,
        classes=None,  # 🆕 从命令行获取
    ):
        """
        运行 YOLO11 和 SAHI 进行视频目标检测，并支持按类别筛选。

        参数:
            weights (str): 模型权重文件路径。
            source (str): 视频文件路径。
            view_img (bool): 是否在窗口中显示检测结果。
            save_img (bool): 是否保存检测结果的视频。
            exist_ok (bool): 是否允许覆盖已有的输出文件。
            classes (list): 只检测这些类别，例如 ["person", "car"]
        """
        # 视频设置
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), "无法读取视频文件"
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))  # 获取视频帧的宽度和高度

        # 输出目录设置
        save_dir = increment_path(Path("ultralytics_results_with_sahi") / "exp", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(
            str(save_dir / f"{Path(source).stem}.mp4"),
            cv2.VideoWriter_fourcc(*"MJPG"),
            int(cap.get(5)),  # 获取视频帧率
            (frame_width, frame_height),
        )

        # 加载模型
        self.load_model(weights)
        print("SAHI 正在使用的设备:", self.detection_model.device)
        if classes:
            print(f"🔍 仅检测以下类别: {classes}")

        # 逐帧处理视频
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            annotator = Annotator(frame)  # 初始化标注工具，用于绘制检测和跟踪结果

            # 进行目标检测，并按类别筛选
            results = get_sliced_prediction(
                frame[..., ::-1],  # BGR -> RGB 颜色转换
                self.detection_model,
                slice_height=512,  # 切片高度
                slice_width=512,  # 切片宽度
                overlap_height_ratio=0.1,  # 🚀 减少重叠，提高速度
                overlap_width_ratio=0.1,
                perform_standard_pred=True,
            )

            # 提取检测结果数据
            detection_data = [
                (det.category.name, det.category.id, (det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy))
                for det in results.object_prediction_list
                if classes is None or det.category.name in classes  # 🆕 只保留需要的类别
            ]

            # 在视频帧上绘制检测框
            for det in detection_data:
                annotator.box_label(det[2], label=str(det[0]), color=colors(int(det[1]), True))

            # 显示检测结果
            if view_img:
                cv2.imshow(Path(source).stem, frame)
            # 保存检测结果
            if save_img:
                video_writer.write(frame)

            # 按 "q" 退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # 释放资源
        video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

    def parse_opt(self):
        """解析命令行参数。"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--weights", type=str, default="yolo11n.pt", help="模型权重文件路径")
        parser.add_argument("--source", type=str, required=True, help="视频文件路径")
        parser.add_argument("--view-img", action="store_true", help="是否显示检测结果")
        parser.add_argument("--save-img", action="store_true", help="是否保存检测结果视频")
        parser.add_argument("--exist-ok", action="store_true", help="是否允许覆盖已有的输出文件")
        parser.add_argument("--classes", nargs="+", type=str, help="要检测的类别名称，例如 --classes person car")
        return parser.parse_args()


if __name__ == "__main__":
    inference = SAHIInference()
    inference.inference(**vars(inference.parse_opt()))
