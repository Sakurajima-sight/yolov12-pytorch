# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import time
from collections import defaultdict
from typing import List, Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from ultralytics import YOLO
from ultralytics.data.loaders import get_best_youtube_url
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.torch_utils import select_device


class TorchVisionVideoClassifier:
    """使用预训练的TorchVision模型分类视频；请参见https://pytorch.org/vision/stable/。"""

    from torchvision.models.video import (
        MViT_V1_B_Weights,
        MViT_V2_S_Weights,
        R3D_18_Weights,
        S3D_Weights,
        Swin3D_B_Weights,
        Swin3D_T_Weights,
        mvit_v1_b,
        mvit_v2_s,
        r3d_18,
        s3d,
        swin3d_b,
        swin3d_t,
    )

    model_name_to_model_and_weights = {
        "s3d": (s3d, S3D_Weights.DEFAULT),  # --num-video-sequence-samples需要设置成16
        "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
        "swin3d_t": (swin3d_t, Swin3D_T_Weights.DEFAULT),
        "swin3d_b": (swin3d_b, Swin3D_B_Weights.DEFAULT),
        "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights.DEFAULT),  # --num-video-sequence-samples需要设置成16
        "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights.DEFAULT),  # --num-video-sequence-samples需要设置成16
    }

    def __init__(self, model_name: str, device: Union[str, torch.device] = ""):
        """
        使用指定的模型名称和设备初始化VideoClassifier。

        参数:
            model_name (str): 使用的模型名称。
            device (str 或 torch.device, 可选): 运行模型的设备。默认值为空字符串。

        异常:
            ValueError: 如果提供了无效的模型名称。
        """
        if model_name not in self.model_name_to_model_and_weights:
            raise ValueError(f"无效的模型名称 '{model_name}'。可用的模型: {self.available_model_names()}")
        model, self.weights = self.model_name_to_model_and_weights[model_name]
        self.device = select_device(device)
        self.model = model(weights=self.weights).to(self.device).eval()

    @staticmethod
    def available_model_names() -> List[str]:
        """
        获取可用模型名称列表。

        返回:
            list: 可用模型名称的列表。
        """
        return list(TorchVisionVideoClassifier.model_name_to_model_and_weights.keys())

    def preprocess_crops_for_video_cls(self, crops: List[np.ndarray], input_size: list = None) -> torch.Tensor:
        """
        对视频分类的裁剪图像进行预处理。

        参数:
            crops (List[np.ndarray]): 要预处理的裁剪图像列表。每个裁剪图像的维度应为 (H, W, C)
            input_size (tuple, 可选): 模型的目标输入尺寸。默认值为 (224, 224)。

        返回:
            torch.Tensor: 预处理后的裁剪图像张量，维度为 (1, T, C, H, W)。
        """
        if input_size is None:
            input_size = [224, 224]
        from torchvision.transforms import v2

        transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(input_size, antialias=True),
                v2.Normalize(mean=self.weights.transforms().mean, std=self.weights.transforms().std),
            ]
        )

        processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]
        return torch.stack(processed_crops).unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)

    def __call__(self, sequences: torch.Tensor):
        """
        对给定的序列进行推理。

        参数:
            sequences (torch.Tensor): 模型的输入序列。预期的输入维度为 (B, T, C, H, W)，
                                      即批量视频帧，或者 (T, C, H, W)，即单一视频帧。

        返回:
            torch.Tensor: 模型的输出。
        """
        with torch.inference_mode():
            return self.model(sequences)

    def postprocess(self, outputs: torch.Tensor) -> Tuple[List[str], List[float]]:
        """
        对模型的批量输出进行后处理。

        参数:
            outputs (torch.Tensor): 模型的输出。

        返回:
            List[str]: 预测的标签。
            List[float]: 预测的置信度。
        """
        pred_labels = []
        pred_confs = []
        for output in outputs:
            pred_class = output.argmax(0).item()
            pred_label = self.weights.meta["categories"][pred_class]
            pred_labels.append(pred_label)
            pred_conf = output.softmax(0)[pred_class].item()
            pred_confs.append(pred_conf)

        return [pred_labels], [pred_confs]


class HuggingFaceVideoClassifier:
    """使用Hugging Face模型进行零样本视频分类，适用于各种设备。"""

    def __init__(self, labels: List[str], model_name: str = "microsoft/xclip-base-patch16-zero-shot", device: Union[str, torch.device] = "", fp16: bool = False):
        """
        使用指定的模型名称初始化 HuggingFaceVideoClassifier。

        参数:
            labels (List[str]): 零样本分类的标签列表。
            model_name (str): 使用的模型名称。默认值为 "microsoft/xclip-base-patch16-zero-shot"。
            device (str 或 torch.device, 可选): 运行模型的设备。默认值为空字符串。
            fp16 (bool, 可选): 是否使用FP16进行推理。默认值为 False。
        """
        self.fp16 = fp16
        self.labels = labels
        self.device = select_device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        if fp16:
            model = model.half()
        self.model = model.eval()

    def preprocess_crops_for_video_cls(self, crops: List[np.ndarray], input_size: list = None) -> torch.Tensor:
        """
        对视频分类的裁剪图像进行预处理。

        参数:
            crops (List[np.ndarray]): 要预处理的裁剪图像列表。每个裁剪图像的维度应为 (H, W, C)
            input_size (tuple, 可选): 模型的目标输入尺寸。默认值为 (224, 224)。

        返回:
            torch.Tensor: 预处理后的裁剪图像张量，维度为 (1, T, C, H, W)。
        """
        if input_size is None:
            input_size = [224, 224]
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.float() / 255.0),
                transforms.Resize(input_size),
                transforms.Normalize(
                    mean=self.processor.image_processor.image_mean, std=self.processor.image_processor.image_std
                ),
            ]
        )

        processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]  # (T, C, H, W)
        output = torch.stack(processed_crops).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        if self.fp16:
            output = output.half()
        return output

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        对给定的序列进行推理。

        参数:
            sequences (torch.Tensor): 模型的输入序列。批量视频帧，形状为 (B, T, H, W, C)。

        返回:
            torch.Tensor: 模型的输出。
        """
        input_ids = self.processor(text=self.labels, return_tensors="pt", padding=True)["input_ids"].to(self.device)

        inputs = {"pixel_values": sequences, "input_ids": input_ids}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        return outputs.logits_per_video

    def postprocess(self, outputs: torch.Tensor) -> Tuple[List[List[str]], List[List[float]]]:
        """
        对模型的批量输出进行后处理。

        参数:
            outputs (torch.Tensor): 模型的输出。

        返回:
            List[List[str]]: 预测的前3个标签。
            List[List[float]]: 预测的前3个置信度。
        """
        pred_labels = []
        pred_confs = []

        with torch.no_grad():
            logits_per_video = outputs  # 假设 outputs 已经是 logits 张量
            probs = logits_per_video.softmax(dim=-1)  # 使用 softmax 将 logits 转换为概率

        for prob in probs:
            top2_indices = prob.topk(2).indices.tolist()
            top2_labels = [self.labels[idx] for idx in top2_indices]
            top2_confs = prob[top2_indices].tolist()
            pred_labels.append(top2_labels)
            pred_confs.append(top2_confs)

        return pred_labels, pred_confs


def crop_and_pad(frame, box, margin_percent):
    """裁剪带有边距的框，并从帧中获取方形裁剪图像。"""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1

    # 添加边距
    margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
    x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)

    # 从帧中获取方形裁剪图像
    size = max(y2 - y1, x2 - x1)
    center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
    half_size = size // 2
    square_crop = frame[
        max(0, center_y - half_size) : min(frame.shape[0], center_y + half_size),
        max(0, center_x - half_size) : min(frame.shape[1], center_x + half_size),
    ]

    return cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_LINEAR)


def run(
    weights: str = "yolo11n.pt",
    device: str = "",
    source: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_path: Optional[str] = None,
    crop_margin_percentage: int = 10,
    num_video_sequence_samples: int = 8,
    skip_frame: int = 2,
    video_cls_overlap_ratio: float = 0.25,
    fp16: bool = False,
    video_classifier_model: str = "microsoft/xclip-base-patch32",
    labels: List[str] = None,
) -> None:
    """
    使用YOLO进行目标检测和视频分类器对视频源进行动作识别。

    参数:
        weights (str): YOLO模型权重的路径。默认值为 "yolo11n.pt"。
        device (str): 运行模型的设备。使用'cuda'表示NVIDIA GPU，'mps'表示Apple Silicon，'cpu'表示CPU。默认值为自动检测。
        source (str): 视频文件路径或YouTube URL。默认值为一个示例YouTube视频。
        output_path (Optional[str], 可选): 输出视频的路径。默认值为 None。
        crop_margin_percentage (int, 可选): 对检测到的物体添加的边距百分比。默认值为10。
        num_video_sequence_samples (int, 可选): 用于分类的视频帧数量。默认值为8。
        skip_frame (int, 可选): 每次检测之间跳过的帧数。默认值为4。
        video_cls_overlap_ratio (float, 可选): 视频序列之间的重叠比例。默认值为0.25。
        fp16 (bool, 可选): 是否使用FP16进行推理。默认值为False。
        video_classifier_model (str, 可选): 视频分类模型名称或路径。默认值为 "microsoft/xclip-base-patch32"。
        labels (List[str], 可选): 零样本分类的标签列表。默认值为预定义列表。

    返回:
        None
    """
    if labels is None:
        labels = [
            "walking",
            "running",
            "brushing teeth",
            "looking into phone",
            "weight lifting",
            "cooking",
            "sitting",
        ]
    # 初始化模型和设备
    device = select_device(device)
    yolo_model = YOLO(weights).to(device)
    if video_classifier_model in TorchVisionVideoClassifier.available_model_names():
        print("'fp16' 不支持 TorchVisionVideoClassifier。将 fp16 设置为 False。")
        print(
            "'labels' 未用于 TorchVisionVideoClassifier。忽略提供的标签，使用 Kinetics-400 标签。"
        )
        video_classifier = TorchVisionVideoClassifier(video_classifier_model, device=device)
    else:
        video_classifier = HuggingFaceVideoClassifier(
            labels, model_name=video_classifier_model, device=device, fp16=fp16
        )

    # 初始化视频捕获
    if source.startswith("http") and urlparse(source).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}:
        source = get_best_youtube_url(source)
    elif not source.endswith(".mp4"):
        raise ValueError("无效的源。支持的源为 YouTube URL 和 MP4 文件。")
    cap = cv2.VideoCapture(source)

    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化 VideoWriter
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 初始化跟踪历史记录
    track_history = defaultdict(list)
    frame_counter = 0

    track_ids_to_infer = []
    crops_to_infer = []
    pred_labels = []
    pred_confs = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_counter += 1

        # 运行 YOLO 跟踪
        results = yolo_model.track(frame, persist=True, classes=[0])  # 仅跟踪人物类

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            # 可视化预测
            annotator = Annotator(frame, line_width=3, font_size=10, pil=False)

            if frame_counter % skip_frame == 0:
                crops_to_infer = []
                track_ids_to_infer = []

            for box, track_id in zip(boxes, track_ids):
                if frame_counter % skip_frame == 0:
                    crop = crop_and_pad(frame, box, crop_margin_percentage)
                    track_history[track_id].append(crop)

                if len(track_history[track_id]) > num_video_sequence_samples:
                    track_history[track_id].pop(0)

                if len(track_history[track_id]) == num_video_sequence_samples and frame_counter % skip_frame == 0:
                    start_time = time.time()
                    crops = video_classifier.preprocess_crops_for_video_cls(track_history[track_id])
                    end_time = time.time()
                    preprocess_time = end_time - start_time
                    print(f"视频分类预处理时间: {preprocess_time:.4f} 秒")
                    crops_to_infer.append(crops)
                    track_ids_to_infer.append(track_id)

            if crops_to_infer and (
                not pred_labels
                or frame_counter % int(num_video_sequence_samples * skip_frame * (1 - video_cls_overlap_ratio)) == 0
            ):
                crops_batch = torch.cat(crops_to_infer, dim=0)

                start_inference_time = time.time()
                output_batch = video_classifier(crops_batch)
                end_inference_time = time.time()
                inference_time = end_inference_time - start_inference_time
                print(f"视频分类推理时间: {inference_time:.4f} 秒")

                pred_labels, pred_confs = video_classifier.postprocess(output_batch)

            if track_ids_to_infer and crops_to_infer:
                for box, track_id, pred_label, pred_conf in zip(boxes, track_ids_to_infer, pred_labels, pred_confs):
                    top2_preds = sorted(zip(pred_label, pred_conf), key=lambda x: x[1], reverse=True)
                    label_text = " | ".join([f"{label} ({conf:.2f})" for label, conf in top2_preds])
                    annotator.box_label(box, label_text, color=(0, 0, 255))

        # 将注释后的帧写入输出视频
        if output_path is not None:
            out.write(frame)

        # 显示注释后的帧
        cv2.imshow("YOLOv11 Tracking with S3D Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if output_path is not None:
        out.release()
    cv2.destroyAllWindows()


def parse_opt():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Ultralytics YOLO11 模型路径，可选模型：YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x")
    parser.add_argument("--device", default="", help='cuda 设备，如 0 或 0,1,2,3 或 cpu/mps，"" 表示自动检测')
    parser.add_argument(
        "--source",
        type=str,
        default="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        help="视频文件路径或 YouTube URL",
    )
    parser.add_argument("--output-path", type=str, default="output_video.mp4", help="输出视频文件路径")
    parser.add_argument(
        "--crop-margin-percentage", type=int, default=10, help="在检测到的物体周围添加的边距百分比"
    )
    parser.add_argument(
        "--num-video-sequence-samples", type=int, default=8, help="用于分类的视频帧数量, 对于 s3d, mvit_v1_b, mvit_v2_s, 需要设置为 16 帧。"
    )
    parser.add_argument("--skip-frame", type=int, default=2, help="每次检测之间跳过的帧数")
    parser.add_argument(
        "--video-cls-overlap-ratio", type=float, default=0.25, help="视频序列之间的重叠比例"
    )
    parser.add_argument("--fp16", action="store_true", help="使用 FP16 进行推理")
    parser.add_argument(
        "--video-classifier-model", type=str, default="microsoft/xclip-base-patch32", help="视频分类模型名称"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        default=["dancing", "singing a song"],
        help="零样本视频分类的标签",
    )
    return parser.parse_args()


def main(opt):
    """主函数。"""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
