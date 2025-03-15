# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license
import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.ops as ops

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml
import tkinter as tk

class RTDETR:
    """RTDETR 目标检测模型类，用于处理推理和可视化。"""

    def __init__(self, model_path, img_path, conf_thres=0.5, iou_thres=0.5):
        """
        使用指定的参数初始化 RTDETR 对象。

        参数：
            model_path: ONNX 模型文件的路径。
            img_path: 输入图像的路径。
            conf_thres: 目标检测的置信度阈值。
            iou_thres: 非最大抑制的 IoU 阈值
        """
        self.model_path = model_path
        self.img_path = img_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # 使用 CUDA 和 CPU 执行提供程序设置 ONNX 运行时会话
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider", "CUDAExecutionProvider"])
        self.model_input = self.session.get_inputs()
        self.input_width = self.model_input[0].shape[2]
        print(self.input_width)
        self.input_height = self.model_input[0].shape[3]
        print(self.input_height)


        # 从 COCO 数据集的 YAML 文件加载类名
        self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]

        # 生成用于绘制边界框的颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, box, score, class_id):
        """
        根据检测到的目标，在输入图像上绘制边界框和标签。

        参数：
            box: 检测到的边界框。
            score: 相应的检测得分。
            class_id: 检测到的目标的类 ID。

        返回：
            None
        """
        # 提取边界框的坐标
        x1, y1, x2, y2 = box

        # 获取类 ID 对应的颜色
        color = self.color_palette[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(self.img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # 创建带有类名和得分的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"

        fontScale = min(self.img_width, self.img_height) * 0.0005
        fontThickness = max(1, int(fontScale * 2))
        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontThickness)

        # 计算文本初始位置
        label_x = int(x1)
        label_y = int(y1) - 10  # 默认放在边界框上方

        # **确保标签不会超出左上角 (0, 0)**
        if label_x < 0:
            label_x = 5  # 向右偏移
        if label_y < label_height:
            label_y = int(y1) + label_height + 5  # 放到底部，并留一点间隙

        # **确保标签不会超出右下角**
        if label_x + label_width > self.img.shape[1]:  # 超出右边界
            label_x = self.img.shape[1] - label_width - 5  # 向左偏移
        if label_y + label_height > self.img.shape[0]:  # 超出底部
            label_y = self.img.shape[0] - 5  # 移动到图像边界内

        # 绘制一个填充矩形作为标签文本的背景
        cv2.rectangle(
            self.img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # 在图像上绘制标签文本
        cv2.putText(
            self.img, label, (int(label_x), int(label_y + fontScale * 10)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), fontThickness, cv2.LINE_AA
        )

    def preprocess(self):
        """
        对输入图像进行预处理，以便进行推理。

        返回：
            image_data: 预处理后的图像数据，准备进行推理。
        """
        # 使用 OpenCV 读取输入图像
        self.img = cv2.imread(self.img_path)

        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.img.shape[:2]

        # 将图像的颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # 调整图像大小以匹配输入形状
        img = cv2.resize(img, (self.input_width, self.input_height))

        # 通过除以 255.0 来归一化图像数据
        image_data = np.array(img) / 255.0

        # 转置图像，使得通道维度为第一个维度
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道在前

        # 扩展图像数据的维度，以匹配预期的输入形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # 返回预处理后的图像数据
        return image_data

    def bbox_cxcywh_to_xyxy(self, boxes):
        """
        将边界框从 (中心 x, 中心 y, 宽度, 高度) 格式转换为 (x_min, y_min, x_max, y_max) 格式。

        参数：
            boxes (numpy.ndarray): 形状为 (N, 4) 的数组，每一行表示一个 (cx, cy, w, h) 格式的边界框。

        返回：
            numpy.ndarray: 形状为 (N, 4) 的数组，每一行表示一个 (x_min, y_min, x_max, y_max) 格式的边界框。
        """
        # 计算边界框的半宽和半高
        half_width = boxes[:, 2] / 2
        half_height = boxes[:, 3] / 2

        # 计算边界框的坐标
        x_min = boxes[:, 0] - half_width
        y_min = boxes[:, 1] - half_height
        x_max = boxes[:, 0] + half_width
        y_max = boxes[:, 1] + half_height

        # 返回 (x_min, y_min, x_max, y_max) 格式的边界框
        return np.column_stack((x_min, y_min, x_max, y_max))

    def postprocess(self, model_output):
        """
        对模型输出进行后处理，以提取检测结果并绘制在输入图像上。

        参数：
            model_output: 模型推理的输出。

        返回：
            np.array: 带有检测结果的标注图像。
        """
        # 压缩模型输出，去除多余的维度
        outputs = np.squeeze(model_output[0]).T

        # 从模型输出中提取边界框和得分
        boxes = outputs[:, :4]
        scores = outputs[:, 4:]

        # 获取每个检测的类标签和得分
        labels = np.argmax(scores, axis=1)
        scores = np.max(scores, axis=1)

        # 应用置信度阈值，过滤掉低置信度的检测
        mask = scores > self.conf_thres
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        boxes[:, 0] /= self.model_input[0].shape[2]   # cx 归一化
        boxes[:, 1] /= self.model_input[0].shape[3]  # cy 归一化
        boxes[:, 2] /= self.model_input[0].shape[2]   # w 归一化
        boxes[:, 3] /= self.model_input[0].shape[3]  # h 归一化

        # 将边界框转换为 (x_min, y_min, x_max, y_max) 格式
        boxes = self.bbox_cxcywh_to_xyxy(boxes)

        # 缩放边界框以匹配原始图像的尺寸
        boxes[:, 0::2] *= self.img_width
        boxes[:, 1::2] *= self.img_height

        # 非极大值抑制 (NMS)
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)

            keep = ops.nms(boxes_tensor, scores_tensor, self.iou_thres)  # 使用 `iou-thres`
            
            boxes = boxes[keep.numpy()]
            scores = scores[keep.numpy()]
            labels = labels[keep.numpy()]


        # 在图像上绘制检测结果
        for box, score, label in zip(boxes, scores, labels):
            self.draw_detections(box, score, label)

        # 返回带有标注的图像
        return self.img

    def main(self):
        """
        使用 ONNX 模型在输入图像上执行目标检测。

        返回：
            np.array: 带有标注的输出图像。
        """
        # 对图像进行预处理，准备模型输入
        image_data = self.preprocess()

        # 运行模型推理
        model_output = self.session.run(None, {self.model_input[0].name: image_data})

        # 处理并返回模型输出
        return self.postprocess(model_output)


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rtdetr-l.onnx", help="ONNX 模型文件的路径。")
    parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="输入图像的路径。")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="目标检测的置信度阈值。")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="非最大抑制的 IoU 阈值。")
    args = parser.parse_args()

    # 检查依赖项并设置 ONNX 运行时
    check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")

    # 创建检测器实例并指定参数
    detection = RTDETR(args.model, args.img, args.conf_thres, args.iou_thres)

    # 执行检测并获取输出图像
    output_image = detection.main()

    # 获取屏幕大小
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()  # 屏幕宽度
    screen_height = root.winfo_screenheight()  # 屏幕高度
    root.destroy()  # 关闭 Tkinter 窗口

    # 计算缩放比例，保持宽高比
    height, width = output_image.shape[:2]
    scale = min((screen_width / width) * 0.8, (screen_height / height) * 0.8)

    # 仅当图片过大时缩小
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        output_image = cv2.resize(output_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 显示带有标注的输出图像
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
