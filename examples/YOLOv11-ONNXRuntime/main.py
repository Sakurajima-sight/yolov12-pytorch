# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

import argparse
import cv2
import numpy as np
import onnxruntime as ort
import torch

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml
import tkinter as tk


class YOLOv11:
    """YOLOv11 目标检测模型类，用于处理推理和可视化任务"""

    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        """
        初始化 YOLOv11 类的实例。

        参数:
            onnx_model: ONNX 模型的路径。
            input_image: 输入图像的路径。
            confidence_thres: 置信度阈值，用于筛选检测结果。
            iou_thres: IoU（交并比）阈值，用于非极大值抑制（NMS）。
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # 从 COCO 数据集加载类别名称
        self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]

        # 为类别生成随机颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        在输入图像上绘制检测到的边界框和标签。

        参数:
            img: 需要绘制检测框的输入图像。
            box: 检测到的边界框坐标。
            score: 该检测框的置信度分数。
            class_id: 该检测目标的类别 ID。

        返回:
            None
        """
        # 提取边界框的坐标
        x1, y1, w, h = box

        # 获取类别 ID 对应的颜色
        color = self.color_palette[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # 生成类别标签文本（包含类别名称和置信度分数）
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 计算文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 计算文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # 绘制一个填充矩形作为标签背景
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # 在图像上绘制文本标签
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self):
        """
        对输入图像进行预处理，以便进行推理。

        返回:
            image_data: 预处理后的图像数据，准备用于推理。
        """
        # 使用 OpenCV 读取输入图像
        self.img = cv2.imread(self.input_image)

        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.img.shape[:2]

        # 将图像的颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # 调整图像大小以匹配模型的输入尺寸
        img = cv2.resize(img, (self.input_width, self.input_height))

        # 归一化图像数据（将像素值缩放到 [0,1]）
        image_data = np.array(img) / 255.0

        # 重新排列维度，将通道维度放到最前面
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先

        # 增加 batch 维度，使其符合模型输入形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # 返回预处理后的图像数据
        return image_data

    def postprocess(self, input_image, output):
        """
        对模型的输出结果进行后处理，提取边界框、置信度和类别 ID。

        参数:
            input_image (numpy.ndarray): 原始输入图像。
            output (numpy.ndarray): 模型输出的数据。

        返回:
            numpy.ndarray: 带有检测结果的图像。
        """
        # 转置并压缩输出数据，使其符合预期形状
        outputs = np.transpose(np.squeeze(output[0]))

        # 获取输出数据的行数
        rows = outputs.shape[0]

        # 存储检测到的边界框、置信度和类别 ID
        boxes = []
        scores = []
        class_ids = []

        # 计算边界框坐标的缩放因子
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # 遍历每一行的输出数据
        for i in range(rows):
            # 提取该行的类别得分
            classes_scores = outputs[i][4:]

            # 获取最高类别得分
            max_score = np.amax(classes_scores)

            # 如果最高得分超过置信度阈值
            if max_score >= self.confidence_thres:
                # 获取得分最高的类别 ID
                class_id = np.argmax(classes_scores)

                # 提取边界框坐标
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 计算缩放后的边界框坐标
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # 存储类别 ID、得分和边界框
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # 应用非极大值抑制（NMS）过滤重叠的边界框
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # 遍历 NMS 过滤后的结果
        for i in indices:
            # 获取对应的边界框、得分和类别 ID
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # 在图像上绘制检测结果
            self.draw_detections(input_image, box, score, class_id)

        # 返回带有检测结果的图像
        return input_image

    def main(self):
        """
        使用 ONNX 模型执行目标检测，并返回带有检测框的图像。

        返回:
            output_img: 处理后的输出图像，带有检测框。
        """
        # 创建 ONNX 运行时会话，支持 GPU（CUDA）或 CPU 运行
        session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # 获取模型输入信息
        model_inputs = session.get_inputs()

        # 读取输入张量的尺寸
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # 预处理输入图像
        img_data = self.preprocess()

        # 执行推理
        outputs = session.run(None, {model_inputs[0].name: img_data})

        # 进行后处理并返回结果
        return self.postprocess(self.img, outputs)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov11n.onnx", help="输入 ONNX 模型路径。")
    parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="输入图像路径。")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU 阈值")
    args = parser.parse_args()

    # 创建 YOLOv11 实例
    detector = YOLOv11(
        onnx_model=args.model,
        input_image=args.img,
        confidence_thres=args.conf_thres,
        iou_thres=args.iou_thres
    )

    # 执行检测
    output_image = detector.main()

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

