# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

import argparse
import cv2.dnn
import numpy as np
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

# 加载 COCO 数据集的类别名称
CLASSES = yaml_load(check_yaml("coco8.yaml"))["names"]
# 为每个类别生成随机颜色
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    在输入图像上绘制边界框。

    参数:
        img (numpy.ndarray): 需要绘制边界框的输入图像。
        class_id (int): 目标的类别 ID。
        confidence (float): 目标的置信度分数。
        x (int): 边界框左上角的 X 坐标。
        y (int): 边界框左上角的 Y 坐标。
        x_plus_w (int): 边界框右下角的 X 坐标。
        y_plus_h (int): 边界框右下角的 Y 坐标。
    """
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    # 在图像上绘制矩形框
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    # 在图像上绘制类别标签
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model, input_image):
    """
    加载 ONNX 模型，执行目标检测，并绘制边界框，最终显示检测结果。

    参数:
        onnx_model (str): ONNX 模型的路径。
        input_image (str): 输入图像的路径。

    返回:
        list: 包含检测信息的字典列表，包括类别 ID、类别名称、置信度、边界框信息等。
    """
    # 加载 ONNX 模型
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)

    # 读取输入图像
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape

    # 处理输入图像，使其成为方形
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # 计算缩放比例
    scale = length / 640

    # 对图像进行预处理并转换为模型输入格式
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    # 进行推理
    outputs = model.forward()

    # 处理推理输出
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    # 存储边界框、置信度和类别 ID
    boxes = []
    scores = []
    class_ids = []

    # 遍历模型输出，提取边界框、置信度和类别 ID
    for i in range(rows):
        # 获取类别得分
        classes_scores = outputs[0][i][4:]
        # 获取最高类别得分及其索引
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)

        # 仅保留置信度高于阈值的目标
        if maxScore >= 0.25:
            # 计算边界框坐标
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),  # 左上角 X 坐标
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),  # 左上角 Y 坐标
                outputs[0][i][2],  # 宽度
                outputs[0][i][3],  # 高度
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # 进行非极大值抑制（NMS），去除重叠检测框
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []

    # 遍历 NMS 过滤后的检测框并绘制
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )

    # 显示带有检测框的图像
    cv2.imshow("image", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov11n.onnx", help="输入 ONNX 模型路径。")
    parser.add_argument("--img", default=str(ASSETS / "bus.jpg"), help="输入图像路径。")
    args = parser.parse_args()

    # 运行主函数
    main(args.model, args.img)
