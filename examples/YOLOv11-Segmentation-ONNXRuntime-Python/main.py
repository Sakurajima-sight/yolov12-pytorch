# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse

import cv2
import numpy as np
import onnxruntime as ort

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.plotting import Colors


class YOLOv11Seg:
    """YOLOv11 分割模型."""

    def __init__(self, onnx_model):
        """
        初始化函数。

        参数:
            onnx_model (str): ONNX 模型的路径。
        """
        # 创建 ONNX 运行时会话（InferenceSession）
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"],
        )

        # 确定 numpy 数据类型：支持 FP32 和 FP16 两种 ONNX 模型格式
        self.ndtype = np.half if self.session.get_inputs()[0].type == "tensor(float16)" else np.single

        # 获取模型的宽度和高度（YOLOv11-seg 仅有一个输入）
        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]

        # 加载 COCO 数据集的类别名称
        self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]

        # 创建颜色调色板
        self.color_palette = Colors()

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        """
        处理整个推理流程：预处理 -> 运行推理 -> 处理后处理结果。

        参数:
            im0 (Numpy.ndarray): 原始输入图像。
            conf_threshold (float): 置信度阈值，用于筛选预测结果。
            iou_threshold (float): IoU 阈值，用于非极大值抑制（NMS）。
            nm (int): 掩码的数量。

        返回:
            boxes (List): 预测的边界框列表。
            segments (List): 预测的分割轮廓列表。
            masks (np.ndarray): [N, H, W] 形状的输出掩码数组。
        """
        # 预处理
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # 运行 ONNX 推理
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})

        # 处理推理结果
        boxes, segments, masks = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
        )
        return boxes, segments, masks

    def preprocess(self, img):
        """
        预处理输入图像。

        参数:
            img (Numpy.ndarray): 需要处理的图像。

        返回:
            img_process (Numpy.ndarray): 经过预处理后可用于推理的图像。
            ratio (tuple): 图像缩放比例 (width, height)。
            pad_w (float): 宽度方向的填充值。
            pad_h (float): 高度方向的填充值。
        """
        # 使用 `letterbox()` 调整大小并填充输入图像（借鉴自 Ultralytics）
        shape = img.shape[:2]  # 原始图像的高度和宽度
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 计算缩放比例
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 计算缩放后的尺寸
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # 计算填充量

        # 如果尺寸不匹配，则调整大小
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 计算填充边界
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

        # 使用灰色 (114,114,114) 填充图像
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 变换: HWC（高度-宽度-通道） -> CHW（通道-高度-宽度） -> BGR 转 RGB -> 归一化 (除以 255) -> 变为连续存储 -> 添加维度
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        """
        处理推理结果。

        参数:
            preds (Numpy.ndarray): 由 `ort.session.run()` 生成的预测结果。
            im0 (Numpy.ndarray): [h, w, c] 原始输入图像。
            ratio (tuple): 缩放比例 (width, height)。
            pad_w (float): 宽度方向的填充量。
            pad_h (float): 高度方向的填充量。
            conf_threshold (float): 置信度阈值。
            iou_threshold (float): IoU 阈值。
            nm (int): 掩码数量。

        返回:
            boxes (List): 预测的边界框列表。
            segments (List): 预测的分割轮廓列表。
            masks (np.ndarray): [N, H, W] 形状的输出掩码数组。
        """
        x, protos = preds[0], preds[1]  # 预测结果包括两个部分：边界框信息和掩码原型

        # 转置维度: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum("bcn->bnc", x)

        # 根据置信度阈值筛选预测结果
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # 创建一个新矩阵，将（边界框、置信度、类别、掩码）合并为一个整体
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        # 进行非极大值抑制（NMS）
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        # 解析并返回结果
        if len(x) > 0:
            # 将边界框格式从中心点 (cxcywh) 转换为左上角-右下角 (xyxy)
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # 将边界框从模型尺寸（model_height, model_width）调整回原始图像尺寸
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # 限制边界框不能超出图像范围
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            # 处理掩码
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)

            # 将掩码转换为轮廓
            segments = self.masks2segments(masks)
            return x[..., :6], segments, masks  # 返回边界框、分割轮廓和掩码
        else:
            return [], [], []


    @staticmethod
    def masks2segments(masks):
        """
        将一组掩码（n, h, w）转换为一组轮廓（n, xy）。
        来源：https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py

        参数:
            masks (numpy.ndarray): 模型的输出，形状为 (batch_size, 160, 160)。

        返回:
            segments (List): 分割掩码的列表。
        """
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # 使用 CHAIN_APPROX_SIMPLE 进行轮廓检测
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)  # 选取最大的轮廓
            else:
                c = np.zeros((0, 2))  # 如果没有找到轮廓，返回空数组
            segments.append(c.astype("float32"))
        return segments

    @staticmethod
    def crop_mask(masks, boxes):
        """
        获取一个掩码和一个边界框，并返回裁剪到该边界框的掩码。
        来源：https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py

        参数:
            masks (Numpy.ndarray): 形状为 [n, h, w] 的掩码张量。
            boxes (Numpy.ndarray): 形状为 [n, 4] 的边界框坐标（相对坐标）。

        返回:
            (Numpy.ndarray): 裁剪后的掩码。
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)  # 分割边界框的坐标
        r = np.arange(w, dtype=x1.dtype)[None, None, :]  # 生成列索引
        c = np.arange(h, dtype=x1.dtype)[None, :, None]  # 生成行索引
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))  # 仅保留在边界框内的掩码部分

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        处理模型输出的掩码，将掩码应用于边界框。这种方法生成的掩码质量更高，但速度较慢。
        来源：https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py

        参数:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w] 形状的掩码原型。
            masks_in (numpy.ndarray): [n, mask_dim]，n 是非极大值抑制（NMS）后的掩码数量。
            bboxes (numpy.ndarray): 重新缩放到原始图像大小的边界框。
            im0_shape (tuple): 输入图像的大小 (h, w, c)。

        返回:
            (numpy.ndarray): 经过上采样的掩码。
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN 格式
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # 重新调整掩码尺寸，使其匹配原始输入图像
        masks = np.einsum("HWN -> NHW", masks)  # 从 HWN 转换为 NHW
        masks = self.crop_mask(masks, bboxes)  # 裁剪掩码
        return np.greater(masks, 0.5)  # 设定阈值，大于 0.5 设为 1，其他设为 0

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        重新调整掩码大小，使其匹配原始输入图像的大小。
        来源：https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py

        参数:
            masks (np.ndarray): 已调整大小并填充的掩码/图像，[h, w, num] 或 [h, w, 3]。
            im0_shape (tuple): 原始图像的大小。
            ratio_pad (tuple): 填充比例。

        返回:
            masks (np.ndarray): 重新缩放后的掩码。
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # 如果没有提供 ratio_pad，计算缩放比例
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain = 旧尺寸 / 新尺寸
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # 计算填充大小
        else:
            pad = ratio_pad[1]

        # 计算掩码的裁剪边界
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        
        # 处理异常情况
        if len(masks.shape) < 2:
            raise ValueError(f'"masks.shape" 长度应为 2 或 3，但得到 {len(masks.shape)}')

        # 裁剪掩码
        masks = masks[top:bottom, left:right]

        # 重新调整掩码大小，使其匹配原始输入图像
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )

        # 处理 2D 掩码
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        
        return masks

    def draw_and_visualize(self, im, bboxes, segments, vis=False, save=True):
        """
        绘制并可视化检测结果。

        参数:
            im (np.ndarray): 原始图像，形状为 [h, w, c]。
            bboxes (numpy.ndarray): 形状为 [n, 4]，n 为边界框数量。
            segments (List): 分割掩码的列表。
            vis (bool): 是否使用 OpenCV 显示图像。
            save (bool): 是否保存标注后的图像。

        返回:
            None
        """
        # 复制原始图像以进行绘制
        im_canvas = im.copy()
        
        # 遍历边界框和对应的分割掩码
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            # 绘制轮廓并填充掩码
            cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)  # 画白色轮廓
            cv2.fillPoly(im_canvas, np.int32([segment]), self.color_palette(int(cls_), bgr=True))

            # 绘制边界框
            cv2.rectangle(
                im,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.color_palette(int(cls_), bgr=True),
                1,
                cv2.LINE_AA,
            )
            
            # 添加类别名称和置信度
            cv2.putText(
                im,
                f"{self.classes[cls_]}: {conf:.3f}",
                (int(box[0]), int(box[1] - 9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.color_palette(int(cls_), bgr=True),
                2,
                cv2.LINE_AA,
            )

        # 合成最终的可视化图像
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

        # 显示图像
        if vis:
            cv2.imshow("demo", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # 保存图像
        if save:
            cv2.imwrite("demo.jpg", im)


if __name__ == "__main__":
    # 创建命令行参数解析器，用于处理命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="ONNX 模型的路径")
    parser.add_argument("--source", type=str, default=str(ASSETS / "bus.jpg"), help="输入图像的路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS（非极大值抑制）的 IoU 阈值")
    args = parser.parse_args()

    # 构建模型
    model = YOLOv11Seg(args.model)

    # 使用 OpenCV 读取图像
    img = cv2.imread(args.source)

    # 进行推理
    boxes, segments, _ = model(img, conf_threshold=args.conf, iou_threshold=args.iou)

    # 绘制边界框和多边形
    if len(boxes) > 0:
        model.draw_and_visualize(img, boxes, segments, vis=True, save=True)

