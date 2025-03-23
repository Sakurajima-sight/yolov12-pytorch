# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops

__all__ = ("RTDETRValidator",)  # 元组或列表


class RTDETRDataset(YOLODataset):
    """
    实时检测与追踪 (RT-DETR) 数据集类，继承自基本的 YOLODataset 类。

    这个专门的数据集类是为了配合 RT-DETR 目标检测模型使用，并且优化了实时检测与追踪任务。
    """

    def __init__(self, *args, data=None, **kwargs):
        """通过继承 YOLODataset 类初始化 RTDETRDataset 类。"""
        super().__init__(*args, data=data, **kwargs)

    # 注意：为 RTDETR 拼接添加图像加载版本
    def load_image(self, i, rect_mode=False):
        """加载数据集中索引 'i' 的一张图片，返回 (im, resized hw)。"""
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """临时，仅用于评估。"""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scaleFill=True)])
            transforms = Compose([])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms


class RTDETRValidator(DetectionValidator):
    """
    RTDETRValidator 扩展了 DetectionValidator 类，为 RT-DETR（实时DETR）目标检测模型提供特定的验证功能。

    该类允许为验证构建一个特定的 RTDETR 数据集，应用非最大抑制（NMS）进行后处理，并相应地更新评估指标。

    示例：
        ```python
        from ultralytics.models.rtdetr import RTDETRValidator

        args = dict(model="rtdetr-l.pt", data="coco8.yaml")
        validator = RTDETRValidator(args=args)
        validator()
        ```

    注意：
        有关属性和方法的更多详细信息，请参考父类 DetectionValidator。
    """

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        构建一个 RTDETR 数据集。

        参数：
            img_path (str): 包含图片的文件夹路径。
            mode (str): `train` 模式或 `val` 模式，用户可以为每种模式定制不同的增强方式。
            batch (int, 可选): 批量大小，这用于 `rect`。默认为 None。
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # 无增强
            hyp=self.args,
            rect=False,  # 无 rect
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def postprocess(self, preds):
        """对预测输出应用非最大抑制（NMS）。"""
        if not isinstance(preds, (list, tuple)):  # 对于 PyTorch 推理是列表，但导出推理是 list[0] Tensor
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            # 评估时不需要阈值，因为这里只有 300 个框
            # idx = score > self.args.conf
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # 过滤
            # 按置信度排序以正确获取内部指标
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred  # [idx]

        return outputs

    def _prepare_batch(self, si, batch):
        """通过应用转换准备一个批次用于训练或推理。"""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox)  # 目标框
            bbox[..., [0, 2]] *= ori_shape[1]  # 原始空间预测
            bbox[..., [1, 3]] *= ori_shape[0]  # 原始空间预测
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """准备并返回一个批次，转换后的边界框和类标签。"""
        predn = pred.clone()
        predn[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # 原始空间预测
        predn[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # 原始空间预测
        return predn.float()
