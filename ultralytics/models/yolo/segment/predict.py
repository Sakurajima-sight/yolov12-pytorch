# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class SegmentationPredictor(DetectionPredictor):
    """
    基于分割模型的预测器类，继承自 DetectionPredictor。

    示例：
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model="yolov8n-seg.pt", source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """使用提供的配置、参数覆盖和回调函数初始化 SegmentationPredictor。"""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"  # 设置任务类型为分割任务

    def postprocess(self, preds, img, orig_imgs):
        """对输入批次中的每张图像应用非极大值抑制并处理检测结果。"""
        p = ops.non_max_suppression(
            preds[0],                  # 检测框预测结果
            self.args.conf,            # 置信度阈值
            self.args.iou,             # IOU 阈值
            agnostic=self.args.agnostic_nms,  # 是否类别无关的NMS
            max_det=self.args.max_det,        # 最大检测数量
            nc=len(self.model.names),         # 类别数量
            classes=self.args.classes,        # 指定检测类别
        )

        # 如果原始图像不是列表格式（而是Tensor），则转换为NumPy数组列表
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        # 提取 mask 原型：若为 PyTorch 模型是元组，若为导出模型是数组
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]

        for i, (pred, orig_img, img_path) in enumerate(zip(p, orig_imgs, self.batch[0])):
            if not len(pred):  # 若没有检测结果，设置 mask 为 None
                masks = None
            elif self.args.retina_masks:
                # 将预测框尺寸从输入图像尺寸映射到原图尺寸
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                # 使用原生方法处理 mask（更精确但更慢）
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC格式
            else:
                # 使用默认方法处理 mask（速度快）
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC格式
                # 再将预测框映射回原图尺寸
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

            # 构造并保存每一张图的结果（包含图像、路径、类别名、预测框、mask）
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results
