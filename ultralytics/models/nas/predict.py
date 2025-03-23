# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class NASPredictor(BasePredictor):
    """
    Ultralytics YOLO NAS预测器，用于目标检测。

    该类继承自Ultralytics引擎中的`BasePredictor`，负责对YOLO NAS模型生成的原始预测结果进行后处理。
    它应用诸如非极大值抑制和调整边界框以适应原始图像尺寸等操作。

    属性:
        args (Namespace): 包含各种后处理配置的命名空间。

    示例:
        ```python
        from ultralytics import NAS

        model = NAS("yolo_nas_s")
        predictor = model.predictor
        # 假设raw_preds, img, orig_imgs已经定义
        results = predictor.postprocess(raw_preds, img, orig_imgs)
        ```

    注意:
        通常，直接实例化该类是不必要的。它在`NAS`类内部使用。
    """

    def postprocess(self, preds_in, img, orig_imgs):
        """对预测结果进行后处理，并返回一个Results对象的列表。"""
        # 拼接框和类别分数
        boxes = ops.xyxy2xywh(preds_in[0][0])
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)

        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # 输入图像是torch.Tensor，而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
