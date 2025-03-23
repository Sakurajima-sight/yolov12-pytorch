# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops

__all__ = ["NASValidator"]


class NASValidator(DetectionValidator):
    """
    Ultralytics YOLO NAS验证器，用于目标检测。

    扩展了Ultralytics模型包中的`DetectionValidator`，并旨在对YOLO NAS模型生成的原始预测进行后处理。
    它执行非极大值抑制，以移除重叠和低置信度的框，最终生成最终的检测结果。

    属性:
        args (Namespace): 包含各种后处理配置的命名空间，例如置信度和IoU。
        lb (torch.Tensor): 可选的张量，用于多标签NMS。

    示例:
        ```python
        from ultralytics import NAS

        model = NAS("yolo_nas_s")
        validator = model.validator
        # 假设raw_preds已经定义
        final_preds = validator.postprocess(raw_preds)
        ```

    注意:
        该类通常不会直接实例化，而是作为`NAS`类内部使用。
    """

    def postprocess(self, preds_in):
        """对预测输出应用非极大值抑制（NMS）。"""
        boxes = ops.xyxy2xywh(preds_in[0][0])
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=False,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            max_time_img=0.5,
        )
