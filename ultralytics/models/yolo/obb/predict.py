# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class OBBPredictor(DetectionPredictor):
    """
    基于 OBB（方向性边界框）模型的预测器类，继承自 DetectionPredictor。

    示例：
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.obb import OBBPredictor

        args = dict(model="yolov8n-obb.pt", source=ASSETS)
        predictor = OBBPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化 OBBPredictor，可通过传参覆盖模型和数据配置。"""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "obb"

    def postprocess(self, preds, img, orig_imgs):
        """对预测结果进行后处理，并返回 Results 对象列表。"""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,              # 置信度阈值
            self.args.iou,               # IoU 阈值
            agnostic=self.args.agnostic_nms,  # 是否类别无关的 NMS
            max_det=self.args.max_det,   # 最大检测数目
            nc=len(self.model.names),    # 类别数量
            classes=self.args.classes,   # 筛选的目标类别
            rotated=True,                # 启用旋转框处理
        )

        if not isinstance(orig_imgs, list):  # 如果输入图像是 torch.Tensor 而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            # 将预测的 [x, y, w, h, θ] 标准化成方向性边界框（OBB）
            rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
            # 将方向框坐标从网络输入尺寸映射回原图尺寸
            rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
            # 最终格式：[x, y, w, h, θ, conf, cls]
            obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
            results.append(Results(orig_img, path=img_path, names=self.model.names, obb=obb))
        return results
