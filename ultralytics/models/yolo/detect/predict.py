# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class DetectionPredictor(BasePredictor):
    """
    基于检测模型的预测器类，继承自BasePredictor。

    示例：
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolo11n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """对预测结果进行后处理，并返回一个由 Results 对象构成的列表。"""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,             # 置信度阈值
            self.args.iou,              # IOU 阈值
            agnostic=self.args.agnostic_nms,  # 类别无关的NMS
            max_det=self.args.max_det,        # 最大检测数
            classes=self.args.classes,        # 指定的检测类别
        )

        # 如果输入的原始图像不是列表（即为Tensor格式），则转换为NumPy数组格式
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            # 将预测框从模型图像尺寸映射回原图尺寸
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            # 构造结果对象并追加到结果列表
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
