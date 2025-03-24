# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops


class PosePredictor(DetectionPredictor):
    """
    基于姿态估计模型的预测器类，继承自 DetectionPredictor。

    示例：
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.pose import PosePredictor

        args = dict(model="yolov8n-pose.pt", source=ASSETS)
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化 PosePredictor，将任务类型设置为 'pose'，并在设备为 'mps' 时给出警告信息。"""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"  # 设置任务为姿态估计
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "⚠️ 警告：Apple MPS 存在已知的 Pose 模型 bug，建议使用 'device=cpu'。"
                "详情参见：https://github.com/ultralytics/ultralytics/issues/4031。"
            )

    def postprocess(self, preds, img, orig_imgs):
        """对输入图像或图像列表的预测结果进行后处理，并返回检测结果。"""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,                # 置信度阈值
            self.args.iou,                 # IOU 阈值
            agnostic=self.args.agnostic_nms,  # 是否类别无关的NMS
            max_det=self.args.max_det,     # 最大检测目标数
            classes=self.args.classes,     # 限定检测的类别
            nc=len(self.model.names),      # 类别总数
        )

        # 如果原始图像不是列表（说明是Tensor格式），转换为NumPy数组格式
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            # 将预测框从模型输入大小映射回原图大小，并四舍五入为整数像素
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            # 提取关键点信息并调整大小
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            # 构造结果对象并添加到结果列表
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts)
            )
        return results
