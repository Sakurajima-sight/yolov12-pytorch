# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class RTDETRPredictor(BasePredictor):
    """
    RT-DETR（实时检测转换器）预测器，扩展了BasePredictor类，用于使用
    百度的RT-DETR模型进行预测。

    该类利用视觉转换器的强大功能，在保持高精度的同时提供实时目标检测。
    它支持高效的混合编码和IoU感知查询选择等关键特性。

    示例:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.rtdetr import RTDETRPredictor

        args = dict(model="rtdetr-l.pt", source=ASSETS)
        predictor = RTDETRPredictor(overrides=args)
        predictor.predict_cli()
        ```

    属性:
        imgsz (int): 推理的图像尺寸（必须是方形并进行比例填充）。
        args (dict): 预测器的参数覆盖。
    """

    def postprocess(self, preds, img, orig_imgs):
        """
        对模型的原始预测结果进行后处理，生成边界框和置信度分数。

        该方法根据置信度和类别对检测结果进行筛选（如果在`self.args`中指定）。

        参数:
            preds (list): 模型输出的[predictions, extra]列表。
            img (torch.Tensor): 处理过的输入图像。
            orig_imgs (list或torch.Tensor): 原始、未经处理的图像。

        返回:
            (list[Results]): 一个包含后处理后的边界框、置信度分数和类别标签的Results对象列表。
        """
        if not isinstance(preds, (list, tuple)):  # 对于PyTorch推理是列表，但对于导出推理是list[0] Tensor
            preds = [preds, None]

        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        if not isinstance(orig_imgs, list):  # 输入图像是torch.Tensor，而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for bbox, score, orig_img, img_path in zip(bboxes, scores, orig_imgs, self.batch[0]):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)  # (300, 1)
            idx = max_score.squeeze(-1) > self.args.conf  # (300, )
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]  # 过滤
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow
            pred[..., [1, 3]] *= oh
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def pre_transform(self, im):
        """
        在将输入图像输入到模型进行推理之前进行预处理。输入图像被填充到方形以确保比例，并进行比例填充。
        尺寸必须是方形的(640)，并且进行比例填充。

        参数:
            im (list[np.ndarray] | torch.Tensor): 输入图像，形状为 (N,3,h,w) 的tensor，或 [(h,w,3) x N] 的列表。

        返回:
            (list): 预处理后的图像列表，准备好进行模型推理。
        """
        letterbox = LetterBox(self.imgsz, auto=False, scaleFill=True)
        return [letterbox(image=x) for x in im]
