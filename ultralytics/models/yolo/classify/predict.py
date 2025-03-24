# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import torch
from PIL import Image

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class ClassificationPredictor(BasePredictor):
    """
    一个继承自 BasePredictor 的分类预测器类，用于基于分类模型进行预测。

    注意：
        - 也可以将 Torchvision 的分类模型传入 'model' 参数，例如 model='resnet18'。

    示例：
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.classify import ClassificationPredictor

        args = dict(model="yolov8n-cls.pt", source=ASSETS)
        predictor = ClassificationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化 ClassificationPredictor，并将任务类型设为 'classify'（分类）。"""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "classify"
        self._legacy_transform_name = "ultralytics.yolo.data.augment.ToTensor"

    def preprocess(self, img):
        """将输入图像转换为模型兼容的数据类型。"""
        if not isinstance(img, torch.Tensor):
            is_legacy_transform = any(
                self._legacy_transform_name in str(transform) for transform in self.transforms.transforms
            )
            if is_legacy_transform:  # 处理旧版数据增强方式
                img = torch.stack([self.transforms(im) for im in img], dim=0)
            else:
                img = torch.stack(
                    [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
                )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # 从 uint8 转为 fp16 或 fp32

    def postprocess(self, preds, img, orig_imgs):
        """对预测结果进行后处理，返回 Results 对象列表。"""
        if not isinstance(orig_imgs, list):  # 输入图像为 torch.Tensor 而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]
