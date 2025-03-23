# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
from PIL import Image

from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG, checks
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import scale_masks

from .utils import adjust_bboxes_to_image_border


class FastSAMPredictor(SegmentationPredictor):
    """
    FastSAMPredictor是专为Fast SAM（Segment Anything Model）分割预测任务在Ultralytics YOLO框架中设计的。

    该类扩展了SegmentationPredictor，特别定制了Fast SAM的预测流程。它调整了后处理步骤，以便整合掩膜预测和非最大抑制，同时优化单类分割。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化一个FastSAMPredictor，用于Ultralytics YOLO框架中的Fast SAM分割任务。"""
        super().__init__(cfg, overrides, _callbacks)
        self.prompts = {}

    def postprocess(self, preds, img, orig_imgs):
        """对FastSAM预测结果应用边界框后处理。"""
        bboxes = self.prompts.pop("bboxes", None)
        points = self.prompts.pop("points", None)
        labels = self.prompts.pop("labels", None)
        texts = self.prompts.pop("texts", None)
        results = super().postprocess(preds, img, orig_imgs)
        for result in results:
            full_box = torch.tensor(
                [0, 0, result.orig_shape[1], result.orig_shape[0]], device=preds[0].device, dtype=torch.float32
            )
            boxes = adjust_bboxes_to_image_border(result.boxes.xyxy, result.orig_shape)
            idx = torch.nonzero(box_iou(full_box[None], boxes) > 0.9).flatten()
            if idx.numel() != 0:
                result.boxes.xyxy[idx] = full_box

        return self.prompt(results, bboxes=bboxes, points=points, labels=labels, texts=texts)

    def prompt(self, results, bboxes=None, points=None, labels=None, texts=None):
        """
        基于边界框、点和掩膜的提示进行图像分割推断的内部函数。
        利用SAM的专用架构进行基于提示的实时分割。

        参数：
            results (Results | List[Results]): 来自FastSAM模型的原始推断结果，不含任何提示。
            bboxes (np.ndarray | List, optional): 边界框，形状为 (N, 4)，XYXY格式。
            points (np.ndarray | List, optional): 指示物体位置的点，形状为 (N, 2)，单位为像素。
            labels (np.ndarray | List, optional): 点提示的标签，形状为 (N, )。1表示前景，0表示背景。
            texts (str | List[str], optional): 文本提示，包含字符串对象的列表。

        返回：
            (List[Results]): 根据提示确定的输出结果。
        """
        if bboxes is None and points is None and texts is None:
            return results
        prompt_results = []
        if not isinstance(results, list):
            results = [results]
        for result in results:
            if len(result) == 0:
                prompt_results.append(result)
                continue
            masks = result.masks.data
            if masks.shape[1:] != result.orig_shape:
                masks = scale_masks(masks[None], result.orig_shape)[0]
            # 边界框提示
            idx = torch.zeros(len(result), dtype=torch.bool, device=self.device)
            if bboxes is not None:
                bboxes = torch.as_tensor(bboxes, dtype=torch.int32, device=self.device)
                bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
                bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
                mask_areas = torch.stack([masks[:, b[1] : b[3], b[0] : b[2]].sum(dim=(1, 2)) for b in bboxes])
                full_mask_areas = torch.sum(masks, dim=(1, 2))

                union = bbox_areas[:, None] + full_mask_areas - mask_areas
                idx[torch.argmax(mask_areas / union, dim=1)] = True
            if points is not None:
                points = torch.as_tensor(points, dtype=torch.int32, device=self.device)
                points = points[None] if points.ndim == 1 else points
                if labels is None:
                    labels = torch.ones(points.shape[0])
                labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
                assert len(labels) == len(points), (
                    f"期待`labels`与`point`大小相同，但分别为 {len(labels)} 和 {len(points)}"
                )
                point_idx = (
                    torch.ones(len(result), dtype=torch.bool, device=self.device)
                    if labels.sum() == 0  # 所有负点
                    else torch.zeros(len(result), dtype=torch.bool, device=self.device)
                )
                for point, label in zip(points, labels):
                    point_idx[torch.nonzero(masks[:, point[1], point[0]], as_tuple=True)[0]] = bool(label)
                idx |= point_idx
            if texts is not None:
                if isinstance(texts, str):
                    texts = [texts]
                crop_ims, filter_idx = [], []
                for i, b in enumerate(result.boxes.xyxy.tolist()):
                    x1, y1, x2, y2 = (int(x) for x in b)
                    if masks[i].sum() <= 100:
                        filter_idx.append(i)
                        continue
                    crop_ims.append(Image.fromarray(result.orig_img[y1:y2, x1:x2, ::-1]))
                similarity = self._clip_inference(crop_ims, texts)
                text_idx = torch.argmax(similarity, dim=-1)  # (M, )
                if len(filter_idx):
                    text_idx += (torch.tensor(filter_idx, device=self.device)[None] <= int(text_idx)).sum(0)
                idx[text_idx] = True

            prompt_results.append(result[idx])

        return prompt_results

    def _clip_inference(self, images, texts):
        """
        CLIP推断过程。

        参数：
            images (List[PIL.Image]): 一组源图像，每个图像应为PIL.Image类型，且使用RGB通道顺序。
            texts (List[str]): 一组提示文本，每个文本应为字符串对象。

        返回：
            (torch.Tensor): 给定图像和文本之间的相似度。
        """
        try:
            import clip
        except ImportError:
            checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip
        if (not hasattr(self, "clip_model")) or (not hasattr(self, "clip_preprocess")):
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        images = torch.stack([self.clip_preprocess(image).to(self.device) for image in images])
        tokenized_text = clip.tokenize(texts).to(self.device)
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # (N, 512)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # (M, 512)
        return (image_features * text_features[:, None]).sum(-1)  # (M, N)

    def set_prompts(self, prompts):
        """提前设置提示。"""
        self.prompts = prompts
