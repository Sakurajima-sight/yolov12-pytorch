# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
使用Segment Anything Model (SAM)生成预测。

SAM是一个先进的图像分割模型，提供诸如可提示分割和零-shot性能等功能。
此模块包含了执行分割所需的预测逻辑和辅助工具。
它是Ultralytics框架的核心部分，旨在高性能、实时图像分割任务中使用。
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.utils.torch_utils import select_device, smart_inference_mode

from .amg import (
    batch_iterator,
    batched_mask_to_box,
    build_all_layer_point_grids,
    calculate_stability_score,
    generate_crop_boxes,
    is_box_near_crop_edge,
    remove_small_regions,
    uncrop_boxes_xyxy,
    uncrop_masks,
)
from .build import build_sam


class Predictor(BasePredictor):
    """
    SAM的预测器类，支持具有提示能力的实时图像分割。

    该类扩展了BasePredictor并实现了Segment Anything Model (SAM)，用于先进的图像
    分割任务。它支持多种输入提示，如点、边界框和掩码，以便对分割结果进行细粒度控制。

    属性：
        args (SimpleNamespace): 预测器的配置参数。
        model (torch.nn.Module): 加载的SAM模型。
        device (torch.device): 模型加载的设备（CPU或GPU）。
        im (torch.Tensor): 预处理后的输入图像。
        features (torch.Tensor): 提取的图像特征。
        prompts (Dict): 存储各种类型提示（例如，边界框、点、掩码）的字典。
        segment_all (bool): 标记是否进行全图分割。
        mean (torch.Tensor): 图像归一化的均值。
        std (torch.Tensor): 图像归一化的标准差。

    方法：
        preprocess: 准备输入图像以进行模型推理。
        pre_transform: 对输入图像进行初步转换。
        inference: 根据输入提示执行分割推理。
        prompt_inference: 用于基于提示的分割推理的内部函数。
        generate: 生成整个图像的分割掩码。
        setup_model: 初始化SAM模型以进行推理。
        get_model: 构建并返回一个SAM模型。
        postprocess: 后处理模型输出以生成最终结果。
        setup_source: 设置推理的数据源。
        set_image: 设置并预处理单张图像以进行推理。
        get_im_features: 使用SAM图像编码器提取图像特征。
        set_prompts: 设置后续推理的提示。
        reset_image: 重置当前图像及其特征。
        remove_small_regions: 移除掩码中小的孤立区域和孔洞。

    示例：
        >>> predictor = Predictor()
        >>> predictor.setup_model(model_path="sam_model.pt")
        >>> predictor.set_image("image.jpg")
        >>> bboxes = [[100, 100, 200, 200]]
        >>> results = predictor(bboxes=bboxes)
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        使用配置、覆盖项和回调函数初始化预测器。

        为SAM（Segment Anything Model）设置预测器对象，并应用提供的任何配置覆盖项或
        回调函数。初始化SAM特定的设置，例如将retina_masks设置为True，以获得最佳结果。

        参数：
            cfg (Dict): 包含默认设置的配置字典。
            overrides (Dict | None): 覆盖默认配置的值的字典。
            _callbacks (Dict | None): 用于自定义行为的回调函数字典。

        示例：
            >>> predictor_example = Predictor(cfg=DEFAULT_CFG)
            >>> predictor_example_with_imgsz = Predictor(overrides={"imgsz": 640})
            >>> predictor_example_with_callback = Predictor(_callbacks={"on_predict_start": custom_callback})
        """
        if overrides is None:
            overrides = {}
        overrides.update(dict(task="segment", mode="predict", batch=1))
        super().__init__(cfg, overrides, _callbacks)
        self.args.retina_masks = True
        self.im = None
        self.features = None
        self.prompts = {}
        self.segment_all = False

    def preprocess(self, im):
        """
        为模型推理预处理输入图像。

        此方法通过应用转换和归一化准备输入图像。它支持torch.Tensor和np.ndarray列表作为输入格式。

        参数：
            im (torch.Tensor | List[np.ndarray]): 输入图像，以BCHW张量格式或HWC的numpy数组列表形式。

        返回：
            im (torch.Tensor): 预处理后的图像张量，经过归一化并转换为适当的数据类型。

        示例：
            >>> predictor = Predictor()
            >>> image = torch.rand(1, 3, 640, 640)
            >>> preprocessed_image = predictor.preprocess(image)
        """
        if self.im is not None:
            return self.im
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        if not_tensor:
            im = (im - self.mean) / self.std
        return im

    def pre_transform(self, im):
        """
        对输入图像进行初步转换以进行预处理。

        此方法应用诸如调整大小等转换，以准备图像进行进一步预处理。
        当前不支持批量推理，因此列表的长度应该为1。

        参数：
            im (List[np.ndarray]): 包含单张图像的列表，图像为HWC格式的numpy数组。

        返回：
            (List[np.ndarray]): 包含转换后图像的列表。

        异常：
            AssertionError: 如果输入列表包含多于一张图像。

        示例：
            >>> predictor = Predictor()
            >>> image = np.random.rand(480, 640, 3)  # 单张HWC图像
            >>> transformed = predictor.pre_transform([image])
            >>> print(len(transformed))
            1
        """
        assert len(im) == 1, "SAM模型当前不支持批量推理"
        letterbox = LetterBox(self.args.imgsz, auto=False, center=False)
        return [letterbox(image=x) for x in im]

    def inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False, *args, **kwargs):
        """
        基于给定的输入提示，使用当前加载的图像进行图像分割推理。

        此方法利用SAM（Segment Anything Model）架构，包括图像编码器、提示编码器和掩码解码器，进行实时和基于提示的分割任务。

        参数：
            im (torch.Tensor): 预处理后的输入图像，tensor格式，形状为(N, C, H, W)。
            bboxes (np.ndarray | List | None): 边界框，形状为(N, 4)，XYXY格式。
            points (np.ndarray | List | None): 表示物体位置的点，形状为(N, 2)，单位为像素。
            labels (np.ndarray | List | None): 点提示的标签，形状为(N,)，1表示前景，0表示背景。
            masks (np.ndarray | None): 来自先前预测的低分辨率掩码，形状为(N, H, W)，对于SAM来说，H=W=256。
            multimask_output (bool): 是否返回多个掩码。对模糊的提示非常有用。
            *args (Any): 其他位置参数。
            **kwargs (Any): 其他关键字参数。

        返回：
            (np.ndarray): 输出掩码，形状为(C, H, W)，其中C是生成的掩码数量。
            (np.ndarray): 长度为C的数组，包含模型为每个掩码预测的质量分数。
            (np.ndarray): 低分辨率的logits，形状为(C, H, W)，用于后续推理，H=W=256。

        示例：
            >>> predictor = Predictor()
            >>> predictor.setup_model(model_path="sam_model.pt")
            >>> predictor.set_image("image.jpg")
            >>> results = predictor(bboxes=[[0, 0, 100, 100]])
        """
        # 如果有任何提示存储在self.prompts中，则覆盖它们
        bboxes = self.prompts.pop("bboxes", bboxes)
        points = self.prompts.pop("points", points)
        masks = self.prompts.pop("masks", masks)
        labels = self.prompts.pop("labels", labels)

        if all(i is None for i in [bboxes, points, masks]):
            return self.generate(im, *args, **kwargs)

        return self.prompt_inference(im, bboxes, points, labels, masks, multimask_output)

    def prompt_inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False):
        """
        基于输入提示使用SAM的专用架构执行图像分割推理。

        此内部函数利用Segment Anything Model（SAM）进行基于提示的实时分割。
        它处理各种输入提示，如边界框、点和掩码，以生成分割掩码。

        参数：
            im (torch.Tensor): 预处理后的输入图像tensor，形状为(N, C, H, W)。
            bboxes (np.ndarray | List | None): 边界框，XYXY格式，形状为(N, 4)。
            points (np.ndarray | List | None): 表示物体位置的点，形状为(N, 2)或(N, num_points, 2)，单位为像素。
            labels (np.ndarray | List | None): 点提示标签，形状为(N)或(N, num_points)。1表示前景，0表示背景。
            masks (np.ndarray | None): 来自先前预测的低分辨率掩码，形状为(N, H, W)。对于SAM来说，H=W=256。
            multimask_output (bool): 是否返回多个掩码，适用于模糊提示。

        异常：
            AssertionError: 如果点的数量与标签的数量不匹配（当标签被传递时）。

        返回：
            (np.ndarray): 输出掩码，形状为(C, H, W)，其中C是生成的掩码数量。
            (np.ndarray): 模型为每个掩码预测的质量分数，长度为C。

        示例：
            >>> predictor = Predictor()
            >>> im = torch.rand(1, 3, 1024, 1024)
            >>> bboxes = [[100, 100, 200, 200]]
            >>> masks, scores, logits = predictor.prompt_inference(im, bboxes=bboxes)
        """
        features = self.get_im_features(im) if self.features is None else self.features

        bboxes, points, labels, masks = self._prepare_prompts(im.shape[2:], bboxes, points, labels, masks)
        points = (points, labels) if points is not None else None
        # 嵌入提示
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points, boxes=bboxes, masks=masks)

        # 预测掩码
        pred_masks, pred_scores = self.model.mask_decoder(
            image_embeddings=features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # (N, d, H, W) --> (N*d, H, W), (N, d) --> (N*d, )
        # `d` 可能是1或3，取决于`multimask_output`。
        return pred_masks.flatten(0, 1), pred_scores.flatten(0, 1)

    def _prepare_prompts(self, dst_shape, bboxes=None, points=None, labels=None, masks=None):
        """
        根据目标形状准备并转换输入的提示信息。

        参数：
            dst_shape (tuple): 提示信息的目标形状（高度，宽度）。
            bboxes (np.ndarray | List | None): 以XYXY格式表示的边界框，形状为(N, 4)。
            points (np.ndarray | List | None): 表示物体位置的点，形状为(N, 2)或(N, num_points, 2)，单位为像素。
            labels (np.ndarray | List | None): 点提示标签，形状为(N)或(N, num_points)。前景为1，背景为0。
            masks (List | np.ndarray, 可选): 物体的掩膜，每个掩膜是一个二维数组。

        异常：
            AssertionError: 如果传入标签时，点的数量与标签的数量不匹配。

        返回：
            (tuple): 返回转换后的边界框、点、标签和掩膜。
        """
        src_shape = self.batch[1][0].shape[:2]
        r = 1.0 if self.segment_all else min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])
        # 转换输入提示信息
        if points is not None:
            points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
            points = points[None] if points.ndim == 1 else points
            # 假设如果用户没有传入标签，则标签都为正样本。
            if labels is None:
                labels = np.ones(points.shape[:-1])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            assert points.shape[-2] == labels.shape[-1], (
                f"点的数量 {points.shape[-2]} 应该与标签的数量 {labels.shape[-1]} 匹配。"
            )
            points *= r
            if points.ndim == 2:
                # (N, 2) --> (N, 1, 2), (N, ) --> (N, 1)
                points, labels = points[:, None, :], labels[:, None]
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32, device=self.device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            bboxes *= r
        if masks is not None:
            masks = torch.as_tensor(masks, dtype=torch.float32, device=self.device).unsqueeze(1)
        return bboxes, points, labels, masks

    def generate(
        self,
        im,
        crop_n_layers=0,
        crop_overlap_ratio=512 / 1500,
        crop_downscale_factor=1,
        point_grids=None,
        points_stride=32,
        points_batch_size=64,
        conf_thres=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=0.95,
        crop_nms_thresh=0.7,
    ):
        """
        使用Segment Anything Model (SAM)进行图像分割。

        该方法通过利用SAM的先进架构和实时性能能力，将整张图像分割为组成部分。它还可以选择在图像裁剪上工作，以实现更精细的分割。

        参数：
            im (torch.Tensor): 输入张量，表示预处理后的图像，形状为(N, C, H, W)。
            crop_n_layers (int): 用于图像裁剪的额外掩膜预测的层数。
            crop_overlap_ratio (float): 裁剪之间的重叠比例，后续层会缩小。
            crop_downscale_factor (int): 每层采样点的缩放因子。
            point_grids (List[np.ndarray] | None): 自定义的点采样网格，归一化到[0,1]。
            points_stride (int): 每侧采样点的数量。
            points_batch_size (int): 批量大小，表示同时处理的点的数量。
            conf_thres (float): 用于根据掩膜质量预测的置信度阈值[0,1]。
            stability_score_thresh (float): 用于基于稳定性进行掩膜过滤的稳定性阈值[0,1]。
            stability_score_offset (float): 计算稳定性分数的偏移值。
            crop_nms_thresh (float): 用于NMS的IoU截止值，用于去除裁剪之间的重复掩膜。

        返回：
            pred_masks (torch.Tensor): 分割后的掩膜，形状为(N, H, W)。
            pred_scores (torch.Tensor): 每个掩膜的置信度分数，形状为(N,)。
            pred_bboxes (torch.Tensor): 每个掩膜的边界框，形状为(N, 4)。

        示例：
            >>> predictor = Predictor()
            >>> im = torch.rand(1, 3, 1024, 1024)  # 示例输入图像
            >>> masks, scores, boxes = predictor.generate(im)
        """
        import torchvision  # 用于更快的 'import ultralytics'

        self.segment_all = True
        ih, iw = im.shape[2:]
        crop_regions, layer_idxs = generate_crop_boxes((ih, iw), crop_n_layers, crop_overlap_ratio)
        if point_grids is None:
            point_grids = build_all_layer_point_grids(points_stride, crop_n_layers, crop_downscale_factor)
        pred_masks, pred_scores, pred_bboxes, region_areas = [], [], [], []
        for crop_region, layer_idx in zip(crop_regions, layer_idxs):
            x1, y1, x2, y2 = crop_region
            w, h = x2 - x1, y2 - y1
            area = torch.tensor(w * h, device=im.device)
            points_scale = np.array([[w, h]])  # w, h
            # 裁剪图像并插值到输入大小
            crop_im = F.interpolate(im[..., y1:y2, x1:x2], (ih, iw), mode="bilinear", align_corners=False)
            # (num_points, 2)
            points_for_image = point_grids[layer_idx] * points_scale
            crop_masks, crop_scores, crop_bboxes = [], [], []
            for (points,) in batch_iterator(points_batch_size, points_for_image):
                pred_mask, pred_score = self.prompt_inference(crop_im, points=points, multimask_output=True)
                # 将预测的掩膜插值到输入大小
                pred_mask = F.interpolate(pred_mask[None], (h, w), mode="bilinear", align_corners=False)[0]
                idx = pred_score > conf_thres
                pred_mask, pred_score = pred_mask[idx], pred_score[idx]

                stability_score = calculate_stability_score(
                    pred_mask, self.model.mask_threshold, stability_score_offset
                )
                idx = stability_score > stability_score_thresh
                pred_mask, pred_score = pred_mask[idx], pred_score[idx]
                # 使用布尔类型更节省内存
                pred_mask = pred_mask > self.model.mask_threshold
                # (N, 4)
                pred_bbox = batched_mask_to_box(pred_mask).float()
                keep_mask = ~is_box_near_crop_edge(pred_bbox, crop_region, [0, 0, iw, ih])
                if not torch.all(keep_mask):
                    pred_bbox, pred_mask, pred_score = pred_bbox[keep_mask], pred_mask[keep_mask], pred_score[keep_mask]

                crop_masks.append(pred_mask)
                crop_bboxes.append(pred_bbox)
                crop_scores.append(pred_score)

            # 对该裁剪区域进行NMS
            crop_masks = torch.cat(crop_masks)
            crop_bboxes = torch.cat(crop_bboxes)
            crop_scores = torch.cat(crop_scores)
            keep = torchvision.ops.nms(crop_bboxes, crop_scores, self.args.iou)  # NMS
            crop_bboxes = uncrop_boxes_xyxy(crop_bboxes[keep], crop_region)
            crop_masks = uncrop_masks(crop_masks[keep], crop_region, ih, iw)
            crop_scores = crop_scores[keep]

            pred_masks.append(crop_masks)
            pred_bboxes.append(crop_bboxes)
            pred_scores.append(crop_scores)
            region_areas.append(area.expand(len(crop_masks)))

        pred_masks = torch.cat(pred_masks)
        pred_bboxes = torch.cat(pred_bboxes)
        pred_scores = torch.cat(pred_scores)
        region_areas = torch.cat(region_areas)

        # 去除裁剪之间的重复掩膜
        if len(crop_regions) > 1:
            scores = 1 / region_areas
            keep = torchvision.ops.nms(pred_bboxes, scores, crop_nms_thresh)
            pred_masks, pred_bboxes, pred_scores = pred_masks[keep], pred_bboxes[keep], pred_scores[keep]

        return pred_masks, pred_scores, pred_bboxes

    def setup_model(self, model=None, verbose=True):
        """
        初始化 Segment Anything Model (SAM) 以进行推理。

        该方法通过将模型分配到适当的设备并初始化图像归一化和其他 Ultralytics 兼容性设置来设置 SAM 模型。

        参数:
            model (torch.nn.Module | None): 预训练的 SAM 模型。如果为 None，则根据配置构建新模型。
            verbose (bool): 如果为 True，则打印选择的设备信息。

        示例:
            >>> predictor = Predictor()
            >>> predictor.setup_model(model=sam_model, verbose=True)
        """
        device = select_device(self.args.device, verbose=verbose)
        if model is None:
            model = self.get_model()
        model.eval()
        self.model = model.to(device)
        self.device = device
        self.mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)

        # Ultralytics 兼容性设置
        self.model.pt = False
        self.model.triton = False
        self.model.stride = 32
        self.model.fp16 = False
        self.done_warmup = True

    def get_model(self):
        """获取或构建用于图像分割任务的 Segment Anything Model (SAM)。"""
        return build_sam(self.args.model)

    def postprocess(self, preds, img, orig_imgs):
        """
        后处理 SAM 推理输出，以生成目标检测掩膜和边界框。

        该方法将掩膜和边框缩放到原始图像大小，并应用掩膜预测的阈值。它利用 SAM 的先进架构进行实时的、可提示的分割任务。

        参数:
            preds (Tuple[torch.Tensor]): 来自 SAM 模型推理的输出，包含：
                - pred_masks (torch.Tensor): 预测的掩膜，形状为 (N, 1, H, W)。
                - pred_scores (torch.Tensor): 每个掩膜的置信度分数，形状为 (N, 1)。
                - pred_bboxes (torch.Tensor, 可选): 如果 segment_all 为 True，则为预测的边界框。
            img (torch.Tensor): 处理后的输入图像张量，形状为 (C, H, W)。
            orig_imgs (List[np.ndarray] | torch.Tensor): 原始的未处理图像。

        返回:
            results (List[Results]): 包含检测掩膜、边界框和每个处理图像的其他元数据的 Results 对象列表。

        示例:
            >>> predictor = Predictor()
            >>> preds = predictor.inference(img)
            >>> results = predictor.postprocess(preds, img, orig_imgs)
        """
        # (N, 1, H, W), (N, 1)
        pred_masks, pred_scores = preds[:2]
        pred_bboxes = preds[2] if self.segment_all else None
        names = dict(enumerate(str(i) for i in range(len(pred_masks))))

        if not isinstance(orig_imgs, list):  # 输入图像是一个 torch.Tensor，而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for masks, orig_img, img_path in zip([pred_masks], orig_imgs, self.batch[0]):
            if len(masks) == 0:
                masks, pred_bboxes = None, torch.zeros((0, 6), device=pred_masks.device)
            else:
                masks = ops.scale_masks(masks[None].float(), orig_img.shape[:2], padding=False)[0]
                masks = masks > self.model.mask_threshold  # 转为布尔值
                if pred_bboxes is not None:
                    pred_bboxes = ops.scale_boxes(img.shape[2:], pred_bboxes.float(), orig_img.shape, padding=False)
                else:
                    pred_bboxes = batched_mask_to_box(masks)
                # 注意：SAM 模型不会返回类别信息。这里的 `cls` 只是为了保持一致性。
                cls = torch.arange(len(pred_masks), dtype=torch.int32, device=pred_masks.device)
                pred_bboxes = torch.cat([pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1)
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=pred_bboxes))
        # 重置 segment-all 模式。
        self.segment_all = False
        return results

    def setup_source(self, source):
        """
        设置推理的数据源。

        该方法配置将用于推理的图像数据源。它支持多种输入类型，如图像文件、目录、视频文件和其他兼容的数据源。

        参数:
            source (str | Path | None): 图像数据源的路径或标识符。可以是文件路径、目录路径、URL 或其他支持的源类型。

        示例:
            >>> predictor = Predictor()
            >>> predictor.setup_source("path/to/images")
            >>> predictor.setup_source("video.mp4")
            >>> predictor.setup_source(None)  # 如果可用，使用默认源

        注意:
            - 如果 source 为 None，则该方法可能会使用默认源（如果已配置）。
            - 该方法适应不同的源类型，并为后续的推理步骤做准备。
            - 支持的源类型可能包括本地文件、目录、URL 和视频流。
        """
        if source is not None:
            super().setup_source(source)

    def set_image(self, image):
        """
        为推理预处理并设置单张图像。

        此方法通过设置模型（如果尚未初始化）、配置数据源，并预处理图像以进行特征提取，
        为单张图像的推理做准备。它确保一次只设置一张图像，并提取图像特征以供后续使用。

        参数：
            image (str | np.ndarray): 图像文件的路径字符串，或代表通过cv2读取的图像的numpy数组。

        异常：
            AssertionError: 如果尝试设置多于一张图像，则抛出此异常。

        示例：
            >>> predictor = Predictor()
            >>> predictor.set_image("path/to/image.jpg")
            >>> predictor.set_image(cv2.imread("path/to/image.jpg"))

        注意：
            - 在对新图像进行推理之前应调用此方法。
            - 提取的特征存储在`self.features`属性中，供以后使用。
        """
        if self.model is None:
            self.setup_model(model=None)
        self.setup_source(image)
        assert len(self.dataset) == 1, "`set_image` 仅支持设置一张图像！"
        for batch in self.dataset:
            im = self.preprocess(batch[1])
            self.features = self.get_im_features(im)
            break

    def get_im_features(self, im):
        """使用SAM模型的图像编码器提取图像特征，以便后续掩码预测。"""
        assert isinstance(self.imgsz, (tuple, list)) and self.imgsz[0] == self.imgsz[1], (
            f"SAM模型仅支持方形图像大小，但得到了 {self.imgsz}。"
        )
        self.model.set_imgsz(self.imgsz)
        return self.model.image_encoder(im)

    def set_prompts(self, prompts):
        """为后续推理操作设置提示。"""
        self.prompts = prompts

    def reset_image(self):
        """重置当前图像及其特征，为后续推理清除它们。"""
        self.im = None
        self.features = None

    @staticmethod
    def remove_small_regions(masks, min_area=0, nms_thresh=0.7):
        """
        从分割掩码中移除小的孤立区域和孔洞。

        此函数对由Segment Anything Model（SAM）生成的分割掩码进行后处理。
        它移除输入掩码中的小孤立区域和孔洞，然后执行非最大抑制（NMS）来去除任何新创建的重复框。

        参数：
            masks (torch.Tensor): 要处理的分割掩码，形状为(N, H, W)，其中N是掩码数量，H是高度，W是宽度。
            min_area (int): 用于移除孤立区域和孔洞的最小面积阈值。小于此阈值的区域将被移除。
            nms_thresh (float): NMS算法的IoU阈值，用于去除重复框。

        返回：
            new_masks (torch.Tensor): 处理后的掩码，移除了小区域，形状为(N, H, W)。
            keep (List[int]): 在NMS之后保留的掩码索引，用于过滤相应的框。

        示例：
            >>> masks = torch.rand(5, 640, 640) > 0.5  # 5个随机二进制掩码
            >>> new_masks, keep = remove_small_regions(masks, min_area=100, nms_thresh=0.7)
            >>> print(f"原始掩码: {masks.shape}, 处理后掩码: {new_masks.shape}")
            >>> print(f"保留的掩码索引: {keep}")
        """
        import torchvision  # 为了更快速地导入 'import ultralytics'

        if len(masks) == 0:
            return masks

        # 过滤掉小的孤立区域和孔洞
        new_masks = []
        scores = []
        for mask in masks:
            mask = mask.cpu().numpy().astype(np.uint8)
            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # 给变化的掩码分配分数0，给未变化的掩码分配分数1，这样NMS会偏好不需要后处理的掩码
            scores.append(float(unchanged))

        # 重新计算框并去除任何新的重复框
        new_masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(new_masks)
        keep = torchvision.ops.nms(boxes.float(), torch.as_tensor(scores), nms_thresh)

        return new_masks[keep].to(device=masks.device, dtype=masks.dtype), keep


class SAM2Predictor(Predictor):
    """
    SAM2Predictor类用于基于Segment Anything Model 2架构的高级图像分割。

    该类扩展了基础Predictor类，以实现SAM2特定的图像分割功能。它提供了模型初始化、特征提取和基于提示的推理方法。

    属性：
        _bb_feat_sizes (List[Tuple[int, int]]): 不同骨干网络层的特征大小。
        model (torch.nn.Module): 加载的SAM2模型。
        device (torch.device): 模型加载的设备（CPU或GPU）。
        features (Dict[str, torch.Tensor]): 用于高效推理的缓存图像特征。
        segment_all (bool): 标志，指示是否应预测所有分割区域。
        prompts (Dict): 用于推理的各种提示的字典。

    方法：
        get_model: 获取并初始化SAM2模型。
        prompt_inference: 基于各种提示执行图像分割推理。
        set_image: 预处理并设置单张图像以进行推理。
        get_im_features: 使用SAM2的图像编码器提取和处理图像特征。

    示例：
        >>> predictor = SAM2Predictor(cfg)
        >>> predictor.set_image("path/to/image.jpg")
        >>> bboxes = [[100, 100, 200, 200]]
        >>> result = predictor(bboxes=bboxes)[0]
        >>> print(f"预测了{len(result.masks)}个掩码，平均得分 {result.boxes.conf.mean():.2f}")
    """

    _bb_feat_sizes = [
        (256, 256),
        (128, 128),
        (64, 64),
    ]

    def get_model(self):
        """获取并初始化Segment Anything Model 2（SAM2）用于图像分割任务。"""
        return build_sam(self.args.model)

    def prompt_inference(
        self,
        im,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
        img_idx=-1,
    ):
        """
        基于各种提示使用SAM2架构执行图像分割推理。

        此方法利用Segment Anything Model 2（SAM2）根据提供的提示（如边界框、点或现有掩码）生成输入图像的分割掩码。它支持单对象和多对象预测场景。

        参数：
            im (torch.Tensor): 预处理后的输入图像tensor，形状为(N, C, H, W)。
            bboxes (np.ndarray | List[List[float]] | None): 边界框，形状为(N, 4)，XYXY格式。
            points (np.ndarray | List[List[float]] | None): 物体位置点，形状为(N, 2)，单位为像素。
            labels (np.ndarray | List[int] | None): 点提示标签，形状为(N,)，1表示前景，0表示背景。
            masks (np.ndarray | None): 来自先前预测的低分辨率掩码，形状为(N, H, W)。
            multimask_output (bool): 是否返回多个掩码，用于模糊提示。
            img_idx (int): 批量中要处理的图像索引。

        返回：
            (np.ndarray): 输出掩码，形状为(C, H, W)，其中C是生成的掩码数量。
            (np.ndarray): 每个掩码的质量分数，长度为C。

        示例：
            >>> predictor = SAM2Predictor(cfg)
            >>> image = torch.rand(1, 3, 640, 640)
            >>> bboxes = [[100, 100, 200, 200]]
            >>> result = predictor(image, bboxes=bboxes)[0]
            >>> print(f"生成了{result.masks.shape[0]}个掩码，平均得分 {result.boxes.conf.mean():.2f}")

        注意：
            - 该方法支持提供点或边界框时的批量推理，用于多对象预测。
            - 输入提示（边界框、点）会自动缩放以匹配输入图像的尺寸。
            - 当同时提供边界框和点时，它们会合并成一个单独的“点”输入给模型。

        参考文献：
            - SAM2论文：[添加SAM2论文链接（如果可用）]
        """
        features = self.get_im_features(im) if self.features is None else self.features

        points, labels, masks = self._prepare_prompts(im.shape[2:], bboxes, points, labels, masks)
        points = (points, labels) if points is not None else None

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=points,
            boxes=None,
            masks=masks,
        )
        # 预测掩码
        batched_mode = points is not None and points[0].shape[0] > 1  # 多对象预测
        high_res_features = [feat_level[img_idx].unsqueeze(0) for feat_level in features["high_res_feats"]]
        pred_masks, pred_scores, _, _ = self.model.sam_mask_decoder(
            image_embeddings=features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        # (N, d, H, W) --> (N*d, H, W), (N, d) --> (N*d, )
        # `d` 可能是1或3，取决于`multimask_output`。
        return pred_masks.flatten(0, 1), pred_scores.flatten(0, 1)

    def _prepare_prompts(self, dst_shape, bboxes=None, points=None, labels=None, masks=None):
        """
        根据目标尺寸对输入提示进行预处理和变换。

        参数：
            dst_shape (tuple): 目标图像的尺寸 (height, width)。
            bboxes (np.ndarray | List | None): 边界框，XYXY 格式，形状为 (N, 4)。
            points (np.ndarray | List | None): 指示物体位置的点，形状为 (N, 2) 或 (N, num_points, 2)，单位为像素。
            labels (np.ndarray | List | None): 点的标签，形状为 (N,) 或 (N, num_points)。前景为 1，背景为 0。
            masks (List | np.ndarray, 可选): 物体对应的掩码，每个掩码是一个二维数组。

        异常：
            AssertionError: 如果传入了标签，但点的数量与标签不匹配，则会抛出异常。

        返回：
            (tuple): 返回一个元组，包含处理后的 points、labels 和 masks。
        """
        bboxes, points, labels, masks = super()._prepare_prompts(dst_shape, bboxes, points, labels, masks)
        if bboxes is not None:
            bboxes = bboxes.view(-1, 2, 2)
            bbox_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=bboxes.device).expand(len(bboxes), -1)
            # 注意：将“boxes”和“points”合并为一个“points”输入，
            # 并作为整体输入传入 model.sam_prompt_encoder。
            if points is not None:
                points = torch.cat([bboxes, points], dim=1)
                labels = torch.cat([bbox_labels, labels], dim=1)
            else:
                points, labels = bboxes, bbox_labels
        return points, labels, masks

    def set_image(self, image):
        """
        预处理并设置单张图像，用于 SAM2 模型推理。

        此方法会在模型未初始化时进行初始化，配置数据源，并提取特征向量。
        每次只能设置一张图像。

        参数：
            image (str | np.ndarray): 图像路径（字符串）或图像数据（NumPy 数组）。

        异常：
            AssertionError: 如果尝试设置超过一张图像，将抛出异常。

        示例：
            >>> predictor = SAM2Predictor()
            >>> predictor.set_image("path/to/image.jpg")
            >>> predictor.set_image(np.array([...]))  # 直接使用 numpy 图像数组

        注意：
            - 在对新图像进行推理之前，必须调用此方法。
            - 提取的图像特征将被缓存，以加速对相同图像的重复推理。
            - 一次仅支持一张图像。如需处理多张图像，请逐张调用该方法。
        """
        if self.model is None:
            self.setup_model(model=None)
        self.setup_source(image)
        assert len(self.dataset) == 1, "`set_image` 只支持一次设置一张图像！"
        for batch in self.dataset:
            im = self.preprocess(batch[1])
            self.features = self.get_im_features(im)
            break

    def get_im_features(self, im):
        """从 SAM 图像编码器中提取图像特征，用于后续处理。"""
        assert isinstance(self.imgsz, (tuple, list)) and self.imgsz[0] == self.imgsz[1], (
            f"SAM 2 模型仅支持方形图像尺寸，但当前为 {self.imgsz}。"
        )
        self.model.set_imgsz(self.imgsz)
        self._bb_feat_sizes = [[x // (4 * i) for x in self.imgsz] for i in [1, 2, 4]]

        backbone_out = self.model.forward_image(im)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}


class SAM2VideoPredictor(SAM2Predictor):
    """
    SAM2VideoPredictor 用于处理视频中的用户交互并管理推理状态。

    本类在 SAM2Predictor 的基础上扩展，支持视频处理，并维护推理操作的状态。
    它包含了一些用于管理掩码不重叠、清除非条件内存、设置预测事件回调等的配置项。

    属性：
        inference_state (Dict): 字典，用于存储当前的推理状态。
        non_overlap_masks (bool): 是否启用非重叠掩码的标志。
        clear_non_cond_mem_around_input (bool): 是否在输入周围清除非条件性记忆的标志。
        clear_non_cond_mem_for_multi_obj (bool): 是否在多目标场景下清除非条件性记忆的标志。
        callbacks (Dict): 包含多个预测生命周期事件的回调函数字典。

    参数：
        cfg (Dict, 可选): 用于初始化的配置项，默认为 DEFAULT_CFG。
        overrides (Dict, 可选): 用于覆盖默认配置的额外设置。
        _callbacks (List, 可选): 自定义回调列表，默认为 None。

    注意：
        属性 `fill_hole_area` 已定义但当前实现中未使用。
    """

    # fill_hole_area = 8  # 未使用

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        使用配置项初始化视频预测器，并设置必要的状态与回调。

        此构造函数会根据传入配置进行初始化，应用任何 override 的参数，
        并设置控制行为的标志及初始化推理状态。

        参数：
            cfg (Dict): 包含默认设置的配置字典。
            overrides (Dict | None): 用于覆盖默认配置的参数字典。
            _callbacks (Dict | None): 包含自定义行为函数的回调字典。

        示例：
            >>> predictor = SAM2VideoPredictor(cfg=DEFAULT_CFG)
            >>> predictor_example_with_imgsz = SAM2VideoPredictor(overrides={"imgsz": 640})
            >>> predictor_example_with_callback = SAM2VideoPredictor(_callbacks={"on_predict_start": custom_callback})
        """
        super().__init__(cfg, overrides, _callbacks)
        self.inference_state = {}
        self.non_overlap_masks = True
        self.clear_non_cond_mem_around_input = False
        self.clear_non_cond_mem_for_multi_obj = False
        self.callbacks["on_predict_start"].append(self.init_state)

    def get_model(self):
        """
        获取并配置启用了二值化的模型。

        注意:
            该方法重写了基类实现，将二值化标志设置为 True。
        """
        model = super().get_model()
        model.set_binarize(True)
        return model

    def inference(self, im, bboxes=None, points=None, labels=None, masks=None):
        """
        基于给定的输入提示执行图像分割推理，使用当前加载的图像。该方法利用 SAM（Segment Anything Model）架构，包括图像编码器、提示编码器和掩膜解码器，用于实时和可提示的分割任务。

        参数:
            im (torch.Tensor): 预处理后的输入图像张量，形状为 (N, C, H, W)。
            bboxes (np.ndarray | List, 可选): 边界框，形状为 (N, 4)，XYXY 格式。
            points (np.ndarray | List, 可选): 指示物体位置的点，形状为 (N, 2)，单位为像素。
            labels (np.ndarray | List, 可选): 点提示的标签，形状为 (N, )。1 = 前景，0 = 背景。
            masks (np.ndarray, 可选): 来自先前预测的低分辨率掩膜，形状为 (N, H, W)。对于 SAM，H=W=256。

        返回:
            (np.ndarray): 输出掩膜，形状为 CxHxW，其中 C 是生成的掩膜数量。
            (np.ndarray): 长度为 C 的数组，包含模型为每个掩膜预测的质量分数。
        """
        # 如果 self.prompts 中有任何存储的提示，则覆盖它们
        bboxes = self.prompts.pop("bboxes", bboxes)
        points = self.prompts.pop("points", points)
        masks = self.prompts.pop("masks", masks)

        frame = self.dataset.frame
        self.inference_state["im"] = im
        output_dict = self.inference_state["output_dict"]
        if len(output_dict["cond_frame_outputs"]) == 0:  # 初始化提示
            points, labels, masks = self._prepare_prompts(im.shape[2:], bboxes, points, labels, masks)
            if points is not None:
                for i in range(len(points)):
                    self.add_new_prompts(obj_id=i, points=points[[i]], labels=labels[[i]], frame_idx=frame)
            elif masks is not None:
                for i in range(len(masks)):
                    self.add_new_prompts(obj_id=i, masks=masks[[i]], frame_idx=frame)
        self.propagate_in_video_preflight()

        consolidated_frame_inds = self.inference_state["consolidated_frame_inds"]
        batch_size = len(self.inference_state["obj_idx_to_id"])
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("没有提供点；请先添加点")

        if frame in consolidated_frame_inds["cond_frame_outputs"]:
            storage_key = "cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
            if self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1):
                # 清除周围帧的非条件内存
                self._clear_non_cond_mem_around_input(frame)
        elif frame in consolidated_frame_inds["non_cond_frame_outputs"]:
            storage_key = "non_cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
        else:
            storage_key = "non_cond_frame_outputs"
            current_out = self._run_single_frame_inference(
                output_dict=output_dict,
                frame_idx=frame,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            output_dict[storage_key][frame] = current_out
        # 为后续与每个单独物体的交互创建每个物体输出的切片
        self._add_output_per_object(frame, current_out, storage_key)
        self.inference_state["frames_already_tracked"].append(frame)
        pred_masks = current_out["pred_masks"].flatten(0, 1)
        pred_masks = pred_masks[(pred_masks > self.model.mask_threshold).sum((1, 2)) > 0]  # 过滤空白掩膜

        return pred_masks, torch.ones(len(pred_masks), dtype=pred_masks.dtype, device=pred_masks.device)

    def postprocess(self, preds, img, orig_imgs):
        """
        对预测结果进行后处理，应用非重叠约束（如果需要）。

        该方法通过在 `non_overlap_masks` 标志设置为 True 时，对预测的掩膜应用非重叠约束，来扩展后处理功能。这样可以确保掩膜不重叠，这对于某些应用非常有用。

        参数:
            preds (Tuple[torch.Tensor]): 模型的预测结果。
            img (torch.Tensor): 处理后的图像张量。
            orig_imgs (List[np.ndarray]): 处理前的原始图像。

        返回:
            results (list): 后处理后的预测结果。

        注意:
            如果 `non_overlap_masks` 为 True，则该方法会应用约束以确保掩膜不重叠。
        """
        results = super().postprocess(preds, img, orig_imgs)
        if self.non_overlap_masks:
            for result in results:
                if result.masks is None or len(result.masks) == 0:
                    continue
                result.masks.data = self.model._apply_non_overlapping_constraints(result.masks.data.unsqueeze(0))[0]
        return results

    @smart_inference_mode()
    def add_new_prompts(
        self,
        obj_id,
        points=None,
        labels=None,
        masks=None,
        frame_idx=0,
    ):
        """
        向特定帧为给定的对象ID添加新的点或掩膜。

        此方法通过新提示（点或掩膜）更新推理状态，针对指定的对象和帧索引进行操作。确保每次调用只能添加一种类型的提示（点或掩膜），并根据提供的提示和现有状态更新内部状态。它还处理基于提供的提示生成新的分割。

        参数：
            obj_id (int): 与提示相关联的对象ID。
            points (torch.Tensor, 可选): 感兴趣点的坐标。默认为 None。
            labels (torch.Tensor, 可选): 对应点的标签。默认为 None。
            masks (torch.Tensor, 可选): 对象的二进制掩膜。默认为 None。
            frame_idx (int, 可选): 应用提示的帧的索引。默认为 0。

        返回：
            (tuple): 一个元组，包含展平的预测掩膜和一个表示对象数量的全1张量。

        异常：
            AssertionError: 如果同时提供了 `masks` 和 `points`，或两者都没有提供。

        注意：
            - 每次调用时只能添加一种类型的提示（点或掩膜）。
            - 如果该帧是首次被追踪，则视为初始条件帧。
            - 此方法处理输出的合并和掩膜的重新调整，以匹配原视频分辨率。
        """
        assert (masks is None) ^ (points is None), "'masks' 和 'points' 提示不能同时提供，也不能都不提供。"
        obj_idx = self._obj_id_to_idx(obj_id)

        point_inputs = None
        pop_key = "point_inputs_per_obj"
        if points is not None:
            point_inputs = {"point_coords": points, "point_labels": labels}
            self.inference_state["point_inputs_per_obj"][obj_idx][frame_idx] = point_inputs
            pop_key = "mask_inputs_per_obj"
        self.inference_state["mask_inputs_per_obj"][obj_idx][frame_idx] = masks
        self.inference_state[pop_key][obj_idx].pop(frame_idx, None)
        
        # 如果此帧尚未被追踪过，则视为初始条件帧，
        # 即输入的点用于在此帧生成分割，不依赖其他帧的内存，类似于 SAM 的行为。
        # 否则（如果已经被追踪过），输入的点用于修正已追踪的掩膜。
        is_init_cond_frame = frame_idx not in self.inference_state["frames_already_tracked"]
        obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
        
        # 如果是初始条件帧，或者模型要求所有帧都作为修正条件帧，则将该帧标记为条件帧
        is_cond = is_init_cond_frame or self.model.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # 获取先前预测的掩膜logits，并将其与新的点击一起输入到 SAM 掩膜解码器。
        prev_sam_mask_logits = None
        # 先查询临时输出字典，它包含最新的输出
        # 如果未找到，再查询条件帧输出和非条件帧输出。
        if point_inputs is not None:
            prev_out = (
                obj_temp_output_dict[storage_key].get(frame_idx)
                or obj_output_dict["cond_frame_outputs"].get(frame_idx)
                or obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
            )

            if prev_out is not None and prev_out.get("pred_masks") is not None:
                prev_sam_mask_logits = prev_out["pred_masks"].to(device=self.device, non_blocking=True)
                # 限制prev_sam_mask_logits的值域，避免罕见的数值问题。
                prev_sam_mask_logits.clamp_(-32.0, 32.0)
        current_out = self._run_single_frame_inference(
            output_dict=obj_output_dict,  # 在单个对象的切片上运行
            frame_idx=frame_idx,
            batch_size=1,  # 在单个对象的切片上运行
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=masks,
            reverse=False,
            # 添加点击或掩膜时跳过内存编码器。我们在`propagate_in_video`的开头执行内存编码器（用户确认他们的点击之后）。
            # 这样可以确保所有对象的非重叠约束被执行，而不是编码成内存。
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # 将输出添加到输出字典中，以供未来作为内存使用
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # 将输出掩膜调整为原始视频分辨率
        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
        )
        pred_masks = consolidated_out["pred_masks"].flatten(0, 1)
        return pred_masks.flatten(0, 1), torch.ones(1, dtype=pred_masks.dtype, device=pred_masks.device)

    @smart_inference_mode()
    def propagate_in_video_preflight(self):
        """
        准备推理状态并合并临时输出，启动跟踪。

        该方法标志着跟踪的开始，并且不允许在会话重置之前添加新对象。
        它将 `temp_output_dict_per_obj` 中的临时输出合并到 `output_dict` 中。
        此外，它还清除了输入帧周围的非条件内存，并确保状态与提供的输入一致。
        """
        # 跟踪已开始，并且在会话重置之前不允许添加新对象。
        self.inference_state["tracking_has_started"] = True
        batch_size = len(self.inference_state["obj_idx_to_id"])

        # 合并每个对象的临时输出（存储在 "temp_output_dict_per_obj" 中），并将它们添加到 "output_dict" 中。
        temp_output_dict_per_obj = self.inference_state["temp_output_dict_per_obj"]
        output_dict = self.inference_state["output_dict"]
        # "consolidated_frame_inds" 包含已合并临时输出的帧索引（这些帧可能是本次调用合并的，也可能是之前调用的 `propagate_in_video_preflight` 合并的）。
        consolidated_frame_inds = self.inference_state["consolidated_frame_inds"]
        for is_cond in {False, True}:
            # 分别合并条件和非条件的临时输出
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # 查找包含任何对象临时输出的帧（这些应该是刚刚接收到点击作为掩膜输入的帧）
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # 合并该帧上所有对象的临时输出
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    frame_idx, is_cond=is_cond, run_mem_encoder=True
                )
                # 将它们合并到 "output_dict" 中，并为每个对象创建切片
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(frame_idx, consolidated_out, storage_key)
                if self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1):
                    # 清除输入帧周围的非条件内存
                    self._clear_non_cond_mem_around_input(frame_idx)

            # 清除 `temp_output_dict_per_obj` 中的临时输出
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # 特殊情况：如果已向 "cond_frame_outputs" 添加输出，则应删除相同帧在 "non_cond_frame_outputs" 中的任何输出
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in self.inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # 确保 "consolidated_frame_inds" 中的帧索引正好是那些有点或掩膜输入的帧（在正确的工作流下应当成立）。
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"] | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in self.inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in self.inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds

    @staticmethod
    def init_state(predictor):
        """
        初始化预测器的推理状态。

        此函数设置执行视频数据推理所需的初始状态。它包括初始化各种字典和有序字典，用于存储与跟踪过程相关的输入、输出和其他元数据。

        参数:
            predictor (SAM2VideoPredictor): 用于初始化状态的预测器对象。
        """
        if len(predictor.inference_state) > 0:  # 表示已初始化
            return
        assert predictor.dataset is not None
        assert predictor.dataset.mode == "video"

        inference_state = {
            "num_frames": predictor.dataset.frames,
            "point_inputs_per_obj": {},  # 每帧上的输入点
            "mask_inputs_per_obj": {},  # 每帧上的输入掩码
            "constants": {},  # 不随帧变化的值（因此只需保留一份）
            # 客户端对象 ID 与模型端对象索引之间的映射
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            # 用于存储每帧上的模型跟踪结果和状态的存储空间
            "output_dict": {
                "cond_frame_outputs": {},  # 字典，包含 {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # 字典，包含 {frame_idx: <out>}
            },
            # 每个对象跟踪结果的切片（视图），与 "output_dict" 共享内存
            "output_dict_per_obj": {},
            # 临时存储，用于保存用户与帧交互时的新输出
            # （它在传播开始前合并到 "output_dict" 中）
            "temp_output_dict_per_obj": {},
            # 已经保存了点击或掩码输入合并结果的帧
            # （我们直接在跟踪过程中使用它们的合并输出）
            "consolidated_frame_inds": {
                "cond_frame_outputs": set(),  # 包含帧索引的集合
                "non_cond_frame_outputs": set(),  # 包含帧索引的集合
            },
            # 每帧跟踪的元数据（例如：跟踪的方向）
            "tracking_has_started": False,
            "frames_already_tracked": [],
        }
        predictor.inference_state = inference_state

    def get_im_features(self, im, batch=1):
        """
        使用 SAM2 的图像编码器提取并处理图像特征，以便后续的分割任务。

        参数:
            im (torch.Tensor): 输入的图像张量。
            batch (int, 可选): 如果有多个提示，则扩展特征的批次大小。默认为 1。

        返回:
            vis_feats (torch.Tensor): 从图像中提取的视觉特征。
            vis_pos_embed (torch.Tensor): 视觉特征的位置信息嵌入。
            feat_sizes (List(Tuple[int])): 包含提取特征大小的列表。

        注意:
            - 如果 `batch` 大于 1，特征将扩展以适应批次大小。
            - 该方法利用模型的 `_prepare_backbone_features` 方法来准备骨干网络特征。
        """
        backbone_out = self.model.forward_image(im)
        if batch > 1:  # 如果有多个提示，扩展特征
            for i, feat in enumerate(backbone_out["backbone_fpn"]):
                backbone_out["backbone_fpn"][i] = feat.expand(batch, -1, -1, -1)
            for i, pos in enumerate(backbone_out["vision_pos_enc"]):
                pos = pos.expand(batch, -1, -1, -1)
                backbone_out["vision_pos_enc"][i] = pos
        _, vis_feats, vis_pos_embed, feat_sizes = self.model._prepare_backbone_features(backbone_out)
        return vis_feats, vis_pos_embed, feat_sizes

    def _obj_id_to_idx(self, obj_id):
        """
        将客户端对象 ID 映射到模型端对象索引。

        参数:
            obj_id (int): 客户端提供的对象的唯一标识符。

        返回:
            obj_idx (int): 模型端的对象索引。

        异常:
            RuntimeError: 如果在跟踪开始后尝试添加新对象，则抛出此异常。

        注意:
            - 该方法更新或检索存储在 `inference_state` 中的对象 ID 与索引之间的映射。
            - 它确保只能在跟踪开始之前添加新对象。
            - 它维护 ID 和索引之间的双向映射（`obj_id_to_idx` 和 `obj_idx_to_id`）。
            - 为新对象初始化了额外的数据结构，用于存储输入和输出。
        """
        obj_idx = self.inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # 这是一个新对象 ID，在服务器之前没有发送过。我们只允许在
        # 跟踪开始前添加新对象。
        allow_new_object = not self.inference_state["tracking_has_started"]
        if allow_new_object:
            # 获取下一个对象槽位
            obj_idx = len(self.inference_state["obj_id_to_idx"])
            self.inference_state["obj_id_to_idx"][obj_id] = obj_idx
            self.inference_state["obj_idx_to_id"][obj_idx] = obj_id
            self.inference_state["obj_ids"] = list(self.inference_state["obj_id_to_idx"])
            # 为该对象设置输入输出结构
            self.inference_state["point_inputs_per_obj"][obj_idx] = {}
            self.inference_state["mask_inputs_per_obj"][obj_idx] = {}
            self.inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # 字典，包含 {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # 字典，包含 {frame_idx: <out>}
            }
            self.inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # 字典，包含 {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # 字典，包含 {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"跟踪开始后无法添加新对象 ID {obj_id}. "
                f"所有现有对象 ID: {self.inference_state['obj_ids']}. "
                f"请调用 'reset_state' 重新开始。"
            )

    def _run_single_frame_inference(
        self,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """
        根据当前输入和先前的记忆在单帧上运行跟踪。

        参数：
            output_dict (Dict): 包含跟踪过程输出状态的字典。
            frame_idx (int): 当前帧的索引。
            batch_size (int): 处理该帧时的批大小。
            is_init_cond_frame (bool): 表示当前帧是否为初始化条件帧。
            point_inputs (Dict, 可选): 输入点及其标签。默认为None。
            mask_inputs (torch.Tensor, 可选): 输入的二值掩码。默认为None。
            reverse (bool): 表示是否应以反向顺序执行跟踪。
            run_mem_encoder (bool): 表示是否应执行内存编码器。
            prev_sam_mask_logits (torch.Tensor, 可选): 当前对象的先前掩码逻辑。默认为None。

        返回：
            current_out (dict): 包含跟踪步骤输出的字典，包括更新的特征和预测。

        异常：
            AssertionError: 如果同时提供`point_inputs`和`mask_inputs`，或者两者都没有提供。

        注意：
            - 该方法假定`point_inputs`和`mask_inputs`是互斥的。
            - 该方法使用`get_im_features`方法检索图像特征。
            - `maskmem_pos_enc`假定在各帧间保持不变，因此只存储一份。
            - 由于需要CUDA扩展，`fill_holes_in_mask_scores`函数被注释掉，当前不支持。
        """
        # 获取正确的图像特征
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self.get_im_features(
            self.inference_state["im"], batch_size
        )

        # 确保同一帧中`point_inputs`和`mask_inputs`不能同时出现
        assert point_inputs is None or mask_inputs is None
        current_out = self.model.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=self.inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            current_out["maskmem_features"] = maskmem_features.to(
                dtype=torch.float16, device=self.device, non_blocking=True
            )
        # 注意：由于需要CUDA扩展，当前不支持`fill_holes_in_mask_scores`函数
        # 可能会填充预测掩码中的空洞
        # if self.fill_hole_area > 0:
        #     pred_masks = current_out["pred_masks"].to(self.device, non_blocking=True)
        #     pred_masks = fill_holes_in_mask_scores(pred_masks, self.fill_hole_area)

        # "maskmem_pos_enc"在各帧间保持一致，因此只需要存储一份
        current_out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(current_out["maskmem_pos_enc"])
        return current_out

    def _get_maskmem_pos_enc(self, out_maskmem_pos_enc):
        """
        在各帧和对象之间缓存和管理掩码记忆的位置信息。

        此方法通过缓存掩码记忆的位置信息（`maskmem_pos_enc`）来优化存储，
        因为该信息在各帧和对象之间是恒定的，因此减少了在推理过程中存储冗余信息的需求。
        它会检查位置信息是否已经缓存，如果没有，则缓存提供的编码片段。
        如果批量大小大于一，则扩展缓存的位置信息以匹配当前的批量大小。

        参数：
            out_maskmem_pos_enc (List[torch.Tensor] 或 None): 掩码记忆的位置信息。
                应该是一个张量列表或者None。

        返回：
            out_maskmem_pos_enc (List[torch.Tensor]): 掩码记忆的位置信息，可能是缓存的或者扩展后的。

        注意：
            - 该方法假定`out_maskmem_pos_enc`是一个张量列表或者None。
            - 由于该编码在对象之间相同，因此只缓存一个对象的切片。
            - 该方法检查位置信息是否已经缓存，并将其存储在会话的常量中。
            - 如果批量大小大于一，则扩展缓存的`maskmem_pos_enc`以适应批量大小。
        """
        model_constants = self.inference_state["constants"]
        # "out_maskmem_pos_enc"应该是一个张量列表或None
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # 只取一个对象的切片，因为它在各对象间是相同的
                maskmem_pos_enc = [x[:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # 将缓存的maskmem_pos_enc扩展到实际的批量大小
            batch_size = out_maskmem_pos_enc[0].size(0)
            if batch_size > 1:
                out_maskmem_pos_enc = [x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc]
        return out_maskmem_pos_enc

    def _consolidate_temp_output_across_obj(
        self,
        frame_idx,
        is_cond=False,
        run_mem_encoder=False,
    ):
        """
        合并每个对象的临时输出，生成所有对象的统一输出。

        该方法将每个对象在给定帧上的临时输出合并成一个统一的输出。
        它会填充缺失的对象，无论是从主输出字典中获取，还是在主输出中不存在时留下占位符。如果需要，方法还可以在应用非重叠约束后重新运行内存编码器。

        参数：
            frame_idx (int): 要合并输出的帧的索引。
            is_cond (bool, 可选): 指示该帧是否被视为条件帧。默认值为False。
            run_mem_encoder (bool, 可选): 指定是否在合并输出后重新运行内存编码器。默认值为False。

        返回：
            consolidated_out (dict): 一个合并后的输出字典，包含所有对象的合并结果。

        备注：
            - 该方法初始化合并输出时使用占位符值来处理缺失的对象。
            - 它在临时输出字典和主输出字典中查找每个对象的输出。
            - 如果 `run_mem_encoder` 为True，它会应用非重叠约束，并重新运行内存编码器。
            - `maskmem_features` 和 `maskmem_pos_enc` 只有在 `run_mem_encoder` 为True时才会被填充。
        """
        batch_size = len(self.inference_state["obj_idx_to_id"])
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # 初始化 `consolidated_out`。它的 "maskmem_features" 和 "maskmem_pos_enc"
        # 将在重新运行内存编码器并应用非重叠约束后填充。它的 "pred_masks" 被预填充为一个大
        # 负值（NO_OBJ_SCORE）以表示缺失的对象。
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "pred_masks": torch.full(
                size=(batch_size, 1, self.imgsz[0] // 4, self.imgsz[1] // 4),
                fill_value=-1024.0,
                dtype=torch.float32,
                device=self.device,
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.model.hidden_dim),
                fill_value=-1024.0,
                dtype=torch.float32,
                device=self.device,
            ),
            "object_score_logits": torch.full(
                size=(batch_size, 1),
                # 默认情况下为10.0，表示对象存在，即sigmoid(10)=1，和 `MaskDecoder` 中的 `predict_masks` 相同。
                fill_value=10.0,
                dtype=torch.float32,
                device=self.device,
            ),
        }
        for obj_idx in range(batch_size):
            obj_temp_output_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
            out = (
                obj_temp_output_dict[storage_key].get(frame_idx)
                # 如果该对象在 "temp_output_dict_per_obj" 中没有出现在该帧，
                # 我们会回退并在 "output_dict_per_obj" 中查找它的前一个输出。
                # 我们在 "output_dict_per_obj" 中查找 "cond_frame_outputs" 和 "non_cond_frame_outputs"
                # 来寻找该对象的前一个输出。
                or obj_output_dict["cond_frame_outputs"].get(frame_idx)
                or obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
            )
            # 如果该对象在 "output_dict_per_obj" 中也没有找到，则跳过它
            # 并将其掩码得分保持为默认得分（即上面的 NO_OBJ_SCORE 占位符），
            # 同时将对象指针设置为一个虚拟指针。
            if out is None:
                # 对于那些在当前帧没有任何输入或跟踪结果的对象（仅在 `run_mem_encoder=True` 下，
                # 即需要为跟踪构建内存时），填充虚拟对象指针。
                if run_mem_encoder:
                    # 填充对象指针为一个虚拟指针（基于空掩码）
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = self._get_empty_mask_ptr(frame_idx)
                continue
            # 将临时对象输出掩码添加到合并输出的掩码中
            consolidated_out["pred_masks"][obj_idx : obj_idx + 1] = out["pred_masks"]
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]

        # 可选地，对合并后的得分应用非重叠约束，并重新运行内存编码器
        if run_mem_encoder:
            high_res_masks = F.interpolate(
                consolidated_out["pred_masks"],
                size=self.imgsz,
                mode="bilinear",
                align_corners=False,
            )
            if self.model.non_overlap_masks_for_mem_enc:
                high_res_masks = self.model._apply_non_overlapping_constraints(high_res_masks)
            consolidated_out["maskmem_features"], consolidated_out["maskmem_pos_enc"] = self._run_memory_encoder(
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                is_mask_from_pts=True,  # 这些帧是用户交互的帧
                object_score_logits=consolidated_out["object_score_logits"],
            )

        return consolidated_out

    def _get_empty_mask_ptr(self, frame_idx):
        """
        基于当前帧的空掩码生成一个虚拟对象指针。

        参数：
            frame_idx (int): 当前帧的索引，用于生成虚拟对象指针。

        返回：
            (torch.Tensor): 一个基于空掩码生成的虚拟对象指针的张量。
        """
        # 获取正确的图像特征
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self.get_im_features(self.inference_state["im"])

        # 将空掩码和上述图像特征输入，获取虚拟对象指针
        current_out = self.model.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            # 使用一个虚拟（空）掩码和单个对象
            mask_inputs=torch.zeros((1, 1, *self.imgsz), dtype=torch.float32, device=self.device),
            output_dict={},
            num_frames=self.inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]

    def _run_memory_encoder(self, batch_size, high_res_masks, object_score_logits, is_mask_from_pts):
        """
        在掩膜上运行内存编码器。

        这通常发生在对对象分数应用非重叠约束之后。由于它们的分数发生了变化，内存也需要使用内存编码器重新计算。

        参数：
            batch_size (int): 处理帧的批次大小。
            high_res_masks (torch.Tensor): 用于计算内存的高分辨率掩膜。
            object_score_logits (torch.Tensor): 表示对象分数的logits。
            is_mask_from_pts (bool): 指示掩膜是否来自点交互。

        返回：
            (tuple[torch.Tensor, torch.Tensor]): 一个包含编码后的掩膜特征和位置编码的元组。
        """
        # 获取正确的图像特征
        current_vision_feats, _, feat_sizes = self.get_im_features(self.inference_state["im"], batch_size)
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=is_mask_from_pts,
            object_score_logits=object_score_logits,
        )

        # "maskmem_pos_enc" 在所有帧中是相同的，因此我们只需要存储一个副本
        maskmem_pos_enc = self._get_maskmem_pos_enc(maskmem_pos_enc)
        return maskmem_features.to(dtype=torch.float16, device=self.device, non_blocking=True), maskmem_pos_enc

    def _add_output_per_object(self, frame_idx, current_out, storage_key):
        """
        将多对象输出拆分为每个对象的输出切片，并将它们添加到 Output_Dict_Per_Obj 中。

        生成的切片共享相同的张量存储。

        参数：
            frame_idx (int): 当前帧的索引。
            current_out (Dict): 当前输出字典，包含多对象的输出。
            storage_key (str): 用于将输出存储在每个对象输出字典中的键。
        """
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        for obj_idx, obj_output_dict in self.inference_state["output_dict_per_obj"].items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

    def _clear_non_cond_mem_around_input(self, frame_idx):
        """
        清除输入帧周围的非条件内存。

        当用户提供修正点击时，周围帧的非条件内存可能仍然包含过时的对象外观信息，并可能会混淆模型。
        这个方法清除与交互帧相邻的非条件内存，以避免给模型提供关于对象的旧信息和新信息。

        参数：
            frame_idx (int): 当前用户交互发生的帧的索引。
        """
        r = self.model.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.model.num_maskmem
        frame_idx_end = frame_idx + r * self.model.num_maskmem
        for t in range(frame_idx_begin, frame_idx_end + 1):
            self.inference_state["output_dict"]["non_cond_frame_outputs"].pop(t, None)
            for obj_output_dict in self.inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)
