# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class SegmentationValidator(DetectionValidator):
    """
    基于分割模型的验证器类，继承自 DetectionValidator。

    示例：
        ```python
        from ultralytics.models.yolo.segment import SegmentationValidator

        args = dict(model="yolov8n-seg.pt", data="coco8-seg.yaml")
        validator = SegmentationValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """初始化 SegmentationValidator 并将任务类型设置为 'segment'，并将度量标准设置为 SegmentMetrics。"""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "segment"  # 设置任务类型为分割任务
        self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)  # 初始化度量标准

    def preprocess(self, batch):
        """通过将掩码转换为浮点数并发送到设备上来预处理批次数据。"""
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()  # 转换掩码为浮点数并发送到设备
        return batch

    def init_metrics(self, model):
        """初始化度量标准，并根据是否保存JSON来选择掩码处理函数。"""
        super().init_metrics(model)
        self.plot_masks = []
        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")  # 检查pycocotools库版本
        # 根据是否保存JSON或TXT，选择更精确或更快速的掩码处理方式
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask
        self.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        """返回格式化的评估指标描述。"""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        """后处理YOLO预测结果，并返回输出检测结果和原型信息。"""
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,                # 置信度阈值
            self.args.iou,                 # IOU阈值
            labels=self.lb,                # 类别标签
            multi_label=True,              # 是否支持多标签
            agnostic=self.args.single_cls or self.args.agnostic_nms,  # 是否使用类别无关的NMS
            max_det=self.args.max_det,     # 最大检测数量
            nc=self.nc,                    # 类别数量
        )
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # 如果是PyTorch模型，第二个输出长度为3；如果是导出的模型则为1
        return p, proto

    def _prepare_batch(self, si, batch):
        """通过处理图像和目标，准备一个批次的数据进行训练或推理。"""
        prepared_batch = super()._prepare_batch(si, batch)
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        prepared_batch["masks"] = batch["masks"][midx]  # 为当前批次添加掩码
        return prepared_batch

    def _prepare_pred(self, pred, pbatch, proto):
        """为训练或推理准备一个批次的数据，通过处理图像和目标。"""
        predn = super()._prepare_pred(pred, pbatch)
        pred_masks = self.process(proto, pred[:, 6:], pred[:, :4], shape=pbatch["imgsz"])  # 处理掩码
        return predn, pred_masks

    def update_metrics(self, preds, batch):
        """更新度量标准。"""
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),  # 初始化空的统计字典
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),  # 真阳性
                tp_m=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),  # 真阳性掩码
            )
            pbatch = self._prepare_batch(si, batch)  # 准备批次
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:  # 如果没有预测框
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # 处理掩码
            gt_masks = pbatch.pop("masks")
            # 预测处理
            if self.args.single_cls:
                pred[:, 5] = 0  # 仅使用单类
            predn, pred_masks = self._prepare_pred(pred, pbatch, proto)
            stat["conf"] = predn[:, 4]  # 置信度
            stat["pred_cls"] = predn[:, 5]  # 预测类别

            # 评估
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)  # 处理框
                stat["tp_m"] = self._process_batch(
                    predn, bbox, cls, pred_masks, gt_masks, self.args.overlap_mask, masks=True  # 处理掩码
                )
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if self.args.plots and self.batch_i < 3:
                self.plot_masks.append(pred_masks[:15].cpu())  # 仅绘制前15个掩码

            # 保存结果
            if self.args.save_json:
                self.pred_to_json(
                    predn,
                    batch["im_file"][si],
                    ops.scale_image(
                        pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                        pbatch["ori_shape"],
                        ratio_pad=batch["ratio_pad"][si],
                    ),
                )
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    pred_masks,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def finalize_metrics(self, *args, **kwargs):
        """设置评估指标中的速度和混淆矩阵。"""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None, gt_masks=None, overlap=False, masks=False):
        """
        基于边界框和可选的掩码计算批次的正确预测矩阵。

        参数：
            detections (torch.Tensor): 形状为 (N, 6) 的张量，表示检测到的边界框以及相关的置信度得分和类别索引。
                每一行格式为 [x1, y1, x2, y2, conf, class]。
            gt_bboxes (torch.Tensor): 形状为 (M, 4) 的张量，表示真实的边界框坐标。
                每一行格式为 [x1, y1, x2, y2]。
            gt_cls (torch.Tensor): 形状为 (M,) 的张量，表示真实的类别索引。
            pred_masks (torch.Tensor | None): 预测的掩码张量，如果有的话。其形状应与真实掩码相匹配。
            gt_masks (torch.Tensor | None): 形状为 (M, H, W) 的张量，表示真实的掩码，如果有的话。
            overlap (bool): 标志，指示是否应考虑重叠的掩码。
            masks (bool): 标志，指示批次中是否包含掩码数据。

        返回：
            (torch.Tensor): 形状为 (N, 10) 的正确预测矩阵，其中 10 代表不同的 IoU 水平。

        注意：
            - 如果 `masks` 为 True，则该函数计算预测掩码与真实掩码之间的 IoU。
            - 如果 `overlap` 为 True 且 `masks` 为 True，则在计算 IoU 时考虑重叠的掩码。

        示例：
            ```python
            detections = torch.tensor([[25, 30, 200, 300, 0.8, 1], [50, 60, 180, 290, 0.75, 0]])
            gt_bboxes = torch.tensor([[24, 29, 199, 299], [55, 65, 185, 295]])
            gt_cls = torch.tensor([1, 0])
            correct_preds = validator._process_batch(detections, gt_bboxes, gt_cls)
            ```
        """
        if masks:
            if overlap:
                nl = len(gt_cls)
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
                gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        else:  # boxes
            iou = box_iou(gt_bboxes, detections[:, :4])

        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def plot_val_samples(self, batch, ni):
        """绘制验证样本图像，并标注边界框标签。"""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """绘制批次预测结果，包括掩码和边界框。"""
        plot_images(
            batch["img"],
            *output_to_target(preds[0], max_det=15),  # 不设置为 self.args.max_det 是因为绘制速度较慢
            torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred
        self.plot_masks.clear()

    def save_one_txt(self, predn, pred_masks, save_conf, shape, file):
        """将 YOLO 检测结果保存到 txt 文件中，使用归一化坐标和特定格式。"""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
            masks=pred_masks,
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename, pred_masks):
        """
        保存一个 JSON 结果。

        示例：
             >>> result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        """
        from pycocotools.mask import encode  # noqa

        def single_encode(x):
            """将预测的掩码编码为 RLE，并将结果追加到 jdict 中。"""
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # 从 xy 中心转换为左上角
        pred_masks = np.transpose(pred_masks, (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                    "segmentation": rles[i],
                }
            )

    def eval_json(self, stats):
        """返回 COCO 风格的目标检测评估指标。"""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/instances_val2017.json"  # 注释
            pred_json = self.save_dir / "predictions.json"  # 预测结果
            LOGGER.info(f"\n使用 {pred_json} 和 {anno_json} 评估 pycocotools mAP...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} 文件未找到"
                anno = COCO(str(anno_json))  # 初始化注释 API
                pred = anno.loadRes(str(pred_json))  # 初始化预测结果 API（必须传递字符串，而非 Path 对象）
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm")]):
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # 图像评估
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    idx = i * 4 + 2
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[
                        :2
                    ]  # 更新 mAP50-95 和 mAP50
            except Exception as e:
                LOGGER.warning(f"无法运行 pycocotools：{e}")
        return stats
