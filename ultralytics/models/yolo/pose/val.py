# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, box_iou, kpt_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class PoseValidator(DetectionValidator):
    """
    姿态估计模型验证器类，继承自目标检测验证器 DetectionValidator。

    示例：
        ```python
        from ultralytics.models.yolo.pose import PoseValidator

        args = dict(model="yolov8n-pose.pt", data="coco8-pose.yaml")
        validator = PoseValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """使用自定义参数初始化一个 PoseValidator 对象，并设置相关属性。"""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"
        self.metrics = PoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "⚠️ 警告：Apple MPS 存在已知的 Pose 模型 bug。建议使用 'device=cpu'。"
                "详情请见：https://github.com/ultralytics/ultralytics/issues/4031。"
            )

    def preprocess(self, batch):
        """对输入 batch 中的关键点数据进行转换为 float，并转移到计算设备上。"""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].to(self.device).float()
        return batch

    def get_desc(self):
        """返回评估指标的格式化描述字符串。"""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        """应用非极大值抑制（NMS），并返回置信度高的预测结果。"""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=self.nc,
        )

    def init_metrics(self, model):
        """初始化 YOLO 姿态估计模型的评估指标。"""
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt
        self.stats = dict(tp_p=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def _prepare_batch(self, si, batch):
        """准备单张图像及其关键点标签，用于后续处理。"""
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        kpts = ops.scale_coords(pbatch["imgsz"], kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        pbatch["kpts"] = kpts
        return pbatch

    def _prepare_pred(self, pred, pbatch):
        """准备并缩放预测关键点，使其适配原图尺寸。"""
        predn = super()._prepare_pred(pred, pbatch)
        nk = pbatch["kpts"].shape[1]
        pred_kpts = predn[:, 6:].view(len(predn), nk, -1)
        ops.scale_coords(pbatch["imgsz"], pred_kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        return predn, pred_kpts

    def update_metrics(self, preds, batch):
        """更新评估指标。"""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_p=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # 预测结果处理
            if self.args.single_cls:
                pred[:, 5] = 0
            predn, pred_kpts = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # 评估计算
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                stat["tp_p"] = self._process_batch(predn, bbox, cls, pred_kpts, pbatch["kpts"])
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # 保存预测结果
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    pred_kpts,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_kpts=None, gt_kpts=None):
        """
        通过计算预测框与真实框之间的 IoU（交并比）返回正确预测矩阵。

        参数：
            detections (torch.Tensor): 形状为 (N, 6) 的张量，表示预测的边界框和置信度，
                每条预测格式为 (x1, y1, x2, y2, conf, class)。
            gt_bboxes (torch.Tensor): 形状为 (M, 4) 的张量，表示真实边界框，
                每条格式为 (x1, y1, x2, y2)。
            gt_cls (torch.Tensor): 形状为 (M,) 的张量，表示每个真实框对应的类别索引。
            pred_kpts (torch.Tensor | None): 可选参数，形状为 (N, 51) 的张量，表示预测关键点，
                其中 51 对应 17 个关键点，每个关键点有 3 个值（x, y, score）。
            gt_kpts (torch.Tensor | None): 可选参数，形状为 (M, 51) 的张量，表示真实关键点。

        返回：
            torch.Tensor: 形状为 (N, 10) 的布尔张量，表示每个预测在 10 个 IoU 阈值下是否为正确预测。

        示例：
            ```python
            detections = torch.rand(100, 6)  # 100 个预测：(x1, y1, x2, y2, conf, class)
            gt_bboxes = torch.rand(50, 4)  # 50 个真实框：(x1, y1, x2, y2)
            gt_cls = torch.randint(0, 2, (50,))  # 50 个真实类别索引
            pred_kpts = torch.rand(100, 51)  # 100 个预测关键点
            gt_kpts = torch.rand(50, 51)  # 50 个真实关键点
            correct_preds = _process_batch(detections, gt_bboxes, gt_cls, pred_kpts, gt_kpts)
            ```

        注意：
            `0.53` 的缩放因子用于计算关键点区域，其来源于：https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384。
        """
        if pred_kpts is not None and gt_kpts is not None:
            # “0.53” 来源于 COCO 的关键点评估标准
            area = ops.xyxy2xywh(gt_bboxes)[:, 2:].prod(1) * 0.53
            iou = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, area=area)
        else:  # 仅使用边界框计算
            iou = box_iou(gt_bboxes, detections[:, :4])

        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def plot_val_samples(self, batch, ni):
        """绘制并保存验证集中样本图像，包括预测边界框与关键点。"""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            kpts=batch["keypoints"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """绘制 YOLO 模型的预测结果（包括关键点）。"""
        pred_kpts = torch.cat([p[:, 6:].view(-1, *self.kpt_shape) for p in preds], 0)
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            kpts=pred_kpts,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # 绘制预测结果

    def save_one_txt(self, predn, pred_kpts, save_conf, shape, file):
        """以 YOLO 所需的格式保存检测结果到 txt 文件，坐标为归一化形式。"""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
            keypoints=pred_kpts,
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """将 YOLO 的预测结果转换为 COCO JSON 格式。"""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # 转换为 xywh 格式
        box[:, :2] -= box[:, 2:] / 2  # 将中心点坐标转换为左上角
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,  # 图像编号
                    "category_id": self.class_map[int(p[5])],  # 类别编号
                    "bbox": [round(x, 3) for x in b],  # 边界框坐标
                    "keypoints": p[6:],  # 关键点坐标
                    "score": round(p[4], 5),  # 检测得分
                }
            )

    def eval_json(self, stats):
        """使用 COCO JSON 格式评估目标检测模型（支持 bbox 和关键点）。"""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # 真实标注路径
            pred_json = self.save_dir / "predictions.json"  # 预测保存路径
            LOGGER.info(f"\n使用 {pred_json} 和 {anno_json} 评估 pycocotools mAP...")
            try:  # 官方示例见：pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} 文件未找到"
                anno = COCO(str(anno_json))  # 初始化真实标注
                pred = anno.loadRes(str(pred_json))  # 加载预测结果（注意：必须传字符串路径）
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "keypoints")]):
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # 设置评估图像列表
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
