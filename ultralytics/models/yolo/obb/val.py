# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OBBMetrics, batch_probiou
from ultralytics.utils.plotting import output_to_rotated_target, plot_images


class OBBValidator(DetectionValidator):
    """
    一个继承自 DetectionValidator 的验证器类，用于支持旋转边界框（Oriented Bounding Box, OBB）模型的验证。

    示例：
        ```python
        from ultralytics.models.yolo.obb import OBBValidator

        args = dict(model="yolov8n-obb.pt", data="dota8.yaml")
        validator = OBBValidator(args=args)
        validator(model=args["model"])
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """初始化 OBBValidator，并将任务类型设为 'obb'，同时使用 OBBMetrics 作为评估指标类。"""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "obb"
        self.metrics = OBBMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)

    def init_metrics(self, model):
        """初始化 YOLO 的评估指标。"""
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # 验证集路径
        self.is_dota = isinstance(val, str) and "DOTA" in val  # 判断是否为 DOTA 数据集

    def postprocess(self, preds):
        """对预测结果应用非极大值抑制（NMS）。"""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            rotated=True,  # 启用旋转框 NMS
        )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        计算一个批次中的预测与真实旋转框之间的匹配关系，输出预测是否正确的布尔矩阵。

        参数：
            detections (torch.Tensor): 张量形状为 (N, 7)，表示检测到的目标，每行格式为
                (x1, y1, x2, y2, conf, class, angle)。
            gt_bboxes (torch.Tensor): 张量形状为 (M, 5)，表示真实框，格式为 (x1, y1, x2, y2, angle)。
            gt_cls (torch.Tensor): 张量形状为 (M,)，表示每个真实框的类别标签。

        返回：
            (torch.Tensor): 形状为 (N, 10) 的布尔矩阵，表示每个预测在 10 个 IoU 阈值下是否为正确预测。

        示例：
            ```python
            detections = torch.rand(100, 7)  # 100 个预测样本
            gt_bboxes = torch.rand(50, 5)  # 50 个真实框
            gt_cls = torch.randint(0, 5, (50,))  # 50 个真实类别标签
            correct_matrix = OBBValidator._process_batch(detections, gt_bboxes, gt_cls)
            ```

        注意：
            此方法依赖于 `batch_probiou` 来计算旋转框之间的 IoU。
        """
        iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def _prepare_batch(self, si, batch):
        """准备并返回一个用于旋转框验证的 batch 数据。"""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            # 缩放目标框到网络输入尺寸
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])
            # 缩放到原图空间的目标框
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=True)
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """准备并返回缩放和填充后的预测结果，用于旋转框验证。"""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True
        )  # 将预测框缩放回原图尺寸
        return predn

    def plot_predictions(self, batch, preds, ni):
        """在输入图像上绘制预测的旋转框并保存结果图像。"""
        plot_images(
            batch["img"],
            *output_to_rotated_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # 绘制预测框

    def pred_to_json(self, predn, filename):
        """将 YOLO 的预测结果序列化为 COCO 格式的 JSON 文件。"""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        rbox = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)  # 合并位置和角度信息
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)  # 转换为 8 点多边形格式
        for i, (r, b) in enumerate(zip(rbox.tolist(), poly.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,  # 图像 ID
                    "category_id": self.class_map[int(predn[i, 5].item())],  # 类别 ID
                    "score": round(predn[i, 4].item(), 5),  # 置信度
                    "rbox": [round(x, 3) for x in r],  # 旋转框信息
                    "poly": [round(x, 3) for x in b],  # 多边形表示
                }
            )

    def save_one_txt(self, predn, save_conf, shape, file):
        """将YOLO的检测结果以归一化坐标的特定格式保存为txt文件。"""
        import numpy as np
        from ultralytics.engine.results import Results

        # 构造旋转框格式：xywh + 旋转角度 + 置信度 + 类别
        rboxes = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)  # xywh + rotation
        obb = torch.cat([rboxes, predn[:, 4:6]], dim=-1)  # xywh + rotation + conf + cls

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),  # 使用零图像占位
            path=None,
            names=self.names,
            obb=obb,  # 设置为旋转框
        ).save_txt(file, save_conf=save_conf)  # 保存到文件，是否保存置信度由 save_conf 控制

    def eval_json(self, stats):
        """评估YOLO输出的JSON格式结果，并返回性能统计指标。"""
        if self.args.save_json and self.is_dota and len(self.jdict):
            import json
            import re
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"  # 预测结果的JSON路径
            pred_txt = self.save_dir / "predictions_txt"  # 保存分片后的DOTA格式预测结果的路径
            pred_txt.mkdir(parents=True, exist_ok=True)
            data = json.load(open(pred_json))  # 加载JSON预测数据

            # 保存分片的DOTA格式结果
            LOGGER.info(f"正在将预测结果保存为 DOTA 格式至 {pred_txt}...")
            for d in data:
                image_id = d["image_id"]  # 图像ID
                score = d["score"]        # 置信度
                classname = self.names[d["category_id"] - 1].replace(" ", "-")  # 类别名称（空格替换为短横线）
                p = d["poly"]  # 八点多边形坐标

                # 写入到对应类别的txt文件中
                with open(f"{pred_txt / f'Task1_{classname}'}.txt", "a") as f:
                    f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")

            # 保存合并后的预测结果。注意：这种方式可能略低于官方脚本的mAP，
            # 原因是使用了近似的 probiou 计算而非官方合并方式。
            pred_merged_txt = self.save_dir / "predictions_merged_txt"  # 合并后的结果路径
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            merged_results = defaultdict(list)

            LOGGER.info(f"正在将合并的预测结果保存为 DOTA 格式至 {pred_merged_txt}...")
            for d in data:
                image_id = d["image_id"].split("__")[0]  # 提取原始图像名（去除分片信息）
                pattern = re.compile(r"\d+___\d+")  # 提取分片的x、y偏移
                x, y = (int(c) for c in re.findall(pattern, d["image_id"])[0].split("___"))
                bbox, score, cls = d["rbox"], d["score"], d["category_id"] - 1
                bbox[0] += x  # x中心坐标加偏移
                bbox[1] += y  # y中心坐标加偏移
                bbox.extend([score, cls])  # 添加score和类别
                merged_results[image_id].append(bbox)  # 保存到对应图像下的结果

            # 对每张图像执行NMS并写入合并结果
            for image_id, bbox in merged_results.items():
                bbox = torch.tensor(bbox)
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh  # 类别映射到空间偏移
                scores = bbox[:, 5]  # 获取分数
                b = bbox[:, :5].clone()  # 提取旋转框
                b[:, :2] += c  # 加上偏移用于类别区分

                # 应用旋转框NMS（设定阈值0.3，与官方合并结果相近）
                i = ops.nms_rotated(b, scores, 0.3)
                bbox = bbox[i]

                # 将xywhr格式转为8点坐标表示
                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    classname = self.names[int(x[-1])].replace(" ", "-")
                    p = [round(i, 3) for i in x[:-2]]  # 多边形坐标保留三位小数
                    score = round(x[-2], 3)

                    # 写入最终的合并结果到对应类别txt文件
                    with open(f"{pred_merged_txt / f'Task1_{classname}'}.txt", "a") as f:
                        f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")

        return stats
