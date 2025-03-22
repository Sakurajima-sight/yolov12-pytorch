# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
检查模型在数据集的测试集或验证集上的准确性。

用法：
    $ yolo mode=val model=yolov8n.pt data=coco8.yaml imgsz=640

用法 - 格式：
    $ yolo mode=val model=yolov8n.pt                 # PyTorch
                          yolov8n.torchscript        # TorchScript
                          yolov8n.onnx               # ONNX Runtime 或 OpenCV DNN 使用 dnn=True
                          yolov8n_openvino_model     # OpenVINO
                          yolov8n.engine             # TensorRT
                          yolov8n.mlpackage          # CoreML（仅限macOS）
                          yolov8n_saved_model        # TensorFlow SavedModel
                          yolov8n.pb                 # TensorFlow GraphDef
                          yolov8n.tflite             # TensorFlow Lite
                          yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolov8n_paddle_model       # PaddlePaddle
                          yolov8n.mnn                # MNN
                          yolov8n_ncnn_model         # NCNN
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode


class BaseValidator:
    """
    基础验证器类。

    用于创建验证器的基类。

    属性：
        args (SimpleNamespace): 验证器的配置。
        dataloader (DataLoader): 用于验证的数据加载器。
        pbar (tqdm): 在验证过程中更新的进度条。
        model (nn.Module): 要验证的模型。
        data (dict): 数据字典。
        device (torch.device): 用于验证的设备。
        batch_i (int): 当前批次索引。
        training (bool): 模型是否处于训练模式。
        names (dict): 类别名称。
        seen: 记录验证过程中看到的图像数量。
        stats: 验证过程中的统计信息占位符。
        confusion_matrix: 混淆矩阵占位符。
        nc: 类别数量。
        iouv: (torch.Tensor): IoU阈值，从0.50到0.95，步长为0.05。
        jdict (dict): 用于存储JSON格式验证结果的字典。
        speed (dict): 包含键 'preprocess'、'inference'、'loss'、'postprocess' 及其对应的
                      每批处理时间（单位：毫秒）的字典。
        save_dir (Path): 保存结果的目录。
        plots (dict): 存储可视化图表的字典。
        callbacks (dict): 存储各种回调函数的字典。
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        初始化一个BaseValidator实例。

        参数：
            dataloader (torch.utils.data.DataLoader): 用于验证的数据加载器。
            save_dir (Path, optional): 用于保存结果的目录。
            pbar (tqdm.tqdm): 用于显示进度的进度条。
            args (SimpleNamespace): 验证器的配置。
            _callbacks (dict): 存储各种回调函数的字典。
        """
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001  # 默认置信度为0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """执行验证过程，在数据加载器上运行推理并计算性能指标。"""
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # 在训练期间强制使用 FP16 验证
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("警告 ⚠️ 验证一个未训练的模型 YAML 将导致 0 mAP。")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            # self.model = model
            self.device = model.device  # 更新设备
            self.args.half = model.fp16  # 更新半精度
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)  # export.py 模型默认批大小为 1
                LOGGER.info(f"设置 batch={self.args.batch} 输入形状为 ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"数据集 '{self.args.data}' 对于任务={self.args.task} 未找到 ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # 更快的 CPU 验证，因为时间主要由推理支配，而非数据加载
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # 在 get_dataloader() 中用于填充
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # 预热

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # 每次验证前清空
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # 预处理
            with dt[0]:
                batch = self.preprocess(batch)

            # 推理
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # 损失
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # 后处理
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # 返回保留 5 位小数的结果
        else:
            LOGGER.info(
                "每张图片的速度: {:.1f}ms 预处理, {:.1f}ms 推理, {:.1f}ms 损失计算, {:.1f}ms 后处理".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"保存 {f.name}...")
                    json.dump(self.jdict, f)  # 扁平化并保存
                stats = self.eval_json(stats)  # 更新统计数据
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"结果已保存到 {colorstr('bold', self.save_dir)}")
            return stats

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        使用 IoU 将预测结果与真实目标进行匹配。

        参数:
            pred_classes (torch.Tensor): 预测的类别索引，形状为(N, )。
            true_classes (torch.Tensor): 真实目标的类别索引，形状为(M, )。
            iou (torch.Tensor): 一个 NxM 的张量，包含预测和真实目标之间的逐对 IoU 值。
            use_scipy (bool): 是否使用 scipy 进行匹配（更精确）。

        返回:
            (torch.Tensor): 一个正确的张量，形状为(N, 10)，对应于 10 个 IoU 阈值。
        """
        # Dx10 矩阵，其中 D - 检测结果数量，10 - IoU 阈值数量
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD 矩阵，其中 L - 标签（行），D - 检测结果（列）
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # 将不匹配的类别置零
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # 警告: 在 https://github.com/ultralytics/ultralytics/pull/4708 中已知问题，可能会减少 mAP
                import scipy  # 限制导入范围，避免所有命令都导入

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > 阈值且类别匹配
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def add_callback(self, event: str, callback):
        """添加给定的回调函数。"""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """运行所有与指定事件关联的回调函数。"""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """根据数据集路径和批大小获取数据加载器。"""
        raise NotImplementedError("未实现该验证器的 get_dataloader 函数")

    def build_dataset(self, img_path):
        """构建数据集。"""
        raise NotImplementedError("未在验证器中实现 build_dataset 函数")

    def preprocess(self, batch):
        """预处理输入批次数据。"""
        return batch

    def postprocess(self, preds):
        """后处理预测结果。"""
        return preds

    def init_metrics(self, model):
        """初始化YOLO模型的性能指标。"""
        pass

    def update_metrics(self, preds, batch):
        """根据预测结果和批次更新指标。"""
        pass

    def finalize_metrics(self, *args, **kwargs):
        """最终确定并返回所有指标。"""
        pass

    def get_stats(self):
        """返回模型性能的统计信息。"""
        return {}

    def check_stats(self, stats):
        """检查统计信息。"""
        pass

    def print_results(self):
        """打印模型预测结果。"""
        pass

    def get_desc(self):
        """获取YOLO模型的描述。"""
        pass

    @property
    def metric_keys(self):
        """返回YOLO训练/验证中使用的指标键。"""
        return []

    def on_plot(self, name, data=None):
        """注册图表（例如，在回调中使用）。"""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    # TODO: 可能需要将以下函数放入回调中
    def plot_val_samples(self, batch, ni):
        """在训练过程中绘制验证样本。"""
        pass

    def plot_predictions(self, batch, preds, ni):
        """在批量图像上绘制YOLO模型的预测结果。"""
        pass

    def pred_to_json(self, preds, batch):
        """将预测结果转换为JSON格式。"""
        pass

    def eval_json(self, stats):
        """评估并返回预测统计信息的JSON格式。"""
        pass
