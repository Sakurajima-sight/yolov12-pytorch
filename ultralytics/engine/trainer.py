# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
在数据集上训练模型。

用法：
    $ yolo mode=train model=yolov8n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
)


class BaseTrainer:
    """
    创建训练器的基类。

    属性:
        args (SimpleNamespace): 训练器的配置。
        validator (BaseValidator): 验证器实例。
        model (nn.Module): 模型实例。
        callbacks (defaultdict): 回调函数的字典。
        save_dir (Path): 保存结果的目录。
        wdir (Path): 保存权重的目录。
        last (Path): 最后检查点的路径。
        best (Path): 最佳检查点的路径。
        save_period (int): 每x个epoch保存一次检查点（如果小于1则禁用）。
        batch_size (int): 训练时的批量大小。
        epochs (int): 训练的总轮数。
        start_epoch (int): 训练的起始轮数。
        device (torch.device): 用于训练的设备。
        amp (bool): 启用自动混合精度（AMP）的标志。
        scaler (amp.GradScaler): AMP的梯度缩放器。
        data (str): 数据路径。
        trainset (torch.utils.data.Dataset): 训练数据集。
        testset (torch.utils.data.Dataset): 测试数据集。
        ema (nn.Module): 模型的EMA（指数移动平均）。
        resume (bool): 是否从检查点恢复训练。
        lf (nn.Module): 损失函数。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。
        best_fitness (float): 达到的最佳适应度值。
        fitness (float): 当前的适应度值。
        loss (float): 当前的损失值。
        tloss (float): 总损失值。
        loss_names (list): 损失名称的列表。
        csv (Path): 结果CSV文件的路径。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        初始化 BaseTrainer 类。

        参数:
            cfg (str, 可选): 配置文件的路径。默认为 DEFAULT_CFG。
            overrides (dict, 可选): 配置的覆盖。默认为 None。
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # 目录
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # 更新日志的名称
        self.wdir = self.save_dir / "weights"  # 权重目录
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # 创建目录
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # 保存运行参数
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # 检查点路径
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # 如果用户意外传递 epochs=None 且有定时训练，则默认为 100
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # 设备
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # 更快的 CPU 训练，因为时间主要被推理占据，而不是数据加载

        # 模型和数据集
        self.model = check_model_file_from_stem(self.args.model)  # 添加后缀，例如 yolov8n -> yolov8n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # 避免数据集自动下载多次
            self.trainset, self.testset = self.get_dataset()
        self.ema = None

        # 优化工具初始化
        self.lf = None
        self.scheduler = None

        # 每个 epoch 的指标
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # HUB
        self.hub_session = None

        # 回调函数
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """添加给定的回调函数。"""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """用给定的回调函数替换现有的回调函数。"""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """运行与特定事件关联的所有回调函数。"""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """在多 GPU 系统中允许 device='' 或 device=None 默认使用 device=0。"""
        if isinstance(self.args.device, str) and len(self.args.device):  # 即 device='0' 或 device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # 即 device=[0, 1, 2, 3]（从命令行传入的多 GPU 列表）
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # 即 device='cpu' 或 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # 即 device=None 或 device='' 或 device=number
            world_size = 1  # 默认使用设备 0
        else:  # 即 device=None 或 device=''
            world_size = 0

        # 如果是 DDP 训练则运行子进程，否则正常训练
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # 参数检查
            if self.args.rect:
                LOGGER.warning("警告 ⚠️ 'rect=True' 与多 GPU 训练不兼容，设置 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning(
                    "警告 ⚠️ 'batch<1' 的 AutoBatch 与多 GPU 训练不兼容，设置默认值 'batch=16'"
                )
                self.args.batch = 16

            # 命令
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} 调试命令 {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train(world_size)

    def _setup_scheduler(self):
        """初始化训练学习率调度器。"""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # 余弦学习率调度 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # 线性学习率调度
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """初始化并设置用于训练的 DistributedDataParallel 参数。"""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # LOGGER.info(f'DDP 信息: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # 设置为强制超时
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 小时
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """在正确的进程中构建数据加载器和优化器。"""
        # 模型
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # 冻结层
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # 始终冻结这些层
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN 变为 0（已注释掉以避免训练结果不稳定）
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"冻结层 '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # 只有浮点型张量才能要求计算梯度
                LOGGER.info(
                    f"警告 ⚠️ 设置 'requires_grad=True' 为被冻结的层 '{k}'。"
                    "查看 ultralytics.engine.trainer 以自定义冻结层。"
                )
                v.requires_grad = True

        # 检查 AMP（自动混合精度）
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True 或 False
        if self.amp and RANK in {-1, 0}:  # 单 GPU 和 DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # 备份回调函数，因为 check_amp() 会重置它们
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # 恢复回调函数
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # 从 rank 0 广播张量到所有其他 ranks（返回 None）
        self.amp = bool(self.amp)  # 转换为布尔值
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)
            self.set_model_attributes()  # 在 DDP 封装后再次设置模型属性

        # 检查图片尺寸
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # 网格大小（最大步幅）
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # 用于多尺度训练

        # 批量大小
        if self.batch_size < 1 and RANK == -1:  # 仅单 GPU，估算最佳批量大小
            self.args.batch = self.batch_size = self.auto_batch()

        # 数据加载器
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if RANK in {-1, 0}:
            # 注意：当训练 DOTA 数据集时，双倍批量大小可能会导致超过 2000 个物体的图像内存溢出（OOM）。
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # 优化器
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # 在优化前累积损失
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # 缩放权重衰减
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # 调度器
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # 不移动
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """训练完成后，如果指定了参数则进行评估和绘图。"""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # 批次数量
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # 预热迭代次数
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"图像尺寸 {self.args.imgsz} 训练, {self.args.imgsz} 验证\n"
            f"使用 {self.train_loader.num_workers * (world_size or 1)} 个数据加载器工作线程\n"
            f"日志结果保存至 {colorstr('bold', self.save_dir)}\n"
            f"开始训练 {f'{self.args.time} 小时...' if self.args.time else f'{self.epochs} 轮...'}"
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # 清除任何恢复的梯度，确保训练开始时的稳定性
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 抑制 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # 更新数据加载器属性（可选）
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # 预热
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x 插值
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # 偏置学习率从 0.1 下降到 lr0，其他所有学习率从 0.0 上升到 lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # 向前传播
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                # 反向传播
                self.scaler.scale(self.loss).backward()

                # 优化 - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # 定时停止
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # 如果是 DDP 训练
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # 广播 'stop' 给所有 ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # 如果训练时间超过
                            break

                # 日志
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU 内存利用率
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # 损失
                            batch["cls"].shape[0],  # 批量大小，例如 8
                            batch["img"].shape[-1],  # 图像尺寸，例如 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # 供日志使用
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # 验证
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # 保存模型
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # 调度器
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # 不移动
                self.stop |= epoch >= self.epochs  # 如果超过 epochs，则停止
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory()

            # 提前停止
            if RANK != -1:  # 如果是 DDP 训练
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # 广播 'stop' 给所有 ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # 必须终止所有 DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # 使用 best.pt 做最终验证
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} 轮训练完成，共耗时 {seconds / 3600:.3f} 小时。")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")

    def auto_batch(self, max_num_obj=0):
        """通过计算模型的内存占用来获取批量大小。"""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # 返回批量大小

    def _get_memory(self):
        """获取加速器的内存使用情况（单位：GB）。"""
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
        elif self.device.type == "cpu":
            memory = 0
        else:
            memory = torch.cuda.memory_reserved()
        return memory / 1e9

    def _clear_memory(self):
        """在不同平台上清理加速器的内存。"""
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        """读取 results.csv 文件并转换为字典格式，使用 pandas。"""
        import pandas as pd  # 作用域内导入以提高 'import ultralytics' 的速度

        return pd.read_csv(self.csv).to_dict(orient="list")

    def save_model(self):
        """保存模型训练的检查点以及附加的元数据。"""
        import io

        # 将检查点序列化为字节缓冲区（比重复调用 torch.save() 更快）
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # 恢复和最终检查点从 EMA 中派生
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # 以字典形式保存训练参数
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # 获取序列化后的内容以保存

        # 保存检查点
        self.last.write_bytes(serialized_ckpt)  # 保存 last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # 保存 best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # 保存 epoch，例如 'epoch3.pt'
        # 如果需要关闭马赛克并且当前为训练倒数第 n 个 epoch（注释掉的部分）
        # if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
        #    (self.wdir / "last_mosaic.pt").write_bytes(serialized_ckpt)  # 保存马赛克检查点

    def get_dataset(self):
        """
        如果数据字典存在，从中获取训练和验证路径。

        如果数据格式无法识别，则返回 None。
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # 用于验证 'yolo train data=url.zip' 的用法
        except Exception as e:
            raise RuntimeError(emojis(f"数据集 '{clean_url(self.args.data)}' 错误 ❌ {e}")) from e
        self.data = data
        return data["train"], data.get("val") or data.get("test")

    def setup_model(self):
        """加载/创建/下载适用于任何任务的模型。"""
        if isinstance(self.model, torch.nn.Module):  # 如果模型已经加载，无需再次设置
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # 调用 Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """执行训练优化器的单步操作，包括梯度裁剪和 EMA 更新。"""
        self.scaler.unscale_(self.optimizer)  # 反缩放梯度
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # 梯度裁剪
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """根据任务类型允许自定义预处理模型输入和真实标签。"""
        return batch

    def validate(self):
        """
        使用 self.validator 在测试集上进行验证。

        返回的字典预计包含 "fitness" 键。
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # 如果未找到 fitness，则使用损失作为 fitness 测量
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """获取模型，并在加载 cfg 文件时引发 NotImplementedError。"""
        raise NotImplementedError("此任务的训练器不支持加载 cfg 文件")

    def get_validator(self):
        """当调用 get_validator 函数时，返回 NotImplementedError。"""
        raise NotImplementedError("训练器中未实现 get_validator 函数")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """返回由torch.data.Dataloader派生的数据加载器。"""
        raise NotImplementedError("get_dataloader函数在训练器中尚未实现")

    def build_dataset(self, img_path, mode="train", batch=None):
        """构建数据集。"""
        raise NotImplementedError("build_dataset函数在训练器中尚未实现")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        返回一个带有标记的训练损失项张量的损失字典。

        注意：
            对于分类任务不需要此功能，但对于分割和检测任务是必要的。
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """在训练前设置或更新模型参数。"""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """为训练 YOLO 模型构建目标张量。"""
        pass

    def progress_string(self):
        """返回描述训练进度的字符串。"""
        return ""

    # TODO: 可能需要将以下函数放入回调中
    def plot_training_samples(self, batch, ni):
        """在 YOLO 训练过程中绘制训练样本。"""
        pass

    def plot_training_labels(self):
        """为 YOLO 模型绘制训练标签。"""
        pass

    def save_metrics(self, metrics):
        """将训练指标保存到 CSV 文件中。"""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # 列的数量
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")  # 表头
        t = time.time() - self.train_time_start
        with open(self.csv, "a") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """绘制并可视化显示指标。"""
        pass

    def on_plot(self, name, data=None):
        """注册图表（例如，以便在回调中使用）。"""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """执行最终评估和验证，适用于目标检测 YOLO 模型。"""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"  # 从 last.pt 更新 best.pt 的 train_metrics
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    LOGGER.info(f"\n正在验证 {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """检查是否存在恢复的检查点，并根据需要更新参数。"""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # 检查恢复数据的 YAML 文件是否存在，否则强制重新下载数据集
                ckpt_args = attempt_load_weights(last).args
                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # 恢复模型
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                ):  # 允许参数更新以减少内存或在恢复时更新设备
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "未找到恢复检查点。请传递一个有效的检查点以恢复训练，"
                    "例如 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        """从给定的 epoch 和最佳 fitness 恢复 YOLO 训练。"""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # 恢复优化器
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # 恢复 EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} 已训练至 {self.epochs} 轮，无法恢复训练。\n"
            f"请开始新的训练而不是恢复，示例：'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"从第 {start_epoch + 1} 轮恢复训练 {self.args.model}，直到总轮次 {self.epochs}")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} 已训练 {ckpt['epoch']} 轮。继续微调 {self.epochs} 轮。"
            )
            self.epochs += ckpt["epoch"]  # 微调额外的轮次
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """更新数据加载器以停止使用 Mosaic 数据增强。"""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("关闭数据加载器的 Mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        为给定的模型构建一个优化器，基于指定的优化器名称、学习率、动量、权重衰减和迭代次数。

        参数:
            model (torch.nn.Module): 要为其构建优化器的模型。
            name (str, 可选): 要使用的优化器名称。如果是 'auto'，则根据迭代次数自动选择优化器。默认值：'auto'。
            lr (float, 可选): 优化器的学习率。默认值：0.001。
            momentum (float, 可选): 优化器的动量因子。默认值：0.9。
            decay (float, 可选): 优化器的权重衰减。默认值：1e-5。
            iterations (float, 可选): 迭代次数，决定如果优化器名称为 'auto' 时选择哪种优化器。默认值：1e5。

        返回:
            (torch.optim.Optimizer): 构建的优化器。
        """
        g = [], [], []  # 优化器参数组
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # 归一化层，例如 BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' 被检测到，"
                f"忽略 'lr0={self.args.lr0}' 和 'momentum={self.args.momentum}'，"
                f"自动选择最佳的 'optimizer'、'lr0' 和 'momentum'... "
            )
            nc = getattr(model, "nc", 10)  # 类别数
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 适配公式，保留 6 位小数
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # 对于 Adam，bias 学习率不超过 0.01

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # 偏置（不衰减）
                    g[2].append(param)
                elif isinstance(module, bn):  # 权重（不衰减）
                    g[1].append(param)
                else:  # 权重（衰减）
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"优化器 '{name}' 未在可用优化器列表 {optimizers} 中找到。"
                "如需支持更多优化器，请访问 https://github.com/ultralytics/ultralytics 提出请求。"
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # 添加 g0 并使用权重衰减
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # 添加 g1（BatchNorm2d 权重）
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) 具有参数组 "
            f"{len(g[1])} 权重(衰减=0.0)，{len(g[0])} 权重(衰减={decay})，{len(g[2])} 偏置(衰减=0.0)"
        )
        return optimizer
