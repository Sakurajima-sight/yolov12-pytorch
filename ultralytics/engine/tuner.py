# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
该模块提供了超参数调优功能，用于 Ultralytics YOLO 模型，适用于目标检测、实例分割、图像分类、姿态估计和多目标跟踪。

超参数调优是系统地搜索最佳超参数集合的过程，以获得最佳的模型性能。这在深度学习模型（如 YOLO）中尤为重要，因为超参数的细微变化可能导致模型准确性和效率的显著差异。

示例：
    在 COCO8 数据集上为 YOLOv8n 调整超参数，imgsz=640，epochs=30，进行 300 次调优迭代。
    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    model.tune(data="coco8.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
    ```
"""

import random
import shutil
import subprocess
import time

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, LOGGER, callbacks, colorstr, remove_colorstr, yaml_print, yaml_save
from ultralytics.utils.plotting import plot_tune_results


class Tuner:
    """
    负责 YOLO 模型超参数调优的类。

    该类通过在给定的迭代次数中根据搜索空间变异超参数，并重新训练模型来评估其性能，从而进化 YOLO 模型的超参数。

    属性:
        space (dict): 包含变异的边界和缩放因子的超参数搜索空间。
        tune_dir (Path): 用于保存进化日志和结果的目录。
        tune_csv (Path): 保存进化日志的 CSV 文件路径。

    方法:
        _mutate(hyp: dict) -> dict:
            在 `self.space` 中指定的边界内变异给定的超参数。

        __call__():
            执行多次迭代的超参数进化。

    示例:
        在 COCO8 数据集上，使用 imgsz=640 和 epochs=30，进行 300 次调优迭代来调整 YOLOv8n 的超参数。
        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.tune(data="coco8.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
        ```

        使用自定义搜索空间进行调优。
        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.tune(space={key1: val1, key2: val2})  # 自定义搜索空间字典
        ```
    """

    def __init__(self, args=DEFAULT_CFG, _callbacks=None):
        """
        使用配置初始化 Tuner。

        参数:
            args (dict, optional): 用于超参数进化的配置。
        """
        self.space = args.pop("space", None) or {  # key: (min, max, gain(optional))
            # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
            "lr0": (1e-5, 1e-1),  # 初始学习率（例如：SGD=1E-2, Adam=1E-3）
            "lrf": (0.0001, 0.1),  # 最终 OneCycleLR 学习率（lr0 * lrf）
            "momentum": (0.7, 0.98, 0.3),  # SGD 动量/Adam beta1
            "weight_decay": (0.0, 0.001),  # 优化器的权重衰减 5e-4
            "warmup_epochs": (0.0, 5.0),  # 预热轮次（可以是小数）
            "warmup_momentum": (0.0, 0.95),  # 预热初始动量
            "box": (1.0, 20.0),  # 边界框损失增益
            "cls": (0.2, 4.0),  # 类别损失增益（与像素的比例）
            "dfl": (0.4, 6.0),  # DFL 损失增益
            "hsv_h": (0.0, 0.1),  # 图像 HSV-Hue 增强（比例）
            "hsv_s": (0.0, 0.9),  # 图像 HSV-Saturation 增强（比例）
            "hsv_v": (0.0, 0.9),  # 图像 HSV-Value 增强（比例）
            "degrees": (0.0, 45.0),  # 图像旋转（+/- 度）
            "translate": (0.0, 0.9),  # 图像平移（+/- 比例）
            "scale": (0.0, 0.95),  # 图像缩放（+/- 增益）
            "shear": (0.0, 10.0),  # 图像剪切（+/- 度）
            "perspective": (0.0, 0.001),  # 图像透视（+/- 比例），范围 0-0.001
            "flipud": (0.0, 1.0),  # 图像上下翻转（概率）
            "fliplr": (0.0, 1.0),  # 图像左右翻转（概率）
            "bgr": (0.0, 1.0),  # 图像通道 BGR（概率）
            "mosaic": (0.0, 1.0),  # 图像拼接（概率）
            "mixup": (0.0, 1.0),  # 图像混合（概率）
            "copy_paste": (0.0, 1.0),  # 分割复制粘贴（概率）
        }
        self.args = get_cfg(overrides=args)
        self.tune_dir = get_save_dir(self.args, name=self.args.name or "tune")
        self.args.name = None  # 重置以避免影响训练目录
        self.tune_csv = self.tune_dir / "tune_results.csv"
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.prefix = colorstr("Tuner: ")
        callbacks.add_integration_callbacks(self)
        LOGGER.info(
            f"{self.prefix}初始化 Tuner 实例，'tune_dir={self.tune_dir}'\n"
            f"{self.prefix}💡 了解超参数调优，请访问 https://docs.ultralytics.com/guides/hyperparameter-tuning"
        )

    def _mutate(self, parent="single", n=5, mutation=0.8, sigma=0.2):
        """
        根据 `self.space` 中指定的边界和缩放因子变异超参数。

        参数:
            parent (str): 父代选择方法：'single' 或 'weighted'。
            n (int): 考虑的父代数量。
            mutation (float): 在任何给定迭代中参数变异的概率。
            sigma (float): 高斯随机数生成器的标准差。

        返回:
            (dict): 一个包含变异超参数的字典。
        """
        if self.tune_csv.exists():  # 如果 CSV 文件存在：选择最佳超参数并变异
            # 选择父代
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            fitness = x[:, 0]  # 第一列
            n = min(n, len(x))  # 考虑的之前结果数量
            x = x[np.argsort(-fitness)][:n]  # 选择前 n 个变异
            w = x[:, 0] - x[:, 0].min() + 1e-6  # 权重（总和 > 0）
            if parent == "single" or len(x) == 1:
                # x = x[random.randint(0, n - 1)]  # 随机选择
                x = x[random.choices(range(n), weights=w)[0]]  # 加权选择
            elif parent == "weighted":
                x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # 加权组合

            # 变异
            r = np.random  # 方法
            r.seed(int(time.time()))
            g = np.array([v[2] if len(v) == 3 else 1.0 for v in self.space.values()])  # 增益 0-1
            ng = len(self.space)
            v = np.ones(ng)
            while all(v == 1):  # 直到发生变异（防止重复）
                v = (g * (r.random(ng) < mutation) * r.randn(ng) * r.random() * sigma + 1).clip(0.3, 3.0)
            hyp = {k: float(x[i + 1] * v[i]) for i, k in enumerate(self.space.keys())}
        else:
            hyp = {k: getattr(self.args, k) for k in self.space.keys()}

        # 限制在边界内
        for k, v in self.space.items():
            hyp[k] = max(hyp[k], v[0])  # 下限
            hyp[k] = min(hyp[k], v[1])  # 上限
            hyp[k] = round(hyp[k], 5)  # 精确到小数点后五位

        return hyp

    def __call__(self, model=None, iterations=10, cleanup=True):
        """
        当调用 Tuner 实例时，执行超参数进化过程。

        该方法通过指定的迭代次数进行迭代，每次迭代执行以下步骤：
        1. 加载现有超参数或初始化新的超参数。
        2. 使用 `mutate` 方法变异超参数。
        3. 使用变异后的超参数训练 YOLO 模型。
        4. 将 fitness 得分和变异后的超参数记录到 CSV 文件。

        参数:
           model (Model): 一个预初始化的 YOLO 模型，用于训练。
           iterations (int): 进化运行的代数。
           cleanup (bool): 是否删除迭代权重以减少调优过程中使用的存储空间。

        注意:
           该方法利用 `self.tune_csv` Path 对象读取和记录超参数及 fitness 分数。
           请确保在 Tuner 实例中正确设置此路径。
        """
        t0 = time.time()
        best_save_dir, best_metrics = None, None
        (self.tune_dir / "weights").mkdir(parents=True, exist_ok=True)
        for i in range(iterations):
            # 变异超参数
            mutated_hyp = self._mutate()
            LOGGER.info(f"{self.prefix}开始第 {i + 1}/{iterations} 轮，超参数为: {mutated_hyp}")

            metrics = {}
            train_args = {**vars(self.args), **mutated_hyp}
            save_dir = get_save_dir(get_cfg(train_args))
            weights_dir = save_dir / "weights"
            try:
                # 使用变异后的超参数训练 YOLO 模型（在子进程中运行以避免数据加载器挂起）
                cmd = ["yolo", "train", *(f"{k}={v}" for k, v in train_args.items())]
                return_code = subprocess.run(" ".join(cmd), check=True, shell=True).returncode
                ckpt_file = weights_dir / ("best.pt" if (weights_dir / "best.pt").exists() else "last.pt")
                metrics = torch.load(ckpt_file)["train_metrics"]
                assert return_code == 0, "训练失败"

            except Exception as e:
                LOGGER.warning(f"警告 ❌️ 第 {i + 1} 轮超参数调优训练失败\n{e}")

            # 保存结果和变异后的超参数到 CSV
            fitness = metrics.get("fitness", 0.0)
            log_row = [round(fitness, 5)] + [mutated_hyp[k] for k in self.space.keys()]
            headers = "" if self.tune_csv.exists() else (",".join(["fitness"] + list(self.space.keys())) + "\n")
            with open(self.tune_csv, "a") as f:
                f.write(headers + ",".join(map(str, log_row)) + "\n")

            # 获取最佳结果
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            fitness = x[:, 0]  # 第一列
            best_idx = fitness.argmax()
            best_is_current = best_idx == i
            if best_is_current:
                best_save_dir = save_dir
                best_metrics = {k: round(v, 5) for k, v in metrics.items()}
                for ckpt in weights_dir.glob("*.pt"):
                    shutil.copy2(ckpt, self.tune_dir / "weights")
            elif cleanup:
                shutil.rmtree(weights_dir, ignore_errors=True)  # 删除迭代权重/目录以减少存储空间

            # 绘制调优结果
            plot_tune_results(self.tune_csv)

            # 保存并打印调优结果
            header = (
                f"{self.prefix}{i + 1}/{iterations} 轮完成 ✅ ({time.time() - t0:.2f}s)\n"
                f"{self.prefix}结果已保存到 {colorstr('bold', self.tune_dir)}\n"
                f"{self.prefix}最佳 fitness={fitness[best_idx]} 出现在第 {best_idx + 1} 轮\n"
                f"{self.prefix}最佳 fitness 的指标为 {best_metrics}\n"
                f"{self.prefix}最佳 fitness 模型为 {best_save_dir}\n"
                f"{self.prefix}最佳 fitness 的超参数如下所示。\n"
            )
            LOGGER.info("\n" + header)
            data = {k: float(x[best_idx, i + 1]) for i, k in enumerate(self.space.keys())}
            yaml_save(
                self.tune_dir / "best_hyperparameters.yaml",
                data=data,
                header=remove_colorstr(header.replace(self.prefix, "# ")) + "\n",
            )
            yaml_print(self.tune_dir / "best_hyperparameters.yaml")
