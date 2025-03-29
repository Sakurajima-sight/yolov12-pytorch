# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.cfg import TASK2DATA, TASK2METRIC, get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, NUM_THREADS, checks


def run_ray_tune(
    model,
    space: dict = None,
    grace_period: int = 10,
    gpu_per_trial: int = None,
    max_samples: int = 10,
    **train_args,
):
    """
    使用 Ray Tune 进行超参数调优。

    参数：
        model (YOLO): 要进行调优的模型。
        space (dict, 可选): 超参数搜索空间。默认为 None。
        grace_period (int, 可选): ASHA 调度器的宽限期（以 epochs 为单位）。默认为 10。
        gpu_per_trial (int, 可选): 每个试验分配的 GPU 数量。默认为 None。
        max_samples (int, 可选): 要运行的最大试验次数。默认为 10。
        train_args (dict, 可选): 传递给 `train()` 方法的额外参数。默认为 {}。

    返回：
        (dict): 包含超参数搜索结果的字典。

    示例：
        ```python
        from ultralytics import YOLO

        # 加载 YOLOv8n 模型
        model = YOLO("yolo11n.pt")

        # 开始为 YOLOv8n 模型在 COCO8 数据集上调优超参数
        result_grid = model.tune(data="coco8.yaml", use_ray=True)
        ```
    """
    LOGGER.info("💡 了解 RayTune 详情，请访问 https://docs.ultralytics.com/integrations/ray-tune")
    if train_args is None:
        train_args = {}

    try:
        checks.check_requirements("ray[tune]")

        import ray
        from ray import tune
        from ray.air import RunConfig
        from ray.air.integrations.wandb import WandbLoggerCallback
        from ray.tune.schedulers import ASHAScheduler
    except ImportError:
        raise ModuleNotFoundError('需要 Ray Tune，但未找到。请运行: pip install "ray[tune]"')

    try:
        import wandb

        assert hasattr(wandb, "__version__")
    except (ImportError, AssertionError):
        wandb = False

    checks.check_version(ray.__version__, ">=2.0.0", "ray")
    default_space = {
        # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
        "lr0": tune.uniform(1e-5, 1e-1),
        "lrf": tune.uniform(0.01, 1.0),  # 最终 OneCycleLR 学习率 (lr0 * lrf)
        "momentum": tune.uniform(0.6, 0.98),  # SGD 动量/Adam beta1
        "weight_decay": tune.uniform(0.0, 0.001),  # 优化器权重衰减 5e-4
        "warmup_epochs": tune.uniform(0.0, 5.0),  # 预热 epochs（可以是分数）
        "warmup_momentum": tune.uniform(0.0, 0.95),  # 预热初始动量
        "box": tune.uniform(0.02, 0.2),  # 框损失增益
        "cls": tune.uniform(0.2, 4.0),  # 类别损失增益（按像素比例缩放）
        "hsv_h": tune.uniform(0.0, 0.1),  # 图像 HSV-色调增强（比例）
        "hsv_s": tune.uniform(0.0, 0.9),  # 图像 HSV-饱和度增强（比例）
        "hsv_v": tune.uniform(0.0, 0.9),  # 图像 HSV-亮度增强（比例）
        "degrees": tune.uniform(0.0, 45.0),  # 图像旋转（度数）
        "translate": tune.uniform(0.0, 0.9),  # 图像平移（比例）
        "scale": tune.uniform(0.0, 0.9),  # 图像缩放（比例）
        "shear": tune.uniform(0.0, 10.0),  # 图像剪切（度数）
        "perspective": tune.uniform(0.0, 0.001),  # 图像透视变换（比例），范围 0-0.001
        "flipud": tune.uniform(0.0, 1.0),  # 图像上下翻转（概率）
        "fliplr": tune.uniform(0.0, 1.0),  # 图像左右翻转（概率）
        "bgr": tune.uniform(0.0, 1.0),  # 图像通道 BGR（概率）
        "mosaic": tune.uniform(0.0, 1.0),  # 图像拼接（概率）
        "mixup": tune.uniform(0.0, 1.0),  # 图像混合（概率）
        "copy_paste": tune.uniform(0.0, 1.0),  # 分割拷贝粘贴（概率）
    }

    # 将模型放入 ray 存储
    task = model.task
    model_in_store = ray.put(model)

    def _tune(config):
        """
        使用指定的超参数和额外参数训练 YOLO 模型。

        参数：
            config (dict): 用于训练的超参数字典。

        返回：
            None
        """
        model_to_train = ray.get(model_in_store)  # 从 ray 存储中获取模型进行调优
        model_to_train.reset_callbacks()
        config.update(train_args)
        results = model_to_train.train(**config)
        return results.results_dict

    # 获取搜索空间
    if not space:
        space = default_space
        LOGGER.warning("警告 ⚠️ 未提供搜索空间，使用默认搜索空间。")

    # 获取数据集
    data = train_args.get("data", TASK2DATA[task])
    space["data"] = data
    if "data" not in train_args:
        LOGGER.warning(f'警告 ⚠️ 未提供数据集，使用默认数据集 "data={data}"。')

    # 定义具有分配资源的可训练函数
    trainable_with_resources = tune.with_resources(_tune, {"cpu": NUM_THREADS, "gpu": gpu_per_trial or 0})

    # 定义用于超参数搜索的 ASHA 调度器
    asha_scheduler = ASHAScheduler(
        time_attr="epoch",
        metric=TASK2METRIC[task],
        mode="max",
        max_t=train_args.get("epochs") or DEFAULT_CFG_DICT["epochs"] or 100,
        grace_period=grace_period,
        reduction_factor=3,
    )

    # 定义用于超参数搜索的回调函数
    tuner_callbacks = [WandbLoggerCallback(project="YOLOv8-tune")] if wandb else []

    # 创建 Ray Tune 超参数搜索调优器
    tune_dir = get_save_dir(
        get_cfg(DEFAULT_CFG, train_args), name=train_args.pop("name", "tune")
    ).resolve()  # 必须是绝对路径
    tune_dir.mkdir(parents=True, exist_ok=True)
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=space,
        tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=max_samples),
        run_config=RunConfig(callbacks=tuner_callbacks, storage_path=tune_dir),
    )

    # 运行超参数搜索
    tuner.fit()

    # 获取超参数搜索的结果
    results = tuner.get_results()

    # 关闭 Ray 清理工作
    ray.shutdown()

    return results
