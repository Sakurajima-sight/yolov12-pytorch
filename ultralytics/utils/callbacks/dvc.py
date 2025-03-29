# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, checks

try:
    assert not TESTS_RUNNING  # 不记录pytest日志
    assert SETTINGS["dvc"] is True  # 确认集成已启用
    import dvclive

    assert checks.check_version("dvclive", "2.11.0", verbose=True)

    import os
    import re
    from pathlib import Path

    # DVCLive日志实例
    live = None
    _processed_plots = {}

    # `on_fit_epoch_end`在最终验证时调用（可能需要修复），目前这是我们区分最佳模型的最终评估与最后一个周期验证的方式
    _training_epoch = False

except (ImportError, AssertionError, TypeError):
    dvclive = None


def _log_images(path, prefix=""):
    """使用DVCLive记录指定路径下的图像，并可选地添加前缀。"""
    if live:
        name = path.name

        # 按批次分组图像，以便在UI中启用滑块
        if m := re.search(r"_batch(\d+)", name):
            ni = m[1]
            new_stem = re.sub(r"_batch(\d+)", "_batch", path.stem)
            name = (Path(new_stem) / ni).with_suffix(path.suffix)

        live.log_image(os.path.join(prefix, name), path)


def _log_plots(plots, prefix=""):
    """记录训练过程中的绘图图像（如果它们尚未处理）。"""
    for name, params in plots.items():
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            _log_images(name, prefix)
            _processed_plots[name] = timestamp


def _log_confusion_matrix(validator):
    """使用DVCLive记录给定验证器的混淆矩阵。"""
    targets = []
    preds = []
    matrix = validator.confusion_matrix.matrix
    names = list(validator.names.values())
    if validator.confusion_matrix.task == "detect":
        names += ["background"]

    for ti, pred in enumerate(matrix.T.astype(int)):
        for pi, num in enumerate(pred):
            targets.extend([names[ti]] * num)
            preds.extend([names[pi]] * num)

    live.log_sklearn_plot("confusion_matrix", targets, preds, name="cf.json", normalized=True)


def on_pretrain_routine_start(trainer):
    """在预训练例程开始时初始化DVCLive日志记录器，以记录训练元数据。"""
    try:
        global live
        live = dvclive.Live(save_dvc_exp=True, cache_images=True)
        LOGGER.info("检测到DVCLive并启用了自动日志记录（运行'yolo settings dvc=False'以禁用）。")
    except Exception as e:
        LOGGER.warning(f"警告 ⚠️ DVCLive已安装，但未正确初始化，未记录此次运行。{e}")


def on_pretrain_routine_end(trainer):
    """在预训练例程结束时记录与训练过程相关的图像。"""
    _log_plots(trainer.plots, "train")


def on_train_start(trainer):
    """如果启用了DVCLive日志记录，则记录训练参数。"""
    if live:
        live.log_params(trainer.args)


def on_train_epoch_start(trainer):
    """在每个训练周期开始时，将全局变量_training_epoch的值设置为True。"""
    global _training_epoch
    _training_epoch = True


def on_fit_epoch_end(trainer):
    """在每个拟合周期结束时记录训练指标和模型信息，并推进到下一步。"""
    global _training_epoch
    if live and _training_epoch:
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics, **trainer.lr}
        for metric, value in all_metrics.items():
            live.log_metric(metric, value)

        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            for metric, value in model_info_for_loggers(trainer).items():
                live.log_metric(metric, value, plot=False)

        _log_plots(trainer.plots, "train")
        _log_plots(trainer.validator.plots, "val")

        live.next_step()
        _training_epoch = False


def on_train_end(trainer):
    """在训练结束时，如果DVCLive激活，则记录最佳指标、图像和混淆矩阵。"""
    if live:
        # 在结束时记录最佳指标。它会在内部运行验证器以验证最佳模型。
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics, **trainer.lr}
        for metric, value in all_metrics.items():
            live.log_metric(metric, value, plot=False)

        _log_plots(trainer.plots, "val")
        _log_plots(trainer.validator.plots, "val")
        _log_confusion_matrix(trainer.validator)

        if trainer.best.exists():
            live.log_artifact(trainer.best, copy=True, type="model")

        live.end()


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_start": on_train_start,
        "on_train_epoch_start": on_train_epoch_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if dvclive
    else {}
)
