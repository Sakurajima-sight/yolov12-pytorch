# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING

try:
    assert not TESTS_RUNNING  # 不记录 pytest 日志
    assert SETTINGS["neptune"] is True  # 验证是否启用了 Neptune 集成
    import neptune
    from neptune.types import File

    assert hasattr(neptune, "__version__")

    run = None  # NeptuneAI 实验日志记录实例

except (ImportError, AssertionError):
    neptune = None


def _log_scalars(scalars, step=0):
    """将标量日志记录到 NeptuneAI 实验日志中。"""
    if run:
        for k, v in scalars.items():
            run[k].append(value=v, step=step)


def _log_images(imgs_dict, group=""):
    """将图像日志记录到 NeptuneAI 实验日志中。"""
    if run:
        for k, v in imgs_dict.items():
            run[f"{group}/{k}"].upload(File(v))


def _log_plot(title, plot_path):
    """
    将图表日志记录到 NeptuneAI 实验日志中。

    参数:
        title (str): 图表的标题。
        plot_path (PosixPath | str): 保存的图像文件路径。
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    img = mpimg.imread(plot_path)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])  # 无刻度
    ax.imshow(img)
    run[f"Plots/{title}"].upload(fig)


def on_pretrain_routine_start(trainer):
    """在训练例程开始之前调用的回调函数。"""
    try:
        global run
        run = neptune.init_run(
            project=trainer.args.project or "Ultralytics",
            name=trainer.args.name,
            tags=["Ultralytics"],
        )
        run["Configuration/Hyperparameters"] = {k: "" if v is None else v for k, v in vars(trainer.args).items()}
    except Exception as e:
        LOGGER.warning(f"警告 ⚠️ NeptuneAI 已安装但未正确初始化，未记录此运行。{e}")


def on_train_epoch_end(trainer):
    """在每个训练周期结束时调用的回调函数。"""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)
    _log_scalars(trainer.lr, trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_images({f.stem: str(f) for f in trainer.save_dir.glob("train_batch*.jpg")}, "Mosaic")


def on_fit_epoch_end(trainer):
    """在每个训练和验证周期结束时调用的回调函数。"""
    if run and trainer.epoch == 0:
        from ultralytics.utils.torch_utils import model_info_for_loggers

        run["Configuration/Model"] = model_info_for_loggers(trainer)
    _log_scalars(trainer.metrics, trainer.epoch + 1)


def on_val_end(validator):
    """在每次验证结束时调用的回调函数。"""
    if run:
        # 记录 val_labels 和 val_pred
        _log_images({f.stem: str(f) for f in validator.save_dir.glob("val*.jpg")}, "Validation")


def on_train_end(trainer):
    """在训练结束时调用的回调函数。"""
    if run:
        # 记录最终结果，混淆矩阵 + 精度/召回曲线
        files = [
            "results.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R")),
        ]
        files = [(trainer.save_dir / f) for f in files if (trainer.save_dir / f).exists()]  # 过滤文件
        for f in files:
            _log_plot(title=f.stem, plot_path=f)
        # 记录最终模型
        run[f"weights/{trainer.args.name or trainer.args.task}/{trainer.best.name}"].upload(File(str(trainer.best)))


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_val_end": on_val_end,
        "on_train_end": on_train_end,
    }
    if neptune
    else {}
)
