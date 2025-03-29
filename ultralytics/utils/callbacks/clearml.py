# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING

try:
    assert not TESTS_RUNNING  # 不记录pytest日志
    assert SETTINGS["clearml"] is True  # 验证是否启用了集成
    import clearml
    from clearml import Task

    assert hasattr(clearml, "__version__")  # 验证包不是目录

except (ImportError, AssertionError):
    clearml = None


def _log_debug_samples(files, title="Debug Samples") -> None:
    """
    将文件（图像）作为调试样本记录到ClearML任务中。

    参数：
        files (list): 一个包含PosixPath格式文件路径的列表。
        title (str): 一个标题，用来将具有相同值的图像分组。
    """
    import re

    if task := Task.current_task():
        for f in files:
            if f.exists():
                it = re.search(r"_batch(\d+)", f.name)
                iteration = int(it.groups()[0]) if it else 0
                task.get_logger().report_image(
                    title=title, series=f.name.replace(it.group(), ""), local_path=str(f), iteration=iteration
                )


def _log_plot(title, plot_path) -> None:
    """
    将图像作为图表记录到ClearML的图表部分。

    参数：
        title (str): 图表的标题。
        plot_path (str): 保存的图像文件路径。
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    img = mpimg.imread(plot_path)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])  # 不显示坐标轴
    ax.imshow(img)

    Task.current_task().get_logger().report_matplotlib_figure(
        title=title, series="", figure=fig, report_interactive=False
    )


def on_pretrain_routine_start(trainer):
    """在预训练过程开始时运行；初始化并连接/记录任务到ClearML。"""
    try:
        if task := Task.current_task():
            # 警告：确保禁用了自动的pytorch和matplotlib绑定！
            # 我们会在集成中手动记录这些图表和模型文件
            from clearml.binding.frameworks.pytorch_bind import PatchPyTorchModelIO
            from clearml.binding.matplotlib_bind import PatchedMatplotlib

            PatchPyTorchModelIO.update_current_task(None)
            PatchedMatplotlib.update_current_task(None)
        else:
            task = Task.init(
                project_name=trainer.args.project or "Ultralytics",
                task_name=trainer.args.name,
                tags=["Ultralytics"],
                output_uri=True,
                reuse_last_task_id=False,
                auto_connect_frameworks={"pytorch": False, "matplotlib": False},
            )
            LOGGER.warning(
                "ClearML初始化了一个新任务。如果您想远程运行，请在初始化YOLO之前添加clearml-init并连接您的参数。"
            )
        task.connect(vars(trainer.args), name="General")
    except Exception as e:
        LOGGER.warning(f"警告 ⚠️ ClearML已安装，但未正确初始化，未记录此运行。{e}")


def on_train_epoch_end(trainer):
    """在YOLO训练的每个epoch结束时记录调试样本并报告当前训练进度。"""
    if task := Task.current_task():
        # 记录调试样本
        if trainer.epoch == 1:
            _log_debug_samples(sorted(trainer.save_dir.glob("train_batch*.jpg")), "Mosaic")
        # 报告当前训练进度
        for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():
            task.get_logger().report_scalar("train", k, v, iteration=trainer.epoch)
        for k, v in trainer.lr.items():
            task.get_logger().report_scalar("lr", k, v, iteration=trainer.epoch)


def on_fit_epoch_end(trainer):
    """在每个epoch结束时报告模型信息到日志中。"""
    if task := Task.current_task():
        # 你应该可以访问到验证集的bbox数据，存储在jdict中
        task.get_logger().report_scalar(
            title="Epoch Time", series="Epoch Time", value=trainer.epoch_time, iteration=trainer.epoch
        )
        for k, v in trainer.metrics.items():
            task.get_logger().report_scalar("val", k, v, iteration=trainer.epoch)
        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            for k, v in model_info_for_loggers(trainer).items():
                task.get_logger().report_single_value(k, v)


def on_val_end(validator):
    """记录验证结果，包括标签和预测结果。"""
    if Task.current_task():
        # 记录val_labels和val_pred
        _log_debug_samples(sorted(validator.save_dir.glob("val*.jpg")), "Validation")


def on_train_end(trainer):
    """在训练完成时记录最终模型及其名称。"""
    if task := Task.current_task():
        # 记录最终结果，CM矩阵 + PR曲线
        files = [
            "results.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R")),
        ]
        files = [(trainer.save_dir / f) for f in files if (trainer.save_dir / f).exists()]  # 过滤存在的文件
        for f in files:
            _log_plot(title=f.stem, plot_path=f)
        # 报告最终的指标
        for k, v in trainer.validator.metrics.results_dict.items():
            task.get_logger().report_single_value(k, v)
        # 记录最终模型
        task.update_output_model(model_path=str(trainer.best), model_name=trainer.args.name, auto_delete_file=False)


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_val_end": on_val_end,
        "on_train_end": on_train_end,
    }
    if clearml
    else {}
)
