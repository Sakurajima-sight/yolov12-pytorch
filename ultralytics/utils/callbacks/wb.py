# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import SETTINGS, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers

try:
    assert not TESTS_RUNNING  # 不记录pytest日志
    assert SETTINGS["wandb"] is True  # 验证wandb集成是否启用
    import wandb as wb

    assert hasattr(wb, "__version__")  # 验证包是否是目录
    _processed_plots = {}

except (ImportError, AssertionError):
    wb = None


def _custom_table(x, y, classes, title="Precision Recall Curve", x_title="Recall", y_title="Precision"):
    """
    创建并记录自定义的指标可视化到wandb.plot.pr_curve。

    该函数创建一个自定义的指标可视化图，模仿默认的wandb精确度-召回率曲线，同时允许更多的自定义。
    这个可视化的指标对于监控模型在不同类别上的性能非常有用。

    参数：
        x (List): x轴的值，长度应为N。
        y (List): 对应的y轴值，长度也应为N。
        classes (List): 用于标识每个点所属类别的标签，长度为N。
        title (str, optional): 图表的标题，默认为“Precision Recall Curve”。
        x_title (str, optional): x轴的标签，默认为“Recall”。
        y_title (str, optional): y轴的标签，默认为“Precision”。

    返回：
        (wandb.Object): 一个适合记录的wandb对象，展示了创建的指标可视化。
    """
    import pandas  # 引入pandas库

    df = pandas.DataFrame({"class": classes, "y": y, "x": x}).round(3)
    fields = {"x": "x", "y": "y", "class": "class"}
    string_fields = {"title": title, "x-axis-title": x_title, "y-axis-title": y_title}
    return wb.plot_table(
        "wandb/area-under-curve/v0", wb.Table(dataframe=df), fields=fields, string_fields=string_fields
    )


def _plot_curve(
    x,
    y,
    names=None,
    id="precision-recall",
    title="Precision Recall Curve",
    x_title="Recall",
    y_title="Precision",
    num_x=100,
    only_mean=False,
):
    """
    记录一个指标曲线可视化。

    该函数基于输入数据生成一个指标曲线，并将可视化结果记录到wandb。
    曲线可以表示聚合数据（均值）或每个类别的数据，取决于'only_mean'标志。

    参数：
        x (np.ndarray): x轴数据点，长度为N。
        y (np.ndarray): 对应的y轴数据点，形状为CxN，其中C是类别数量。
        names (list, optional): 与y轴数据对应的类别名称，长度为C，默认为[]。
        id (str, optional): 在wandb中记录的数据的唯一标识符，默认为“precision-recall”。
        title (str, optional): 可视化图表的标题，默认为“Precision Recall Curve”。
        x_title (str, optional): x轴标签，默认为“Recall”。
        y_title (str, optional): y轴标签，默认为“Precision”。
        num_x (int, optional): 可视化插值的x数据点数量，默认为100。
        only_mean (bool, optional): 是否只绘制均值曲线的标志，默认为True。

    注意：
        该函数使用'_custom_table'函数来生成实际的可视化。
    """
    import numpy as np

    # 创建新的x
    if names is None:
        names = []
    x_new = np.linspace(x[0], x[-1], num_x).round(5)

    # 创建用于记录的数组
    x_log = x_new.tolist()
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()

    if only_mean:
        table = wb.Table(data=list(zip(x_log, y_log)), columns=[x_title, y_title])
        wb.run.log({title: wb.plot.line(table, x_title, y_title, title=title)})
    else:
        classes = ["mean"] * len(x_log)
        for i, yi in enumerate(y):
            x_log.extend(x_new)  # 添加新的x
            y_log.extend(np.interp(x_new, x, yi))  # 将y插值到新的x
            classes.extend([names[i]] * len(x_new))  # 添加类别名称
        wb.log({id: _custom_table(x_log, y_log, classes, title, x_title, y_title)}, commit=False)


def _log_plots(plots, step):
    """如果指定步长下的图表尚未记录，则记录输入字典中的图表。"""
    for name, params in plots.copy().items():  # 使用浅拷贝以防止在遍历过程中修改plots字典
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            wb.run.log({name.stem: wb.Image(str(name))}, step=step)
            _processed_plots[name] = timestamp


def on_pretrain_routine_start(trainer):
    """如果模块存在，则初始化并启动项目。"""
    if not wb.run:
        wb.init(
            project=str(trainer.args.project).replace("/", "-") if trainer.args.project else "Ultralytics",
            name=str(trainer.args.name).replace("/", "-"),
            config=vars(trainer.args),
        )


def on_fit_epoch_end(trainer):
    """在每个epoch结束时记录训练指标和模型信息。"""
    wb.run.log(trainer.metrics, step=trainer.epoch + 1)
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    if trainer.epoch == 0:
        wb.run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)


def on_train_epoch_end(trainer):
    """在每个训练epoch结束时记录指标并保存图像。"""
    wb.run.log(trainer.label_loss_items(trainer.tloss, prefix="train"), step=trainer.epoch + 1)
    wb.run.log(trainer.lr, step=trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_plots(trainer.plots, step=trainer.epoch + 1)


def on_train_end(trainer):
    """在训练结束时将最佳模型作为artifact保存。"""
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    art = wb.Artifact(type="model", name=f"run_{wb.run.id}_model")
    if trainer.best.exists():
        art.add_file(trainer.best)
        wb.run.log_artifact(art, aliases=["best"])
    # 检查是否有实际的图表需要保存
    if trainer.args.plots and hasattr(trainer.validator.metrics, "curves_results"):
        for curve_name, curve_values in zip(trainer.validator.metrics.curves, trainer.validator.metrics.curves_results):
            x, y, x_title, y_title = curve_values
            _plot_curve(
                x,
                y,
                names=list(trainer.validator.metrics.names.values()),
                id=f"curves/{curve_name}",
                title=curve_name,
                x_title=x_title,
                y_title=y_title,
            )
    wb.run.finish()  # 必需，否则run会继续显示在仪表板上


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if wb
    else {}
)
