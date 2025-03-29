# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
MLflow日志记录用于Ultralytics YOLO。

此模块启用Ultralytics YOLO的MLflow日志记录。它记录指标、参数和模型工件。
要进行设置，应该指定一个跟踪URI。可以使用环境变量自定义日志记录。

命令：
    1. 设置项目名称：
        `export MLFLOW_EXPERIMENT_NAME=<your_experiment_name>` 或使用 project=<project> 参数

    2. 设置运行名称：
        `export MLFLOW_RUN=<your_run_name>` 或使用 name=<name> 参数

    3. 启动本地MLflow服务器：
        mlflow server --backend-store-uri runs/mlflow
       它默认会启动一个本地服务器，地址为 http://127.0.0.1:5000。
       若要指定不同的URI，设置 MLFLOW_TRACKING_URI 环境变量。

    4. 杀死所有运行中的MLflow服务器实例：
        ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
"""

from ultralytics.utils import LOGGER, RUNS_DIR, SETTINGS, TESTS_RUNNING, colorstr

try:
    import os

    assert not TESTS_RUNNING or "test_mlflow" in os.environ.get("PYTEST_CURRENT_TEST", "")  # 不记录pytest日志
    assert SETTINGS["mlflow"] is True  # 验证MLflow集成是否启用
    import mlflow

    assert hasattr(mlflow, "__version__")  # 验证mlflow包是否是正确的包

    from pathlib import Path

    PREFIX = colorstr("MLflow: ")

except (ImportError, AssertionError):
    mlflow = None


def sanitize_dict(x):
    """清理字典的键，去除括号并将值转换为浮动数值。"""
    return {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}


def on_pretrain_routine_end(trainer):
    """
    在预训练例程结束时将训练参数记录到MLflow。

    该函数根据环境变量和训练器参数设置MLflow日志记录。它设置跟踪URI、实验名称和运行名称，
    然后启动MLflow运行（如果尚未启动）。最后，它记录训练器中的参数。

    参数：
        trainer (ultralytics.engine.trainer.BaseTrainer): 训练对象，包含要记录的参数和参数。

    全局变量：
        mlflow: 用于记录的mlflow模块。

    环境变量：
        MLFLOW_TRACKING_URI: 用于MLflow跟踪的URI。如果未设置，默认使用'runs/mlflow'。
        MLFLOW_EXPERIMENT_NAME: MLflow实验名称。如果未设置，默认使用trainer.args.project。
        MLFLOW_RUN: MLflow运行名称。如果未设置，默认使用trainer.args.name。
        MLFLOW_KEEP_RUN_ACTIVE: 一个布尔值，指示训练结束后是否保持MLflow运行活跃。
    """
    global mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI") or str(RUNS_DIR / "mlflow")
    LOGGER.debug(f"{PREFIX} 跟踪URI: {uri}")
    mlflow.set_tracking_uri(uri)

    # 设置实验和运行名称
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME") or trainer.args.project or "/Shared/Ultralytics"
    run_name = os.environ.get("MLFLOW_RUN") or trainer.args.name
    mlflow.set_experiment(experiment_name)

    mlflow.autolog()
    try:
        active_run = mlflow.active_run() or mlflow.start_run(run_name=run_name)
        LOGGER.info(f"{PREFIX} 记录run_id({active_run.info.run_id})到 {uri}")
        if Path(uri).is_dir():
            LOGGER.info(f"{PREFIX} 在 http://127.0.0.1:5000 查看，使用 'mlflow server --backend-store-uri {uri}'")
        LOGGER.info(f"{PREFIX} 使用 'yolo settings mlflow=False' 禁用")
        mlflow.log_params(dict(trainer.args))
    except Exception as e:
        LOGGER.warning(f"{PREFIX} 警告 ⚠️ 初始化失败: {e}\n{PREFIX} 警告 ⚠️ 未记录此运行")


def on_train_epoch_end(trainer):
    """在每个训练轮次结束时，将训练指标记录到MLflow。"""
    if mlflow:
        mlflow.log_metrics(
            metrics={
                **sanitize_dict(trainer.lr),
                **sanitize_dict(trainer.label_loss_items(trainer.tloss, prefix="train")),
            },
            step=trainer.epoch,
        )


def on_fit_epoch_end(trainer):
    """在每个拟合轮次结束时，将训练指标记录到MLflow。"""
    if mlflow:
        mlflow.log_metrics(metrics=sanitize_dict(trainer.metrics), step=trainer.epoch)


def on_train_end(trainer):
    """在训练结束时，记录模型工件到MLflow。"""
    if not mlflow:
        return
    mlflow.log_artifact(str(trainer.best.parent))  # 记录保存目录/权重目录中的best.pt和last.pt
    for f in trainer.save_dir.glob("*"):  # 记录保存目录中的其他文件
        if f.suffix in {".png", ".jpg", ".csv", ".pt", ".yaml"}:
            mlflow.log_artifact(str(f))
    keep_run_active = os.environ.get("MLFLOW_KEEP_RUN_ACTIVE", "False").lower() == "true"
    if keep_run_active:
        LOGGER.info(f"{PREFIX} MLflow运行仍然活跃，记得使用 mlflow.end_run() 来关闭它")
    else:
        mlflow.end_run()
        LOGGER.debug(f"{PREFIX} MLflow运行结束")

    LOGGER.info(
        f"{PREFIX} 结果已记录到 {mlflow.get_tracking_uri()}\n{PREFIX} 使用 'yolo settings mlflow=False' 禁用"
    )


callbacks = (
    {
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if mlflow
    else {}
)
