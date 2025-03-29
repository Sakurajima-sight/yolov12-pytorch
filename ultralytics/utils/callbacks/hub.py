# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
from time import time

from ultralytics.hub import HUB_WEB_ROOT, PREFIX, HUBTrainingSession, events
from ultralytics.utils import LOGGER, RANK, SETTINGS


def on_pretrain_routine_start(trainer):
    """创建一个远程的Ultralytics HUB会话来记录本地模型训练。"""
    if RANK in {-1, 0} and SETTINGS["hub"] is True and SETTINGS["api_key"] and trainer.hub_session is None:
        trainer.hub_session = HUBTrainingSession.create_session(trainer.args.model, trainer.args)


def on_pretrain_routine_end(trainer):
    """在开始上传速率限制定时器之前记录信息。"""
    if session := getattr(trainer, "hub_session", None):
        # 启动上传速率限制定时器
        session.timers = {"metrics": time(), "ckpt": time()}  # 在session.rate_limit上启动定时器


def on_fit_epoch_end(trainer):
    """在每个epoch结束时上传训练进度指标。"""
    if session := getattr(trainer, "hub_session", None):
        # 在验证结束后上传指标
        all_plots = {
            **trainer.label_loss_items(trainer.tloss, prefix="train"),
            **trainer.metrics,
        }
        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            all_plots = {**all_plots, **model_info_for_loggers(trainer)}

        session.metrics_queue[trainer.epoch] = json.dumps(all_plots)

        # 如果有任何指标上传失败，将其添加到队列中，尝试重新上传。
        if session.metrics_upload_failed_queue:
            session.metrics_queue.update(session.metrics_upload_failed_queue)

        if time() - session.timers["metrics"] > session.rate_limits["metrics"]:
            session.upload_metrics()
            session.timers["metrics"] = time()  # 重置定时器
            session.metrics_queue = {}  # 重置队列


def on_model_save(trainer):
    """使用速率限制将检查点保存到Ultralytics HUB。"""
    if session := getattr(trainer, "hub_session", None):
        # 使用速率限制上传检查点
        is_best = trainer.best_fitness == trainer.fitness
        if time() - session.timers["ckpt"] > session.rate_limits["ckpt"]:
            LOGGER.info(f"{PREFIX}正在上传检查点 {HUB_WEB_ROOT}/models/{session.model.id}")
            session.upload_model(trainer.epoch, trainer.last, is_best)
            session.timers["ckpt"] = time()  # 重置定时器


def on_train_end(trainer):
    """在训练结束时将最终模型和指标上传到Ultralytics HUB。"""
    if session := getattr(trainer, "hub_session", None):
        # 使用指数回退上传最终模型和指标
        LOGGER.info(f"{PREFIX}同步最终模型中...")
        session.upload_model(
            trainer.epoch,
            trainer.best,
            map=trainer.metrics.get("metrics/mAP50-95(B)", 0),
            final=True,
        )
        session.alive = False  # 停止心跳
        LOGGER.info(f"{PREFIX}完成 ✅\n{PREFIX}查看模型：{session.model_url} 🚀")


def on_train_start(trainer):
    """在训练开始时运行事件。"""
    events(trainer.args)


def on_val_start(validator):
    """在验证开始时运行事件。"""
    events(validator.args)


def on_predict_start(predictor):
    """在预测开始时运行事件。"""
    events(predictor.args)


def on_export_start(exporter):
    """在导出开始时运行事件。"""
    events(exporter.args)


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_model_save": on_model_save,
        "on_train_end": on_train_end,
        "on_train_start": on_train_start,
        "on_val_start": on_val_start,
        "on_predict_start": on_predict_start,
        "on_export_start": on_export_start,
    }
    if SETTINGS["hub"] is True
    else {}
)  # 验证是否启用
