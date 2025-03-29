# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""基础回调函数。"""

from collections import defaultdict
from copy import deepcopy

# 训练器回调函数 ----------------------------------------------------------------------------------------------------

def on_pretrain_routine_start(trainer):
    """在预训练例程开始前调用。"""
    pass


def on_pretrain_routine_end(trainer):
    """在预训练例程结束后调用。"""
    pass


def on_train_start(trainer):
    """在训练开始时调用。"""
    pass


def on_train_epoch_start(trainer):
    """在每个训练周期开始时调用。"""
    pass


def on_train_batch_start(trainer):
    """在每个训练批次开始时调用。"""
    pass


def optimizer_step(trainer):
    """在优化器进行一步更新时调用。"""
    pass


def on_before_zero_grad(trainer):
    """在梯度被清零之前调用。"""
    pass


def on_train_batch_end(trainer):
    """在每个训练批次结束时调用。"""
    pass


def on_train_epoch_end(trainer):
    """在每个训练周期结束时调用。"""
    pass


def on_fit_epoch_end(trainer):
    """在每个拟合周期结束时调用（包括训练和验证）。"""
    pass


def on_model_save(trainer):
    """在模型保存时调用。"""
    pass


def on_train_end(trainer):
    """在训练结束时调用。"""
    pass


def on_params_update(trainer):
    """在模型参数更新时调用。"""
    pass


def teardown(trainer):
    """在训练过程的拆解阶段调用。"""
    pass


# 验证器回调函数 --------------------------------------------------------------------------------------------------

def on_val_start(validator):
    """在验证开始时调用。"""
    pass


def on_val_batch_start(validator):
    """在每个验证批次开始时调用。"""
    pass


def on_val_batch_end(validator):
    """在每个验证批次结束时调用。"""
    pass


def on_val_end(validator):
    """在验证结束时调用。"""
    pass


# 预测器回调函数 --------------------------------------------------------------------------------------------------

def on_predict_start(predictor):
    """在预测开始时调用。"""
    pass


def on_predict_batch_start(predictor):
    """在每个预测批次开始时调用。"""
    pass


def on_predict_batch_end(predictor):
    """在每个预测批次结束时调用。"""
    pass


def on_predict_postprocess_end(predictor):
    """在预测后处理结束时调用。"""
    pass


def on_predict_end(predictor):
    """在预测结束时调用。"""
    pass


# 导出器回调函数 ---------------------------------------------------------------------------------------------------

def on_export_start(exporter):
    """在模型导出开始时调用。"""
    pass


def on_export_end(exporter):
    """在模型导出结束时调用。"""
    pass


default_callbacks = {
    # 在训练器中运行
    "on_pretrain_routine_start": [on_pretrain_routine_start],
    "on_pretrain_routine_end": [on_pretrain_routine_end],
    "on_train_start": [on_train_start],
    "on_train_epoch_start": [on_train_epoch_start],
    "on_train_batch_start": [on_train_batch_start],
    "optimizer_step": [optimizer_step],
    "on_before_zero_grad": [on_before_zero_grad],
    "on_train_batch_end": [on_train_batch_end],
    "on_train_epoch_end": [on_train_epoch_end],
    "on_fit_epoch_end": [on_fit_epoch_end],  # fit = 训练 + 验证
    "on_model_save": [on_model_save],
    "on_train_end": [on_train_end],
    "on_params_update": [on_params_update],
    "teardown": [teardown],
    # 在验证器中运行
    "on_val_start": [on_val_start],
    "on_val_batch_start": [on_val_batch_start],
    "on_val_batch_end": [on_val_batch_end],
    "on_val_end": [on_val_end],
    # 在预测器中运行
    "on_predict_start": [on_predict_start],
    "on_predict_batch_start": [on_predict_batch_start],
    "on_predict_postprocess_end": [on_predict_postprocess_end],
    "on_predict_batch_end": [on_predict_batch_end],
    "on_predict_end": [on_predict_end],
    # 在导出器中运行
    "on_export_start": [on_export_start],
    "on_export_end": [on_export_end],
}


def get_default_callbacks():
    """
    返回一个带有默认值（空列表）的默认回调字典的副本。

    返回:
        (defaultdict): 一个使用default_callbacks键并以空列表为默认值的defaultdict。
    """
    return defaultdict(list, deepcopy(default_callbacks))


def add_integration_callbacks(instance):
    """
    向实例的回调字典添加来自不同来源的集成回调。

    参数:
        instance (Trainer, Predictor, Validator, Exporter): 一个具有'callbacks'属性的对象，属性是一个回调列表的字典。
    """
    # 加载 HUB 回调
    from .hub import callbacks as hub_cb

    callbacks_list = [hub_cb]

    # 加载训练回调
    if "Trainer" in instance.__class__.__name__:
        from .clearml import callbacks as clear_cb
        from .comet import callbacks as comet_cb
        from .dvc import callbacks as dvc_cb
        from .mlflow import callbacks as mlflow_cb
        from .neptune import callbacks as neptune_cb
        from .raytune import callbacks as tune_cb
        from .tensorboard import callbacks as tb_cb
        from .wb import callbacks as wb_cb

        callbacks_list.extend([clear_cb, comet_cb, dvc_cb, mlflow_cb, neptune_cb, tune_cb, tb_cb, wb_cb])

    # 将回调添加到回调字典
    for callbacks in callbacks_list:
        for k, v in callbacks.items():
            if v not in instance.callbacks[k]:
                instance.callbacks[k].append(v)
