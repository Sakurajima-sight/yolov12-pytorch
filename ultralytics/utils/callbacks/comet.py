# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, RANK, SETTINGS, TESTS_RUNNING, ops
from ultralytics.utils.metrics import ClassifyMetrics, DetMetrics, OBBMetrics, PoseMetrics, SegmentMetrics

try:
    assert not TESTS_RUNNING  # 不记录pytest日志
    assert SETTINGS["comet"] is True  # 验证Comet集成是否启用
    import comet_ml

    assert hasattr(comet_ml, "__version__")  # 验证comet_ml包不是目录

    import os
    from pathlib import Path

    # 确保某些日志记录函数只在支持的任务中运行
    COMET_SUPPORTED_TASKS = ["detect"]

    # Ultralytics创建的图表名称，这些图表会被记录到Comet
    CONFUSION_MATRIX_PLOT_NAMES = "confusion_matrix", "confusion_matrix_normalized"
    EVALUATION_PLOT_NAMES = "F1_curve", "P_curve", "R_curve", "PR_curve"
    LABEL_PLOT_NAMES = "labels", "labels_correlogram"
    SEGMENT_METRICS_PLOT_PREFIX = "Box", "Mask"
    POSE_METRICS_PLOT_PREFIX = "Box", "Pose"

    _comet_image_prediction_count = 0

except (ImportError, AssertionError):
    comet_ml = None


def _get_comet_mode():
    """返回环境变量中设置的Comet模式，如果未设置，则默认返回'在线'模式."""
    return os.getenv("COMET_MODE", "online")


def _get_comet_model_name():
    """返回Comet的模型名称，可以从环境变量COMET_MODEL_NAME获取，默认为'Ultralytics'."""
    return os.getenv("COMET_MODEL_NAME", "Ultralytics")


def _get_eval_batch_logging_interval():
    """从环境变量中获取评估批次日志记录间隔，默认为1."""
    return int(os.getenv("COMET_EVAL_BATCH_LOGGING_INTERVAL", 1))


def _get_max_image_predictions_to_log():
    """从环境变量中获取最大图像预测日志记录数量."""
    return int(os.getenv("COMET_MAX_IMAGE_PREDICTIONS", 100))


def _scale_confidence_score(score):
    """根据环境变量中指定的因子来缩放给定的置信度分数."""
    scale = float(os.getenv("COMET_MAX_CONFIDENCE_SCORE", 100.0))
    return score * scale


def _should_log_confusion_matrix():
    """根据环境变量设置，判断是否记录混淆矩阵."""
    return os.getenv("COMET_EVAL_LOG_CONFUSION_MATRIX", "false").lower() == "true"


def _should_log_image_predictions():
    """根据指定的环境变量，判断是否记录图像预测."""
    return os.getenv("COMET_EVAL_LOG_IMAGE_PREDICTIONS", "true").lower() == "true"


def _get_experiment_type(mode, project_name):
    """根据模式和项目名称返回一个实验实例."""
    if mode == "offline":
        return comet_ml.OfflineExperiment(project_name=project_name)

    return comet_ml.Experiment(project_name=project_name)


def _create_experiment(args):
    """确保实验对象只在分布式训练中的一个进程中创建."""
    if RANK not in {-1, 0}:
        return
    try:
        comet_mode = _get_comet_mode()
        _project_name = os.getenv("COMET_PROJECT_NAME", args.project)
        experiment = _get_experiment_type(comet_mode, _project_name)
        experiment.log_parameters(vars(args))
        experiment.log_others(
            {
                "eval_batch_logging_interval": _get_eval_batch_logging_interval(),
                "log_confusion_matrix_on_eval": _should_log_confusion_matrix(),
                "log_image_predictions": _should_log_image_predictions(),
                "max_image_predictions": _get_max_image_predictions_to_log(),
            }
        )
        experiment.log_other("Created from", "ultralytics")

    except Exception as e:
        LOGGER.warning(f"警告 ⚠️ Comet已安装，但未正确初始化，未记录此运行。{e}")


def _fetch_trainer_metadata(trainer):
    """返回YOLO训练的元数据，包括当前训练轮次和资产保存状态."""
    curr_epoch = trainer.epoch + 1

    train_num_steps_per_epoch = len(trainer.train_loader.dataset) // trainer.batch_size
    curr_step = curr_epoch * train_num_steps_per_epoch
    final_epoch = curr_epoch == trainer.epochs

    save = trainer.args.save
    save_period = trainer.args.save_period
    save_interval = curr_epoch % save_period == 0
    save_assets = save and save_period > 0 and save_interval and not final_epoch

    return dict(curr_epoch=curr_epoch, curr_step=curr_step, save_assets=save_assets, final_epoch=final_epoch)


def _scale_bounding_box_to_original_image_shape(box, resized_image_shape, original_image_shape, ratio_pad):
    """
    YOLO在训练期间会调整图像大小，并且标签值是基于调整后的图像形状进行归一化的。

    该函数将边界框标签重新缩放到原始图像的形状。
    """
    resized_image_height, resized_image_width = resized_image_shape

    # 将归一化的xywh格式预测转换为调整后尺度的xyxy格式
    box = ops.xywhn2xyxy(box, h=resized_image_height, w=resized_image_width)
    # 将边界框预测从调整后的图像尺度缩放回原始图像尺度
    box = ops.scale_boxes(resized_image_shape, box, original_image_shape, ratio_pad)
    # 将边界框格式从xyxy转换为xywh，以便于Comet日志记录
    box = ops.xyxy2xywh(box)
    # 调整xy中心，使其对应左上角
    box[:2] -= box[2:] / 2
    box = box.tolist()

    return box


def _format_ground_truth_annotations_for_detection(img_idx, image_path, batch, class_name_map=None):
    """格式化检测任务的真实标签注释。"""
    indices = batch["batch_idx"] == img_idx
    bboxes = batch["bboxes"][indices]
    if len(bboxes) == 0:
        LOGGER.debug(f"COMET WARNING: 图像: {image_path} 没有边界框标签")
        return None

    cls_labels = batch["cls"][indices].squeeze(1).tolist()
    if class_name_map:
        cls_labels = [str(class_name_map[label]) for label in cls_labels]

    original_image_shape = batch["ori_shape"][img_idx]
    resized_image_shape = batch["resized_shape"][img_idx]
    ratio_pad = batch["ratio_pad"][img_idx]

    data = []
    for box, label in zip(bboxes, cls_labels):
        box = _scale_bounding_box_to_original_image_shape(box, resized_image_shape, original_image_shape, ratio_pad)
        data.append(
            {
                "boxes": [box],
                "label": f"gt_{label}",
                "score": _scale_confidence_score(1.0),
            }
        )

    return {"name": "ground_truth", "data": data}


def _format_prediction_annotations_for_detection(image_path, metadata, class_label_map=None):
    """格式化YOLO模型的检测预测结果，用于可视化。"""
    stem = image_path.stem
    image_id = int(stem) if stem.isnumeric() else stem

    predictions = metadata.get(image_id)
    if not predictions:
        LOGGER.debug(f"COMET WARNING: 图像: {image_path} 没有边界框预测")
        return None

    data = []
    for prediction in predictions:
        boxes = prediction["bbox"]
        score = _scale_confidence_score(prediction["score"])
        cls_label = prediction["category_id"]
        if class_label_map:
            cls_label = str(class_label_map[cls_label])

        data.append({"boxes": [boxes], "label": cls_label, "score": score})

    return {"name": "prediction", "data": data}


def _fetch_annotations(img_idx, image_path, batch, prediction_metadata_map, class_label_map):
    """如果存在，联合真实标签和预测标签。"""
    ground_truth_annotations = _format_ground_truth_annotations_for_detection(
        img_idx, image_path, batch, class_label_map
    )
    prediction_annotations = _format_prediction_annotations_for_detection(
        image_path, prediction_metadata_map, class_label_map
    )

    annotations = [
        annotation for annotation in [ground_truth_annotations, prediction_annotations] if annotation is not None
    ]
    return [annotations] if annotations else None


def _create_prediction_metadata_map(model_predictions):
    """根据图像ID将模型预测结果分组，创建元数据映射。"""
    pred_metadata_map = {}
    for prediction in model_predictions:
        pred_metadata_map.setdefault(prediction["image_id"], [])
        pred_metadata_map[prediction["image_id"]].append(prediction)

    return pred_metadata_map


def _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch):
    """将混淆矩阵记录到Comet实验中。"""
    conf_mat = trainer.validator.confusion_matrix.matrix
    names = list(trainer.data["names"].values()) + ["background"]
    experiment.log_confusion_matrix(
        matrix=conf_mat, labels=names, max_categories=len(names), epoch=curr_epoch, step=curr_step
    )


def _log_images(experiment, image_paths, curr_step, annotations=None):
    """将图像记录到实验中，带有可选的注释。"""
    if annotations:
        for image_path, annotation in zip(image_paths, annotations):
            experiment.log_image(image_path, name=image_path.stem, step=curr_step, annotations=annotation)

    else:
        for image_path in image_paths:
            experiment.log_image(image_path, name=image_path.stem, step=curr_step)


def _log_image_predictions(experiment, validator, curr_step):
    """在训练期间，记录单张图像的预测边界框。"""
    global _comet_image_prediction_count

    task = validator.args.task
    if task not in COMET_SUPPORTED_TASKS:
        return

    jdict = validator.jdict
    if not jdict:
        return

    predictions_metadata_map = _create_prediction_metadata_map(jdict)
    dataloader = validator.dataloader
    class_label_map = validator.names

    batch_logging_interval = _get_eval_batch_logging_interval()
    max_image_predictions = _get_max_image_predictions_to_log()

    for batch_idx, batch in enumerate(dataloader):
        if (batch_idx + 1) % batch_logging_interval != 0:
            continue

        image_paths = batch["im_file"]
        for img_idx, image_path in enumerate(image_paths):
            if _comet_image_prediction_count >= max_image_predictions:
                return

            image_path = Path(image_path)
            annotations = _fetch_annotations(
                img_idx,
                image_path,
                batch,
                predictions_metadata_map,
                class_label_map,
            )
            _log_images(
                experiment,
                [image_path],
                curr_step,
                annotations=annotations,
            )
            _comet_image_prediction_count += 1


def _log_plots(experiment, trainer):
    """记录评估图表和标签图表到实验中。"""
    plot_filenames = None
    if isinstance(trainer.validator.metrics, SegmentMetrics) and trainer.validator.metrics.task == "segment":
        plot_filenames = [
            trainer.save_dir / f"{prefix}{plots}.png"
            for plots in EVALUATION_PLOT_NAMES
            for prefix in SEGMENT_METRICS_PLOT_PREFIX
        ]
    elif isinstance(trainer.validator.metrics, PoseMetrics):
        plot_filenames = [
            trainer.save_dir / f"{prefix}{plots}.png"
            for plots in EVALUATION_PLOT_NAMES
            for prefix in POSE_METRICS_PLOT_PREFIX
        ]
    elif isinstance(trainer.validator.metrics, (DetMetrics, OBBMetrics)):
        plot_filenames = [trainer.save_dir / f"{plots}.png" for plots in EVALUATION_PLOT_NAMES]

    if plot_filenames is not None:
        _log_images(experiment, plot_filenames, None)

    confusion_matrix_filenames = [trainer.save_dir / f"{plots}.png" for plots in CONFUSION_MATRIX_PLOT_NAMES]
    _log_images(experiment, confusion_matrix_filenames, None)

    if not isinstance(trainer.validator.metrics, ClassifyMetrics):
        label_plot_filenames = [trainer.save_dir / f"{labels}.jpg" for labels in LABEL_PLOT_NAMES]
        _log_images(experiment, label_plot_filenames, None)


def _log_model(experiment, trainer):
    """将训练的最佳模型记录到Comet.ml中。"""
    model_name = _get_comet_model_name()
    experiment.log_model(model_name, file_or_folder=str(trainer.best), file_name="best.pt", overwrite=True)


def on_pretrain_routine_start(trainer):
    """在YOLO预训练例程开始时创建或恢复CometML实验。"""
    experiment = comet_ml.get_global_experiment()
    is_alive = getattr(experiment, "alive", False)
    if not experiment or not is_alive:
        _create_experiment(trainer.args)


def on_train_epoch_end(trainer):
    """在训练周期结束时记录指标并保存批次图像。"""
    experiment = comet_ml.get_global_experiment()
    if not experiment:
        return

    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]

    experiment.log_metrics(trainer.label_loss_items(trainer.tloss, prefix="train"), step=curr_step, epoch=curr_epoch)


def on_fit_epoch_end(trainer):
    """在每个周期结束时记录模型资产。"""
    experiment = comet_ml.get_global_experiment()
    if not experiment:
        return

    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]
    save_assets = metadata["save_assets"]

    experiment.log_metrics(trainer.metrics, step=curr_step, epoch=curr_epoch)
    experiment.log_metrics(trainer.lr, step=curr_step, epoch=curr_epoch)
    if curr_epoch == 1:
        from ultralytics.utils.torch_utils import model_info_for_loggers

        experiment.log_metrics(model_info_for_loggers(trainer), step=curr_step, epoch=curr_epoch)

    if not save_assets:
        return

    _log_model(experiment, trainer)
    if _should_log_confusion_matrix():
        _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)
    if _should_log_image_predictions():
        _log_image_predictions(experiment, trainer.validator, curr_step)


def on_train_end(trainer):
    """在训练结束时执行操作。"""
    experiment = comet_ml.get_global_experiment()
    if not experiment:
        return

    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]
    plots = trainer.args.plots

    _log_model(experiment, trainer)
    if plots:
        _log_plots(experiment, trainer)

    _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)
    _log_image_predictions(experiment, trainer.validator, curr_step)
    _log_images(experiment, trainer.save_dir.glob("train_batch*.jpg"), curr_step)
    _log_images(experiment, trainer.save_dir.glob("val_batch*.jpg"), curr_step)
    experiment.end()

    global _comet_image_prediction_count
    _comet_image_prediction_count = 0


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if comet_ml
    else {}
)
