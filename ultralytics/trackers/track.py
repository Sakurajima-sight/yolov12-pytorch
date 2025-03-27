# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

from functools import partial
from pathlib import Path

import torch

from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

# 跟踪器类型与对应跟踪器类的映射关系
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}


def on_predict_start(predictor: object, persist: bool = False) -> None:
    """
    在预测开始时初始化目标跟踪器。

    参数:
        predictor (object): 要为其初始化跟踪器的预测器对象。
        persist (bool): 如果已有跟踪器，是否保留（不重新初始化）。

    异常:
        AssertionError: 如果 tracker_type 不是 'bytetrack' 或 'botsort'。

    示例:
        为预测器对象初始化跟踪器：
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor, persist=True)
    """
    if hasattr(predictor, "trackers") and persist:
        return

    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))

    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise AssertionError(f"目前仅支持 'bytetrack' 和 'botsort'，但收到的是 '{cfg.tracker_type}'")

    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        if predictor.dataset.mode != "stream":  # 对于非流式模式，只需一个跟踪器
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs  # 用于判断何时在新视频上重置跟踪器


def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """
    后处理检测框，并进行目标跟踪更新。

    参数:
        predictor (object): 包含预测结果的预测器对象。
        persist (bool): 如果已有跟踪器，是否保留（不重新初始化）。

    示例:
        后处理预测结果并进行跟踪更新：
        >>> predictor = YourPredictorClass()
        >>> on_predict_postprocess_end(predictor, persist=True)
    """
    path, im0s = predictor.batch[:2]

    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"
    for i in range(len(im0s)):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(path[i]).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes).cpu().numpy()
        if len(det) == 0:
            continue
        tracks = tracker.update(det, im0s[i])
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]

        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)


def register_tracker(model: object, persist: bool) -> None:
    """
    为模型注册目标跟踪的回调函数，用于预测阶段的跟踪操作。

    参数:
        model (object): 要注册跟踪回调的模型对象。
        persist (bool): 如果已有跟踪器，是否保留（不重新初始化）。

    示例:
        为 YOLO 模型注册跟踪回调：
        >>> model = YOLOModel()
        >>> register_tracker(model, persist=True)
    """
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))
