# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""该模块定义了 YOLO 中目标跟踪的基础类和结构。"""

from collections import OrderedDict
import numpy as np


class TrackState:
    """
    表示被跟踪目标可能状态的枚举类。

    属性:
        New (int): 新检测到的目标状态。
        Tracked (int): 成功在后续帧中被跟踪的目标状态。
        Lost (int): 无法继续跟踪目标的状态。
        Removed (int): 被移除出跟踪列表的目标状态。

    示例:
        >>> state = TrackState.New
        >>> if state == TrackState.New:
        >>>     print("目标是新检测到的。")
    """

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """
    目标跟踪的基类，提供基本的属性和方法。

    属性:
        _count (int): 用于生成唯一 track ID 的类级别计数器。
        track_id (int): 跟踪目标的唯一标识符。
        is_activated (bool): 当前跟踪是否处于激活状态。
        state (TrackState): 跟踪目标的当前状态。
        history (OrderedDict): 跟踪状态的有序历史记录。
        features (List): 从目标中提取的特征列表，用于辅助跟踪。
        curr_feature (Any): 当前帧中目标的特征。
        score (float): 跟踪置信度评分。
        start_frame (int): 跟踪开始的帧编号。
        frame_id (int): 当前最新的一帧帧编号。
        time_since_update (int): 自上次更新以来经过的帧数。
        location (tuple): 在多摄像头场景下，目标的位置。

    方法:
        end_frame: 返回目标最后一次被跟踪的帧编号。
        next_id: 增加并返回下一个全局唯一的 track ID。
        activate: 抽象方法，激活该 track。
        predict: 抽象方法，预测下一帧的状态。
        update: 抽象方法，用新数据更新当前 track。
        mark_lost: 将该 track 标记为丢失状态。
        mark_removed: 将该 track 标记为已移除状态。
        reset_id: 重置全局 track ID 计数器。

    示例:
        初始化一个新的 track 并标记为丢失：
        >>> track = BaseTrack()
        >>> track.mark_lost()
        >>> print(track.state)  # 输出: 2 (TrackState.Lost)
    """

    _count = 0

    def __init__(self):
        """
        初始化一个新的 track，分配唯一 ID，并初始化基础属性。

        示例:
            初始化一个新的 track
            >>> track = BaseTrack()
            >>> print(track.track_id)
            0
        """
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)

    @property
    def end_frame(self):
        """返回目标最后一次被成功跟踪的帧编号。"""
        return self.frame_id

    @staticmethod
    def next_id():
        """递增并返回下一个全局唯一的 track ID。"""
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        """激活该 track，并根据传入参数初始化必要的属性。"""
        raise NotImplementedError

    def predict(self):
        """根据当前状态和模型预测下一帧的状态。"""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """使用新观测数据更新该 track，并相应更新其状态和属性。"""
        raise NotImplementedError

    def mark_lost(self):
        """将该 track 标记为丢失状态（TrackState.Lost）。"""
        self.state = TrackState.Lost

    def mark_removed(self):
        """将该 track 标记为已移除状态（TrackState.Removed）。"""
        self.state = TrackState.Removed

    @staticmethod
    def reset_id():
        """重置全局 track ID 计数器为初始值。"""
        BaseTrack._count = 0
