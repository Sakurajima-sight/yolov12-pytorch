# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class TrackZone(BaseSolution):
    """
    用于在视频流中管理基于区域的目标跟踪的类。

    本类继承自 BaseSolution，提供在特定区域（由多边形定义）内进行目标跟踪的功能。
    区域之外的目标将被忽略。支持区域的动态初始化，可使用默认区域，也可使用用户自定义的多边形区域。

    属性：
        region (ndarray): 用于跟踪的多边形区域，以凸包形式表示。

    方法：
        trackzone: 处理视频的每一帧，执行基于区域的目标跟踪。

    示例：
        >>> tracker = TrackZone()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = tracker.trackzone(frame)
        >>> cv2.imshow("Tracked Frame", processed_frame)
    """

    def __init__(self, **kwargs):
        """初始化 TrackZone 类，用于在定义区域内进行视频目标跟踪。"""
        super().__init__(**kwargs)
        default_region = [(150, 150), (1130, 150), (1130, 570), (150, 570)]
        self.region = cv2.convexHull(np.array(self.region or default_region, dtype=np.int32))

    def trackzone(self, im0):
        """
        处理输入帧，在定义区域内进行目标跟踪。

        本方法初始化绘图工具，为指定区域创建掩膜，只提取区域内的目标轨迹，并更新跟踪信息。
        区域外的目标将被忽略。

        参数：
            im0 (numpy.ndarray): 要处理的输入图像或帧。

        返回：
            (numpy.ndarray): 处理后的图像，带有跟踪 ID 和边界框注释。

        示例：
            >>> tracker = TrackZone()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> tracker.trackzone(frame)
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # 初始化绘图工具
        # 为区域创建掩膜，仅从掩膜区域中提取目标
        masked_frame = cv2.bitwise_and(im0, im0, mask=cv2.fillPoly(np.zeros_like(im0[:, :, 0]), [self.region], 255))
        self.extract_tracks(masked_frame)

        # 绘制区域边界
        cv2.polylines(im0, [self.region], isClosed=True, color=(255, 255, 255), thickness=self.line_width * 2)

        # 遍历所有边界框、跟踪 ID 和类别索引，绘制目标框和标签
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.annotator.box_label(box, label=f"{self.names[cls]}:{track_id}", color=colors(track_id, True))

        self.display_output(im0)  # 使用基类方法显示处理结果

        return im0  # 返回处理后的图像，便于后续使用
