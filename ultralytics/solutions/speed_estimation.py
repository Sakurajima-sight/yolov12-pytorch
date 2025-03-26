# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from time import time

import numpy as np

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class SpeedEstimator(BaseSolution):
    """
    一个用于根据目标轨迹估算实时视频流中物体速度的类。

    该类继承自 BaseSolution，提供了使用视频流中的跟踪数据估算目标速度的功能。

    属性：
        spd (Dict[int, float])：存储已跟踪目标的速度数据的字典。
        trkd_ids (List[int])：已完成速度估计的目标 ID 列表。
        trk_pt (Dict[int, float])：存储跟踪目标上一次时间戳的字典。
        trk_pp (Dict[int, Tuple[float, float]])：存储跟踪目标上一次位置的字典。
        annotator (Annotator)：用于图像绘图的注释器对象。
        region (List[Tuple[int, int]])：定义速度估计区域的点列表。
        track_line (List[Tuple[float, float]])：表示目标轨迹的点列表。
        r_s (LineString)：表示速度估计区域的线对象。

    方法：
        initialize_region：初始化速度估计区域。
        estimate_speed：根据跟踪数据估算目标速度。
        store_tracking_history：存储目标的跟踪历史。
        extract_tracks：从当前帧中提取目标轨迹。
        display_output：显示带注释的输出图像。

    示例：
        >>> estimator = SpeedEstimator()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = estimator.estimate_speed(frame)
        >>> cv2.imshow("Speed Estimation", processed_frame)
    """

    def __init__(self, **kwargs):
        """初始化 SpeedEstimator 对象，设置速度估计相关参数和数据结构。"""
        super().__init__(**kwargs)

        self.initialize_region()  # 初始化速度估计区域

        self.spd = {}  # 用于存储速度数据的字典
        self.trkd_ids = []  # 存储已进行过速度估计的跟踪 ID 列表
        self.trk_pt = {}  # 存储目标上一次时间戳的字典
        self.trk_pp = {}  # 存储目标上一次坐标点的字典

    def estimate_speed(self, im0):
        """
        根据跟踪数据估算目标速度。

        参数：
            im0 (np.ndarray)：输入图像，通常形状为 (H, W, C)。

        返回：
            (np.ndarray)：包含速度估计和注释的处理后图像。

        示例：
            >>> estimator = SpeedEstimator()
            >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> processed_image = estimator.estimate_speed(image)
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # 初始化注释器
        self.extract_tracks(im0)  # 提取跟踪数据

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )  # 绘制速度估计区域

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)  # 存储轨迹历史

            # 如果 track_id 不在 trk_pt 或 trk_pp 中，则进行初始化
            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = 0
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]

            speed_label = f"{int(self.spd[track_id])} km/h" if track_id in self.spd else self.names[int(cls)]
            self.annotator.box_label(box, label=speed_label, color=colors(track_id, True))  # 绘制边框和标签

            # 绘制目标的轨迹线
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width
            )

            # 判断轨迹是否穿过指定区域，用于确定移动方向
            if self.LineString([self.trk_pp[track_id], self.track_line[-1]]).intersects(self.r_s):
                direction = "known"
            else:
                direction = "unknown"

            # 如果方向可判定且该目标未估算过速度，则执行速度估计
            if direction == "known" and track_id not in self.trkd_ids:
                self.trkd_ids.append(track_id)
                time_difference = time() - self.trk_pt[track_id]
                if time_difference > 0:
                    self.spd[track_id] = np.abs(self.track_line[-1][1] - self.trk_pp[track_id][1]) / time_difference

            # 更新该目标的时间和位置
            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = self.track_line[-1]

        self.display_output(im0)  # 调用基类方法显示带注释的输出图像

        return im0  # 返回输出图像用于进一步使用
