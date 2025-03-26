# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import cv2
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class DistanceCalculation(BaseSolution):
    """
    一个用于在实时视频流中根据目标轨迹计算两个物体之间距离的类。

    本类继承自 BaseSolution，提供了选择目标并基于 YOLO 检测与跟踪功能计算它们之间距离的能力。

    属性：
        left_mouse_count (int): 鼠标左键点击计数器。
        selected_boxes (Dict[int, List[float]]): 用于存储选中目标的边界框及其 track ID 的字典。
        annotator (Annotator): Annotator 类的实例，用于在图像上绘制。
        boxes (List[List[float]]): 检测到的目标边界框列表。
        track_ids (List[int]): 检测到的目标的跟踪 ID 列表。
        clss (List[int]): 检测到目标的类别索引列表。
        names (List[str]): 模型可检测的类别名称列表。
        centroids (List[List[int]]): 用于存储选中边界框中心点的列表。

    方法：
        mouse_event_for_distance: 处理视频流中的鼠标事件，用于选择目标。
        calculate: 处理视频帧并计算选中目标之间的距离。

    示例：
        >>> distance_calc = DistanceCalculation()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = distance_calc.calculate(frame)
        >>> cv2.imshow("Distance Calculation", processed_frame)
        >>> cv2.waitKey(0)
    """

    def __init__(self, **kwargs):
        """初始化 DistanceCalculation 类，用于测量视频流中目标之间的距离。"""
        super().__init__(**kwargs)

        # 鼠标事件信息
        self.left_mouse_count = 0
        self.selected_boxes = {}

        self.centroids = []  # 初始化空列表，用于存储中心点

    def mouse_event_for_distance(self, event, x, y, flags, param):
        """
        处理鼠标事件，在实时视频流中选择区域用于距离计算。

        参数：
            event (int): 鼠标事件的类型（如 cv2.EVENT_MOUSEMOVE、cv2.EVENT_LBUTTONDOWN）。
            x (int): 鼠标指针的 X 坐标。
            y (int): 鼠标指针的 Y 坐标。
            flags (int): 与事件相关的标志（如 cv2.EVENT_FLAG_CTRLKEY、cv2.EVENT_FLAG_SHIFTKEY）。
            param (Dict): 传递给该函数的附加参数。

        示例：
            >>> # 假设 'dc' 是 DistanceCalculation 的实例
            >>> cv2.setMouseCallback("window_name", dc.mouse_event_for_distance)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_mouse_count += 1
            if self.left_mouse_count <= 2:
                for box, track_id in zip(self.boxes, self.track_ids):
                    if box[0] < x < box[2] and box[1] < y < box[3] and track_id not in self.selected_boxes:
                        self.selected_boxes[track_id] = box

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_boxes = {}
            self.left_mouse_count = 0

    def calculate(self, im0):
        """
        处理视频帧并计算两个选中边界框之间的距离。

        此方法从输入图像中提取跟踪信息，绘制边界框注释，如果已选择两个目标，则计算它们之间的距离。

        参数：
            im0 (numpy.ndarray): 要处理的输入图像帧。

        返回：
            (numpy.ndarray): 带有注释和距离信息的处理后图像帧。

        示例：
            >>> import numpy as np
            >>> from ultralytics.solutions import DistanceCalculation
            >>> dc = DistanceCalculation()
            >>> frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> processed_frame = dc.calculate(frame)
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # 初始化绘图工具
        self.extract_tracks(im0)  # 提取目标跟踪信息

        # 遍历所有边界框、track_id 和类别索引
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.annotator.box_label(box, color=colors(int(cls), True), label=self.names[int(cls)])

            if len(self.selected_boxes) == 2:
                for trk_id in self.selected_boxes.keys():
                    if trk_id == track_id:
                        self.selected_boxes[track_id] = box

        if len(self.selected_boxes) == 2:
            # 存储用户选中目标的中心点
            self.centroids.extend(
                [[int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)] for box in self.selected_boxes.values()]
            )
            # 计算像素距离
            pixels_distance = math.sqrt(
                (self.centroids[0][0] - self.centroids[1][0]) ** 2 + (self.centroids[0][1] - self.centroids[1][1]) ** 2
            )
            self.annotator.plot_distance_and_line(pixels_distance, self.centroids)

        self.centroids = []

        self.display_output(im0)  # 使用基类函数显示输出结果
        cv2.setMouseCallback("Ultralytics Solutions", self.mouse_event_for_distance)

        return im0  # 返回处理后的图像，可用于其他用途
