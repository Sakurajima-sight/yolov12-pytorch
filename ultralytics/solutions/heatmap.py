# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import numpy as np

from ultralytics.solutions.object_counter import ObjectCounter
from ultralytics.utils.plotting import Annotator


class Heatmap(ObjectCounter):
    """
    用于在实时视频流中绘制热力图的类，基于目标轨迹进行热力图生成。

    该类继承自 ObjectCounter 类，用于在视频流中基于目标跟踪位置创建和可视化热力图，
    利用累计的目标位置生成动态热力图效果。

    属性:
        initialized (bool): 标记热力图是否已初始化。
        colormap (int): 用于热力图可视化的 OpenCV 颜色映射类型。
        heatmap (np.ndarray): 储存累计热力图数据的数组。
        annotator (Annotator): 用于在图像上绘制注释的对象。

    方法:
        heatmap_effect: 计算并更新某个边界框区域的热力图效果。
        generate_heatmap: 生成并将热力图效果应用到每一帧图像。

    示例:
        >>> from ultralytics.solutions import Heatmap
        >>> heatmap = Heatmap(model="yolov8n.pt", colormap=cv2.COLORMAP_JET)
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = heatmap.generate_heatmap(frame)
    """

    def __init__(self, **kwargs):
        """初始化 Heatmap 类，用于基于目标轨迹生成实时视频热力图。"""
        super().__init__(**kwargs)

        self.initialized = False  # 标志热力图是否已初始化
        if self.region is not None:  # 如果用户提供了计数区域
            self.initialize_region()

        # 储存颜色映射方式
        self.colormap = cv2.COLORMAP_PARULA if self.CFG["colormap"] is None else self.CFG["colormap"]
        self.heatmap = None

    def heatmap_effect(self, box):
        """
        高效计算热力图区域及作用位置，并应用颜色映射。

        参数:
            box (List[float]): 边界框坐标 [x0, y0, x1, y1]。

        示例:
            >>> heatmap = Heatmap()
            >>> box = [100, 100, 200, 200]
            >>> heatmap.heatmap_effect(box)
        """
        x0, y0, x1, y1 = map(int, box)
        radius_squared = (min(x1 - x0, y1 - y0) // 2) ** 2

        # 使用 meshgrid 在目标区域内创建用于矢量化距离计算的网格
        xv, yv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))

        # 计算每个点到中心的平方距离
        dist_squared = (xv - ((x0 + x1) // 2)) ** 2 + (yv - ((y0 + y1) // 2)) ** 2

        # 创建一个布尔掩码，标记半径范围内的点
        within_radius = dist_squared <= radius_squared

        # 在边界框区域内对满足条件的区域进行累计更新（矢量化操作）
        self.heatmap[y0:y1, x0:x1][within_radius] += 2

    def generate_heatmap(self, im0):
        """
        使用 Ultralytics 提供的功能为每一帧图像生成热力图。

        参数:
            im0 (np.ndarray): 输入图像数组。

        返回:
            (np.ndarray): 添加热力图叠加层以及目标计数信息（若设置区域）的图像。

        示例:
            >>> heatmap = Heatmap()
            >>> im0 = cv2.imread("image.jpg")
            >>> result = heatmap.generate_heatmap(im0)
        """
        if not self.initialized:
            self.heatmap = np.zeros_like(im0, dtype=np.float32) * 0.99  # 初始化热力图
        self.initialized = True  # 确保只初始化一次

        self.annotator = Annotator(im0, line_width=self.line_width)  # 初始化注释器
        self.extract_tracks(im0)  # 提取目标轨迹

        # 遍历所有边界框、跟踪ID 和类别索引
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # 应用热力图效果
            self.heatmap_effect(box)

            if self.region is not None:
                # 绘制计数区域
                self.annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)
                self.store_tracking_history(track_id, box)  # 存储跟踪历史
                self.store_classwise_counts(cls)  # 按类别存储计数信息
                current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                # 记录上一个位置，执行目标计数逻辑
                prev_position = None
                if len(self.track_history[track_id]) > 1:
                    prev_position = self.track_history[track_id][-2]
                self.count_objects(current_centroid, track_id, prev_position, cls)  # 执行计数

        if self.region is not None:
            self.display_counts(im0)  # 在图像上显示计数信息

        # 标准化热力图，应用颜色映射，并与原图融合
        if self.track_data.id is not None:
            im0 = cv2.addWeighted(
                im0,
                0.5,
                cv2.applyColorMap(
                    cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), self.colormap
                ),
                0.5,
                0,
            )

        self.display_output(im0)  # 调用基类方法显示结果图像
        return im0  # 返回处理后图像以供后续使用
