# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class ObjectCounter(BaseSolution):
    """
    一个用于在实时视频流中基于对象轨迹进行计数的类。

    该类继承自 BaseSolution，提供了在视频流中统计对象进出指定区域数量的功能。
    支持使用多边形区域或线段区域进行计数。

    属性:
        in_count (int): 进入区域的对象计数器。
        out_count (int): 离开区域的对象计数器。
        counted_ids (List[int]): 已经被计数的对象 ID 列表。
        classwise_counts (Dict[str, Dict[str, int]]): 按对象类别分类的计数字典。
        region_initialized (bool): 表示计数区域是否已初始化的标志。
        show_in (bool): 是否显示进入计数的标志。
        show_out (bool): 是否显示离开计数的标志。

    方法:
        count_objects: 在多边形或线段区域中对对象进行计数。
        store_classwise_counts: 如果尚未存在，初始化按类别的计数结构。
        display_counts: 在图像帧上显示计数信息。
        count: 处理输入帧或对象轨迹并更新计数。

    示例:
        >>> counter = ObjectCounter()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = counter.count(frame)
        >>> print(f"Inward count: {counter.in_count}, Outward count: {counter.out_count}")
    """

    def __init__(self, **kwargs):
        """初始化 ObjectCounter 类，用于在视频流中进行实时对象计数。"""
        super().__init__(**kwargs)

        self.in_count = 0  # 进入区域的对象数量
        self.out_count = 0  # 离开区域的对象数量
        self.counted_ids = []  # 已计数对象的 ID 列表
        self.classwise_counts = {}  # 按类别分类的对象计数字典
        self.region_initialized = False  # 区域是否初始化的布尔值标志

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]

    def count_objects(self, current_centroid, track_id, prev_position, cls):
        """
        基于对象轨迹，在多边形或线段区域内对对象进行计数。

        参数:
            current_centroid (Tuple[float, float]): 当前帧中对象的中心点坐标。
            track_id (int): 当前跟踪对象的唯一标识符。
            prev_position (Tuple[float, float]): 上一帧中对象的位置坐标 (x, y)。
            cls (int): 当前对象的类别索引，用于按类别计数。

        示例:
            >>> counter = ObjectCounter()
            >>> track_line = {1: [100, 200], 2: [110, 210], 3: [120, 220]}
            >>> box = [130, 230, 150, 250]
            >>> track_id = 1
            >>> prev_position = (120, 220)
            >>> cls = 0
            >>> counter.count_objects(current_centroid, track_id, prev_position, cls)
        """
        if prev_position is None or track_id in self.counted_ids:
            return

        if len(self.region) == 2:  # 区域为线段（两个点定义）
            line = self.LineString(self.region)  # 检查线段是否与对象轨迹相交
            if line.intersects(self.LineString([prev_position, current_centroid])):
                # 判断区域是垂直还是水平
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    # 垂直区域：比较 x 坐标判断方向
                    if current_centroid[0] > prev_position[0]:  # 向右移动
                        self.in_count += 1
                        self.classwise_counts[self.names[cls]]["IN"] += 1
                    else:  # 向左移动
                        self.out_count += 1
                        self.classwise_counts[self.names[cls]]["OUT"] += 1
                # 水平区域：比较 y 坐标判断方向
                elif current_centroid[1] > prev_position[1]:  # 向下移动
                    self.in_count += 1
                    self.classwise_counts[self.names[cls]]["IN"] += 1
                else:  # 向上移动
                    self.out_count += 1
                    self.classwise_counts[self.names[cls]]["OUT"] += 1
                self.counted_ids.append(track_id)

        elif len(self.region) > 2:  # 区域为多边形
            polygon = self.Polygon(self.region)
            if polygon.contains(self.Point(current_centroid)):
                # 判断多边形是更偏垂直还是水平
                region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
                region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)

                if (
                    region_width < region_height
                    and current_centroid[0] > prev_position[0]
                    or region_width >= region_height
                    and current_centroid[1] > prev_position[1]
                ):  # 向右或向下移动，视区域方向而定
                    self.in_count += 1
                    self.classwise_counts[self.names[cls]]["IN"] += 1
                else:  # 向左或向上移动
                    self.out_count += 1
                    self.classwise_counts[self.names[cls]]["OUT"] += 1
                self.counted_ids.append(track_id)

    def store_classwise_counts(self, cls):
        """
        如果指定类别尚未记录，则初始化该类别的进出计数。

        参数：
            cls (int): 用于更新类别计数的类别索引。

        此方法确保 classwise_counts 字典中包含指定类别的条目，
        如果该类别尚未存在，则将 "IN" 和 "OUT" 计数初始化为 0。

        示例：
            >>> counter = ObjectCounter()
            >>> counter.store_classwise_counts(0)  # 初始化类别索引为 0 的计数
            >>> print(counter.classwise_counts)
            {'person': {'IN': 0, 'OUT': 0}}
        """
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

    def display_counts(self, im0):
        """
        在输入图像或帧上显示对象的进出计数。

        参数：
            im0 (numpy.ndarray): 要显示计数信息的图像或帧。

        示例：
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("image.jpg")
            >>> counter.display_counts(frame)
        """
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
            f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_counts.items()
            if value["IN"] != 0 or value["OUT"] != 0
        }

        if labels_dict:
            self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

    def count(self, im0):
        """
        处理输入数据（图像帧或目标轨迹）并更新对象计数。

        此方法初始化计数区域、提取目标轨迹、绘制边框与区域、
        更新目标计数，并将结果显示在输入图像上。

        参数：
            im0 (numpy.ndarray): 要处理的图像或帧。

        返回：
            (numpy.ndarray): 处理后的图像，带有注释和计数信息。

        示例：
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> processed_frame = counter.count(frame)
        """
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.annotator = Annotator(im0, line_width=self.line_width)  # 初始化绘图工具
        self.extract_tracks(im0)  # 提取目标轨迹

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )  # 绘制计数区域

        # 遍历所有目标框、跟踪ID和类别索引
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # 绘制边框与计数区域
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
            self.store_tracking_history(track_id, box)  # 存储轨迹历史
            self.store_classwise_counts(cls)  # 将类别进出计数存入字典

            # 绘制目标的轨迹
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(cls), True), track_thickness=self.line_width
            )
            current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            # 存储该目标上一次的位置用于计数
            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(current_centroid, track_id, prev_position, cls)  # 执行目标计数

        self.display_counts(im0)  # 在图像上显示计数信息
        self.display_output(im0)  # 调用基类方法显示输出

        return im0  # 返回处理后的图像以供后续使用
