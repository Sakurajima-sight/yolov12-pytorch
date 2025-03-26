# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class QueueManager(BaseSolution):
    """
    基于目标轨迹，在实时视频流中进行排队计数的管理类。

    本类继承自 BaseSolution，用于在视频帧中指定区域内追踪和统计对象数量。

    属性:
        counts (int): 当前排队区域中的对象数量。
        rect_color (Tuple[int, int, int]): 绘制排队区域矩形的 RGB 颜色。
        region_length (int): 定义排队区域的点的数量。
        annotator (Annotator): Annotator 类的实例，用于在图像上绘制可视化内容。
        track_line (List[Tuple[int, int]]): 轨迹线的坐标列表。
        track_history (Dict[int, List[Tuple[int, int]]]): 储存每个对象跟踪历史的字典。

    方法:
        initialize_region: 初始化排队区域。
        process_queue: 处理单帧图像中的排队检测逻辑。
        extract_tracks: 从当前帧图像中提取目标轨迹。
        store_tracking_history: 储存单个对象的跟踪历史。
        display_output: 显示处理后的结果图像。

    示例:
        >>> cap = cv2.VideoCapture("Path/to/video/file.mp4")
        >>> queue_manager = QueueManager(region=[100, 100, 200, 200, 300, 300])
        >>> while cap.isOpened():
        >>>     success, im0 = cap.read()
        >>>     if not success:
        >>>         break
        >>>     out = queue.process_queue(im0)
    """

    def __init__(self, **kwargs):
        """初始化 QueueManager，配置用于视频流中目标跟踪和计数的参数。"""
        super().__init__(**kwargs)
        self.initialize_region()
        self.counts = 0  # 当前帧中的排队人数统计
        self.rect_color = (255, 255, 255)  # 绘制区域的颜色（白色）
        self.region_length = len(self.region)  # 保存区域点数用于后续判断

    def process_queue(self, im0):
        """
        处理单帧视频图像中的排队管理逻辑。

        参数:
            im0 (numpy.ndarray): 输入图像，通常是一帧视频图像。

        返回:
            (numpy.ndarray): 处理后的图像，包含标注、边框以及排队人数信息。

        此方法执行以下步骤：
        1. 重置当前帧的排队计数；
        2. 初始化 Annotator 用于图像绘制；
        3. 从图像中提取目标轨迹；
        4. 在图像上绘制计数区域；
        5. 对于每个检测到的对象：
           - 绘制边界框和标签；
           - 储存跟踪历史；
           - 绘制目标中心点和轨迹；
           - 判断目标是否进入计数区域并更新人数；
        6. 在图像上显示排队人数；
        7. 调用基类方法显示处理结果。

        示例:
            >>> queue_manager = QueueManager()
            >>> frame = cv2.imread("frame.jpg")
            >>> processed_frame = queue_manager.process_queue(frame)
        """
        self.counts = 0  # 每帧开始时重置人数统计
        self.annotator = Annotator(im0, line_width=self.line_width)  # 初始化绘图注释器
        self.extract_tracks(im0)  # 提取当前帧中的轨迹

        self.annotator.draw_region(
            reg_pts=self.region, color=self.rect_color, thickness=self.line_width * 2
        )  # 绘制区域框

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # 绘制边界框和标签
            self.annotator.box_label(box, label=self.names[cls], color=colors(track_id, True))
            self.store_tracking_history(track_id, box)  # 存储跟踪历史

            # 绘制目标的中心点和轨迹线
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width
            )

            # 快速访问对象历史
            track_history = self.track_history.get(track_id, [])

            # 获取目标之前的位置，检查是否进入了计数区域
            prev_position = None
            if len(track_history) > 1:
                prev_position = track_history[-2]
            if self.region_length >= 3 and prev_position and self.r_s.contains(self.Point(self.track_line[-1])):
                self.counts += 1

        # 显示排队人数统计信息
        self.annotator.queue_counts_display(
            f"Queue Counts : {str(self.counts)}",
            points=self.region,
            region_color=self.rect_color,
            txt_color=(104, 31, 17),
        )
        self.display_output(im0)  # 调用基类方法显示输出图像

        return im0  # 返回处理后的图像以供后续使用
