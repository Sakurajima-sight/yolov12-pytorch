# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import defaultdict

import cv2

from ultralytics import YOLO
from ultralytics.utils import ASSETS_URL, DEFAULT_CFG_DICT, DEFAULT_SOL_DICT, LOGGER
from ultralytics.utils.checks import check_imshow, check_requirements


class BaseSolution:
    """
    Ultralytics 解决方案的基础类。

    此类为多个 Ultralytics 解决方案提供核心功能，包括模型加载、对象跟踪和区域初始化等。

    属性:
        LineString (shapely.geometry.LineString): 用于创建线段几何图形的类。
        Polygon (shapely.geometry.Polygon): 用于创建多边形几何图形的类。
        Point (shapely.geometry.Point): 用于创建点几何图形的类。
        CFG (Dict): 从 YAML 配置文件加载并由关键字参数更新的配置字典。
        region (List[Tuple[int, int]]): 定义感兴趣区域的坐标元组列表。
        line_width (int): 可视化中用于绘制线条的宽度。
        model (ultralytics.YOLO): 已加载的 YOLO 模型实例。
        names (Dict[int, str]): 类索引与类名的映射字典。
        env_check (bool): 标志位，用于指示环境是否支持图像显示。
        track_history (collections.defaultdict): 存储每个对象的跟踪历史的字典。

    方法:
        extract_tracks: 执行对象跟踪，并从输入图像中提取跟踪信息。
        store_tracking_history: 存储给定对象的跟踪历史。
        initialize_region: 根据配置初始化计数区域或线段。
        display_output: 显示处理结果，包括显示图像帧或保存结果。

    示例:
        >>> solution = BaseSolution(model="yolov8n.pt", region=[(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> solution.initialize_region()
        >>> image = cv2.imread("image.jpg")
        >>> solution.extract_tracks(image)
        >>> solution.display_output(image)
    """

    def __init__(self, IS_CLI=False, **kwargs):
        """
        初始化 `BaseSolution` 类，加载配置并载入 YOLO 模型，用于 Ultralytics 的各类解决方案。

        IS_CLI（可选）: 若设为 True，则启用命令行模式。
        """
        check_requirements("shapely>=2.0.0")
        from shapely.geometry import LineString, Point, Polygon
        from shapely.prepared import prep

        self.LineString = LineString
        self.Polygon = Polygon
        self.Point = Point
        self.prep = prep
        self.annotator = None  # 初始化绘图工具
        self.tracks = None
        self.track_data = None
        self.boxes = []
        self.clss = []
        self.track_ids = []
        self.track_line = None
        self.r_s = None

        # 加载配置并用传入参数更新
        DEFAULT_SOL_DICT.update(kwargs)
        DEFAULT_CFG_DICT.update(kwargs)
        self.CFG = {**DEFAULT_SOL_DICT, **DEFAULT_CFG_DICT}
        LOGGER.info(f"Ultralytics Solutions: ✅ {DEFAULT_SOL_DICT}")

        self.region = self.CFG["region"]  # 存储区域数据，供其他类使用
        self.line_width = (
            self.CFG["line_width"] if self.CFG["line_width"] is not None else 2
        )  # 存储线宽，供绘制使用

        # 加载模型并获取类别名称
        if self.CFG["model"] is None:
            self.CFG["model"] = "yolo11n.pt"
        self.model = YOLO(self.CFG["model"])
        self.names = self.model.names

        self.track_add_args = {  # 追踪器的附加参数，用于高级配置
            k: self.CFG[k] for k in ["verbose", "iou", "conf", "device", "max_det", "half", "tracker"]
        }

        # 命令行模式下，如果未指定输入源，则使用默认视频
        if IS_CLI and self.CFG["source"] is None:
            d_s = "solutions_ci_demo.mp4" if "-pose" not in self.CFG["model"] else "solution_ci_pose_demo.mp4"
            LOGGER.warning(f"⚠️ WARNING: 未提供输入源，默认使用 {ASSETS_URL}/{d_s}")
            from ultralytics.utils.downloads import safe_download

            safe_download(f"{ASSETS_URL}/{d_s}")  # 从 Ultralytics 下载默认资源
            self.CFG["source"] = d_s  # 设置默认输入源

        # 初始化环境检测与区域设置
        self.env_check = check_imshow(warn=True)
        self.track_history = defaultdict(list)

    def extract_tracks(self, im0):
        """
        应用对象跟踪，从输入图像或帧中提取跟踪信息。

        参数:
            im0 (ndarray): 输入图像或视频帧。

        示例:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.extract_tracks(frame)
        """
        self.tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"], **self.track_add_args)

        # 提取用于 OBB 或常规目标检测的跟踪数据
        self.track_data = self.tracks[0].obb or self.tracks[0].boxes

        if self.track_data and self.track_data.id is not None:
            self.boxes = self.track_data.xyxy.cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()
        else:
            LOGGER.warning("⚠️ 警告：未发现任何跟踪对象！")
            self.boxes, self.clss, self.track_ids = [], [], []

    def store_tracking_history(self, track_id, box):
        """
        存储某个对象的跟踪历史。

        此方法将当前边界框的中心点添加到对应对象的历史轨迹中，
        并限制历史点数量最多为30个。

        参数:
            track_id (int): 跟踪对象的唯一 ID。
            box (List[float]): 对象的边界框坐标，格式为 [x1, y1, x2, y2]。

        示例:
            >>> solution = BaseSolution()
            >>> solution.store_tracking_history(1, [100, 200, 300, 400])
        """
        # 保存跟踪历史
        self.track_line = self.track_history[track_id]
        self.track_line.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))  # 添加中心点
        if len(self.track_line) > 30:
            self.track_line.pop(0)  # 保持最多 30 个历史点

    def initialize_region(self):
        """根据配置初始化计数区域或线段。"""
        if self.region is None:
            self.region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
        self.r_s = (
            self.Polygon(self.region) if len(self.region) >= 3 else self.LineString(self.region)
        )  # 根据点数判断区域为多边形或线段

    def display_output(self, im0):
        """
        显示处理结果，包括目标检测和跟踪后的注释图像帧。

        此方法用于将处理结果可视化。若配置中启用显示且环境支持图像显示，则弹出窗口显示帧图。
        用户可通过按下 'q' 键关闭窗口。

        参数:
            im0 (numpy.ndarray): 已处理并注释的图像或帧。

        示例:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.display_output(frame)

        注意:
            - 仅在配置中开启 'show' 且环境支持显示时，才会显示输出。
            - 按下键盘 'q' 可关闭窗口。
        """
        if self.CFG.get("show") and self.env_check:
            cv2.imshow("Ultralytics Solutions", im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
