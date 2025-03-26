# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from itertools import cycle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from ultralytics.solutions.solutions import BaseSolution  # 导入父类


class Analytics(BaseSolution):
    """
    一个用于创建和更新各种类型图表的可视化分析类。

    该类继承自 BaseSolution，用于基于目标检测和跟踪数据生成折线图、柱状图、饼图和面积图。

    属性:
        type (str): 要生成的分析图表类型（'line'、'bar'、'pie' 或 'area'）。
        x_label (str): x轴标签。
        y_label (str): y轴标签。
        bg_color (str): 图表背景颜色。
        fg_color (str): 图表前景颜色。
        title (str): 图表窗口标题。
        max_points (int): 图表中最多显示的数据点数量。
        fontsize (int): 文本字体大小。
        color_cycle (cycle): 用于图表的循环颜色迭代器。
        total_counts (int): 检测到的目标总数（用于折线图）。
        clswise_count (Dict[str, int]): 按类别统计的目标数量字典。
        fig (Figure): Matplotlib 图表对象。
        ax (Axes): Matplotlib 坐标轴对象。
        canvas (FigureCanvas): 用于渲染图表的画布。

    方法:
        process_data: 处理图像数据并更新图表。
        update_graph: 使用新数据点更新图表。

    示例:
        >>> analytics = Analytics(analytics_type="line")
        >>> frame = cv2.imread("image.jpg")
        >>> processed_frame = analytics.process_data(frame, frame_number=1)
        >>> cv2.imshow("Analytics", processed_frame)
    """

    def __init__(self, **kwargs):
        """初始化 Analytics 类，支持各种图表类型的可视化数据表示。"""
        super().__init__(**kwargs)

        self.type = self.CFG["analytics_type"]  # 提取分析类型
        self.x_label = "Classes" if self.type in {"bar", "pie"} else "Frame#"  # 设置 x 轴标签
        self.y_label = "Total Counts"  # 设置 y 轴标签

        # 预设配置
        self.bg_color = "#F3F3F3"  # 窗口背景色
        self.fg_color = "#111E68"  # 窗口前景色
        self.title = "Ultralytics Solutions"  # 窗口标题
        self.max_points = 45  # 窗口中显示的最大点数
        self.fontsize = 25  # 文本字体大小
        figsize = (19.2, 10.8)  # 输出图像尺寸为 1920 * 1080
        self.color_cycle = cycle(["#DD00BA", "#042AFF", "#FF4447", "#7D24FF", "#BD00FF"])

        self.total_counts = 0  # 用于折线图的总计数量
        self.clswise_count = {}  # 按类别统计的数量字典

        # 针对折线图和面积图初始化
        if self.type in {"line", "area"}:
            self.lines = {}
            self.fig = Figure(facecolor=self.bg_color, figsize=figsize)
            self.canvas = FigureCanvas(self.fig)  # 设置通用轴属性
            self.ax = self.fig.add_subplot(111, facecolor=self.bg_color)
            if self.type == "line":
                (self.line,) = self.ax.plot([], [], color="cyan", linewidth=self.line_width)
        elif self.type in {"bar", "pie"}:
            # 初始化柱状图或饼图
            self.fig, self.ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
            self.canvas = FigureCanvas(self.fig)  # 设置通用轴属性
            self.ax.set_facecolor(self.bg_color)
            self.color_mapping = {}

            if self.type == "pie":  # 确保饼图为圆形
                self.ax.axis("equal")

    def process_data(self, im0, frame_number):
        """
        处理图像数据并运行目标跟踪以更新分析图表。

        参数:
            im0 (np.ndarray): 输入图像。
            frame_number (int): 当前视频帧编号，用于绘图。

        返回:
            (np.ndarray): 添加了分析图表的图像帧。

        异常:
            ModuleNotFoundError: 如果指定了不支持的图表类型，将抛出异常。

        示例:
            >>> analytics = Analytics(analytics_type="line")
            >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> processed_frame = analytics.process_data(frame, frame_number=1)
        """
        self.extract_tracks(im0)  # 提取目标跟踪信息

        if self.type == "line":
            for _ in self.boxes:
                self.total_counts += 1
            im0 = self.update_graph(frame_number=frame_number)
            self.total_counts = 0
        elif self.type in {"pie", "bar", "area"}:
            self.clswise_count = {}
            for box, cls in zip(self.boxes, self.clss):
                if self.names[int(cls)] in self.clswise_count:
                    self.clswise_count[self.names[int(cls)]] += 1
                else:
                    self.clswise_count[self.names[int(cls)]] = 1
            im0 = self.update_graph(frame_number=frame_number, count_dict=self.clswise_count, plot=self.type)
        else:
            raise ModuleNotFoundError(f"{self.type} chart is not supported ❌")
        return im0

def update_graph(self, frame_number, count_dict=None, plot="line"):
    """
    使用新的数据更新图表，支持单类别或多类别数据。

    参数:
        frame_number (int): 当前帧编号。
        count_dict (Dict[str, int] | None): 类别名称为键、计数为值的字典，用于多类别情况。若为 None，则更新单线图。
        plot (str): 图表类型，可选值为 'line'、'bar'、'pie' 或 'area'。

    返回:
        (np.ndarray): 包含更新后图表的图像。

    示例:
        >>> analytics = Analytics()
        >>> frame_number = 10
        >>> count_dict = {"person": 5, "car": 3}
        >>> updated_image = analytics.update_graph(frame_number, count_dict, plot="bar")
    """
    if count_dict is None:
        # 单线图更新
        x_data = np.append(self.line.get_xdata(), float(frame_number))
        y_data = np.append(self.line.get_ydata(), float(self.total_counts))

        if len(x_data) > self.max_points:
            x_data, y_data = x_data[-self.max_points:], y_data[-self.max_points:]

        self.line.set_data(x_data, y_data)
        self.line.set_label("Counts")
        self.line.set_color("#7b0068")  # 粉色
        self.line.set_marker("*")
        self.line.set_markersize(self.line_width * 5)
    else:
        labels = list(count_dict.keys())
        counts = list(count_dict.values())
        if plot == "area":
            color_cycle = cycle(["#DD00BA", "#042AFF", "#FF4447", "#7D24FF", "#BD00FF"])
            # 多线图或面积图更新
            x_data = self.ax.lines[0].get_xdata() if self.ax.lines else np.array([])
            y_data_dict = {key: np.array([]) for key in count_dict.keys()}
            if self.ax.lines:
                for line, key in zip(self.ax.lines, count_dict.keys()):
                    y_data_dict[key] = line.get_ydata()

            x_data = np.append(x_data, float(frame_number))
            max_length = len(x_data)
            for key in count_dict.keys():
                y_data_dict[key] = np.append(y_data_dict[key], float(count_dict[key]))
                if len(y_data_dict[key]) < max_length:
                    y_data_dict[key] = np.pad(y_data_dict[key], (0, max_length - len(y_data_dict[key])))
            if len(x_data) > self.max_points:
                x_data = x_data[1:]
                for key in count_dict.keys():
                    y_data_dict[key] = y_data_dict[key][1:]

            self.ax.clear()
            for key, y_data in y_data_dict.items():
                color = next(color_cycle)
                self.ax.fill_between(x_data, y_data, color=color, alpha=0.7)
                self.ax.plot(
                    x_data,
                    y_data,
                    color=color,
                    linewidth=self.line_width,
                    marker="o",
                    markersize=self.line_width * 5,
                    label=f"{key} 数据点",
                )
        if plot == "bar":
            self.ax.clear()  # 清除柱状图内容
            for label in labels:  # 将标签映射到颜色
                if label not in self.color_mapping:
                    self.color_mapping[label] = next(self.color_cycle)
            colors = [self.color_mapping[label] for label in labels]
            bars = self.ax.bar(labels, counts, color=colors)
            for bar, count in zip(bars, counts):
                self.ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    str(count),
                    ha="center",
                    va="bottom",
                    color=self.fg_color,
                )
            # 使用柱状图的标签创建图例
            for bar, label in zip(bars, labels):
                bar.set_label(label)  # 给每个柱子分配标签
            self.ax.legend(loc="upper left", fontsize=13, facecolor=self.fg_color, edgecolor=self.fg_color)
        if plot == "pie":
            total = sum(counts)
            percentages = [size / total * 100 for size in counts]
            start_angle = 90
            self.ax.clear()

            # 创建饼图，并生成带有百分比的图例标签
            wedges, autotexts = self.ax.pie(
                counts, labels=labels, startangle=start_angle, textprops={"color": self.fg_color}, autopct=None
            )
            legend_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(labels, percentages)]

            # 使用 wedge 和自定义标签设置图例
            self.ax.legend(wedges, legend_labels, title="类别", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            self.fig.subplots_adjust(left=0.1, right=0.75)  # 调整布局以容纳图例

    # 公共图表设置
    self.ax.set_facecolor("#f0f0f0")  # 设置为浅灰色或其他喜欢的颜色
    self.ax.set_title(self.title, color=self.fg_color, fontsize=self.fontsize)
    self.ax.set_xlabel(self.x_label, color=self.fg_color, fontsize=self.fontsize - 3)
    self.ax.set_ylabel(self.y_label, color=self.fg_color, fontsize=self.fontsize - 3)

    # 添加和格式化图例
    legend = self.ax.legend(loc="upper left", fontsize=13, facecolor=self.bg_color, edgecolor=self.bg_color)
    for text in legend.get_texts():
        text.set_color(self.fg_color)

    # 重新计算图表范围、更新视图、渲染并显示图像
    self.ax.relim()
    self.ax.autoscale_view()
    self.canvas.draw()
    im0 = np.array(self.canvas.renderer.buffer_rgba())
    im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)
    self.display_output(im0)

    return im0  # 返回图像
