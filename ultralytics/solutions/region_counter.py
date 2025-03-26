# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors


class RegionCounter(BaseSolution):
    """
    一个用于在视频流中对用户自定义区域内对象进行实时计数的类。

    本类继承自 `BaseSolution`，提供了在视频帧中定义多边形区域、跟踪对象、
    并统计经过每个定义区域的对象数量的功能。适用于需要区域统计的场景，
    如监控分区、区域分析等。

    属性:
        region_template (dict): 用于创建新的计数区域的模板，包含默认属性，如区域名称、
                                多边形坐标和显示颜色。
        counting_regions (list): 存储所有已定义区域的列表，每个区域基于 `region_template`，
                                 并包含具体的设置（名称、坐标、颜色等）。

    方法:
        add_region: 添加一个新的计数区域，可指定区域名称、多边形点、区域颜色和文字颜色。
        count: 处理视频帧，检测并计数每个区域中的对象，同时绘制区域并显示计数信息。
    """

    def __init__(self, **kwargs):
        """初始化 RegionCounter 类，用于视频流中不同区域的实时计数。"""
        super().__init__(**kwargs)
        self.region_template = {
            "name": "Default Region",        # 区域名称
            "polygon": None,                 # 区域多边形
            "counts": 0,                     # 当前帧内该区域的对象计数
            "dragging": False,               # 拖动状态（用于UI交互）
            "region_color": (255, 255, 255), # 区域颜色（BGR）
            "text_color": (0, 0, 0),         # 文字颜色（BGR）
        }
        self.counting_regions = []  # 计数区域列表

    def add_region(self, name, polygon_points, region_color, text_color):
        """
        基于模板添加一个新的计数区域，包含指定的属性。

        参数:
            name (str): 区域的名称。
            polygon_points (list[tuple]): 定义区域的多边形点坐标列表（x, y）。
            region_color (tuple): 区域绘制时使用的 BGR 颜色。
            text_color (tuple): 区域内部显示文字的 BGR 颜色。
        """
        region = self.region_template.copy()
        region.update(
            {
                "name": name,
                "polygon": self.Polygon(polygon_points),
                "region_color": region_color,
                "text_color": text_color,
            }
        )
        self.counting_regions.append(region)

    def count(self, im0):
        """
        处理输入图像帧，检测并统计每个已定义区域内的对象数量。

        参数:
            im0 (numpy.ndarray): 输入图像帧，用于对象检测、区域绘制和显示计数。

        返回:
            im0 (numpy.ndarray): 添加了计数信息和注释的处理后图像帧。
        """
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)  # 提取对象跟踪结果

        # 区域初始化与格式转换
        if self.region is None:
            self.initialize_region()
            regions = {"Region#01": self.region}
        else:
            regions = self.region if isinstance(self.region, dict) else {"Region#01": self.region}

        # 绘制每个区域，并将其添加到计数区域列表中
        for idx, (region_name, reg_pts) in enumerate(regions.items(), start=1):
            if not isinstance(reg_pts, list) or not all(isinstance(pt, tuple) for pt in reg_pts):
                LOGGER.warning(f"无效的区域点：{region_name}: {reg_pts}")
                continue  # 跳过无效区域
            color = colors(idx, True)
            self.annotator.draw_region(reg_pts=reg_pts, color=color, thickness=self.line_width * 2)
            self.add_region(region_name, reg_pts, color, self.annotator.get_txt_color())

        # 预处理区域以便后续包含检测（多边形包围中心点）
        for region in self.counting_regions:
            region["prepared_polygon"] = self.prep(region["polygon"])

        # 遍历所有检测框，对每个区域内的对象进行计数
        for box, cls in zip(self.boxes, self.clss):
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
            bbox_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)  # 计算框中心点

            for region in self.counting_regions:
                if region["prepared_polygon"].contains(self.Point(bbox_center)):
                    region["counts"] += 1  # 若中心点在区域内，增加计数

        # 显示每个区域的计数信息，并在每帧后重置计数
        for region in self.counting_regions:
            self.annotator.text_label(
                region["polygon"].bounds,      # 文字位置
                label=str(region["counts"]),  # 显示的计数字符串
                color=region["region_color"], # 区域颜色
                txt_color=region["text_color"]# 文字颜色
            )
            region["counts"] = 0  # 每帧后重置区域计数

        self.display_output(im0)  # 显示处理后图像（若环境支持）
        return im0
