# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.plotting import Annotator


class ParkingPtsSelection:
    """
    一个用于在图像上选择并管理停车区域点的类，使用基于 Tkinter 的图形界面。

    该类提供上传图像、选择点以定义停车区域、并将所选点保存为 JSON 文件的功能。
    使用 Tkinter 实现图形用户界面。

    属性:
        tk (module): Tkinter 模块，用于 GUI 操作。
        filedialog (module): Tkinter 的文件对话框模块，用于文件选择。
        messagebox (module): Tkinter 的消息框模块，用于显示提示信息。
        master (tk.Tk): 主 Tkinter 窗口。
        canvas (tk.Canvas): 用于显示图像和绘制边界框的画布控件。
        image (PIL.Image.Image): 上传的图像对象。
        canvas_image (ImageTk.PhotoImage): 显示在画布上的图像。
        rg_data (List[List[Tuple[int, int]]]): 所有边界框点的列表，每个框由 4 个点定义。
        current_box (List[Tuple[int, int]]): 当前正在绘制的边界框的临时点集合。
        imgw (int): 上传图像的原始宽度。
        imgh (int): 上传图像的原始高度。
        canvas_max_width (int): 画布允许的最大宽度。
        canvas_max_height (int): 画布允许的最大高度。

    方法:
        initialize_properties: 初始化必要的属性。
        upload_image: 上传图像并缩放以适应画布，然后显示图像。
        on_canvas_click: 处理鼠标点击事件，用于添加边界框点。
        draw_box: 在画布上绘制边界框。
        remove_last_bounding_box: 删除最后一个边界框，并重新绘制画布。
        redraw_canvas: 使用图像和所有边界框重新绘制画布。
        save_to_json: 将所有边界框数据保存为 JSON 文件。

    示例:
        >>> parking_selector = ParkingPtsSelection()
        >>> # 使用图形界面上传图像，选择停车区域并保存数据
    """

    def __init__(self):
        """初始化 ParkingPtsSelection 类，设置界面和停车区域点选取相关属性。"""
        check_requirements("tkinter")
        import tkinter as tk
        from tkinter import filedialog, messagebox

        self.tk, self.filedialog, self.messagebox = tk, filedialog, messagebox
        self.master = self.tk.Tk()  # 主窗口引用
        self.master.title("Ultralytics Parking Zones Points Selector")
        self.master.resizable(False, False)  # 禁用窗口缩放

        self.canvas = self.tk.Canvas(self.master, bg="white")  # 创建用于显示图像的画布控件
        self.canvas.pack(side=self.tk.BOTTOM)

        self.image = None  # 存储加载的图像对象
        self.canvas_image = None  # 画布上显示的图像对象
        self.canvas_max_width = None  # 画布最大宽度限制
        self.canvas_max_height = None  # 画布最大高度限制
        self.rg_data = None  # 存储区域（边界框）数据
        self.current_box = None  # 当前正在绘制的边界框
        self.imgh = None  # 当前图像高度
        self.imgw = None  # 当前图像宽度

        # 创建按钮区域并添加按钮
        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        for text, cmd in [
            ("Upload Image", self.upload_image),
            ("Remove Last BBox", self.remove_last_bounding_box),
            ("Save", self.save_to_json),
        ]:
            self.tk.Button(button_frame, text=text, command=cmd).pack(side=self.tk.LEFT)

        self.initialize_properties()
        self.master.mainloop()

    def initialize_properties(self):
        """初始化图像、画布、边界框和尺寸等相关属性。"""
        self.image = self.canvas_image = None
        self.rg_data, self.current_box = [], []
        self.imgw = self.imgh = 0
        self.canvas_max_width, self.canvas_max_height = 1280, 720

    def upload_image(self):
        """上传图像并将其显示在画布上，自动调整尺寸以适配画布。"""
        from PIL import Image, ImageTk  # 局部导入，因为 ImageTk 依赖 tkinter 包

        self.image = Image.open(self.filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")]))
        if not self.image:
            return

        self.imgw, self.imgh = self.image.size
        aspect_ratio = self.imgw / self.imgh
        canvas_width = (
            min(self.canvas_max_width, self.imgw) if aspect_ratio > 1 else int(self.canvas_max_height * aspect_ratio)
        )
        canvas_height = (
            min(self.canvas_max_height, self.imgh) if aspect_ratio <= 1 else int(canvas_width / aspect_ratio)
        )

        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas_image = ImageTk.PhotoImage(self.image.resize((canvas_width, canvas_height)))
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.rg_data.clear(), self.current_box.clear()

    def on_canvas_click(self, event):
        """处理鼠标点击事件，在画布上添加用于标记边界框的点。"""
        self.current_box.append((event.x, event.y))
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red")
        if len(self.current_box) == 4:
            self.rg_data.append(self.current_box.copy())
            self.draw_box(self.current_box)
            self.current_box.clear()

    def draw_box(self, box):
        """使用提供的坐标在画布上绘制一个边界框。"""
        for i in range(4):
            self.canvas.create_line(box[i], box[(i + 1) % 4], fill="blue", width=2)

    def remove_last_bounding_box(self):
        """移除最后一个边界框，并重绘画布。"""
        if not self.rg_data:
            self.messagebox.showwarning("Warning", "没有可移除的边界框。")
            return
        self.rg_data.pop()
        self.redraw_canvas()

    def redraw_canvas(self):
        """重新绘制画布，包括图像和所有边界框。"""
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        for box in self.rg_data:
            self.draw_box(box)

    def save_to_json(self):
        """将选定的停车区域坐标以缩放后形式保存为 JSON 文件。"""
        scale_w, scale_h = self.imgw / self.canvas.winfo_width(), self.imgh / self.canvas.winfo_height()
        data = [{"points": [(int(x * scale_w), int(y * scale_h)) for x, y in box]} for box in self.rg_data]

        from io import StringIO  # 函数级导入，仅用于保存坐标，而非每一帧图像

        write_buffer = StringIO()
        json.dump(data, write_buffer, indent=4)
        with open("bounding_boxes.json", "w", encoding="utf-8") as f:
            f.write(write_buffer.getvalue())
        self.messagebox.showinfo("Success", "边界框已保存至 bounding_boxes.json")


class ParkingManagement(BaseSolution):
    """
    使用 YOLO 模型进行实时监测与可视化，管理停车位占用情况。

    此类继承自 BaseSolution，提供停车管理相关功能，包括检测被占用车位、可视化车位区域、
    显示车位使用统计等。

    属性：
        json_file (str): 包含停车区域信息的 JSON 文件路径。
        json (List[Dict]): 加载的 JSON 数据，存储停车区域坐标信息。
        pr_info (Dict[str, int]): 储存停车信息的字典（包括已占用与空闲数量）。
        arc (Tuple[int, int, int]): 空闲车位区域的 RGB 颜色。
        occ (Tuple[int, int, int]): 已占用车位区域的 RGB 颜色。
        dc (Tuple[int, int, int]): 检测目标中心点的 RGB 颜色。

    方法：
        process_data: 处理模型检测数据，实现停车管理与可视化。

    示例：
        >>> from ultralytics.solutions import ParkingManagement
        >>> parking_manager = ParkingManagement(model="yolov8n.pt", json_file="parking_regions.json")
        >>> print(f"已占用车位: {parking_manager.pr_info['Occupancy']}")
        >>> print(f"空闲车位: {parking_manager.pr_info['Available']}")
    """

    def __init__(self, **kwargs):
        """初始化停车管理系统，包括 YOLO 模型与可视化配置。"""
        super().__init__(**kwargs)

        self.json_file = self.CFG["json_file"]  # 加载 JSON 数据
        if self.json_file is None:
            LOGGER.warning("❌ 缺少 json_file 参数。需要提供停车区域的详情。")
            raise ValueError("❌ Json 文件路径不能为空")

        with open(self.json_file) as f:
            self.json = json.load(f)

        self.pr_info = {"Occupancy": 0, "Available": 0}  # 初始化车位信息字典

        self.arc = (0, 0, 255)  # 空闲车位区域颜色（红色）
        self.occ = (0, 255, 0)  # 占用车位区域颜色（绿色）
        self.dc = (255, 0, 189)  # 检测目标中心点颜色（粉色）

    def process_data(self, im0):
        """
        处理模型检测数据，用于停车区域管理。

        本函数分析输入图像，提取目标轨迹，根据 JSON 中定义的停车区域判断是否被占用，
        并在图像上标注占用与空闲区域，同时更新车位统计信息。

        参数：
            im0 (np.ndarray): 输入图像帧（用于推理的图像）。

        示例：
            >>> parking_manager = ParkingManagement(json_file="parking_regions.json")
            >>> image = cv2.imread("parking_lot.jpg")
            >>> parking_manager.process_data(image)
        """
        self.extract_tracks(im0)  # 提取图像中的目标轨迹
        es, fs = len(self.json), 0  # 空闲车位数量，总占用数量初始化
        annotator = Annotator(im0, self.line_width)  # 初始化绘图工具

        for region in self.json:
            # 将点转换为 NumPy 数组，确保数据类型正确并重塑维度
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            rg_occupied = False  # 初始化为未被占用
            for box, cls in zip(self.boxes, self.clss):
                xc, yc = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                dist = cv2.pointPolygonTest(pts_array, (xc, yc), False)
                if dist >= 0:
                    # cv2.circle(im0, (xc, yc), radius=self.line_width * 4, color=self.dc, thickness=-1)
                    annotator.display_objects_labels(
                        im0, self.model.names[int(cls)], (104, 31, 17), (255, 255, 255), xc, yc, 10
                    )
                    rg_occupied = True
                    break
            fs, es = (fs + 1, es - 1) if rg_occupied else (fs, es)
            # 绘制区域边框
            cv2.polylines(im0, [pts_array], isClosed=True, color=self.occ if rg_occupied else self.arc, thickness=2)

        self.pr_info["Occupancy"], self.pr_info["Available"] = fs, es

        annotator.display_analytics(im0, self.pr_info, (104, 31, 17), (255, 255, 255), 10)
        self.display_output(im0)  # 使用基类函数显示输出结果
        return im0  # 返回处理后的图像，可用于其他用途
