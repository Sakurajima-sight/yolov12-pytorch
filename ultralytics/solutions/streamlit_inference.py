# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
from typing import Any

import cv2

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS


class Inference:
    """
    使用 Streamlit 和 Ultralytics YOLO 模型进行目标检测、图像分类、图像分割和姿态估计推理的类。

    提供模型加载、配置设置、上传视频文件、实时推理等功能。

    属性:
        st (module): Streamlit 模块，用于创建用户界面。
        temp_dict (dict): 用于临时存储模型路径的字典。
        model_path (str): 加载的模型路径。
        model (YOLO): YOLO 模型实例。
        source (str): 选择的视频源。
        enable_trk (str): 是否启用目标跟踪。
        conf (float): 置信度阈值。
        iou (float): 非极大值抑制的 IoU 阈值。
        vid_file_name (str): 上传的视频文件名。
        selected_ind (list): 被选中的类别索引列表。

    方法:
        web_ui: 设置 Streamlit 网页界面和自定义 HTML 元素。
        sidebar: 配置 Streamlit 侧边栏中的模型和推理设置。
        source_upload: 通过 Streamlit 接口上传视频文件。
        configure: 配置模型并加载选择的类别。
        inference: 执行实时目标检测推理。

    示例:
        >>> inf = solutions.Inference(model="path/to/model.pt")  # 模型参数不是必须的
        >>> inf.inference()
    """

    def __init__(self, **kwargs: Any):
        """
        初始化 Inference 类，检查 Streamlit 依赖并设置模型路径。

        参数:
            **kwargs (Any): 模型配置的其他关键字参数。
        """
        check_requirements("streamlit>=1.29.0")  # 作用域导入以提升 ultralytics 包加载速度
        import streamlit as st

        self.st = st  # Streamlit 实例引用
        self.source = None  # 视频或摄像头源
        self.enable_trk = False  # 是否启用跟踪功能
        self.conf = 0.25  # 检测的置信度阈值
        self.iou = 0.45  # 非极大值抑制的 IoU 阈值
        self.org_frame = None  # 原始帧显示容器
        self.ann_frame = None  # 带注释帧显示容器
        self.vid_file_name = None  # 视频文件名
        self.selected_ind = []  # 被选中的类别索引
        self.model = None  # 加载的模型实例

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None  # 存储模型文件路径
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        """设置 Streamlit 网页界面和自定义 HTML 元素。"""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # 隐藏主菜单

        # 页面主标题
        main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""

        # 页面副标题
        sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-top:-15px; margin-bottom:50px;">体验基于 Ultralytics YOLO 的实时目标检测！🚀</h4></div>"""

        # 页面配置和注入 HTML 样式
        self.st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self):
        """配置 Streamlit 侧边栏中的模型和推理设置。"""
        with self.st.sidebar:  # 添加 LOGO
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title("用户配置")  # 设置侧边栏标题
        self.source = self.st.sidebar.selectbox("视频源", ("webcam", "video"))  # 下拉框选择视频来源
        self.enable_trk = self.st.sidebar.radio("启用目标跟踪", ("Yes", "No"))  # 是否启用跟踪
        self.conf = float(
            self.st.sidebar.slider("置信度阈值", 0.0, 1.0, self.conf, 0.01)
        )  # 滑块控制置信度阈值
        self.iou = float(self.st.sidebar.slider("IoU 阈值", 0.0, 1.0, self.iou, 0.01))  # 滑块控制 IoU 阈值

        col1, col2 = self.st.columns(2)
        self.org_frame = col1.empty()  # 显示原始帧
        self.ann_frame = col2.empty()  # 显示注释帧

    def source_upload(self):
        """通过 Streamlit 接口处理视频文件上传。"""
        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("上传视频文件", type=["mp4", "mov", "avi", "mkv"])
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())  # 创建 BytesIO 对象
                with open("ultralytics.mp4", "wb") as out:  # 将字节写入临时文件
                    out.write(g.read())
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "webcam":
            self.vid_file_name = 0  # 摄像头编号 0

    def configure(self):
        """配置模型并加载所选类别用于推理。"""
        # 从 GitHub 下载模型列表，进行选择
        available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path:  # 如果用户提供了自定义模型，则添加到列表前面
            available_models.insert(0, self.model_path.split(".pt")[0])
        selected_model = self.st.sidebar.selectbox("模型选择", available_models)

        with self.st.spinner("模型加载中..."):
            self.model = YOLO(f"{selected_model.lower()}.pt")  # 加载模型
            class_names = list(self.model.names.values())  # 获取模型的类别名称列表
        self.st.success("模型加载成功！")

        # 类别多选框，获取用户选择的类别索引
        selected_classes = self.st.sidebar.multiselect("选择类别", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, list):  # 确保 selected_ind 是列表类型
            self.selected_ind = list(self.selected_ind)

    def inference(self):
        """执行实时目标检测推理过程。"""
        self.web_ui()        # 初始化界面
        self.sidebar()       # 设置侧边栏
        self.source_upload() # 上传视频源
        self.configure()     # 加载模型和类别

        if self.st.sidebar.button("开始"):
            stop_button = self.st.button("停止")  # 停止按钮
            cap = cv2.VideoCapture(self.vid_file_name)  # 打开视频流
            if not cap.isOpened():
                self.st.error("无法打开摄像头。")
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    self.st.warning("无法读取帧。请确认摄像头连接正常。")
                    break

                # 使用模型进行推理
                if self.enable_trk == "Yes":
                    results = self.model.track(
                        frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                    )
                else:
                    results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated_frame = results[0].plot()  # 在帧上绘制注释

                if stop_button:
                    cap.release()  # 释放摄像头资源
                    self.st.stop()  # 停止 streamlit 应用

                self.org_frame.image(frame, channels="BGR")  # 显示原始帧
                self.ann_frame.image(annotated_frame, channels="BGR")  # 显示带注释帧

            cap.release()  # 释放资源
        cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口


if __name__ == "__main__":
    import sys  # 导入 sys 模块，用于访问命令行参数

    # 判断是否通过命令行传入模型名
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None  # 获取第一个命令行参数作为模型路径
    # 创建 Inference 类实例并执行推理
    Inference(model=model).inference()
