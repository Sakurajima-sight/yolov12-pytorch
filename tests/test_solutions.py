# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

import cv2
import pytest

from tests import TMP
from ultralytics import YOLO, solutions
from ultralytics.utils import ASSETS_URL, WEIGHTS_DIR
from ultralytics.utils.downloads import safe_download

DEMO_VIDEO = "solutions_ci_demo.mp4"
POSE_VIDEO = "solution_ci_pose_demo.mp4"


@pytest.mark.slow
def test_major_solutions():
    """测试目标计数、热力图、速度估算、轨迹区域和队列管理等解决方案。"""
    safe_download(url=f"{ASSETS_URL}/{DEMO_VIDEO}", dir=TMP)  # 下载演示视频
    cap = cv2.VideoCapture(str(TMP / DEMO_VIDEO))  # 读取视频文件
    assert cap.isOpened(), "读取视频文件时出错"
    
    # 定义目标区域
    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
    
    # 初始化不同的分析方案
    counter = solutions.ObjectCounter(region=region_points, model="yolo11n.pt", show=False)  # 目标计数
    heatmap = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, model="yolo11n.pt", show=False)  # 热力图
    heatmap_count = solutions.Heatmap(
        colormap=cv2.COLORMAP_PARULA, model="yolo11n.pt", show=False, region=region_points
    )  # 带目标计数的热力图
    speed = solutions.SpeedEstimator(region=region_points, model="yolo11n.pt", show=False)  # 速度估算
    queue = solutions.QueueManager(region=region_points, model="yolo11n.pt", show=False)  # 队列管理
    line_analytics = solutions.Analytics(analytics_type="line", model="yolo11n.pt", show=False)  # 线形分析
    pie_analytics = solutions.Analytics(analytics_type="pie", model="yolo11n.pt", show=False)  # 饼图分析
    bar_analytics = solutions.Analytics(analytics_type="bar", model="yolo11n.pt", show=False)  # 柱状图分析
    area_analytics = solutions.Analytics(analytics_type="area", model="yolo11n.pt", show=False)  # 面积分析
    trackzone = solutions.TrackZone(region=region_points, model="yolo11n.pt", show=False)  # 轨迹区域

    frame_count = 0  # 统计帧数（用于分析）

    # 逐帧读取视频
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        frame_count += 1
        original_im0 = im0.copy()

        # 调用不同的分析方法
        _ = counter.count(original_im0.copy())  # 目标计数
        _ = heatmap.generate_heatmap(original_im0.copy())  # 生成热力图
        _ = heatmap_count.generate_heatmap(original_im0.copy())  # 生成带目标计数的热力图
        _ = speed.estimate_speed(original_im0.copy())  # 速度估算
        _ = queue.process_queue(original_im0.copy())  # 处理队列管理
        _ = line_analytics.process_data(original_im0.copy(), frame_count)  # 线形分析
        _ = pie_analytics.process_data(original_im0.copy(), frame_count)  # 饼图分析
        _ = bar_analytics.process_data(original_im0.copy(), frame_count)  # 柱状图分析
        _ = area_analytics.process_data(original_im0.copy(), frame_count)  # 面积分析
        _ = trackzone.trackzone(original_im0.copy())  # 轨迹区域分析
    
    cap.release()  # 释放视频资源

    # 测试健身监测解决方案
    safe_download(url=f"{ASSETS_URL}/{POSE_VIDEO}", dir=TMP)  # 下载姿态估计视频
    cap = cv2.VideoCapture(str(TMP / POSE_VIDEO))  # 读取姿态估计视频
    assert cap.isOpened(), "读取视频文件时出错"
    
    gym = solutions.AIGym(kpts=[5, 11, 13], show=False)  # 监测关键点 5, 11, 13

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        _ = gym.monitor(im0)  # 监测姿势
    
    cap.release()  # 释放视频资源


@pytest.mark.slow
def test_instance_segmentation():
    """测试实例分割解决方案。"""
    from ultralytics.utils.plotting import Annotator, colors

    model = YOLO(WEIGHTS_DIR / "yolo11n-seg.pt")  # 加载分割模型
    names = model.names  # 获取类别名称
    cap = cv2.VideoCapture(TMP / DEMO_VIDEO)  # 读取演示视频
    assert cap.isOpened(), "读取视频文件时出错"

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        
        results = model.predict(im0)  # 进行实例分割预测
        annotator = Annotator(im0, line_width=2)  # 标注工具
        
        # 如果检测到了掩码
        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()  # 获取类别
            masks = results[0].masks.xy  # 获取掩码坐标
            for mask, cls in zip(masks, clss):
                color = colors(int(cls), True)  # 选取颜色
                annotator.seg_bbox(mask=mask, mask_color=color, label=names[int(cls)])  # 绘制分割结果
    
    cap.release()  # 释放视频资源
    cv2.destroyAllWindows()  # 关闭所有窗口


@pytest.mark.slow
def test_streamlit_predict():
    """测试 Streamlit 直播推理解决方案。"""
    solutions.Inference().inference()  # 运行 Streamlit 推理
