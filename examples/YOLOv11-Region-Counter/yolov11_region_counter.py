# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(list)

current_region = None
counting_regions = [
    {
        "name": "YOLOv11 Polygon Region",
        "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  # 多边形区域点坐标
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # 区域颜色（BGR）
        "text_color": (255, 255, 255),  # 区域文本颜色
        "active_ids": set(),
    },
    {
        "name": "YOLOv11 Rectangle Region",
        "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),  # 矩形区域点坐标
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # 区域颜色（BGR）
        "text_color": (0, 0, 0),  # 区域文本颜色
        "active_ids": set(),
    },
]


def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    """
    处理区域拖动的鼠标事件。

    参数:
        event (int): 鼠标事件类型（如 cv2.EVENT_LBUTTONDOWN）。
        x (int): 鼠标指针的 x 坐标。
        y (int): 鼠标指针的 y 坐标。
        flags (int): OpenCV 传递的附加标志。
        param (Any): 传递给回调的附加参数。

    全局变量:
        current_region (dict): 当前选中的区域信息。

    说明:
        该函数用于 OpenCV 鼠标事件回调，支持在视频帧中选择和拖动计数区域。

    示例:
        >>> cv2.setMouseCallback(window_name, mouse_callback)
    """
    global current_region

    # 鼠标左键按下事件
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    # 鼠标移动事件
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    # 鼠标左键释放事件
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False


def run(
    weights: str = "yolo11n.pt",
    source: str = None,
    device: str = "cpu",
    view_img: bool = False,
    save_img: bool = False,
    exist_ok: bool = False,
    classes: List[int] = None,
    line_thickness: int = 2,
    track_thickness: int = 2,
    region_thickness: int = 2,
) -> None:
    """
    使用 YOLOv11 和 ByteTrack 在视频中进行区域计数。

    参数:
        weights (str): 模型权重文件路径。
        source (str): 视频文件路径。
        device (str): 处理设备：'cpu'、'0'、'1' 等。
        view_img (bool): 是否显示结果。
        save_img (bool): 是否保存结果视频。
        exist_ok (bool): 是否允许覆盖现有文件。
        classes (List[int]): 需要检测和跟踪的类别。
        line_thickness (int): 边界框厚度。
        track_thickness (int): 轨迹线厚度。
        region_thickness (int): 区域边框厚度。

    说明:
        - 支持实时调整的计数区域。
        - 支持多个区域计数。
        - 计数区域可以是多边形或矩形。
    """
    vid_frame_count = 0

    # 检查视频源路径
    if not Path(source).exists():
        raise FileNotFoundError(f"源文件路径 '{source}' 不存在。")

    # 加载模型
    print(f"正在加载模型 {weights} 到 { 'CUDA' if device == '0' else 'CPU' }...")
    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")

    # 获取类别名称
    names = model.names

    # 视频处理
    videocapture = cv2.VideoCapture(source)
    frame_width = int(videocapture.get(3))
    frame_height = int(videocapture.get(4))
    fps = int(videocapture.get(5))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 输出目录
    save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))

    # 逐帧处理视频
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # 获取检测结果
        results = model.track(frame, persist=True, classes=classes)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # 计算目标中心点

                track = track_history[track_id]  # 绘制跟踪轨迹
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                current_track_ids = set(track_ids)
                if not any(region["dragging"] for region in counting_regions):
                    for region in counting_regions:
                        point = Point((bbox_center[0], bbox_center[1]))  # 目标中心点

                        if region["polygon"].contains(point):
                            # 目标进入区域，确保 track_id 记录
                            region["active_ids"].add(track_id)

                        else:
                            # 目标离开区域，删除 track_id（使用 discard 避免 KeyError）
                            region["active_ids"].discard(track_id)

                        # 移除丢失的目标
                        region["active_ids"] = {tid for tid in region["active_ids"] if tid in current_track_ids}

                        # 更新当前区域内的目标数量
                        region["counts"] = len(region["active_ids"])


        # 绘制区域（多边形/矩形）
        for region in counting_regions:
            polygon_coordinates = np.array(region["polygon"].exterior.coords, dtype=np.int32)

            # 计算区域中心点
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            # 在区域中心绘制计数数字
            region_label = str(region["counts"])  # 转成字符串以便绘制
            text_size, _ = cv2.getTextSize(region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2)

            # 计算文本绘制位置，使其居中
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2

            # 先绘制一个带背景的矩形，以提高可读性
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region["region_color"],  # 用区域颜色作为背景色
                -1,  # 填充矩形
            )

            # 在区域中心绘制计数值
            cv2.putText(
                frame,
                region_label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # 字体大小
                region["text_color"],  # 文字颜色
                2,  # 文字粗细
            )

            # 绘制区域边框
            cv2.polylines(frame, [polygon_coordinates], isClosed=True, color=region["region_color"], thickness=region_thickness)


        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Ultralytics YOLOv11 Region Counter Movable")
                cv2.setMouseCallback("Ultralytics YOLOv11 Region Counter Movable", mouse_callback)
            cv2.imshow("Ultralytics YOLOv11 Region Counter Movable", frame)

        if save_img:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()

def parse_opt() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="模型权重文件路径")
    parser.add_argument("--device", default="", help="CUDA 设备，例如 0 或 0,1,2,3 或 cpu")
    parser.add_argument("--source", type=str, required=True, help="视频文件路径")
    parser.add_argument("--view-img", action="store_true", help="是否显示检测结果")
    parser.add_argument("--save-img", action="store_true", help="是否保存检测结果")
    parser.add_argument("--exist-ok", action="store_true", help="是否允许覆盖已有文件")
    parser.add_argument("--classes", nargs="+", type=int, help="指定检测类别，例如 --classes 0 或 --classes 0 2 3")
    parser.add_argument("--line-thickness", type=int, default=2, help="边界框线条厚度")
    parser.add_argument("--track-thickness", type=int, default=2, help="跟踪轨迹线条厚度")
    parser.add_argument("--region-thickness", type=int, default=4, help="区域边界线条厚度")

    return parser.parse_args()


def main(options: argparse.Namespace) -> None:
    """使用解析的参数执行区域计数功能。"""
    run(**vars(options))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
