# 使用 Ultralytics YOLO 进行多目标跟踪

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418637-1d6250fd-1515-4c10-a844-a32818ae6d46.png" alt="YOLOv8 trackers visualization">

在视频分析领域，对象跟踪是一项关键任务，它不仅识别出图像中对象的位置和类别，还能在视频过程中为每个检测到的对象保持唯一 ID。其应用场景几乎无所不在——从安防监控到实时体育分析。

---

## 为什么选择 Ultralytics YOLO 进行对象跟踪？

Ultralytics 的跟踪器输出格式与标准目标检测一致，但增加了对象 ID，使得在视频流中跟踪对象变得轻松，并便于后续分析。以下是你应该考虑使用 Ultralytics YOLO 进行对象跟踪的理由：

- **高效性**：可在保证精度的同时实现实时视频处理。
- **灵活性**：支持多种跟踪算法和配置。
- **易用性**：提供简单的 Python API 和命令行工具，方便快速集成和部署。
- **可定制性**：可与自定义训练的 YOLO 模型结合使用，便于面向特定领域的集成。

📹 **视频教程**：[使用 Ultralytics YOLO 进行目标检测与跟踪](https://www.youtube.com/embed/hHyHmOtmEgs?si=VNZtXmm45Nb9s-N-)

---

## 功能一览

Ultralytics YOLO 不仅提供强大的目标检测能力，还扩展了对稳健灵活目标跟踪的支持：

- **实时跟踪**：可在高帧率视频中无缝进行对象跟踪。
- **支持多种跟踪器**：可选择多种主流跟踪算法。
- **跟踪器配置可定制**：通过调整参数灵活适配不同场景需求。

---

## 可用的跟踪器

Ultralytics YOLO 支持以下跟踪算法。通过传递对应的 YAML 配置文件启用，例如使用 `tracker=tracker_type.yaml`：

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - 使用 `botsort.yaml` 启用此跟踪器。
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - 使用 `bytetrack.yaml` 启用此跟踪器。

默认使用 BoT-SORT 作为跟踪器。

---

## 跟踪示例

要在视频流中运行目标跟踪，可使用 Detect、Segment 或 Pose 类型的训练模型，如 YOLO11n、YOLO11n-seg 和 YOLO11n-pose。

### Python 示例

```python
from ultralytics import YOLO

# 加载官方模型或自定义模型
model = YOLO("yolo11n.pt")  # 加载官方检测模型
model = YOLO("yolo11n-seg.pt")  # 加载官方分割模型
model = YOLO("yolo11n-pose.pt")  # 加载官方姿态模型
model = YOLO("path/to/best.pt")  # 加载自定义训练模型

# 执行跟踪任务
results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # 使用默认跟踪器进行跟踪
results = model.track(
    source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml"
)  # 使用 ByteTrack 跟踪器进行跟踪
```

### 命令行示例（CLI）

```bash
# 使用命令行接口对不同模型进行目标跟踪
yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4"  # 检测模型
yolo track model=yolo11n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # 分割模型
yolo track model=yolo11n-pose.pt source="https://youtu.be/LNwODJXcvt4"  # 姿态模型
yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4"  # 自定义模型

# 使用 ByteTrack 跟踪器
yolo track model=path/to/best.pt tracker="bytetrack.yaml"
```

如上所示，所有用于检测、分割、姿态估计的模型在视频或流媒体中均可实现目标跟踪。

---

## 配置参数

### 跟踪参数说明

跟踪模式的配置与预测模式类似，例如支持 `conf`（置信度）、`iou`（IoU阈值）、`show`（显示结果）等参数。更多配置说明可参考 [Predict 模式文档](https://docs.ultralytics.com/modes/predict/)。

#### Python 示例

```python
from ultralytics import YOLO

# 设置跟踪参数并执行跟踪
model = YOLO("yolo11n.pt")
results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
```

#### CLI 示例

```bash
# 使用命令行设置参数并运行跟踪
yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3, iou=0.5 show
```

---

### 自定义跟踪器配置

Ultralytics 还允许你使用自定义的跟踪器配置文件。只需复制已有的配置文件（例如 `custom_tracker.yaml`，可从 [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) 下载），然后根据需求修改内容（注意不要改动 `tracker_type` 字段）。

#### Python 示例

```python
from ultralytics import YOLO

# 使用自定义 tracker 配置文件执行跟踪
model = YOLO("yolo11n.pt")
results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker="custom_tracker.yaml")
```

#### CLI 示例

```bash
# 使用自定义 tracker 配置文件执行跟踪
yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
```

所有可配置参数详见：[ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)

---

## Python 示例：持续跟踪（持久化轨迹）

以下是一个使用 OpenCV（`cv2`）和 YOLO11 在视频帧上运行目标跟踪的脚本。它假设你已安装了 `opencv-python` 和 `ultralytics`。其中的 `persist=True` 参数表示当前帧是连续图像序列的一部分，允许跟踪器保留前一帧的轨迹。

#### Python 示例

```python
import cv2

from ultralytics import YOLO

# 加载 YOLO11 模型
model = YOLO("yolo11n.pt")

# 打开视频文件
video_path = "path/to/video.mp4"
cap = cv2.VideoCapture(video_path)

# 遍历视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()

    if success:
        # 在该帧上运行 YOLO11 跟踪，保持轨迹延续
        results = model.track(frame, persist=True)

        # 可视化该帧结果
        annotated_frame = results[0].plot()

        # 显示结果帧
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # 如果按下 'q' 键，则跳出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 视频读取完毕，结束循环
        break
```

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()
```

请注意，从 `model(frame)` 改为 `model.track(frame)`，可以启用目标跟踪而非简单检测。该脚本将在每一帧上运行跟踪器、可视化结果并在窗口中显示。按下 `q` 键可退出循环。

---

### 随时间绘制轨迹

在连续帧中可视化目标轨迹，有助于深入理解视频中检测目标的运动模式与行为。在 Ultralytics YOLO11 中，绘制这些轨迹是高效且简单的过程。

以下示例展示了如何使用 YOLO11 的跟踪功能来绘制目标在多个视频帧中的移动路径。该脚本会打开一个视频文件，逐帧读取，通过 YOLO 模型识别并跟踪多个目标。通过保留检测框中心点并连接这些点，我们可以绘制出跟踪目标的路径线。

---

#### Python

```python
from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# 加载 YOLO11 模型
model = YOLO("yolo11n.pt")

# 打开视频文件
video_path = "path/to/video.mp4"
cap = cv2.VideoCapture(video_path)

# 存储跟踪历史轨迹
track_history = defaultdict(lambda: [])

# 遍历视频帧
while cap.isOpened():
    # 读取一帧图像
    success, frame = cap.read()

    if success:
        # 在该帧上运行 YOLO11 跟踪（启用跨帧轨迹保持）
        results = model.track(frame, persist=True)

        # 获取边界框与对应的 Track ID
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # 可视化结果
        annotated_frame = results[0].plot()

        # 绘制轨迹线
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # 中心点坐标
            if len(track) > 30:  # 保留最近 30 帧轨迹
                track.pop(0)

            # 绘制轨迹折线
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                annotated_frame,
                [points],
                isClosed=False,
                color=(230, 230, 230),
                thickness=10,
            )

        # 显示带注释的帧
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 视频读取完毕，退出循环
        break

# 释放视频捕获对象并关闭窗口
cap.release()
cv2.destroyAllWindows()
```

---

### 多线程跟踪

多线程跟踪支持同时对多个视频流进行目标跟踪，特别适合处理来自多个摄像头的监控输入，能大幅提升效率和实时性。

以下 Python 脚本利用 Python 的 `threading` 模块并发运行多个跟踪器。每个线程处理一个视频文件，使用指定模型进行目标检测与跟踪，同时在各自窗口中显示结果。

为了确保每个线程正确接收所需参数（即视频文件和模型），我们定义了一个函数 `run_tracker_in_thread`，该函数封装了视频读取、跟踪处理及结果显示流程。

在本示例中，我们使用了两个不同模型：`yolo11n.pt` 和 `yolo11n-seg.pt`，分别对不同视频进行跟踪。视频文件路径由 `video_file1` 和 `video_file2` 指定。

使用 `daemon=True` 意味着当主线程结束时，所有子线程将自动关闭。`start()` 启动线程，`join()` 使主线程等待所有子线程执行完成。

所有线程执行完后，调用 `cv2.destroyAllWindows()` 来关闭所有图像窗口。

---

#### Python

```python
import threading

import cv2

from ultralytics import YOLO


def run_tracker_in_thread(filename, model):
    """在新线程中启动目标跟踪，逐帧读取 filename 视频，并使用指定模型进行目标检测与可视化。"""
    video = cv2.VideoCapture(filename)
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(frames):
        ret, frame = video.read()
        if ret:
            results = model.track(source=frame, persist=True)
            res_plotted = results[0].plot()
            cv2.imshow("p", res_plotted)
            if cv2.waitKey(1) == ord("q"):
                break


# 加载模型
model1 = YOLO("yolo11n.pt")
model2 = YOLO("yolo11n-seg.pt")

# 指定两个视频文件
video_file1 = "path/to/video1.mp4"
video_file2 = "path/to/video2.mp4"

# 创建跟踪线程
tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1), daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model2), daemon=True)

# 启动跟踪线程
tracker_thread1.start()
tracker_thread2.start()

# 等待两个线程完成
tracker_thread1.join()
tracker_thread2.join()

# 清理资源，关闭所有窗口
cv2.destroyAllWindows()
```

---

### 贡献你自己的跟踪器模块

你擅长多目标跟踪吗？是否基于 Ultralytics YOLO 实现或改进过跟踪算法？我们诚邀你参与贡献，在 [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) 中提交你的自定义跟踪器！

你的实际应用和解决方案，可能对许多处理跟踪任务的开发者有重要参考价值。

请阅读我们的 [贡献指南](https://docs.ultralytics.com/help/contributing/)，了解如何提交 Pull Request (PR) 🛠️，我们期待你的加入！

让我们一起壮大 Ultralytics YOLO 的跟踪能力生态 🙏！
