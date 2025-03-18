# **使用 YOLOv11 进行区域计数（视频推理）**

> **区域计数（Region Counter）** 现已成为 **[Ultralytics 解决方案](https://docs.ultralytics.com/solutions/)** 的一部分，提供更丰富的功能和定期更新。享受更强大的功能和持续改进！

🔗 **[点击这里探索区域内的目标计数](https://docs.ultralytics.com/guides/region-counting/)**

> 🔔 **通知：**  
> GitHub 示例代码仍然可用，但**将不再进行主动维护**。如需最新更新和改进，请使用官方[链接](https://docs.ultralytics.com/guides/region-counting/)。感谢您的支持！

---

**区域计数（Region Counting）** 是一种用于计算特定区域内目标数量的方法，当同时考虑多个区域时，可以进行更高级的分析。这些区域可以通过**鼠标左键点击**进行交互式调整，并且计算过程是**实时**的。您可以根据自己的需求调整区域，以适应不同的应用场景。

<div>
<p align="center">
  <img src="https://github.com/RizwanMunawar/ultralytics/assets/62513924/5ab3bbd7-fd12-4849-928e-5f294d6c3fcf" width="45%" alt="YOLOv11 区域计数示例 1">
  <img src="https://github.com/RizwanMunawar/ultralytics/assets/62513924/e7c1aea7-474d-4d78-8d48-b50854ffe1ca" width="45%" alt="YOLOv11 区域计数示例 2">
</p>
</div>

---

## **目录**
- [步骤 1：安装所需库](#步骤-1安装所需库)
- [步骤 2：使用 Ultralytics YOLOv11 运行区域计数](#步骤-2使用-ultralytics-yolov11-运行区域计数)
- [使用选项](#使用选项)
- [常见问题（FAQ）](#常见问题faq)

---

## **步骤 1：安装所需库**

克隆仓库，安装依赖项，并进入本地目录，以便运行 **步骤 2** 中的命令。

```bash
# 克隆 Ultralytics 仓库
git clone https://github.com/ultralytics/ultralytics

# 进入本地目录
cd ultralytics/examples/YOLOv11-Region-Counter
```

---

## **步骤 2：使用 Ultralytics YOLOv11 运行区域计数**

以下是运行推理的基本命令：

### **注意**
当视频播放后，您可以**自由拖动区域**，只需**按住鼠标左键**并拖动即可。

```bash
# 如果希望保存结果
python yolov11_region_counter.py --source "path/to/video.mp4" --save-img --view-img

# 如果希望在 CPU 上运行模型
python yolov11_region_counter.py --source "path/to/video.mp4" --save-img --view-img --device cpu

# 如果希望更换模型文件
python yolov11_region_counter.py --source "path/to/video.mp4" --save-img --weights "path/to/model.pt"

# 如果只想检测特定类别（如第一类和第三类）
python yolov11_region_counter.py --source "path/to/video.mp4" --classes 0 2 --weights "path/to/model.pt"

# 如果不想保存结果，仅进行可视化
python yolov11_region_counter.py --source "path/to/video.mp4" --view-img
```

---

## **使用选项**

- `--source`：指定要运行推理的视频文件路径。
- `--device`：指定计算设备，可选 `cpu` 或 `0`（GPU）。
- `--save-img`：是否保存检测结果的图片。
- `--weights`：指定不同的 YOLOv11 模型文件（例如 `yolov11n.pt`, `yolov11s.pt`, `yolov11m.pt`, `yolov11l.pt`, `yolov11x.pt`）。
- `--classes`：指定要检测的类别。
- `--line-thickness`：设置目标框的线条粗细。
- `--region-thickness`：设置区域框的线条粗细。
- `--track-thickness`：设置目标轨迹线的粗细。

---

## **常见问题（FAQ）**

### **1. 什么是区域计数？**
区域计数是一种计算方法，用于确定特定区域内的目标数量，无论是在**录制的视频**还是**实时流**中。这种技术在**图像处理、计算机视觉和模式识别**领域非常常见，可以用于分析目标的**空间关系**和**分布情况**。

---

### **2. 区域计数是否支持自定义绘制区域？**
是的，区域计数支持**多种区域格式**，例如 **多边形（Polygon）和矩形（Rectangle）**。您可以**自由修改区域的属性**，如坐标、颜色等。例如：

```python
from shapely.geometry import Polygon

counting_regions = [
    {
        "name": "YOLOv11 多边形区域",
        "polygon": Polygon(
            [(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]
        ),  # 由五个点组成的五边形
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # BGR 颜色值
        "text_color": (255, 255, 255),  # 文字颜色
    },
    {
        "name": "YOLOv11 矩形区域",
        "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),  # 由四个点组成的矩形
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # BGR 颜色值
        "text_color": (0, 0, 0),  # 文字颜色
    },
]
```
---

### **3. 为什么要将 YOLOv11 与区域计数结合？**
YOLOv11 擅长**检测和跟踪视频流中的目标**，而区域计数能够**在指定区域内进行目标计数**。两者结合后，可以**准确统计**区域内的目标数量，并支持**实时交互调整区域**，因此适用于**交通监控、人员流量分析、工业检测等场景**。

---

### **4. 如何排查运行时的问题？**
如果在运行推理时遇到问题，可以添加 `--debug` 选项，以获取更详细的日志信息：
```bash
python yolov11_region_counter.py --source "path/to/video.mp4" --debug
```

---

### **5. 是否可以使用其他 YOLO 版本？**
可以！您可以使用 `--weights` 选项来指定不同的 YOLO 版本，例如：
```bash
python yolov11_region_counter.py --source "path/to/video.mp4" --weights yolov5s.pt
```
---

### **6. 哪里可以获取更多信息？**
如果您想了解 **YOLOv11 在多目标跟踪（MOT）方面的应用**，请访问：
[Ultralytics YOLO 多目标跟踪教程](https://docs.ultralytics.com/modes/track/)

---

🚀 **现在，您可以使用 YOLOv11 在 OpenCV 上进行高效的区域计数！**