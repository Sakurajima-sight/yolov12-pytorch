# **YOLO11 与 SAHI（视频推理）**

[SAHI](https://docs.ultralytics.com/guides/sahi-tiled-inference/) 旨在优化目标检测算法，使其适用于大规模和高分辨率图像。它通过将图像分割成可管理的小块，对每个小块进行目标检测，然后将结果拼接在一起。 本教程将指导您如何使用 SAHI 在视频文件上运行 YOLO11 推理。

---

## **目录**
- [步骤 1：安装所需的库](#步骤-1安装所需的库)
- [步骤 2：使用 Ultralytics YOLO11 结合 SAHI 运行推理](#步骤-2使用-ultralytics-yolo11-结合-sahi-运行推理)
- [使用选项](#使用选项)
- [常见问题](#常见问题)

---

## **步骤 1：安装所需的库**
首先克隆代码仓库，安装所需的依赖，并 `cd` 进入相关目录，以便执行后续命令。

```bash
# 克隆 ultralytics 代码仓库
git clone https://github.com/ultralytics/ultralytics

# 安装依赖
pip install -U sahi ultralytics

# 进入本地目录
cd ultralytics/examples/YOLOv11-SAHI-Inference-Video
```

---

## **步骤 2：使用 Ultralytics YOLO11 结合 SAHI 运行推理**
以下是运行推理的基本命令：

```bash
# 如果想要保存检测结果
python yolov11_sahi.py --source "path/to/video.mp4" --save-img

# 如果想要更换模型文件
python yolov11_sahi.py --source "path/to/video.mp4" --save-img --weights "yolo11n.pt"
```

---

## **使用选项**
- `--source`：指定要运行推理的视频文件路径。
- `--save-img`：使用该标志可以将检测结果保存为图片。
- `--weights`：指定不同的 YOLO11 模型权重文件（例如 `yolo11n.pt`、`yolov11s.pt`、`yolo11m.pt`、`yolo11l.pt`、`yolo11x.pt`）。

---

## **常见问题**

### **1. 什么是 SAHI？**
SAHI（Slicing Aided Hyper Inference）是一种优化目标检测算法的库，专门用于处理大规模和高分辨率图像。SAHI 的源代码可在 [GitHub](https://github.com/obss/sahi) 上找到。

### **2. 为什么要在 YOLO11 中使用 SAHI？**
SAHI 通过将大图像切片成较小的、易于管理的块进行检测，而不会影响检测质量。这使得它成为 YOLO11 的理想搭配，特别适用于高分辨率视频处理。

### **3. 如何调试问题？**
您可以在命令中添加 `--debug` 标志，以便在推理过程中输出更多调试信息：

```bash
python yolov11_sahi.py --source "path/to/video.mp4" --debug
```

### **4. 可以使用其他 YOLO 版本吗？**
可以，您可以使用 `--weights` 选项指定不同的 YOLO 模型权重。

### **5. 哪里可以找到更多信息？**
完整的 YOLO11 结合 SAHI 的指南请参考：[https://docs.ultralytics.com/guides/sahi-tiled-inference](https://docs.ultralytics.com/guides/sahi-tiled-inference/)。🚀