# **YOLOv11 - ONNX Runtime**

本项目使用 **ONNX Runtime** 实现 **YOLOv11** 目标检测。

---

## **安装指南**

要运行此项目，你需要安装必要的依赖项。以下指南将帮助你完成安装过程。

### **安装必要的依赖项**

你可以通过运行以下命令安装所需的依赖项：

```bash
pip install -r requirements.txt
```

---

### **安装 `onnxruntime-gpu`（适用于 NVIDIA GPU 加速）**

如果你的系统有 **NVIDIA GPU**，并希望利用 GPU 进行加速，可以使用以下命令安装 **onnxruntime-gpu** 版本：

```bash
pip install onnxruntime-gpu
```

> **注意**：请确保你的系统已经安装了适用于 **NVIDIA GPU** 的 **CUDA 驱动程序**，否则 `onnxruntime-gpu` 可能无法正常运行。

---

### **安装 `onnxruntime`（CPU 版本）**

如果你的设备 **没有 NVIDIA GPU**，或者你更倾向于使用 **CPU** 进行计算，你可以安装 **onnxruntime CPU 版本**：

```bash
pip install onnxruntime
```

---

## **使用方法**

成功安装必要的软件包后，你可以使用以下命令运行 YOLOv11 推理：

```bash
python main.py --model yolov11n.onnx --img image.jpg --conf-thres 0.5 --iou-thres 0.5
```

**参数说明：**
- `--model yolov11n.onnx`：指定 **YOLOv11 ONNX 模型文件** 的路径。
- `--img image.jpg`：指定 **输入图像** 的路径。
- `--conf-thres 0.5`：设置 **置信度阈值**，用于筛选低置信度的检测结果。
- `--iou-thres 0.5`：设置 **IoU（交并比）阈值**，用于非极大值抑制（NMS）处理重叠框。

> 请确保 `yolov11n.onnx` 是你的 **YOLOv11 ONNX 模型文件**，`image.jpg` 是你的 **输入图像**，并根据需求调整 **置信度阈值（conf-thres）** 和 **IoU 阈值（iou-thres）**。