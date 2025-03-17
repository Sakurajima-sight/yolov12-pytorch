# RTDETR - ONNX 运行时

本项目实现了使用 ONNX 运行时的 RTDETR。

## 安装

要运行此项目，您需要安装所需的依赖项。以下是安装过程的指导说明。

### 安装所需依赖项

您可以通过运行以下命令安装所需的依赖项：

```bash
pip install -r requirements.txt
```

### 安装 `onnxruntime-gpu`

如果您拥有 NVIDIA GPU 并希望利用 GPU 加速，可以使用以下命令安装 `onnxruntime-gpu` 包：

```bash
pip install onnxruntime-gpu
```

注意：确保您系统上安装了适当的 GPU 驱动程序。

### 安装 `onnxruntime`（CPU 版本）

如果您没有 NVIDIA GPU 或者更喜欢使用 CPU 版本的 onnxruntime，可以使用以下命令安装 `onnxruntime` 包：

```bash
pip install onnxruntime
```

### 使用方法

在成功安装所需的包后，您可以使用以下命令运行 RTDETR 实现：

```bash
python main.py --model rtdetr-l.onnx --img image.jpg --conf-thres 0.5 --iou-thres 0.5
```

"确保将 `rtdetr-l.onnx` 替换为您的 RTDETR ONNX 模型文件路径，将 `image.jpg` 替换为您的输入图像路径，并根据需要调整置信度阈值（`conf-thres`）和 IoU 阈值（`iou-thres`）。RTDETRv2 ONNX 模型文件可在 [RT-DETRv2 GitHub 仓库](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch) 获取。"