# 零样本动作识别与 YOLOv11（视频推理）

- **动作识别** 是一种用于识别和分类视频中人物执行的动作的技术。此过程在考虑多个动作时能够实现更高级的分析。动作可以在实时中被检测并分类。
- 系统可以根据用户的偏好和需求进行定制，识别特定的动作。

## 目录

- [步骤 1：安装所需的库](#步骤-1：安装所需的库)
- [步骤 2：使用 Ultralytics YOLOv11 运行动作识别](#步骤-2：使用-ultralytics-yolov11-运行动作识别)
- [使用选项](#使用选项)
- [常见问题](#常见问题)

## 步骤 1：安装所需的库

克隆仓库，安装依赖项并使用 `cd` 命令进入本地目录进行步骤 2 中的命令。

```bash
# 克隆 ultralytics 仓库
git clone https://github.com/ultralytics/ultralytics

# 进入本地目录
cd examples/YOLOv11-Action-Recognition

# 安装依赖项
pip install -U -r requirements.txt
```

## 步骤 2：使用 Ultralytics YOLOv11 运行动作识别

以下是运行推理的基本命令：

### 注意

该动作识别模型将自动检测并跟踪视频中的人物，并根据指定的标签分类他们的动作。结果将实时显示在视频输出上。您可以通过修改运行脚本时的 `--labels` 参数来自定义动作标签。

```bash
# 快速开始
python action_recognition.py

# 基本用法
python action_recognition.py --source "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --labels "dancing" "singing a song"

# 使用本地视频文件
python action_recognition.py --source path/to/video.mp4

# 提升检测性能
python action_recognition.py --weights yolov11m.pt

# 在 CPU 上运行
python action_recognition.py --device cpu

# 使用不同的视频分类模型
python action_recognition.py --video-classifier-model "s3d"

# 使用 FP16 进行推理（仅适用于 HuggingFace 模型）
python action_recognition.py --fp16

# 导出输出为 mp4 文件
python action_recognition.py --output-path output.mp4

# 组合多个选项
python action_recognition.py --source "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --device 0 --video-classifier-model "microsoft/xclip-base-patch32" --labels "dancing" "singing a song" --fp16
```

## 使用选项

- `--weights`：YOLO 模型权重的路径（默认：`yolov11n.pt`）
- `--device`：Cuda 设备， 如 0 或 0,1,2,3 或 cpu（默认：自动检测）
- `--source`：视频文件路径或 YouTube URL（默认：[rickroll](https://www.youtube.com/watch?v=dQw4w9WgXcQ)）
- `--output-path`：输出视频文件路径
- `--crop-margin-percentage`：检测到的物体周围的边距百分比（默认：10）
- `--num-video-sequence-samples`：用于分类的视频帧数（默认：8）, 对于 s3d, mvit_v1_b, mvit_v2_s, 需要设置为 16 帧
- `--skip-frame`：检测之间跳过的帧数（默认：1）
- `--video-cls-overlap-ratio`：视频序列之间的重叠比例（默认：0.25）
- `--fp16`：使用 FP16 进行推理（仅适用于 HuggingFace 模型）
- `--video-classifier-model`：视频分类模型名称或路径（默认："microsoft/xclip-base-patch32"）
- `--labels`：零样本视频分类标签（默认：`["dancing" "singing a song"]`）

## 常见问题

**1. 动作识别是什么？**

动作识别是一种计算方法，用于识别和分类录制视频或实时流中的个体所执行的动作或活动。这项技术广泛应用于视频分析、监控和人机交互中，通过分析运动模式和上下文来检测和理解人类行为。

**2. 动作识别支持自定义标签吗？**

是的，动作识别系统支持自定义动作标签。`action_recognition.py` 脚本允许用户为零样本视频分类指定自己的自定义标签。这可以通过在运行脚本时使用 `--labels` 参数来完成。例如：

```bash
python action_recognition.py --source https://www.youtube.com/watch?v=dQw4w9WgXcQ --labels "dancing" "singing" "jumping"
```

您可以调整这些标签以匹配您要在视频中识别的特定动作。系统将尝试根据这些自定义标签对检测到的动作进行分类。

此外，您还可以选择不同的视频分类模型：

1. 对于 Hugging Face 模型，您可以使用任何兼容的视频分类模型。默认设置为：

   - `"microsoft/xclip-base-patch32"`

2. 对于 TorchVision 模型（不支持零样本标签），您可以从以下选项中选择：

   - `"s3d"`
   - `"r3d_18"`
   - `"swin3d_t"`
   - `"swin3d_b"`
   - `"mvit_v1_b"`
   - `"mvit_v2_s"`

**3. 为什么将动作识别与 YOLOv11 结合使用？**

YOLOv11 专注于视频流中的物体检测和跟踪。动作识别则通过识别个体执行的动作并进行分类，能够补充 YOLOv11 的目标检测功能，使其成为一个更有价值的应用。

**4. 我可以使用其他版本的 YOLO 吗？**

当然，您可以通过 `--weights` 选项指定不同的 YOLO 模型权重。