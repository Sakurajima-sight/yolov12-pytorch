# YOLOv11 - OpenCV

使用 ONNX 格式在 OpenCV 上实现 YOLOv11。

只需克隆仓库并运行：

```bash
pip install -r requirements.txt
python main.py --model yolov11n.onnx --img image.jpg
```

如果你从零开始：

```bash
pip install ultralytics
yolo export model=yolov11n.pt imgsz=640 format=onnx opset=12
```

_\*请确保包含 `"opset=12"`_