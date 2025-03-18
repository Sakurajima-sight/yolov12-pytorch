from ultralytics import YOLO

# 加载 YOLOv11 的分割模型，支持的模型有：YOLOv11n-seg（nano 版本）、YOLOv11s-seg（small 版本）、YOLOv11m-seg（medium 版本）、YOLOv11i-seg（intermediate 版本）、YOLOv11x-seg（extra large 版本）。
model = YOLO("yolo11x-seg.pt")  # 加载官方模型权重文件

# 导出为 ONNX 格式
model.export(format="onnx") 
