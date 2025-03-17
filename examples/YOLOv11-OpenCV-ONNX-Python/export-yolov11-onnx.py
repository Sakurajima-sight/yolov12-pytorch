from ultralytics import YOLO  # 导入YOLO库

# 加载YOLO11模型，"yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"
model = YOLO("yolo11x.pt") 

# 将模型导出为ONNX格式
model.export(format="onnx", dynamic=False, half=False, simplify=False)  

# 加载导出的ONNX模型
onnx_model = YOLO("yolo11x.onnx", task='detect')  

# 运行推理（推理输入是一张图片）
results = onnx_model("https://ultralytics.com/images/bus.jpg")  

print("类别索引:", results[0].boxes.cls.cpu().numpy())  

results[0].show()  