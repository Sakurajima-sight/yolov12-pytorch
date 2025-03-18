# **YOLOv11-Segmentation-ONNXRuntime-Python 演示**

本仓库提供了一个 **YOLOv11** 使用 **ONNX Runtime** 进行**分割推理**的 Python 演示，展示了 **YOLOv11 模型的跨框架兼容性**，无需完整的 PyTorch 运行环境即可执行推理。

---

## **✨ 主要特点**

- **🔹 无框架依赖**：仅使用 **ONNX Runtime** 进行推理，无需导入 **PyTorch**。
- **⚡ 高效推理**：支持 **FP32（单精度）** 和 **FP16（半精度）** ONNX 模型，适应不同计算需求。
- **📌 易于使用**：通过简单的 **命令行参数** 进行模型推理。
- **🖼️ 广泛兼容**：依赖 **Numpy** 和 **OpenCV** 进行图像处理，可适用于多种环境。

---

## **📥 安装依赖**

使用 `pip` 安装必要的依赖包。  
你需要 **`ultralytics`** 用于 **导出 YOLOv11-seg ONNX 模型**，**`onnxruntime-gpu`** 进行 **GPU 加速推理**，以及 **`opencv-python`** 进行 **图像处理**。

```bash
pip install ultralytics
pip install onnxruntime-gpu  # 适用于支持 NVIDIA GPU 的环境
# pip install onnxruntime    # 如果没有 NVIDIA GPU，请使用这个
pip install numpy
pip install opencv-python
```

---

## **🚀 快速开始**

### **1️⃣ 导出 YOLOv11 ONNX 模型**
使用 `ultralytics` 将 **YOLOv11 分割模型** 转换为 **ONNX 格式**。

```bash
yolo export model=yolov11s-seg.pt imgsz=640 format=onnx opset=12 simplify
```

### **2️⃣ 运行推理**
使用导出的 ONNX 模型在你的图片上执行推理。

```bash
python main.py --model <模型路径> --source <图片路径>
```

---

## **🎯 运行示例**
执行上述命令后，你应该会看到类似如下的**目标分割结果**：

<img src="https://user-images.githubusercontent.com/51357717/279988626-eb74823f-1563-4d58-a8e4-0494025b7c9a.jpg" alt="分割示例" width="800">

---

## **📌 进阶用法**
如果你需要更高级的功能，例如 **实时视频处理**，请参考 `main.py` 脚本中的 **命令行参数** 说明。

---

## **🤝 贡献**
我们欢迎社区贡献者帮助改进本演示项目！  
如果你发现 **bug**、有 **功能建议**，或者想提交 **优化算法**，请提交 **Issue** 或 **Pull Request** 进行交流。

---

## **📜 许可证**
本项目基于 **AGPL-3.0 许可证** 进行开源，详情请查看 [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 文件。

---

## **🎗️ 致谢**
- **YOLOv11-Segmentation-ONNXRuntime-Python** 由 GitHub 用户 [jamjamjon](https://github.com/jamjamjon) 贡献。
- 感谢 **ONNX Runtime** 社区提供了强大高效的推理引擎。

🚀 **希望你喜欢本演示，并在你的项目中成功运行！🎉**