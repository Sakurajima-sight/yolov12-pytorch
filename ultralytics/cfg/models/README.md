## **模型（Models）**

欢迎来到 [Ultralytics](https://www.ultralytics.com/) **模型目录**！在这里，您可以找到各种预配置的模型配置文件（`*.yaml`），可用于创建**自定义的 YOLO 模型**。这些模型由 **Ultralytics 团队**精心设计和调整，以在**目标检测**和**图像分割**等任务上提供最佳性能。

这些模型涵盖了各种应用场景，从**简单的目标检测**到更复杂的**实例分割和目标跟踪**任务。同时，它们经过优化，可以**高效运行在多种硬件平台上**，包括 CPU 和 GPU。无论您是**深度学习专家**还是**刚刚接触 YOLO**，这里的模型都可以为您的自定义开发提供良好的起点。

要开始使用，只需浏览本目录并找到最符合您需求的模型。选择模型后，您可以使用对应的 `*.yaml` 文件**轻松训练和部署您的自定义 YOLO 模型**。完整详情请参阅 **[Ultralytics 官方文档](https://docs.ultralytics.com/models/)**，如果有任何问题或需要帮助，欢迎联系 **Ultralytics 团队**。  
🚀 **不要犹豫，现在就开始创建您的自定义 YOLO 模型吧！** 🚀

---

## **💡 使用方法（Usage）**

**模型 `*.yaml` 文件可以直接用于 [命令行接口（CLI）](https://docs.ultralytics.com/usage/cli/) 中，通过 `yolo` 命令进行调用：**

```bash
# 使用 coco8 数据集训练 YOLO11n 模型 100 轮
yolo task=detect mode=train model=yolo11n.yaml data=coco8.yaml epochs=100
```

**也可以在 Python 环境中直接使用，并且接受与 CLI 相同的参数：[配置参数](https://docs.ultralytics.com/usage/cfg/)**

```python
from ultralytics import YOLO

# 从 YAML 配置文件初始化 YOLO11n 模型
model = YOLO("model.yaml")

# 如果有可用的预训练模型，建议使用：
# model = YOLO("model.pt")

# 显示模型信息
model.info()

# 使用 COCO8 数据集训练模型 100 轮
model.train(data="coco8.yaml", epochs=100)
```

---

## **🧩 预训练模型架构（Pre-trained Model Architectures）**

Ultralytics **支持多种模型架构**。您可以访问 [Ultralytics 模型文档](https://docs.ultralytics.com/models/) 查看详细信息和使用方法。  
任何这些模型都可以通过加载其 **配置文件（`yaml`）** 或 **预训练权重（`pt`）** 直接使用。

---

## **📢 贡献新模型（Contribute New Models）**

**您是否训练了新的 YOLO 变体？或者通过特定的优化实现了最先进的性能？** 🎯  
我们欢迎您的贡献，并愿意在我们的 **模型专区** 展示您的成果！👏  

来自社区的贡献对于扩展 YOLO 模型的选择和优化至关重要。无论是**新模型架构**、**优化方法**还是**高效的超参数调整**，您的贡献都将有助于整个 YOLO 生态系统的发展。通过贡献，您不仅可以**分享知识**，还能让 YOLO 变得更加强大和多样化。

**如何贡献？**
📌 请参阅我们的 **[贡献指南](https://docs.ultralytics.com/help/contributing/)**，其中包含提交 **Pull Request（PR）** 的详细步骤 🛠️。  

🌍 **让我们携手拓展 Ultralytics YOLO 模型的能力和适用范围吧！🙏** 🚀