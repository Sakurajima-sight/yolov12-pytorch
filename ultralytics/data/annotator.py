# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics import SAM, YOLO


def auto_annotate(
    data,
    det_model="yolo11x.pt",
    sam_model="sam_b.pt",
    device="",
    conf=0.25,
    iou=0.45,
    imgsz=640,
    max_det=300,
    classes=None,
    output_dir=None,
):
    """
    使用 YOLO 物体检测模型和 SAM 分割模型自动标注图像。

    该函数处理指定目录中的图像，使用 YOLO 模型进行物体检测，然后使用 SAM 模型生成分割掩码。
    生成的标注结果将保存为文本文件。

    参数:
        data (str): 包含待标注图像的文件夹路径。
        det_model (str): 预训练的 YOLO 检测模型路径或名称。
        sam_model (str): 预训练的 SAM 分割模型路径或名称。
        device (str): 运行模型的设备（例如，'cpu'、'cuda'、'0'）。
        conf (float): 检测模型的置信度阈值；默认值为 0.25。
        iou (float): 用于过滤重叠框的 IoU 阈值；默认值为 0.45。
        imgsz (int): 输入图像的重置尺寸；默认值为 640。
        max_det (int): 限制每张图像的检测数，以控制密集场景中的输出。
        classes (list): 将预测限制为指定的类 ID，仅返回相关的检测结果。
        output_dir (str | None): 保存标注结果的目录。如果为 None，则会创建默认目录。

    示例:
        >>> from ultralytics.data.annotator import auto_annotate
        >>> auto_annotate(data="ultralytics/assets", det_model="yolo11n.pt", sam_model="mobile_sam.pt")

    备注:
        - 如果未指定输出目录，函数将创建一个新的目录。
        - 标注结果将保存为与输入图像同名的文本文件。
        - 输出文本文件中的每一行代表一个检测到的物体，包含其类 ID 和分割点。
    """
    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)

    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    det_results = det_model(
        data, stream=True, device=device, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=classes
    )

    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()  # noqa
        if len(class_ids):
            boxes = result.boxes.xyxy  # 框对象，用于边框输出
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
            segments = sam_results[0].masks.xyn  # noqa

            with open(f"{Path(output_dir) / Path(result.path).stem}.txt", "w") as f:
                for i in range(len(segments)):
                    s = segments[i]
                    if len(s) == 0:
                        continue
                    segment = map(str, segments[i].reshape(-1).tolist())
                    f.write(f"{class_ids[i]} " + " ".join(segment) + "\n")
