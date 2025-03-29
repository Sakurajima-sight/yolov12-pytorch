# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
基准测试YOLO模型在不同格式下的速度和准确性。

用法：
    from ultralytics.utils.benchmarks import ProfileModels, benchmark
    ProfileModels(['yolov8n.yaml', 'yolov8s.yaml']).profile()
    benchmark(model='yolov8n.pt', imgsz=160)

格式                     | `format=argument`         | 模型
---                      | ---                       | ---
PyTorch                  | -                         | yolov8n.pt
TorchScript              | `torchscript`             | yolov8n.torchscript
ONNX                     | `onnx`                    | yolov8n.onnx
OpenVINO                 | `openvino`                | yolov8n_openvino_model/
TensorRT                 | `engine`                  | yolov8n.engine
CoreML                   | `coreml`                  | yolov8n.mlpackage
TensorFlow SavedModel    | `saved_model`             | yolov8n_saved_model/
TensorFlow GraphDef      | `pb`                      | yolov8n.pb
TensorFlow Lite          | `tflite`                  | yolov8n.tflite
TensorFlow Edge TPU      | `edgetpu`                 | yolov8n_edgetpu.tflite
TensorFlow.js            | `tfjs`                    | yolov8n_web_model/
PaddlePaddle             | `paddle`                  | yolov8n_paddle_model/
MNN                      | `mnn`                     | yolov8n.mnn
NCNN                     | `ncnn`                    | yolov8n_ncnn_model/
"""

import glob
import os
import platform
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch.cuda
import yaml

from ultralytics import YOLO, YOLOWorld
from ultralytics.cfg import TASK2DATA, TASK2METRIC
from ultralytics.engine.exporter import export_formats
from ultralytics.utils import ARM64, ASSETS, IS_JETSON, IS_RASPBERRYPI, LINUX, LOGGER, MACOS, TQDM, WEIGHTS_DIR
from ultralytics.utils.checks import IS_PYTHON_3_12, check_requirements, check_yolo
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import get_cpu_info, select_device


def benchmark(
    model=WEIGHTS_DIR / "yolo11n.pt",
    data=None,
    imgsz=160,
    half=False,
    int8=False,
    device="cpu",
    verbose=False,
    eps=1e-3,
):
    """
    基准测试YOLO模型在不同格式下的速度和准确性。

    参数:
        model (str | Path): 模型文件或目录的路径。
        data (str | None): 要评估的数据集，如果未传递则继承自TASK2DATA。
        imgsz (int): 基准测试的图像大小。
        half (bool): 如果为True，则使用半精度模型。
        int8 (bool): 如果为True，则使用int8精度模型。
        device (str): 用于运行基准测试的设备，可以是'cpu'或'cuda'。
        verbose (bool | float): 如果为True或float，则在给定度量值时断言基准测试通过。
        eps (float): 防止除以零的epsilon值。

    返回:
        (pandas.DataFrame): 一个包含每个格式的基准测试结果的pandas DataFrame，结果包括文件大小、度量值和推理时间。

    示例:
        使用默认设置基准测试YOLO模型：
        >>> from ultralytics.utils.benchmarks import benchmark
        >>> benchmark(model="yolo11n.pt", imgsz=640)
    """
    import pandas as pd  # 在这里导入以加快 'import ultralytics'

    pd.options.display.max_columns = 10
    pd.options.display.width = 120
    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    is_end2end = getattr(model.model.model[-1], "end2end", False)

    y = []
    t0 = time.time()
    for i, (name, format, suffix, cpu, gpu, _) in enumerate(zip(*export_formats().values())):
        emoji, filename = "❌", None  # 默认导出
        try:
            # 检查
            if i == 7:  # TF GraphDef
                assert model.task != "obb", "TensorFlow GraphDef不支持OBB任务"
            elif i == 9:  # Edge TPU
                assert LINUX and not ARM64, "仅支持在非aarch64 Linux上导出Edge TPU"
            elif i in {5, 10}:  # CoreML 和 TF.js
                assert MACOS or LINUX, "CoreML和TF.js导出仅支持在macOS和Linux上"
                assert not IS_RASPBERRYPI, "Raspberry Pi不支持CoreML和TF.js导出"
                assert not IS_JETSON, "NVIDIA Jetson不支持CoreML和TF.js导出"
            if i in {5}:  # CoreML
                assert not IS_PYTHON_3_12, "CoreML不支持Python 3.12"
            if i in {6, 7, 8}:  # TF SavedModel, TF GraphDef, 和 TFLite
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 TensorFlow导出还不支持onnx2tf"
            if i in {9, 10}:  # TF EdgeTPU 和 TF.js
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 TensorFlow导出还不支持onnx2tf"
            if i == 11:  # Paddle
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 Paddle导出还不支持"
                assert not is_end2end, "End-to-end模型暂不支持PaddlePaddle"
                assert LINUX or MACOS, "Windows不支持Paddle导出"
            if i == 12:  # MNN
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 MNN导出还不支持"
            if i == 13:  # NCNN
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 NCNN导出还不支持"
            if i == 14:  # IMX
                assert not is_end2end
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 IMX导出不支持"
                assert model.task == "detect", "IMX仅支持检测任务"
                assert "C2f" in model.__str__(), "IMX仅支持YOLOv8"
            if "cpu" in device.type:
                assert cpu, "CPU上不支持推理"
            if "cuda" in device.type:
                assert gpu, "GPU上不支持推理"

            # 导出
            if format == "-":
                filename = model.ckpt_path or model.cfg
                exported_model = model  # PyTorch格式
            else:
                filename = model.export(imgsz=imgsz, format=format, half=half, int8=int8, device=device, verbose=False)
                exported_model = YOLO(filename, task=model.task)
                assert suffix in str(filename), "导出失败"
            emoji = "❎"  # 表示导出成功

            # 预测
            assert model.task != "pose" or i != 7, "GraphDef Pose推理不支持"
            assert i not in {9, 10}, "不支持推理"  # Edge TPU和TF.js不支持
            assert i != 5 or platform.system() == "Darwin", "推理仅支持在macOS>=10.13上"  # CoreML
            if i in {13}:
                assert not is_end2end, "End-to-end torch.topk操作暂不支持NCNN预测"
            exported_model.predict(ASSETS / "bus.jpg", imgsz=imgsz, device=device, half=half)

            # 验证
            data = data or TASK2DATA[model.task]  # 任务到数据集，例如任务=detect时为coco8.yaml
            key = TASK2METRIC[model.task]  # 任务到度量，例如任务=detect时为metrics/mAP50-95(B)
            results = exported_model.val(
                data=data, batch=1, imgsz=imgsz, plots=False, device=device, half=half, int8=int8, verbose=False
            )
            metric, speed = results.results_dict[key], results.speed["inference"]
            fps = round(1000 / (speed + eps), 2)  # 每秒帧数
            y.append([name, "✅", round(file_size(filename), 1), round(metric, 4), round(speed, 2), fps])
        except Exception as e:
            if verbose:
                assert type(e) is AssertionError, f"基准测试失败 {name}: {e}"
            LOGGER.warning(f"错误 ❌️ 基准测试失败 {name}: {e}")
            y.append([name, emoji, round(file_size(filename), 1), None, None, None])  # mAP, t_inference

    # 打印结果
    check_yolo(device=device)  # 打印系统信息
    df = pd.DataFrame(y, columns=["格式", "状态❔", "大小 (MB)", key, "推理时间 (ms/张)", "FPS"])

    name = Path(model.ckpt_path).name
    s = f"\n基准测试完成：{name}，数据集：{data}，imgsz={imgsz} ({time.time() - t0:.2f}s)\n{df}\n"
    LOGGER.info(s)
    with open("benchmarks.log", "a", errors="ignore", encoding="utf-8") as f:
        f.write(s)

    if verbose and isinstance(verbose, float):
        metrics = df[key].array  # 用于与地板值比较的值
        floor = verbose  # 最低性能标准值，例如：= 0.29 mAP 对于 YOLOv5n
        assert all(x > floor for x in metrics if pd.notna(x)), f"基准测试失败：指标 < 地板值 {floor}"

    return df


class RF100Benchmark:
    """基准测试 YOLO 模型在不同格式下的速度和精度表现。"""

    def __init__(self):
        """初始化 RF100Benchmark 类，用于在不同格式下基准测试 YOLO 模型的性能。"""
        self.ds_names = []
        self.ds_cfg_list = []
        self.rf = None
        self.val_metrics = ["类别", "图像", "目标", "精度", "召回率", "mAP50", "mAP95"]

    def set_key(self, api_key):
        """
        设置 Roboflow API 密钥进行数据集处理。

        参数：
            api_key (str): API 密钥。

        示例：
            设置 Roboflow API 密钥以访问数据集：
            >>> benchmark = RF100Benchmark()
            >>> benchmark.set_key("your_roboflow_api_key")
        """
        check_requirements("roboflow")
        from roboflow import Roboflow

        self.rf = Roboflow(api_key=api_key)

    def parse_dataset(self, ds_link_txt="datasets_links.txt"):
        """
        解析数据集链接并下载数据集。

        参数：
            ds_link_txt (str): 包含数据集链接的文件路径。

        示例：
            >>> benchmark = RF100Benchmark()
            >>> benchmark.set_key("api_key")
            >>> benchmark.parse_dataset("datasets_links.txt")
        """
        (shutil.rmtree("rf-100"), os.mkdir("rf-100")) if os.path.exists("rf-100") else os.mkdir("rf-100")
        os.chdir("rf-100")
        os.mkdir("ultralytics-benchmarks")
        safe_download("https://github.com/ultralytics/assets/releases/download/v0.0.0/datasets_links.txt")

        with open(ds_link_txt) as file:
            for line in file:
                try:
                    _, url, workspace, project, version = re.split("/+", line.strip())
                    self.ds_names.append(project)
                    proj_version = f"{project}-{version}"
                    if not Path(proj_version).exists():
                        self.rf.workspace(workspace).project(project).version(version).download("yolov8")
                    else:
                        print("数据集已下载。")
                    self.ds_cfg_list.append(Path.cwd() / proj_version / "data.yaml")
                except Exception:
                    continue

        return self.ds_names, self.ds_cfg_list

    @staticmethod
    def fix_yaml(path):
        """
        修复给定 YAML 文件中的训练和验证路径。

        参数：
            path (str): 要修复的 YAML 文件路径。

        示例：
            >>> RF100Benchmark.fix_yaml("path/to/data.yaml")
        """
        with open(path) as file:
            yaml_data = yaml.safe_load(file)
        yaml_data["train"] = "train/images"
        yaml_data["val"] = "valid/images"
        with open(path, "w") as file:
            yaml.safe_dump(yaml_data, file)

    def evaluate(self, yaml_path, val_log_file, eval_log_file, list_ind):
        """
        在验证结果上评估模型性能。

        参数：
            yaml_path (str): YAML 配置文件路径。
            val_log_file (str): 验证日志文件路径。
            eval_log_file (str): 评估日志文件路径。
            list_ind (int): 当前数据集在列表中的索引。

        返回：
            (float): 评估模型的平均精度 (mAP) 值。

        示例：
            在特定数据集上评估模型：
            >>> benchmark = RF100Benchmark()
            >>> benchmark.evaluate("path/to/data.yaml", "path/to/val_log.txt", "path/to/eval_log.txt", 0)
        """
        skip_symbols = ["🚀", "⚠️", "💡", "❌"]
        with open(yaml_path) as stream:
            class_names = yaml.safe_load(stream)["names"]
        with open(val_log_file, encoding="utf-8") as f:
            lines = f.readlines()
            eval_lines = []
            for line in lines:
                if any(symbol in line for symbol in skip_symbols):
                    continue
                entries = line.split(" ")
                entries = list(filter(lambda val: val != "", entries))
                entries = [e.strip("\n") for e in entries]
                eval_lines.extend(
                    {
                        "类别": entries[0],
                        "图像": entries[1],
                        "目标": entries[2],
                        "精度": entries[3],
                        "召回率": entries[4],
                        "mAP50": entries[5],
                        "mAP95": entries[6],
                    }
                    for e in entries
                    if e in class_names or (e == "all" and "(AP)" not in entries and "(AR)" not in entries)
                )
        map_val = 0.0
        if len(eval_lines) > 1:
            print("有多个字典")
            for lst in eval_lines:
                if lst["类别"] == "all":
                    map_val = lst["mAP50"]
        else:
            print("只有一个字典结果")
            map_val = [res["mAP50"] for res in eval_lines][0]

        with open(eval_log_file, "a") as f:
            f.write(f"{self.ds_names[list_ind]}: {map_val}\n")


class ProfileModels:
    """
    ProfileModels类，用于在ONNX和TensorRT上分析不同模型的性能。

    这个类用于分析不同模型的性能，返回模型的速度和FLOPs等结果。

    属性：
        paths (List[str]): 要分析的模型路径列表。
        num_timed_runs (int): 用于分析的定时运行次数。
        num_warmup_runs (int): 分析前的预热运行次数。
        min_time (float): 最小的分析时间（秒）。
        imgsz (int): 模型使用的图像大小。
        half (bool): 标志位，指示是否在TensorRT分析中使用FP16半精度。
        trt (bool): 标志位，指示是否使用TensorRT进行分析。
        device (torch.device): 用于分析的设备。

    方法：
        profile: 分析模型并打印结果。

    示例：
        分析模型并打印结果
        >>> from ultralytics.utils.benchmarks import ProfileModels
        >>> profiler = ProfileModels(["yolov8n.yaml", "yolov8s.yaml"], imgsz=640)
        >>> profiler.profile()
    """

    def __init__(
        self,
        paths: list,
        num_timed_runs=100,
        num_warmup_runs=10,
        min_time=60,
        imgsz=640,
        half=True,
        trt=True,
        device=None,
    ):
        """
        初始化ProfileModels类以分析模型。

        参数：
            paths (List[str]): 要分析的模型路径列表。
            num_timed_runs (int): 用于分析的定时运行次数。
            num_warmup_runs (int): 分析前的预热运行次数。
            min_time (float): 最小分析时间（秒）。
            imgsz (int): 分析时使用的图像大小。
            half (bool): 标志位，指示是否在TensorRT分析中使用FP16半精度。
            trt (bool): 标志位，指示是否使用TensorRT进行分析。
            device (torch.device | None): 用于分析的设备。如果为None，则自动确定设备。

        注意：
            FP16 'half'参数选项已移除，因为在CPU上使用FP16比FP32慢。

        示例：
            初始化并分析模型
            >>> from ultralytics.utils.benchmarks import ProfileModels
            >>> profiler = ProfileModels(["yolov8n.yaml", "yolov8s.yaml"], imgsz=640)
            >>> profiler.profile()
        """
        self.paths = paths
        self.num_timed_runs = num_timed_runs
        self.num_warmup_runs = num_warmup_runs
        self.min_time = min_time
        self.imgsz = imgsz
        self.half = half
        self.trt = trt  # 运行TensorRT分析
        self.device = device or torch.device(0 if torch.cuda.is_available() else "cpu")

    def profile(self):
        """分析YOLO模型在不同格式（包括ONNX和TensorRT）上的速度和准确性。"""
        files = self.get_files()

        if not files:
            print("未找到匹配的*.pt或*.onnx文件。")
            return

        table_rows = []
        output = []
        for file in files:
            engine_file = file.with_suffix(".engine")
            if file.suffix in {".pt", ".yaml", ".yml"}:
                model = YOLO(str(file))
                model.fuse()  # 以正确报告参数和GFLOPs
                model_info = model.info()
                if self.trt and self.device.type != "cpu" and not engine_file.is_file():
                    engine_file = model.export(
                        format="engine",
                        half=self.half,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                    )
                onnx_file = model.export(
                    format="onnx",
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
            elif file.suffix == ".onnx":
                model_info = self.get_onnx_model_info(file)
                onnx_file = file
            else:
                continue

            t_engine = self.profile_tensorrt_model(str(engine_file))
            t_onnx = self.profile_onnx_model(str(onnx_file))
            table_rows.append(self.generate_table_row(file.stem, t_onnx, t_engine, model_info))
            output.append(self.generate_results_dict(file.stem, t_onnx, t_engine, model_info))

        self.print_table(table_rows)
        return output

    def get_files(self):
        """返回用户指定的所有相关模型文件的路径列表。"""
        files = []
        for path in self.paths:
            path = Path(path)
            if path.is_dir():
                extensions = ["*.pt", "*.onnx", "*.yaml"]
                files.extend([file for ext in extensions for file in glob.glob(str(path / ext))])
            elif path.suffix in {".pt", ".yaml", ".yml"}:  # 添加非现有文件
                files.append(str(path))
            else:
                files.extend(glob.glob(str(path)))

        print(f"正在分析: {sorted(files)}")
        return [Path(file) for file in sorted(files)]

    @staticmethod
    def get_onnx_model_info(onnx_file: str):
        """从ONNX模型文件中提取元数据，包括参数、GFLOPs和输入形状。"""
        return 0.0, 0.0, 0.0, 0.0  # 返回（层数、参数数、梯度数、FLOPs）

    @staticmethod
    def iterative_sigma_clipping(data, sigma=2, max_iters=3):
        """对数据应用迭代sigma裁剪，以根据指定的sigma和迭代次数去除异常值。"""
        data = np.array(data)
        for _ in range(max_iters):
            mean, std = np.mean(data), np.std(data)
            clipped_data = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
            if len(clipped_data) == len(data):
                break
            data = clipped_data
        return data

    def profile_tensorrt_model(self, engine_file: str, eps: float = 1e-3):
        """使用TensorRT对YOLO模型性能进行基准测试，测量平均运行时间和标准偏差。"""
        if not self.trt or not Path(engine_file).is_file():
            return 0.0, 0.0

        # 模型和输入数据
        model = YOLO(engine_file)
        input_data = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)  # 对于分类任务使用uint8

        # 热身运行
        elapsed = 0.0
        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                model(input_data, imgsz=self.imgsz, verbose=False)
            elapsed = time.time() - start_time

        # 计算运行次数，取最小时间或定时运行次数中的较大值
        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs * 50)

        # 定时运行
        run_times = []
        for _ in TQDM(range(num_runs), desc=engine_file):
            results = model(input_data, imgsz=self.imgsz, verbose=False)
            run_times.append(results[0].speed["inference"])  # 转换为毫秒

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)  # sigma修剪
        return np.mean(run_times), np.std(run_times)

    def profile_onnx_model(self, onnx_file: str, eps: float = 1e-3):
        """对ONNX模型进行基准测试，测量多次运行的平均推理时间和标准偏差。"""
        check_requirements("onnxruntime")
        import onnxruntime as ort

        # 使用'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'之一的会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 8  # 限制线程数
        sess = ort.InferenceSession(onnx_file, sess_options, providers=["CPUExecutionProvider"])

        input_tensor = sess.get_inputs()[0]
        input_type = input_tensor.type
        dynamic = not all(isinstance(dim, int) and dim >= 0 for dim in input_tensor.shape)  # 动态输入形状
        input_shape = (1, 3, self.imgsz, self.imgsz) if dynamic else input_tensor.shape

        # 将ONNX数据类型映射到numpy数据类型
        if "float16" in input_type:
            input_dtype = np.float16
        elif "float" in input_type:
            input_dtype = np.float32
        elif "double" in input_type:
            input_dtype = np.float64
        elif "int64" in input_type:
            input_dtype = np.int64
        elif "int32" in input_type:
            input_dtype = np.int32
        else:
            raise ValueError(f"不支持的ONNX数据类型 {input_type}")

        input_data = np.random.rand(*input_shape).astype(input_dtype)
        input_name = input_tensor.name
        output_name = sess.get_outputs()[0].name

        # 热身运行
        elapsed = 0.0
        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                sess.run([output_name], {input_name: input_data})
            elapsed = time.time() - start_time

        # 计算运行次数，取最小时间或定时运行次数中的较大值
        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs)

        # 定时运行
        run_times = []
        for _ in TQDM(range(num_runs), desc=onnx_file):
            start_time = time.time()
            sess.run([output_name], {input_name: input_data})
            run_times.append((time.time() - start_time) * 1000)  # 转换为毫秒

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=5)  # sigma修剪
        return np.mean(run_times), np.std(run_times)

    def generate_table_row(self, model_name, t_onnx, t_engine, model_info):
        """生成包含模型性能指标（包括推理时间和模型细节）的表格行字符串。"""
        layers, params, gradients, flops = model_info
        return (
            f"| {model_name:18s} | {self.imgsz} | - | {t_onnx[0]:.1f}±{t_onnx[1]:.1f} ms | {t_engine[0]:.1f}±"
            f"{t_engine[1]:.1f} ms | {params / 1e6:.1f} | {flops:.1f} |"
        )

    @staticmethod
    def generate_results_dict(model_name, t_onnx, t_engine, model_info):
        """生成包含模型名称、参数、GFLOPs和速度指标的性能结果字典。"""
        layers, params, gradients, flops = model_info
        return {
            "model/name": model_name,
            "model/parameters": params,
            "model/GFLOPs": round(flops, 3),
            "model/speed_ONNX(ms)": round(t_onnx[0], 3),
            "model/speed_TensorRT(ms)": round(t_engine[0], 3),
        }

    @staticmethod
    def print_table(table_rows):
        """打印格式化的模型基准测试结果表格，包括速度和准确性指标。"""
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
        headers = [
            "模型",
            "尺寸<br><sup>(像素)",
            "mAP<sup>val<br>50-95",
            f"速度<br><sup>CPU ({get_cpu_info()}) ONNX<br>(ms)",
            f"速度<br><sup>{gpu} TensorRT<br>(ms)",
            "参数<br><sup>(百万)",
            "FLOPs<br><sup>(十亿)",
        ]
        header = "|" + "|".join(f" {h} " for h in headers) + "|"
        separator = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"

        print(f"\n\n{header}")
        print(separator)
        for row in table_rows:
            print(row)
