# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import ast
import json
import platform
import zipfile
from collections import OrderedDict, namedtuple
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ultralytics.utils import ARM64, IS_JETSON, IS_RASPBERRYPI, LINUX, LOGGER, ROOT, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_version, check_yaml
from ultralytics.utils.downloads import attempt_download_asset, is_url


def check_class_names(names):
    """
    检查类名。

    如果需要，将 imagenet 类代码映射到可读的类名。将列表转换为字典。
    """
    if isinstance(names, list):  # 如果 names 是一个列表
        names = dict(enumerate(names))  # 转换为字典
    if isinstance(names, dict):
        # 转换 1) 字符串类型的键为整数，例如 '0' 转为 0，非字符串类型的值转换为字符串，例如 True 转为 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n} 类数据集需要类索引在 0 到 {n - 1} 之间，但在你的数据集 YAML 文件中定义的类索引 "
                f"是 {min(names.keys())}-{max(names.keys())}，存在无效的类索引。"
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):  # 如果是 imagenet 类代码，例如 'n01440764'
            names_map = yaml_load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # 可读的类名映射
            names = {k: names_map[v] for k, v in names.items()}
    return names


def default_class_names(data=None):
    """应用默认的类名到输入的 YAML 文件，或者返回数字类名。"""
    if data:
        try:
            return yaml_load(check_yaml(data))["names"]
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}  # 如果上面的操作失败，返回默认的类名


class AutoBackend(nn.Module):
    """
    处理动态后端选择，用于运行 Ultralytics YOLO 模型进行推理。

    AutoBackend 类旨在为各种推理引擎提供一个抽象层。它支持多种格式，每种格式都有特定的命名约定，详细信息如下：

        支持的格式和命名约定：
            | 格式                   | 文件后缀           |
            |------------------------|--------------------|
            | PyTorch                | *.pt               |
            | TorchScript            | *.torchscript      |
            | ONNX Runtime           | *.onnx             |
            | ONNX OpenCV DNN        | *.onnx (dnn=True)  |
            | OpenVINO               | *openvino_model/   |
            | CoreML                 | *.mlpackage        |
            | TensorRT               | *.engine           |
            | TensorFlow SavedModel  | *_saved_model/     |
            | TensorFlow GraphDef    | *.pb               |
            | TensorFlow Lite        | *.tflite           |
            | TensorFlow Edge TPU    | *_edgetpu.tflite   |
            | PaddlePaddle           | *_paddle_model/    |
            | MNN                    | *.mnn              |
            | NCNN                   | *_ncnn_model/      |

    该类根据输入的模型格式提供动态后端切换功能，简化了在各种平台上部署模型的工作。
    """

    @torch.no_grad()
    def __init__(
        self,
        weights="yolo11n.pt",
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=False,
        batch=1,
        fuse=True,
        verbose=True,
    ):
        """
        初始化 AutoBackend 进行推理。

        参数：
            weights (str | torch.nn.Module): 模型权重文件的路径或模块实例。默认为 'yolo11n.pt'。
            device (torch.device): 模型运行所在的设备。默认为 CPU。
            dnn (bool): 使用 OpenCV DNN 模块进行 ONNX 推理。默认为 False。
            data (str | Path | optional): 可选的额外数据.yaml 文件路径，其中包含类名。
            fp16 (bool): 启用半精度推理，仅支持特定后端。默认为 False。
            batch (int): 假设的推理批次大小。
            fuse (bool): 是否融合 Conv2D + BatchNorm 层以优化性能。默认为 True。
            verbose (bool): 启用详细日志记录。默认为 True。
        """
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        (
            pt,
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
            paddle,
            mnn,
            ncnn,
            imx,
            triton,
        ) = self._model_type(w)
        fp16 &= pt or jit or onnx or xml or engine or nn_module or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC 格式（与 torch 的 BCWH 格式不同）
        stride = 32  # 默认步幅
        model, metadata, task = None, None, None

        # 设置设备
        cuda = torch.cuda.is_available() and device.type != "cpu"  # 使用 CUDA
        if cuda and not any([nn_module, pt, jit, engine, onnx, paddle]):  # 如果是 GPU 数据加载格式
            device = torch.device("cpu")
            cuda = False

        # 如果本地没有，下载模型
        if not (pt or triton or nn_module):
            w = attempt_download_asset(w)

        # 内存中的PyTorch模型
        if nn_module:
            model = weights.to(device)
            if fuse:
                model = model.fuse(verbose=verbose)
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape  # 仅姿态检测
            stride = max(int(model.stride.max()), 32)  # 模型步幅
            names = model.module.names if hasattr(model, "module") else model.names  # 获取类别名称
            model.half() if fp16 else model.float()
            self.model = model  # 显式赋值，以便使用to()、cpu()、cuda()、half()等
            pt = True

        # PyTorch模型
        elif pt:
            from ultralytics.nn.tasks import attempt_load_weights

            model = attempt_load_weights(
                weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
            )
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape  # 仅姿态检测
            stride = max(int(model.stride.max()), 32)  # 模型步幅
            names = model.module.names if hasattr(model, "module") else model.names  # 获取类别名称
            model.half() if fp16 else model.float()
            self.model = model  # 显式赋值，以便使用to()、cpu()、cuda()、half()等

        # TorchScript模型
        elif jit:
            LOGGER.info(f"加载 {w} 进行TorchScript推理...")
            extra_files = {"config.txt": ""}  # 模型元数据
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # 加载元数据字典
                metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))

        # ONNX OpenCV DNN
        elif dnn:
            LOGGER.info(f"加载 {w} 进行ONNX OpenCV DNN推理...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)

        # ONNX Runtime 和 IMX
        elif onnx or imx:
            LOGGER.info(f"加载 {w} 进行ONNX Runtime推理...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            if IS_RASPBERRYPI or IS_JETSON:
                # 解决'numpy.linalg._umath_linalg'没有'_ilp64'属性的问题，这对RPi和Jetson的TF SavedModel有影响
                check_requirements("numpy==1.23.5")
            import onnxruntime

            providers = ["CPUExecutionProvider"]
            if cuda and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            elif cuda:  # 如果请求了CUDA但不可用，则只记录警告
                LOGGER.warning("警告 ⚠️ 无法使用CUDA启动ONNX Runtime，改为使用CPU...")
                device = torch.device("cpu")
                cuda = False
            LOGGER.info(f"使用ONNX Runtime {providers[0]}")
            if onnx:
                session = onnxruntime.InferenceSession(w, providers=providers)
            else:
                check_requirements(
                    ["model-compression-toolkit==2.1.1", "sony-custom-layers[torch]==0.2.0", "onnxruntime-extensions"]
                )
                w = next(Path(w).glob("*.onnx"))
                LOGGER.info(f"加载 {w} 进行ONNX IMX推理...")
                import mct_quantizers as mctq
                from sony_custom_layers.pytorch.object_detection import nms_ort  # type: ignore # noqa

                session = onnxruntime.InferenceSession(
                    w, mctq.get_ort_session_options(), providers=["CPUExecutionProvider"]
                )
                task = "detect"

            output_names = [x.name for x in session.get_outputs()]
            metadata = session.get_modelmeta().custom_metadata_map
            dynamic = isinstance(session.get_outputs()[0].shape[0], str)
            if not dynamic:
                io = session.io_binding()
                bindings = []
                for output in session.get_outputs():
                    y_tensor = torch.empty(output.shape, dtype=torch.float16 if fp16 else torch.float32).to(device)
                    io.bind_output(
                        name=output.name,
                        device_type=device.type,
                        device_id=device.index if cuda else 0,
                        element_type=np.float16 if fp16 else np.float32,
                        shape=tuple(y_tensor.shape),
                        buffer_ptr=y_tensor.data_ptr(),
                    )
                    bindings.append(y_tensor)

        # OpenVINO
        elif xml:
            LOGGER.info(f"加载 {w} 进行OpenVINO推理...")
            check_requirements("openvino>=2024.0.0")
            import openvino as ov

            core = ov.Core()
            w = Path(w)
            if not w.is_file():  # 如果不是*.xml文件
                w = next(w.glob("*.xml"))  # 从*_openvino_model目录中获取*.xml文件
            ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))
            if ov_model.get_parameters()[0].get_layout().empty:
                ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))

            # OpenVINO推理模式为'LATENCY'，'THROUGHPUT'（不推荐）或'CUMULATIVE_THROUGHPUT'
            inference_mode = "CUMULATIVE_THROUGHPUT" if batch > 1 else "LATENCY"
            LOGGER.info(f"使用OpenVINO {inference_mode} 模式进行批次={batch}推理...")
            ov_compiled_model = core.compile_model(
                ov_model,
                device_name="AUTO",  # AUTO选择最佳可用设备，不要修改
                config={"PERFORMANCE_HINT": inference_mode},
            )
            input_name = ov_compiled_model.input().get_any_name()
            metadata = w.parent / "metadata.yaml"

        # TensorRT
        elif engine:
            LOGGER.info(f"加载 {w} 进行 TensorRT 推理...")
            try:
                import tensorrt as trt  # noqa https://developer.nvidia.com/nvidia-tensorrt-download
            except ImportError:
                if LINUX:
                    check_requirements("tensorrt>7.0.0,!=10.1.0")
                import tensorrt as trt  # noqa
            check_version(trt.__version__, ">=7.0.0", hard=True)
            check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            # 读取文件
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                try:
                    meta_len = int.from_bytes(f.read(4), byteorder="little")  # 读取元数据长度
                    metadata = json.loads(f.read(meta_len).decode("utf-8"))  # 读取元数据
                except UnicodeDecodeError:
                    f.seek(0)  # 引擎文件可能缺少嵌入的Ultralytics元数据
                model = runtime.deserialize_cuda_engine(f.read())  # 读取引擎

            # 模型上下文
            try:
                context = model.create_execution_context()
            except Exception as e:  # 模型为空
                LOGGER.error(f"错误: TensorRT 模型与 {trt.__version__} 版本不兼容\n")
                raise e

            bindings = OrderedDict()
            output_names = []
            fp16 = False  # 默认情况下，下面会更新
            dynamic = False
            is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))
                        if dtype == np.float16:
                            fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_tensor_shape(name))
                else:  # TensorRT < 10.0
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    is_input = model.binding_is_input(i)
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # 动态
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))
                        if dtype == np.float16:
                            fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # 如果是动态的，这里是最大批次大小

        # CoreML
        elif coreml:
            LOGGER.info(f"加载 {w} 进行 CoreML 推理...")
            import coremltools as ct

            model = ct.models.MLModel(w)
            metadata = dict(model.user_defined_metadata)

        # TF SavedModel
        elif saved_model:
            LOGGER.info(f"加载 {w} 进行 TensorFlow SavedModel 推理...")
            import tensorflow as tf

            keras = False  # 假设是TF1的saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
            metadata = Path(w) / "metadata.yaml"

        # TF GraphDef
        elif pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f"加载 {w} 进行 TensorFlow GraphDef 推理...")
            import tensorflow as tf

            from ultralytics.engine.exporter import gd_outputs

            def wrap_frozen_graph(gd, inputs, outputs):
                """包装冻结图以进行部署。"""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # 包装
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
            try:  # 尝试在 SavedModel 中查找元数据和 GraphDef
                metadata = next(Path(w).resolve().parent.rglob(f"{Path(w).stem}_saved_model*/metadata.yaml"))
            except StopIteration:
                pass

        # TFLite 或 TFLite Edge TPU
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                device = device[3:] if str(device).startswith("tpu") else ":0"
                LOGGER.info(f"加载 {w} 在设备 {device[1:]} 上进行 TensorFlow Lite Edge TPU 推理...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[platform.system()]
                interpreter = Interpreter(
                    model_path=w,
                    experimental_delegates=[load_delegate(delegate, options={"device": device})],
                )
                device = "cpu"  # 必须这样，否则 PyTorch 会尝试使用错误的设备
            else:  # TFLite
                LOGGER.info(f"加载 {w} 进行 TensorFlow Lite 推理...")
                interpreter = Interpreter(model_path=w)  # 加载TFLite模型
            interpreter.allocate_tensors()  # 分配张量
            input_details = interpreter.get_input_details()  # 输入
            output_details = interpreter.get_output_details()  # 输出
            # 加载元数据
            try:
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    metadata = ast.literal_eval(model.read(meta_file).decode("utf-8"))
            except zipfile.BadZipFile:
                pass

        # TF.js
        elif tfjs:
            raise NotImplementedError("YOLOv8 TF.js 推理当前不支持。")

        # PaddlePaddle
        elif paddle:
            LOGGER.info(f"加载 {w} 进行 PaddlePaddle 推理...")
            check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")
            import paddle.inference as pdi  # noqa

            w = Path(w)
            if not w.is_file():  # 如果不是 *.pdmodel 文件
                w = next(w.rglob("*.pdmodel"))  # 从 *_paddle_model 目录获取 *.pdmodel 文件
            config = pdi.Config(str(w), str(w.with_suffix(".pdiparams")))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
            metadata = w.parents[1] / "metadata.yaml"

        # MNN
        elif mnn:
            LOGGER.info(f"加载 {w} 进行 MNN 推理...")
            check_requirements("MNN")  # 需要 MNN
            import os

            import MNN

            config = {"precision": "low", "backend": "CPU", "numThread": (os.cpu_count() + 1) // 2}
            rt = MNN.nn.create_runtime_manager((config,))
            net = MNN.nn.load_module_from_file(w, [], [], runtime_manager=rt, rearrange=True)

            def torch_to_mnn(x):
                return MNN.expr.const(x.data_ptr(), x.shape)

            metadata = json.loads(net.get_info()["bizCode"])

        # NCNN
        elif ncnn:
            LOGGER.info(f"加载 {w} 进行 NCNN 推理...")
            check_requirements("git+https://github.com/Tencent/ncnn.git" if ARM64 else "ncnn")  # 需要 NCNN
            import ncnn as pyncnn

            net = pyncnn.Net()
            net.opt.use_vulkan_compute = cuda
            w = Path(w)
            if not w.is_file():  # 如果不是 *.param 文件
                w = next(w.glob("*.param"))  # 从 *_ncnn_model 目录获取 *.param 文件
            net.load_param(str(w))
            net.load_model(str(w.with_suffix(".bin")))
            metadata = w.parent / "metadata.yaml"

        # NVIDIA Triton Inference Server
        elif triton:
            check_requirements("tritonclient[all]")
            from ultralytics.utils.triton import TritonRemoteModel

            model = TritonRemoteModel(w)
            metadata = model.metadata

        # 其他任何格式（不支持的格式）
        else:
            from ultralytics.engine.exporter import export_formats

            raise TypeError(
                f"模型='{w}' 不是一个支持的模型格式。Ultralytics 支持的格式：{export_formats()['Format']}\n"
                f"请参阅 https://docs.ultralytics.com/modes/predict 获取帮助。"
            )

        # 加载外部元数据 YAML 文件
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = yaml_load(metadata)
        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                if k in {"stride", "batch"}:
                    metadata[k] = int(v)
                elif k in {"imgsz", "names", "kpt_shape"} and isinstance(v, str):
                    metadata[k] = eval(v)
            stride = metadata["stride"]
            task = metadata["task"]
            batch = metadata["batch"]
            imgsz = metadata["imgsz"]
            names = metadata["names"]
            kpt_shape = metadata.get("kpt_shape")
        elif not (pt or triton or nn_module):
            LOGGER.warning(f"WARNING ⚠️ 未找到 'model={weights}' 的元数据")

        # 检查类别名称
        if "names" not in locals():  # 如果没有找到类别名称
            names = default_class_names(data)
        names = check_class_names(names)

        # 禁用梯度
        if pt:
            for p in model.parameters():
                p.requires_grad = False

        self.__dict__.update(locals())  # 将所有变量赋值给 self

    def forward(self, im, augment=False, visualize=False, embed=None):
        """
        对YOLOv8 MultiBackend模型进行推理。

        参数:
            im (torch.Tensor): 用于推理的图像张量。
            augment (bool): 是否在推理过程中执行数据增强，默认为False。
            visualize (bool): 是否可视化输出预测结果，默认为False。
            embed (list, optional): 一个包含特征向量/嵌入的列表，用于返回。

        返回:
            (tuple): 返回一个元组，包含原始输出张量，以及用于可视化的处理后输出（如果visualize=True）。
        """
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # 转为FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW 转为 numpy BHWC 形状(1,320,192,3)

        # PyTorch
        if self.pt or self.nn_module:
            y = self.model(im, augment=augment, visualize=visualize, embed=embed)

        # TorchScript
        elif self.jit:
            y = self.model(im)

        # ONNX OpenCV DNN
        elif self.dnn:
            im = im.cpu().numpy()  # 从torch转为numpy
            self.net.setInput(im)
            y = self.net.forward()

        # ONNX Runtime
        elif self.onnx or self.imx:
            if self.dynamic:
                im = im.cpu().numpy()  # 从torch转为numpy
                y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
            else:
                if not self.cuda:
                    im = im.cpu()
                self.io.bind_input(
                    name="images",
                    device_type=im.device.type,
                    device_id=im.device.index if im.device.type == "cuda" else 0,
                    element_type=np.float16 if self.fp16 else np.float32,
                    shape=tuple(im.shape),
                    buffer_ptr=im.data_ptr(),
                )
                self.session.run_with_iobinding(self.io)
                y = self.bindings
            if self.imx:
                # boxes, conf, cls
                y = np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None]], axis=-1)

        # OpenVINO
        elif self.xml:
            im = im.cpu().numpy()  # FP32

            if self.inference_mode in {"THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:  # 对大批量进行优化
                n = im.shape[0]  # 批次中的图像数量
                results = [None] * n  # 预分配列表，与图像数量匹配

                def callback(request, userdata):
                    """使用userdata索引将结果放入预分配的列表中。"""
                    results[userdata] = request.results

                # 创建异步推理队列，设置回调函数并开始每个输入图像的异步推理
                async_queue = self.ov.runtime.AsyncInferQueue(self.ov_compiled_model)
                async_queue.set_callback(callback)
                for i in range(n):
                    # 启动异步推理，并使用userdata=i指定结果列表中的位置
                    async_queue.start_async(inputs={self.input_name: im[i : i + 1]}, userdata=i)  # 保持图像为BCHW格式
                async_queue.wait_all()  # 等待所有推理请求完成
                y = np.concatenate([list(r.values())[0] for r in results])

            else:  # inference_mode = "LATENCY"，对单个批次进行最快的结果优化
                y = list(self.ov_compiled_model(im).values())

        # TensorRT
        elif self.engine:
            if self.dynamic and im.shape != self.bindings["images"].shape:
                if self.is_trt10:
                    self.context.set_input_shape("images", im.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                    for name in self.output_names:
                        self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
                else:
                    i = self.model.get_binding_index("images")
                    self.context.set_binding_shape(i, im.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                    for name in self.output_names:
                        i = self.model.get_binding_index(name)
                        self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))

            s = self.bindings["images"].shape
            assert im.shape == s, f"输入尺寸 {im.shape} {'>' if self.dynamic else '不等于'} 最大模型尺寸 {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]

        # CoreML
        elif self.coreml:
            im = im[0].cpu().numpy()
            im_pil = Image.fromarray((im * 255).astype("uint8"))
            # im = im.resize((192, 320), Image.BILINEAR)
            y = self.model.predict({"image": im_pil})  # 坐标是xywh规范化的
            if "confidence" in y:
                raise TypeError(
                    "Ultralytics 只支持推理未管道化的CoreML模型，这些模型是通过'nms=False'导出的，"
                    f"但'model={w}'包含由'nms=True'导出的NMS管道。"
                )
                # TODO: CoreML NMS推理处理
                # from ultralytics.utils.ops import xywh2xyxy
                # box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy像素
                # conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float32)
                # y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            y = list(y.values())
            if len(y) == 2 and len(y[1].shape) != 4:  # 分割模型
                y = list(reversed(y))  # 对于分割模型（预测，原型），需要反转

        # PaddlePaddle
        elif self.paddle:
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]

        # MNN
        elif self.mnn:
            input_var = self.torch_to_mnn(im)
            output_var = self.net.onForward([input_var])
            y = [x.read() for x in output_var]

        # NCNN
        elif self.ncnn:
            mat_in = self.pyncnn.Mat(im[0].cpu().numpy())
            with self.net.create_extractor() as ex:
                ex.input(self.net.input_names()[0], mat_in)
                # 警告：'output_names' 被排序，作为 https://github.com/pnnx/pnnx/issues/130 的临时修复
                y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.output_names())]

        # NVIDIA Triton 推理服务器
        elif self.triton:
            im = im.cpu().numpy()  # 从 torch 转换到 numpy
            y = self.model(im)

        # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
        else:
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
                if not isinstance(y, list):
                    y = [y]
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite 或 Edge TPU
                details = self.input_details[0]
                is_int = details["dtype"] in {np.int8, np.int16}  # 判断是否为 TFLite 量化 int8 或 int16 模型
                if is_int:
                    scale, zero_point = details["quantization"]
                    im = (im / scale + zero_point).astype(details["dtype"])  # 反缩放
                self.interpreter.set_tensor(details["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if is_int:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # 重新缩放
                    if x.ndim == 3:  # 如果任务不是分类，排除掩码（ndim=4）等
                        # 按照图像大小反归一化 xywh。参考 https://github.com/ultralytics/ultralytics/pull/1695
                        # xywh 在 TFLite/EdgeTPU 中被归一化，以减轻整数模型的量化误差
                        if x.shape[-1] == 6:  # 端到端模型
                            x[:, :, [0, 2]] *= w
                            x[:, :, [1, 3]] *= h
                        else:
                            x[:, [0, 2]] *= w
                            x[:, [1, 3]] *= h
                            if self.task == "pose":
                                x[:, 5::3] *= w
                                x[:, 6::3] *= h
                    y.append(x)
            # TF 段修复：导出顺序与 ONNX 导出相反，protos 已经转置
            if len(y) == 2:  # 处理 (det, proto) 输出顺序反转的情况
                if len(y[1].shape) != 4:
                    y = list(reversed(y))  # 应该是 y = (1, 116, 8400), (1, 160, 160, 32)
                if y[1].shape[-1] == 6:  # 端到端模型
                    y = [y[1]]
                else:
                    y[1] = np.transpose(y[1], (0, 3, 1, 2))  # 应该是 y = (1, 116, 8400), (1, 32, 160, 160)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]

        # for x in y:
        #     print(type(x), len(x)) 如果是 (list, tuple) 则打印类型和长度，否则打印类型和形状  # 调试形状
        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # 分割和名称未定义
                nc = y[0].shape[1] - y[1].shape[1] - 4  # y = (1, 32, 160, 160), (1, 116, 8400)
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
        将 numpy 数组转换为张量。

        参数:
            x (np.ndarray): 需要转换的数组。

        返回:
            (torch.Tensor): 转换后的张量
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        通过运行一次前向传播来预热模型，使用一个虚拟输入。

        参数:
            imgsz (tuple): 虚拟输入张量的形状，格式为(batch_size, channels, height, width)
        """
        import torchvision  # noqa (在此导入，避免记录 torchvision 导入时间)

        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # 输入
            for _ in range(2 if self.jit else 1):
                self.forward(im)  # 预热

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        给定模型文件的路径并返回模型类型。可能的类型包括 pt、jit、onnx、xml、engine、coreml、
        saved_model、pb、tflite、edgetpu、tfjs、ncnn 或 paddle。

        参数:
            p: 模型文件的路径，默认为 "path/to/model.pt"

        示例:
            >>> model = AutoBackend(weights="path/to/model.onnx")
            >>> model_type = model._model_type()  # 返回 "onnx"
        """
        from ultralytics.engine.exporter import export_formats

        sf = export_formats()["Suffix"]  # 导出后缀
        if not is_url(p) and not isinstance(p, str):
            check_suffix(p, sf)  # 检查后缀
        name = Path(p).name
        types = [s in name for s in sf]
        types[5] |= name.endswith(".mlmodel")  # 保留对旧版 Apple CoreML *.mlmodel 格式的支持
        types[8] &= not types[9]  # tflite &= not edgetpu
        if any(types):
            triton = False
        else:
            from urllib.parse import urlsplit

            url = urlsplit(p)
            triton = bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}

        return types + [triton]

