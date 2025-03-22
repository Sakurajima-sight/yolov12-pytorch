# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license
"""
将 YOLO PyTorch 模型导出为其他格式。TensorFlow 导出由 https://github.com/zldrobit 编写。

格式                    | `format=argument`         | 模型
---                     | ---                       | ---
PyTorch                 | -                         | yolo11n.pt
TorchScript             | `torchscript`             | yolo11n.torchscript
ONNX                    | `onnx`                    | yolo11n.onnx
OpenVINO                | `openvino`                | yolo11n_openvino_model/
TensorRT                | `engine`                  | yolo11n.engine
CoreML                  | `coreml`                  | yolo11n.mlpackage
TensorFlow SavedModel   | `saved_model`             | yolo11n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolo11n.pb
TensorFlow Lite         | `tflite`                  | yolo11n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolo11n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolo11n_web_model/
PaddlePaddle            | `paddle`                  | yolo11n_paddle_model/
MNN                     | `mnn`                     | yolo11n.mnn
NCNN                    | `ncnn`                    | yolo11n_ncnn_model/
IMX                     | `imx`                     | yolo11n_imx_model/

要求:
    $ pip install "ultralytics[export]"

Python:
    from ultralytics import YOLO
    model = YOLO('yolo11n.pt')
    results = model.export(format='onnx')

CLI:
    $ yolo mode=export model=yolo11n.pt format=onnx

推理:
    $ yolo predict model=yolo11n.pt                 # PyTorch
                         yolo11n.torchscript        # TorchScript
                         yolo11n.onnx               # ONNX Runtime 或 OpenCV DNN，设置 dnn=True
                         yolo11n_openvino_model     # OpenVINO
                         yolo11n.engine             # TensorRT
                         yolo11n.mlpackage          # CoreML (仅限 macOS)
                         yolo11n_saved_model        # TensorFlow SavedModel
                         yolo11n.pb                 # TensorFlow GraphDef
                         yolo11n.tflite             # TensorFlow Lite
                         yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                         yolo11n_paddle_model       # PaddlePaddle
                         yolo11n.mnn                # MNN
                         yolo11n_ncnn_model         # NCNN
                         yolo11n_imx_model          # IMX

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolo11n_web_model public/yolo11n_web_model
    $ npm start
"""

import gc
import json
import os
import shutil
import subprocess
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import TASK2DATA, get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import check_class_names, default_class_names
from ultralytics.nn.modules import C2f, Classify, Detect, RTDETRDecoder
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, WorldModel
from ultralytics.utils import (
    ARM64,
    DEFAULT_CFG,
    IS_JETSON,
    LINUX,
    LOGGER,
    MACOS,
    PYTHON_VERSION,
    ROOT,
    WINDOWS,
    __version__,
    callbacks,
    colorstr,
    get_default_args,
    yaml_save,
)
from ultralytics.utils.checks import (
    check_imgsz,
    check_is_path_safe,
    check_requirements,
    check_version,
    is_sudo_available,
)
from ultralytics.utils.downloads import attempt_download_asset, get_github_assets, safe_download
from ultralytics.utils.files import file_size, spaces_in_path
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import TORCH_1_13, get_latest_opset, select_device


def export_formats():
    """Ultralytics YOLO 导出格式。"""
    x = [
        ["PyTorch", "-", ".pt", True, True, []],
        ["TorchScript", "torchscript", ".torchscript", True, True, ["batch", "optimize"]],
        ["ONNX", "onnx", ".onnx", True, True, ["batch", "dynamic", "half", "opset", "simplify"]],
        ["OpenVINO", "openvino", "_openvino_model", True, False, ["batch", "dynamic", "half", "int8"]],
        ["TensorRT", "engine", ".engine", False, True, ["batch", "dynamic", "half", "int8", "simplify"]],
        ["CoreML", "coreml", ".mlpackage", True, False, ["batch", "half", "int8", "nms"]],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True, ["batch", "int8", "keras"]],
        ["TensorFlow GraphDef", "pb", ".pb", True, True, ["batch"]],
        ["TensorFlow Lite", "tflite", ".tflite", True, False, ["batch", "half", "int8"]],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", True, False, []],
        ["TensorFlow.js", "tfjs", "_web_model", True, False, ["batch", "half", "int8"]],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True, ["batch"]],
        ["MNN", "mnn", ".mnn", True, True, ["batch", "half", "int8"]],
        ["NCNN", "ncnn", "_ncnn_model", True, True, ["batch", "half"]],
        ["IMX", "imx", "_imx_model", True, True, ["int8"]],
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments"], zip(*x)))


def validate_args(format, passed_args, valid_args):
    """
    根据格式验证参数。

    参数:
        format (str): 导出格式。
        passed_args (Namespace): 导出过程中使用的参数。
        valid_args (dict): 格式的有效参数列表。

    异常:
        AssertionError: 如果使用了不支持的参数，或者格式没有列出支持的参数时。
    """
    # 只检查这些参数的有效使用
    export_args = ["half", "int8", "dynamic", "keras", "nms", "batch"]

    assert valid_args is not None, f"错误 ❌️ '{format}' 的有效参数未列出。"
    custom = {"batch": 1, "data": None, "device": None}  # 导出器默认值
    default_args = get_cfg(DEFAULT_CFG, custom)
    for arg in export_args:
        not_default = getattr(passed_args, arg, None) != getattr(default_args, arg, None)
        if not_default:
            assert arg in valid_args, f"错误 ❌️ 参数 '{arg}' 不支持格式='{format}'"


def gd_outputs(gd):
    """TensorFlow GraphDef 模型的输出节点名称。"""
    name_list, input_list = [], []
    for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
        name_list.append(node.name)
        input_list.extend(node.input)
    return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))


def try_export(inner_func):
    """YOLO 导出装饰器，即 @try_export。"""
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """导出模型。"""
        prefix = inner_args["prefix"]
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f"{prefix} 导出成功 ✅ {dt.t:.1f}s, 已保存为 '{f}' ({file_size(f):.1f} MB)")
            return f, model
        except Exception as e:
            LOGGER.error(f"{prefix} 导出失败 ❌ {dt.t:.1f}s: {e}")
            raise e

    return outer_func


class Exporter:
    """
    一个用于导出模型的类。

    属性:
        args (SimpleNamespace): 导出器的配置。
        callbacks (list, 可选): 回调函数列表。默认为 None。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        初始化 Exporter 类。

        参数:
            cfg (str, 可选): 配置文件的路径。默认为 DEFAULT_CFG。
            overrides (dict, 可选): 配置覆盖项。默认为 None。
            _callbacks (dict, 可选): 回调函数字典。默认为 None。
        """
        self.args = get_cfg(cfg, overrides)
        if self.args.format.lower() in {"coreml", "mlmodel"}:  # 修复 protobuf<3.20.x 错误的尝试
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # 必须在 TensorBoard 回调之前运行

        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def __call__(self, model=None) -> str:
        """返回在运行回调函数后导出的文件/目录列表。"""
        self.run_callbacks("on_export_start")
        t = time.time()
        fmt = self.args.format.lower()  # 转为小写
        if fmt in {"tensorrt", "trt"}:  # 'engine' 别名
            fmt = "engine"
        if fmt in {"mlmodel", "mlpackage", "mlprogram", "apple", "ios", "coreml"}:  # 'coreml' 别名
            fmt = "coreml"
        fmts_dict = export_formats()
        fmts = tuple(fmts_dict["Argument"][1:])  # 可用的导出格式
        if fmt not in fmts:
            import difflib

            # 如果格式无效，获取最接近的匹配
            matches = difflib.get_close_matches(fmt, fmts, n=1, cutoff=0.6)  # 需要 60% 的相似度
            if not matches:
                raise ValueError(f"无效的导出格式='{fmt}'。有效的格式是 {fmts}")
            LOGGER.warning(f"警告 ⚠️ 无效的导出格式='{fmt}'，已更新为格式='{matches[0]}'")
            fmt = matches[0]
        flags = [x == fmt for x in fmts]
        if sum(flags) != 1:
            raise ValueError(f"无效的导出格式='{fmt}'。有效的格式是 {fmts}")
        (
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
        ) = flags  # 导出布尔值
        is_tf_format = any((saved_model, pb, tflite, edgetpu, tfjs))

        # 设备
        dla = None
        if fmt == "engine" and self.args.device is None:
            LOGGER.warning("警告 ⚠️ TensorRT 需要 GPU 导出，自动分配 device=0")
            self.args.device = "0"
        if fmt == "engine" and "dla" in str(self.args.device):  # 首先将 int/list 转为 str
            dla = self.args.device.split(":")[-1]
            self.args.device = "0"  # 更新设备为 "0"
            assert dla in {"0", "1"}, f"期望 self.args.device='dla:0' 或 'dla:1'，但得到了 {self.args.device}."
        self.device = select_device("cpu" if self.args.device is None else self.args.device)

        # 参数兼容性检查
        fmt_keys = fmts_dict["Arguments"][flags.index(True) + 1]
        validate_args(fmt, self.args, fmt_keys)
        if imx and not self.args.int8:
            LOGGER.warning("警告 ⚠️ IMX 仅支持 int8 导出，设置 int8=True。")
            self.args.int8 = True
        if not hasattr(model, "names"):
            model.names = default_class_names()
        model.names = check_class_names(model.names)
        if self.args.half and self.args.int8:
            LOGGER.warning("警告 ⚠️ half=True 和 int8=True 互斥，设置 half=False。")
            self.args.half = False
        if self.args.half and onnx and self.device.type == "cpu":
            LOGGER.warning("警告 ⚠️ half=True 仅与 GPU 导出兼容，即使用 device=0")
            self.args.half = False
            assert not self.args.dynamic, "half=True 与 dynamic=True 不兼容，即只能使用其中一个。"
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # 检查图像大小
        if self.args.int8 and engine:
            self.args.dynamic = True  # 强制动态导出 TensorRT INT8
        if self.args.optimize:
            assert not ncnn, "optimize=True 与格式='ncnn' 不兼容，即使用 optimize=False"
            assert self.device.type == "cpu", "optimize=True 与 CUDA 设备不兼容，即使用 device='cpu'"
        if self.args.int8 and tflite:
            assert not getattr(model, "end2end", False), "TFLite INT8 导出不支持端到端模型。"
        if edgetpu:
            if not LINUX:
                raise SystemError("Edge TPU 导出仅支持 Linux。请查看 https://coral.ai/docs/edgetpu/compiler")
            elif self.args.batch != 1:  # 见 github.com/ultralytics/ultralytics/pull/13420
                LOGGER.warning("警告 ⚠️ Edge TPU 导出要求批量大小为 1，设置 batch=1。")
                self.args.batch = 1
        if isinstance(model, WorldModel):
            LOGGER.warning(
                "警告 ⚠️ YOLOWorld（原版）导出不支持任何格式。\n"
                "警告 ⚠️ YOLOWorldv2 模型（即 'yolov8s-worldv2.pt'）仅支持导出为 "
                "(torchscript, onnx, openvino, engine, coreml) 格式。 "
                "详情请参见 https://docs.ultralytics.com/models/yolo-world"
            )
            model.clip_model = None  # openvino int8 导出错误：https://github.com/ultralytics/ultralytics/pull/18445
        if self.args.int8 and not self.args.data:
            self.args.data = DEFAULT_CFG.data or TASK2DATA[getattr(model, "task", "detect")]  # 分配默认数据
            LOGGER.warning(
                "警告 ⚠️ INT8 导出需要缺少的 'data' 参数进行校准。 "
                f"使用默认 'data={self.args.data}'。"
            )

        # 输入
        im = torch.zeros(self.args.batch, 3, *self.imgsz).to(self.device)
        file = Path(
            getattr(model, "pt_path", None) or getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "")
        )
        if file.suffix in {".yaml", ".yml"}:
            file = Path(file.name)

        # 更新模型
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()

        if imx:
            from ultralytics.utils.torch_utils import FXModel

            model = FXModel(model)
        for m in model.modules():
            if isinstance(m, Classify):
                m.export = True
            if isinstance(m, (Detect, RTDETRDecoder)):  # 包括所有 Detect 子类，如 Segment, Pose, OBB
                m.dynamic = self.args.dynamic
                m.export = True
                m.format = self.args.format
                m.max_det = self.args.max_det
            elif isinstance(m, C2f) and not is_tf_format:
                # EdgeTPU 不支持 FlexSplitV，而 split 提供更清晰的 ONNX 图
                m.forward = m.forward_split
            if isinstance(m, Detect) and imx:
                from ultralytics.utils.tal import make_anchors

                m.anchors, m.strides = (
                    x.transpose(0, 1)
                    for x in make_anchors(
                        torch.cat([s / m.stride.unsqueeze(-1) for s in self.imgsz], dim=1), m.stride, 0.5
                    )
                )

        y = None
        for _ in range(2):
            y = model(im)  # 干运行
        if self.args.half and onnx and self.device.type != "cpu":
            im, model = im.half(), model.half()  # 转为 FP16

        # 过滤警告
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # 忽略 TracerWarning
        warnings.filterwarnings("ignore", category=UserWarning)  # 忽略形状 prim::Constant 缺失的 ONNX 警告
        warnings.filterwarnings("ignore", category=DeprecationWarning)  # 忽略 CoreML np.bool 弃用警告

        # 分配
        self.im = im
        self.model = model
        self.file = file
        self.output_shape = (
            tuple(y.shape)
            if isinstance(y, torch.Tensor)
            else tuple(tuple(x.shape if isinstance(x, torch.Tensor) else []) for x in y)
        )
        self.pretty_name = Path(self.model.yaml.get("yaml_file", self.file)).stem.replace("yolo", "YOLO")
        data = model.args["data"] if hasattr(model, "args") and isinstance(model.args, dict) else ""
        description = f"Ultralytics {self.pretty_name} 模型 {f'训练于 {data}' if data else ''}"
        self.metadata = {
            "description": description,
            "author": "Ultralytics",
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 许可证 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "stride": int(max(model.stride)),
            "task": model.task,
            "batch": self.args.batch,
            "imgsz": self.imgsz,
            "names": model.names,
            "args": {k: v for k, v in self.args if k in fmt_keys},
        }  # 模型元数据
        if model.task == "pose":
            self.metadata["kpt_shape"] = model.model[-1].kpt_shape

        LOGGER.info(
            f"\n{colorstr('PyTorch:')} 从 '{file}' 开始，输入形状 {tuple(im.shape)} BCHW 和 "
            f"输出形状 {self.output_shape} ({file_size(file):.1f} MB)"
        )

        # 导出
        f = [""] * len(fmts)  # 导出的文件名
        if jit or ncnn:  # TorchScript
            f[0], _ = self.export_torchscript()
        if engine:  # TensorRT 需要在 ONNX 之前
            f[1], _ = self.export_engine(dla=dla)
        if onnx:  # ONNX
            f[2], _ = self.export_onnx()
        if xml:  # OpenVINO
            f[3], _ = self.export_openvino()
        if coreml:  # CoreML
            f[4], _ = self.export_coreml()
        if is_tf_format:  # TensorFlow 格式
            self.args.int8 |= edgetpu
            f[5], keras_model = self.export_saved_model()
            if pb or tfjs:  # pb 是 tfjs 的前提
                f[6], _ = self.export_pb(keras_model=keras_model)
            if tflite:
                f[7], _ = self.export_tflite(keras_model=keras_model, nms=False, agnostic_nms=self.args.agnostic_nms)
            if edgetpu:
                f[8], _ = self.export_edgetpu(tflite_model=Path(f[5]) / f"{self.file.stem}_full_integer_quant.tflite")
            if tfjs:
                f[9], _ = self.export_tfjs()
        if paddle:  # PaddlePaddle
            f[10], _ = self.export_paddle()
        if mnn:  # MNN
            f[11], _ = self.export_mnn()
        if ncnn:  # NCNN
            f[12], _ = self.export_ncnn()
        if imx:
            f[13], _ = self.export_imx()

        # 完成
        f = [str(x) for x in f if x]  # 过滤掉 '' 和 None
        if any(f):
            f = str(Path(f[-1]))
            square = self.imgsz[0] == self.imgsz[1]
            s = (
                ""
                if square
                else f"警告 ⚠️ 非 PyTorch 验证需要方形图像，'imgsz={self.imgsz}' 无法使用。"
                f"如果需要验证，请使用导出 'imgsz={max(self.imgsz)}'。"
            )
            imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(" ", "")
            predict_data = f"data={data}" if model.task == "segment" and fmt == "pb" else ""
            q = "int8" if self.args.int8 else "half" if self.args.half else ""  # 量化
            LOGGER.info(
                f"\n导出完成 ({time.time() - t:.1f}s)"
                f"\n结果已保存到 {colorstr('bold', file.parent.resolve())}"
                f"\n预测:         yolo predict task={model.task} model={f} imgsz={imgsz} {q} {predict_data}"
                f"\n验证:         yolo val task={model.task} model={f} imgsz={imgsz} data={data} {q} {s}"
                f"\n可视化:       https://netron.app"
            )

        self.run_callbacks("on_export_end")
        return f  # 返回导出文件/目录列表

    def get_int8_calibration_dataloader(self, prefix=""):
        """构建并返回适用于 INT8 模型校准的 dataloader。"""
        LOGGER.info(f"{prefix} 从 'data={self.args.data}' 收集 INT8 校准图像")
        data = (check_cls_dataset if self.model.task == "classify" else check_det_dataset)(self.args.data)
        # TensorRT INT8 校准应使用 2 倍批量大小
        batch = self.args.batch * (2 if self.args.format == "engine" else 1)
        dataset = YOLODataset(
            data[self.args.split or "val"],
            data=data,
            task=self.model.task,
            imgsz=self.imgsz[0],
            augment=False,
            batch_size=batch,
        )
        n = len(dataset)
        if n < self.args.batch:
            raise ValueError(
                f"校准数据集 ({n} 张图像) 必须至少与批量大小一样多（'batch={self.args.batch}'）。"
            )
        elif n < 300:
            LOGGER.warning(f"{prefix} 警告 ⚠️ >300 张图像推荐用于 INT8 校准，找到 {n} 张图像。")
        return build_dataloader(dataset, batch=batch, workers=0)  # 需要批量加载

    @try_export
    def export_torchscript(self, prefix=colorstr("TorchScript:")):
        """YOLO TorchScript 模型导出。"""
        LOGGER.info(f"\n{prefix} 正在使用 torch {torch.__version__} 开始导出...")
        f = self.file.with_suffix(".torchscript")

        ts = torch.jit.trace(self.model, self.im, strict=False)
        extra_files = {"config.txt": json.dumps(self.metadata)}  # torch._C.ExtraFilesMap()
        if self.args.optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            LOGGER.info(f"{prefix} 正在为移动设备进行优化...")
            from torch.utils.mobile_optimizer import optimize_for_mobile

            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        else:
            ts.save(str(f), _extra_files=extra_files)
        return f, None

    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        """YOLO ONNX 导出。"""
        requirements = ["onnx>=1.12.0"]
        if self.args.simplify:
            requirements += ["onnxslim", "onnxruntime" + ("-gpu" if torch.cuda.is_available() else "")]
        check_requirements(requirements)
        import onnx  # noqa

        opset_version = self.args.opset or get_latest_opset()
        LOGGER.info(f"\n{prefix} 正在使用 onnx {onnx.__version__} 和 opset {opset_version} 开始导出...")
        f = str(self.file.with_suffix(".onnx"))

        output_names = ["output0", "output1"] if isinstance(self.model, SegmentationModel) else ["output0"]
        dynamic = self.args.dynamic
        if dynamic:
            dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
            if isinstance(self.model, SegmentationModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 116, 8400)
                dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
            elif isinstance(self.model, DetectionModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 84, 8400)

        torch.onnx.export(
            self.model.cpu() if dynamic else self.model,  # dynamic=True 仅支持 cpu
            self.im.cpu() if dynamic else self.im,
            f,
            verbose=False,
            opset_version=opset_version,
            do_constant_folding=True,  # 注意：torch>=1.12 的 DNN 推理可能需要 do_constant_folding=False
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=dynamic or None,
        )

        # 检查
        model_onnx = onnx.load(f)  # 加载 onnx 模型

        # 简化
        if self.args.simplify:
            try:
                import onnxslim

                LOGGER.info(f"{prefix} 正在使用 onnxslim {onnxslim.__version__} 进行瘦身...")
                model_onnx = onnxslim.slim(model_onnx)

            except Exception as e:
                LOGGER.warning(f"{prefix} 简化失败: {e}")

        # 元数据
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        onnx.save(model_onnx, f)
        return f, model_onnx

    @try_export
    def export_openvino(self, prefix=colorstr("OpenVINO:")):
        """YOLO OpenVINO 导出。"""
        check_requirements("openvino>=2024.5.0")
        import openvino as ov

        LOGGER.info(f"\n{prefix} 正在使用 openvino {ov.__version__} 开始导出...")
        assert TORCH_1_13, f"OpenVINO 导出需要 torch>=1.13.0，但当前安装的 torch 版本为 {torch.__version__}"
        ov_model = ov.convert_model(
            self.model,
            input=None if self.args.dynamic else [self.im.shape],
            example_input=self.im,
        )

        def serialize(ov_model, file):
            """设置运行时信息，序列化并保存元数据 YAML。"""
            ov_model.set_rt_info("YOLO", ["model_info", "model_type"])
            ov_model.set_rt_info(True, ["model_info", "reverse_input_channels"])
            ov_model.set_rt_info(114, ["model_info", "pad_value"])
            ov_model.set_rt_info([255.0], ["model_info", "scale_values"])
            ov_model.set_rt_info(self.args.iou, ["model_info", "iou_threshold"])
            ov_model.set_rt_info([v.replace(" ", "_") for v in self.model.names.values()], ["model_info", "labels"])
            if self.model.task != "classify":
                ov_model.set_rt_info("fit_to_window_letterbox", ["model_info", "resize_type"])

            ov.runtime.save_model(ov_model, file, compress_to_fp16=self.args.half)
            yaml_save(Path(file).parent / "metadata.yaml", self.metadata)  # 添加 metadata.yaml

        if self.args.int8:
            fq = str(self.file).replace(self.file.suffix, f"_int8_openvino_model{os.sep}")
            fq_ov = str(Path(fq) / self.file.with_suffix(".xml").name)
            check_requirements("nncf>=2.14.0")
            import nncf

            def transform_fn(data_item) -> np.ndarray:
                """量化变换函数。"""
                data_item: torch.Tensor = data_item["img"] if isinstance(data_item, dict) else data_item
                assert data_item.dtype == torch.uint8, "输入图像必须是 uint8 类型才能进行量化预处理"
                im = data_item.numpy().astype(np.float32) / 255.0  # uint8 转 fp16/32，并且将 0-255 转为 0.0-1.0
                return np.expand_dims(im, 0) if im.ndim == 3 else im

            # 为整数量化生成校准数据
            ignored_scope = None
            if isinstance(self.model.model[-1], Detect):
                # 包括所有 Detect 子类，如 Segment, Pose, OBB, WorldDetect
                head_module_name = ".".join(list(self.model.named_modules())[-1][0].split(".")[:2])
                ignored_scope = nncf.IgnoredScope(  # 忽略操作
                    patterns=[
                        f".*{head_module_name}/.*/Add",
                        f".*{head_module_name}/.*/Sub*",
                        f".*{head_module_name}/.*/Mul*",
                        f".*{head_module_name}/.*/Div*",
                        f".*{head_module_name}\\.dfl.*",
                    ],
                    types=["Sigmoid"],
                )

            quantized_ov_model = nncf.quantize(
                model=ov_model,
                calibration_dataset=nncf.Dataset(self.get_int8_calibration_dataloader(prefix), transform_fn),
                preset=nncf.QuantizationPreset.MIXED,
                ignored_scope=ignored_scope,
            )
            serialize(quantized_ov_model, fq_ov)
            return fq, None

        f = str(self.file).replace(self.file.suffix, f"_openvino_model{os.sep}")
        f_ov = str(Path(f) / self.file.with_suffix(".xml").name)

        serialize(ov_model, f_ov)
        return f, None

    @try_export
    def export_paddle(self, prefix=colorstr("PaddlePaddle:")):
        """YOLO Paddle 导出。"""
        check_requirements(("paddlepaddle-gpu" if torch.cuda.is_available() else "paddlepaddle", "x2paddle"))
        import x2paddle  # noqa
        from x2paddle.convert import pytorch2paddle  # noqa

        LOGGER.info(f"\n{prefix} 正在使用 X2Paddle {x2paddle.__version__} 开始导出...")
        f = str(self.file).replace(self.file.suffix, f"_paddle_model{os.sep}")

        pytorch2paddle(module=self.model, save_dir=f, jit_type="trace", input_examples=[self.im])  # 导出
        yaml_save(Path(f) / "metadata.yaml", self.metadata)  # 添加 metadata.yaml
        return f, None

    @try_export
    def export_mnn(self, prefix=colorstr("MNN:")):
        """YOLOv8 使用 MNN 导出，使用 MNN：https://github.com/alibaba/MNN。"""
        f_onnx, _ = self.export_onnx()  # 首先导出 onnx 模型

        check_requirements("MNN>=2.9.6")
        import MNN  # noqa
        from MNN.tools import mnnconvert

        # 设置和检查
        LOGGER.info(f"\n{prefix} 正在使用 MNN {MNN.version()} 开始导出...")
        assert Path(f_onnx).exists(), f"导出 ONNX 文件失败: {f_onnx}"
        f = str(self.file.with_suffix(".mnn"))  # MNN 模型文件
        args = ["", "-f", "ONNX", "--modelFile", f_onnx, "--MNNModel", f, "--bizCode", json.dumps(self.metadata)]
        if self.args.int8:
            args.extend(("--weightQuantBits", "8"))
        if self.args.half:
            args.append("--fp16")
        mnnconvert.convert(args)
        # 移除转换优化时的临时文件
        convert_scratch = Path(self.file.parent / ".__convert_external_data.bin")
        if convert_scratch.exists():
            convert_scratch.unlink()
        return f, None

    @try_export
    def export_ncnn(self, prefix=colorstr("NCNN:")):
        """YOLO 使用 PNNX 导出 NCNN：https://github.com/pnnx/pnnx。"""
        check_requirements("ncnn")
        import ncnn  # noqa

        LOGGER.info(f"\n{prefix} 正在使用 NCNN {ncnn.__version__} 开始导出...")
        f = Path(str(self.file).replace(self.file.suffix, f"_ncnn_model{os.sep}"))
        f_ts = self.file.with_suffix(".torchscript")

        name = Path("pnnx.exe" if WINDOWS else "pnnx")  # PNNX 文件名
        pnnx = name if name.is_file() else (ROOT / name)
        if not pnnx.is_file():
            LOGGER.warning(
                f"{prefix} 警告 ⚠️ 未找到 PNNX。正在尝试从 "
                "https://github.com/pnnx/pnnx/ 下载二进制文件。\n注意：PNNX 二进制文件必须放在当前工作目录或 {ROOT} 下。查看 PNNX 仓库获取完整的安装说明。"
            )
            system = "macos" if MACOS else "windows" if WINDOWS else "linux-aarch64" if ARM64 else "linux"
            try:
                release, assets = get_github_assets(repo="pnnx/pnnx")
                asset = [x for x in assets if f"{system}.zip" in x][0]
                assert isinstance(asset, str), "无法获取 PNNX 仓库的资产"  # 即 pnnx-20240410-macos.zip
                LOGGER.info(f"{prefix} 成功找到最新的 PNNX 资产文件 {asset}")
            except Exception as e:
                release = "20240410"
                asset = f"pnnx-{release}-{system}.zip"
                LOGGER.warning(f"{prefix} 警告 ⚠️ 未找到 PNNX GitHub 资产: {e}，使用默认值 {asset}")
            unzip_dir = safe_download(f"https://github.com/pnnx/pnnx/releases/download/{release}/{asset}", delete=True)
            if check_is_path_safe(Path.cwd(), unzip_dir):  # 避免路径遍历安全漏洞
                shutil.move(src=unzip_dir / name, dst=pnnx)  # 将二进制文件移动到 ROOT
                pnnx.chmod(0o777)  # 为所有用户设置读、写和执行权限
                shutil.rmtree(unzip_dir)  # 删除解压目录

        ncnn_args = [
            f"ncnnparam={f / 'model.ncnn.param'}",
            f"ncnnbin={f / 'model.ncnn.bin'}",
            f"ncnnpy={f / 'model_ncnn.py'}",
        ]

        pnnx_args = [
            f"pnnxparam={f / 'model.pnnx.param'}",
            f"pnnxbin={f / 'model.pnnx.bin'}",
            f"pnnxpy={f / 'model_pnnx.py'}",
            f"pnnxonnx={f / 'model.pnnx.onnx'}",
        ]

        cmd = [
            str(pnnx),
            str(f_ts),
            *ncnn_args,
            *pnnx_args,
            f"fp16={int(self.args.half)}",
            f"device={self.device.type}",
            f'inputshape="{[self.args.batch, 3, *self.imgsz]}"',
        ]
        f.mkdir(exist_ok=True)  # 创建 ncnn_model 目录
        LOGGER.info(f"{prefix} 正在运行命令 '{' '.join(cmd)}'")
        subprocess.run(cmd, check=True)

        # 删除调试文件
        pnnx_files = [x.split("=")[-1] for x in pnnx_args]
        for f_debug in ("debug.bin", "debug.param", "debug2.bin", "debug2.param", *pnnx_files):
            Path(f_debug).unlink(missing_ok=True)

        yaml_save(f / "metadata.yaml", self.metadata)  # 添加 metadata.yaml
        return str(f), None

    @try_export
    def export_coreml(self, prefix=colorstr("CoreML:")):
        """YOLO CoreML 导出。"""
        mlmodel = self.args.format.lower() == "mlmodel"  # 请求旧版 *.mlmodel 导出格式
        check_requirements("coremltools>=6.0,<=6.2" if mlmodel else "coremltools>=7.0")
        import coremltools as ct  # noqa

        LOGGER.info(f"\n{prefix} 使用 coremltools {ct.__version__} 开始导出...")
        assert not WINDOWS, "CoreML 导出不支持 Windows，请在 macOS 或 Linux 上运行。"
        assert self.args.batch == 1, "CoreML 不支持批量大小大于 1。请使用 'batch=1' 重试。"
        f = self.file.with_suffix(".mlmodel" if mlmodel else ".mlpackage")
        if f.is_dir():
            shutil.rmtree(f)
        if self.args.nms and getattr(self.model, "end2end", False):
            LOGGER.warning(f"{prefix} 警告 ⚠️ 'nms=True' 不适用于端到端模型，强制设置 'nms=False'。")
            self.args.nms = False

        bias = [0.0, 0.0, 0.0]
        scale = 1 / 255
        classifier_config = None
        if self.model.task == "classify":
            classifier_config = ct.ClassifierConfig(list(self.model.names.values())) if self.args.nms else None
            model = self.model
        elif self.model.task == "detect":
            model = IOSDetectModel(self.model, self.im) if self.args.nms else self.model
        else:
            if self.args.nms:
                LOGGER.warning(f"{prefix} 警告 ⚠️ 'nms=True' 仅适用于检测模型，例如 'yolov8n.pt'。")
                # TODO CoreML 分割和姿态模型流水线
            model = self.model

        ts = torch.jit.trace(model.eval(), self.im, strict=False)  # TorchScript 模型
        ct_model = ct.convert(
            ts,
            inputs=[ct.ImageType("image", shape=self.im.shape, scale=scale, bias=bias)],
            classifier_config=classifier_config,
            convert_to="neuralnetwork" if mlmodel else "mlprogram",
        )
        bits, mode = (8, "kmeans") if self.args.int8 else (16, "linear") if self.args.half else (32, None)
        if bits < 32:
            if "kmeans" in mode:
                check_requirements("scikit-learn")  # 需要 scikit-learn 包用于 k-means 量化
            if mlmodel:
                ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
            elif bits == 8:  # mlprogram 已经量化为 FP16
                import coremltools.optimize.coreml as cto

                op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=bits, weight_threshold=512)
                config = cto.OptimizationConfig(global_config=op_config)
                ct_model = cto.palettize_weights(ct_model, config=config)
        if self.args.nms and self.model.task == "detect":
            if mlmodel:
                # coremltools<=6.2 NMS 导出需要 Python<3.11
                check_version(PYTHON_VERSION, "<3.11", name="Python ", hard=True)
                weights_dir = None
            else:
                ct_model.save(str(f))  # 否则保存，因为 weights_dir 不存在
                weights_dir = str(f / "Data/com.apple.CoreML/weights")
            ct_model = self._pipeline_coreml(ct_model, weights_dir=weights_dir)

        m = self.metadata  # 元数据字典
        ct_model.short_description = m.pop("description")
        ct_model.author = m.pop("author")
        ct_model.license = m.pop("license")
        ct_model.version = m.pop("version")
        ct_model.user_defined_metadata.update({k: str(v) for k, v in m.items()})
        try:
            ct_model.save(str(f))  # 保存 *.mlpackage
        except Exception as e:
            LOGGER.warning(
                f"{prefix} 警告 ⚠️ CoreML 导出到 *.mlpackage 失败 ({e})，正在回退到 *.mlmodel 导出。"
                f"已知的 coremltools Python 3.11 和 Windows 错误 https://github.com/apple/coremltools/issues/1928."
            )
            f = f.with_suffix(".mlmodel")
            ct_model.save(str(f))
        return f, ct_model

    @try_export
    def export_engine(self, dla=None, prefix=colorstr("TensorRT:")):
        """YOLO TensorRT 导出 https://developer.nvidia.com/tensorrt。"""
        assert self.im.device.type != "cpu", "导出在 CPU 上运行，但必须在 GPU 上运行，即使用 'device=0'"
        f_onnx, _ = self.export_onnx()  # 在 TRT 导入之前运行 https://github.com/ultralytics/ultralytics/issues/7016

        try:
            import tensorrt as trt  # noqa
        except ImportError:
            if LINUX:
                check_requirements("tensorrt>7.0.0,!=10.1.0")
            import tensorrt as trt  # noqa
        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")

        # 设置和检查
        LOGGER.info(f"\n{prefix} 使用 TensorRT {trt.__version__} 开始导出...")
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # 是否是 TensorRT >= 10
        assert Path(f_onnx).exists(), f"导出 ONNX 文件失败: {f_onnx}"
        f = self.file.with_suffix(".engine")  # TensorRT 引擎文件
        logger = trt.Logger(trt.Logger.INFO)
        if self.args.verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        # 引擎构建器
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        workspace = int(self.args.workspace * (1 << 30)) if self.args.workspace is not None else 0
        if is_trt10 and workspace > 0:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        elif workspace > 0:  # TensorRT 版本 7, 8
            config.max_workspace_size = workspace
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        half = builder.platform_has_fast_fp16 and self.args.half
        int8 = builder.platform_has_fast_int8 and self.args.int8

        # 可选启用 DLA（深度学习加速器）
        if dla is not None:
            if not IS_JETSON:
                raise ValueError("DLA 仅在 NVIDIA Jetson 设备上可用")
            LOGGER.info(f"{prefix} 启用 DLA 在核心 {dla} 上运行...")
            if not self.args.half and not self.args.int8:
                raise ValueError(
                    "DLA 需要启用 'half=True' (FP16) 或 'int8=True' (INT8)。请启用其中一个选项并重试。"
                )
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = int(dla)
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        # 读取 ONNX 文件
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(f_onnx):
            raise RuntimeError(f"加载 ONNX 文件失败: {f_onnx}")

        # 网络输入
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            LOGGER.info(f'{prefix} 输入 "{inp.name}" 形状{inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix} 输出 "{out.name}" 形状{out.shape} {out.dtype}')

        if self.args.dynamic:
            shape = self.im.shape
            if shape[0] <= 1:
                LOGGER.warning(f"{prefix} 警告 ⚠️ 'dynamic=True' 模型需要最大批次大小，即 'batch=16'")
            profile = builder.create_optimization_profile()
            min_shape = (1, shape[1], 32, 32)  # 最小输入形状
            max_shape = (*shape[:2], *(int(max(1, workspace) * d) for d in shape[2:]))  # 最大输入形状
            for inp in inputs:
                profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
            config.add_optimization_profile(profile)

        LOGGER.info(f"{prefix} 正在构建 {'INT8' if int8 else 'FP' + ('16' if half else '32')} 引擎，输出为 {f}")
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_calibration_profile(profile)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

            class EngineCalibrator(trt.IInt8Calibrator):
                def __init__(
                    self,
                    dataset,  # ultralytics.data.build.InfiniteDataLoader
                    batch: int,
                    cache: str = "",
                ) -> None:
                    trt.IInt8Calibrator.__init__(self)
                    self.dataset = dataset
                    self.data_iter = iter(dataset)
                    self.algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
                    self.batch = batch
                    self.cache = Path(cache)

                def get_algorithm(self) -> trt.CalibrationAlgoType:
                    """获取要使用的校准算法。"""
                    return self.algo

                def get_batch_size(self) -> int:
                    """获取要使用的批次大小。"""
                    return self.batch or 1

                def get_batch(self, names) -> list:
                    """获取下一个批次，用于校准，作为设备内存指针的列表。"""
                    try:
                        im0s = next(self.data_iter)["img"] / 255.0
                        im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s
                        return [int(im0s.data_ptr())]
                    except StopIteration:
                        # 返回 [] 或 None，向 TensorRT 信号没有剩余的校准数据
                        return None

                def read_calibration_cache(self) -> bytes:
                    """使用现有缓存而不是重新校准，否则隐式返回 None。"""
                    if self.cache.exists() and self.cache.suffix == ".cache":
                        return self.cache.read_bytes()

                def write_calibration_cache(self, cache) -> None:
                    """将校准缓存写入磁盘。"""
                    _ = self.cache.write_bytes(cache)

            # 使用构建器加载数据集（用于批处理）并进行校准
            config.int8_calibrator = EngineCalibrator(
                dataset=self.get_int8_calibration_dataloader(prefix),
                batch=2 * self.args.batch,  # TensorRT INT8 校准应该使用 2 倍批次大小
                cache=str(self.file.with_suffix(".cache")),

            )

        elif half:
            config.set_flag(trt.BuilderFlag.FP16)

        # 释放 CUDA 内存
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        # 写入文件
        build = builder.build_serialized_network if is_trt10 else builder.build_engine
        with build(network, config) as engine, open(f, "wb") as t:
            # 元数据
            meta = json.dumps(self.metadata)
            t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
            t.write(meta.encode())
            # 模型
            t.write(engine if is_trt10 else engine.serialize())

        return f, None

    @try_export
    def export_saved_model(self, prefix=colorstr("TensorFlow SavedModel:")):
        """YOLO TensorFlow SavedModel 导出。"""
        cuda = torch.cuda.is_available()
        try:
            import tensorflow as tf  # noqa
        except ImportError:
            suffix = "-macos" if MACOS else "-aarch64" if ARM64 else "" if cuda else "-cpu"
            version = ">=2.0.0"
            check_requirements(f"tensorflow{suffix}{version}")
            import tensorflow as tf  # noqa
        check_requirements(
            (
                "keras",  # 'onnx2tf' 包所需
                "tf_keras",  # 'onnx2tf' 包所需
                "sng4onnx>=1.0.1",  # 'onnx2tf' 包所需
                "onnx_graphsurgeon>=0.3.26",  # 'onnx2tf' 包所需
                "onnx>=1.12.0",
                "onnx2tf>1.17.5,<=1.26.3",
                "onnxslim>=0.1.31",
                "tflite_support<=0.4.3" if IS_JETSON else "tflite_support",  # 修复 ImportError 'GLIBCXX_3.4.29'
                "flatbuffers>=23.5.26,<100",  # 更新 tensorflow 包中包含的旧 'flatbuffers'
                "onnxruntime-gpu" if cuda else "onnxruntime",
            ),
            cmds="--extra-index-url https://pypi.ngc.nvidia.com",  # 仅在 NVIDIA 上使用 onnx_graphsurgeon
        )

        LOGGER.info(f"\n{prefix} 正在使用 tensorflow {tf.__version__} 开始导出...")
        check_version(
            tf.__version__,
            ">=2.0.0",
            name="tensorflow",
            verbose=True,
            msg="https://github.com/ultralytics/ultralytics/issues/5161",
        )
        import onnx2tf

        f = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if f.is_dir():
            shutil.rmtree(f)  # 删除输出文件夹

        # 预下载校准文件以修复 https://github.com/PINTO0309/onnx2tf/issues/545
        onnx2tf_file = Path("calibration_image_sample_data_20x128x128x3_float32.npy")
        if not onnx2tf_file.exists():
            attempt_download_asset(f"{onnx2tf_file}.zip", unzip=True, delete=True)

        # 导出到 ONNX
        self.args.simplify = True
        f_onnx, _ = self.export_onnx()

        # 导出到 TensorFlow
        np_data = None
        if self.args.int8:
            tmp_file = f / "tmp_tflite_int8_calibration_images.npy"  # int8 校准图像文件
            if self.args.data:
                f.mkdir()
                images = [batch["img"] for batch in self.get_int8_calibration_dataloader(prefix)]
                images = torch.nn.functional.interpolate(torch.cat(images, 0).float(), size=self.imgsz).permute(
                    0, 2, 3, 1
                )
                np.save(str(tmp_file), images.numpy().astype(np.float32))  # BHWC
                np_data = [["images", tmp_file, [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]]

        LOGGER.info(f"{prefix} 正在使用 onnx2tf {onnx2tf.__version__} 开始 TFLite 导出...")
        keras_model = onnx2tf.convert(
            input_onnx_file_path=f_onnx,
            output_folder_path=str(f),
            not_use_onnxsim=True,
            verbosity="error",  # 注意 INT8-FP16 激活问题 https://github.com/ultralytics/ultralytics/issues/15873
            output_integer_quantized_tflite=self.args.int8,
            quant_type="per-tensor",  # "per-tensor"（更快）或 "per-channel"（较慢但更准确）
            custom_input_op_name_np_data_path=np_data,
            disable_group_convolution=True,  # 为了与端到端模型兼容
            enable_batchmatmul_unfold=True,  # 为了与端到端模型兼容
        )
        yaml_save(f / "metadata.yaml", self.metadata)  # 添加 metadata.yaml

        # 删除/重命名 TFLite 模型
        if self.args.int8:
            tmp_file.unlink(missing_ok=True)
            for file in f.rglob("*_dynamic_range_quant.tflite"):
                file.rename(file.with_name(file.stem.replace("_dynamic_range_quant", "_int8") + file.suffix))
            for file in f.rglob("*_integer_quant_with_int16_act.tflite"):
                file.unlink()  # 删除额外的 fp16 激活 TFLite 文件

        # 添加 TFLite 元数据
        for file in f.rglob("*.tflite"):
            f.unlink() if "quant_with_int16_act.tflite" in str(f) else self._add_tflite_metadata(file)

        return str(f), keras_model  # 或 keras_model = tf.saved_model.load(f, tags=None, options=None)

    @try_export
    def export_pb(self, keras_model, prefix=colorstr("TensorFlow GraphDef:")):
        """YOLO TensorFlow GraphDef *.pb 导出 https://github.com/leimao/Frozen_Graph_TensorFlow。"""
        import tensorflow as tf  # noqa
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # noqa

        LOGGER.info(f"\n{prefix} 正在使用 tensorflow {tf.__version__} 开始导出...")
        f = self.file.with_suffix(".pb")

        m = tf.function(lambda x: keras_model(x))  # 完整模型
        m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(m)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)
        return f, None

    @try_export
    def export_tflite(self, keras_model, nms, agnostic_nms, prefix=colorstr("TensorFlow Lite:")):
        """YOLO TensorFlow Lite 导出。"""
        # BUG https://github.com/ultralytics/ultralytics/issues/13436
        import tensorflow as tf  # noqa

        LOGGER.info(f"\n{prefix} 正在使用 tensorflow {tf.__version__} 开始导出...")
        saved_model = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if self.args.int8:
            f = saved_model / f"{self.file.stem}_int8.tflite"  # fp32 输入/输出
        elif self.args.half:
            f = saved_model / f"{self.file.stem}_float16.tflite"  # fp32 输入/输出
        else:
            f = saved_model / f"{self.file.stem}_float32.tflite"
        return str(f), None

    @try_export
    def export_edgetpu(self, tflite_model="", prefix=colorstr("Edge TPU:")):
        """YOLO Edge TPU 导出 https://coral.ai/docs/edgetpu/models-intro/。"""
        LOGGER.warning(f"{prefix} 警告 ⚠️ Edge TPU 已知问题 https://github.com/ultralytics/ultralytics/issues/1185")

        cmd = "edgetpu_compiler --version"
        help_url = "https://coral.ai/docs/edgetpu/compiler/"
        assert LINUX, f"仅支持在 Linux 上导出。请参阅 {help_url}"
        if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True).returncode != 0:
            LOGGER.info(f"\n{prefix} 导出需要 Edge TPU 编译器。尝试从 {help_url} 安装")
            for c in (
                "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -",
                'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | '
                "sudo tee /etc/apt/sources.list.d/coral-edgetpu.list",
                "sudo apt-get update",
                "sudo apt-get install edgetpu-compiler",
            ):
                subprocess.run(c if is_sudo_available() else c.replace("sudo ", ""), shell=True, check=True)
        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

        LOGGER.info(f"\n{prefix} 正在使用 Edge TPU 编译器 {ver} 开始导出...")
        f = str(tflite_model).replace(".tflite", "_edgetpu.tflite")  # Edge TPU 模型

        cmd = (
            "edgetpu_compiler "
            f'--out_dir "{Path(f).parent}" '
            "--show_operations "
            "--search_delegate "
            "--delegate_search_step 30 "
            "--timeout_sec 180 "
            f'"{tflite_model}"'
        )
        LOGGER.info(f"{prefix} 正在运行命令 '{cmd}'")
        subprocess.run(cmd, shell=True)
        self._add_tflite_metadata(f)
        return f, None

    @try_export
    def export_tfjs(self, prefix=colorstr("TensorFlow.js:")):
        """YOLO TensorFlow.js 导出。"""
        check_requirements("tensorflowjs")
        if ARM64:
            # 修复错误：`np.object` 是导出到 TF.js 时对 ARM64 的过时别名
            check_requirements("numpy==1.23.5")
        import tensorflow as tf
        import tensorflowjs as tfjs  # noqa

        LOGGER.info(f"\n{prefix} 使用 tensorflowjs {tfjs.__version__} 开始导出...")
        f = str(self.file).replace(self.file.suffix, "_web_model")  # js 目录
        f_pb = str(self.file.with_suffix(".pb"))  # *.pb 路径

        gd = tf.Graph().as_graph_def()  # TF GraphDef
        with open(f_pb, "rb") as file:
            gd.ParseFromString(file.read())
        outputs = ",".join(gd_outputs(gd))
        LOGGER.info(f"\n{prefix} 输出节点名称：{outputs}")

        quantization = "--quantize_float16" if self.args.half else "--quantize_uint8" if self.args.int8 else ""
        with spaces_in_path(f_pb) as fpb_, spaces_in_path(f) as f_:  # 导出器不能处理路径中的空格
            cmd = (
                "tensorflowjs_converter "
                f'--input_format=tf_frozen_model {quantization} --output_node_names={outputs} "{fpb_}" "{f_}"'
            )
            LOGGER.info(f"{prefix} 运行 '{cmd}'")
            subprocess.run(cmd, shell=True)

        if " " in f:
            LOGGER.warning(f"{prefix} 警告 ⚠️ 你的模型可能因为路径中包含空格而无法正常工作 '{f}'。")

        # 添加元数据
        yaml_save(Path(f) / "metadata.yaml", self.metadata)  # 添加 metadata.yaml
        return f, None

    @try_export
    def export_imx(self, prefix=colorstr("IMX:")):
        """YOLO IMX 导出。"""
        gptq = False
        assert LINUX, (
            "仅在 Linux 上支持导出。请参阅 https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/documentation/imx500-converter"
        )
        if getattr(self.model, "end2end", False):
            raise ValueError("IMX 导出不支持 end2end 模型。")
        if "C2f" not in self.model.__str__():
            raise ValueError("IMX 导出仅支持 YOLOv8n 检测模型")
        check_requirements(("model-compression-toolkit==2.1.1", "sony-custom-layers==0.2.0", "tensorflow==2.12.0"))
        check_requirements("imx500-converter[pt]==3.14.3")  # imx500-converter 的独立要求

        import model_compression_toolkit as mct
        import onnx
        from sony_custom_layers.pytorch.object_detection.nms import multiclass_nms

        try:
            out = subprocess.run(
                ["java", "--version"], check=True, capture_output=True
            )  # imx500-converter 需要 Java 17
            if "openjdk 17" not in str(out.stdout):
                raise FileNotFoundError
        except FileNotFoundError:
            c = ["apt", "install", "-y", "openjdk-17-jdk", "openjdk-17-jre"]
            if is_sudo_available():
                c.insert(0, "sudo")
            subprocess.run(c, check=True)

        def representative_dataset_gen(dataloader=self.get_int8_calibration_dataloader(prefix)):
            for batch in dataloader:
                img = batch["img"]
                img = img / 255.0
                yield [img]

        tpc = mct.get_target_platform_capabilities(
            fw_name="pytorch", target_platform_name="imx500", target_platform_version="v1"
        )

        config = mct.core.CoreConfig(
            mixed_precision_config=mct.core.MixedPrecisionQuantizationConfig(num_of_images=10),
            quantization_config=mct.core.QuantizationConfig(concat_threshold_update=True),
        )

        resource_utilization = mct.core.ResourceUtilization(weights_memory=3146176 * 0.76)

        quant_model = (
            mct.gptq.pytorch_gradient_post_training_quantization(  # 执行基于梯度的后训练量化
                model=self.model,
                representative_data_gen=representative_dataset_gen,
                target_resource_utilization=resource_utilization,
                gptq_config=mct.gptq.get_pytorch_gptq_config(n_epochs=1000, use_hessian_based_weights=False),
                core_config=config,
                target_platform_capabilities=tpc,
            )[0]
            if gptq
            else mct.ptq.pytorch_post_training_quantization(  # 执行后训练量化
                in_module=self.model,
                representative_data_gen=representative_dataset_gen,
                target_resource_utilization=resource_utilization,
                core_config=config,
                target_platform_capabilities=tpc,
            )[0]
        )

        class NMSWrapper(torch.nn.Module):
            def __init__(self,
                         model: torch.nn.Module,
                         score_threshold: float = 0.001,
                         iou_threshold: float = 0.7,
                         max_detections: int = 300,
                         ):
                """
                用 sony_custom_layers 中的 multiclass_nms 层包装 PyTorch 模块。

                参数：
                    model (nn.Module): 模型实例。
                    score_threshold (float): 非最大抑制的得分阈值。
                    iou_threshold (float): 非最大抑制的交并比阈值。
                    max_detections (float): 返回的最大检测数。
                """
                super().__init__()
                self.model = model
                self.score_threshold = score_threshold
                self.iou_threshold = iou_threshold
                self.max_detections = max_detections

            def forward(self, images):
                # 模型推理
                outputs = self.model(images)

                boxes = outputs[0]
                scores = outputs[1]
                nms = multiclass_nms(
                    boxes=boxes,
                    scores=scores,
                    score_threshold=self.score_threshold,
                    iou_threshold=self.iou_threshold,
                    max_detections=self.max_detections,
                )
                return nms

        quant_model = NMSWrapper(
            model=quant_model,
            score_threshold=self.args.conf or 0.001,
            iou_threshold=self.args.iou,
            max_detections=self.args.max_det,
        ).to(self.device)

        f = Path(str(self.file).replace(self.file.suffix, "_imx_model"))
        f.mkdir(exist_ok=True)
        onnx_model = f / Path(str(self.file).replace(self.file.suffix, "_imx.onnx"))  # js 目录
        mct.exporter.pytorch_export_model(
            model=quant_model, save_model_path=onnx_model, repr_dataset=representative_dataset_gen
        )

        model_onnx = onnx.load(onnx_model)  # 加载 onnx 模型
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        onnx.save(model_onnx, onnx_model)

        subprocess.run(
            ["imxconv-pt", "-i", str(onnx_model), "-o", str(f), "--no-input-persistency", "--overwrite-output"],
            check=True,
        )

        # IMX 模型需要的操作。
        with open(f / "labels.txt", "w") as file:
            file.writelines([f"{name}\n" for _, name in self.model.names.items()])

        return f, None

    def _add_tflite_metadata(self, file):
        """根据 https://www.tensorflow.org/lite/models/convert/metadata 为 *.tflite 模型添加元数据。"""
        import flatbuffers

        try:
            # TFLite 支持 bug https://github.com/tensorflow/tflite-support/issues/954#issuecomment-2108570845
            from tensorflow_lite_support.metadata import metadata_schema_py_generated as schema  # noqa
            from tensorflow_lite_support.metadata.python import metadata  # noqa
        except ImportError:  # ARM64 系统可能没有 'tensorflow_lite_support' 包
            from tflite_support import metadata  # noqa
            from tflite_support import metadata_schema_py_generated as schema  # noqa

        # 创建模型信息
        model_meta = schema.ModelMetadataT()
        model_meta.name = self.metadata["description"]
        model_meta.version = self.metadata["version"]
        model_meta.author = self.metadata["author"]
        model_meta.license = self.metadata["license"]

        # 标签文件
        tmp_file = Path(file).parent / "temp_meta.txt"
        with open(tmp_file, "w") as f:
            f.write(str(self.metadata))

        label_file = schema.AssociatedFileT()
        label_file.name = tmp_file.name
        label_file.type = schema.AssociatedFileType.TENSOR_AXIS_LABELS

        # 创建输入信息
        input_meta = schema.TensorMetadataT()
        input_meta.name = "image"
        input_meta.description = "输入的待检测图像。"
        input_meta.content = schema.ContentT()
        input_meta.content.contentProperties = schema.ImagePropertiesT()
        input_meta.content.contentProperties.colorSpace = schema.ColorSpaceType.RGB
        input_meta.content.contentPropertiesType = schema.ContentProperties.ImageProperties

        # 创建输出信息
        output1 = schema.TensorMetadataT()
        output1.name = "output"
        output1.description = "检测到的对象的坐标、类别标签和置信度"
        output1.associatedFiles = [label_file]
        if self.model.task == "segment":
            output2 = schema.TensorMetadataT()
            output2.name = "output"
            output2.description = "掩码原型"
            output2.associatedFiles = [label_file]

        # 创建子图信息
        subgraph = schema.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [input_meta]
        subgraph.outputTensorMetadata = [output1, output2] if self.model.task == "segment" else [output1]
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(model_meta.Pack(b), metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        populator = metadata.MetadataPopulator.with_model_file(str(file))
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files([str(tmp_file)])
        populator.populate()
        tmp_file.unlink()

    def _pipeline_coreml(self, model, weights_dir=None, prefix=colorstr("CoreML Pipeline:")):
        """YOLO CoreML 流水线。"""
        import coremltools as ct  # noqa

        LOGGER.info(f"{prefix} 使用 coremltools {ct.__version__} 开始流水线...")
        _, _, h, w = list(self.im.shape)  # BCHW

        # 输出形状
        spec = model.get_spec()
        out0, out1 = iter(spec.description.output)
        if MACOS:
            from PIL import Image

            img = Image.new("RGB", (w, h))  # w=192, h=320
            out = model.predict({"image": img})
            out0_shape = out[out0.name].shape  # (3780, 80)
            out1_shape = out[out1.name].shape  # (3780, 4)
        else:  # Linux 和 Windows 无法运行 model.predict()，从 PyTorch 模型输出 y 获取大小
            out0_shape = self.output_shape[2], self.output_shape[1] - 4  # (3780, 80)
            out1_shape = self.output_shape[2], 4  # (3780, 4)

        # 检查
        names = self.metadata["names"]
        nx, ny = spec.description.input[0].type.imageType.width, spec.description.input[0].type.imageType.height
        _, nc = out0_shape  # 锚点数，类别数
        assert len(names) == nc, f"{len(names)} 个名字与 nc={nc} 不匹配"  # 检查

        # 定义输出形状（缺失）
        out0.type.multiArrayType.shape[:] = out0_shape  # (3780, 80)
        out1.type.multiArrayType.shape[:] = out1_shape  # (3780, 4)

        # 从 spec 创建模型
        model = ct.models.MLModel(spec, weights_dir=weights_dir)

        # 3. 创建 NMS protobuf
        nms_spec = ct.proto.Model_pb2.Model()
        nms_spec.specificationVersion = 5
        for i in range(2):
            decoder_output = model._spec.description.output[i].SerializeToString()
            nms_spec.description.input.add()
            nms_spec.description.input[i].ParseFromString(decoder_output)
            nms_spec.description.output.add()
            nms_spec.description.output[i].ParseFromString(decoder_output)

        nms_spec.description.output[0].name = "confidence"
        nms_spec.description.output[1].name = "coordinates"

        output_sizes = [nc, 4]
        for i in range(2):
            ma_type = nms_spec.description.output[i].type.multiArrayType
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[0].lowerBound = 0
            ma_type.shapeRange.sizeRanges[0].upperBound = -1
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
            ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
            del ma_type.shape[:]

        nms = nms_spec.nonMaximumSuppression
        nms.confidenceInputFeatureName = out0.name  # 1x507x80
        nms.coordinatesInputFeatureName = out1.name  # 1x507x4
        nms.confidenceOutputFeatureName = "confidence"
        nms.coordinatesOutputFeatureName = "coordinates"
        nms.iouThresholdInputFeatureName = "iouThreshold"
        nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
        nms.iouThreshold = 0.45
        nms.confidenceThreshold = 0.25
        nms.pickTop.perClass = True
        nms.stringClassLabels.vector.extend(names.values())
        nms_model = ct.models.MLModel(nms_spec)

        # 4. 流水线模型
        pipeline = ct.models.pipeline.Pipeline(
            input_features=[
                ("image", ct.models.datatypes.Array(3, ny, nx)),
                ("iouThreshold", ct.models.datatypes.Double()),
                ("confidenceThreshold", ct.models.datatypes.Double()),
            ],
            output_features=["confidence", "coordinates"],
        )
        pipeline.add_model(model)
        pipeline.add_model(nms_model)

        # 修正数据类型
        pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
        pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
        pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

        # 更新元数据
        pipeline.spec.specificationVersion = 5
        pipeline.spec.description.metadata.userDefined.update(
            {"IoU 阈值": str(nms.iouThreshold), "置信度阈值": str(nms.confidenceThreshold)}
        )

        # 保存模型
        model = ct.models.MLModel(pipeline.spec, weights_dir=weights_dir)
        model.input_description["image"] = "输入图像"
        model.input_description["iouThreshold"] = f"(可选) IoU 阈值覆盖（默认：{nms.iouThreshold}）"
        model.input_description["confidenceThreshold"] = (
            f"(可选) 置信度阈值覆盖（默认：{nms.confidenceThreshold}）"
        )
        model.output_description["confidence"] = '框 × 类别置信度（见用户定义的元数据 "classes"）'
        model.output_description["coordinates"] = "框 × [x, y, 宽度, 高度]（相对于图像大小）"
        LOGGER.info(f"{prefix} 流水线成功")
        return model

    def add_callback(self, event: str, callback):
        """追加给定的回调函数。"""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """执行给定事件的所有回调函数。"""
        for callback in self.callbacks.get(event, []):
            callback(self)


class IOSDetectModel(torch.nn.Module):
    """包装一个 Ultralytics YOLO 模型，用于 Apple iOS CoreML 导出。"""

    def __init__(self, model, im):
        """初始化 IOSDetectModel 类，接受 YOLO 模型和示例图像。"""
        super().__init__()
        _, _, h, w = im.shape  # 批次，通道，高度，宽度
        self.model = model
        self.nc = len(model.names)  # 类别数
        if w == h:
            self.normalize = 1.0 / w  # 缩放因子
        else:
            self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])  # 广播（较慢，较小）

    def forward(self, x):
        """根据输入尺寸的相关因素，对目标检测模型的预测进行归一化。"""
        xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
        return cls, xywh * self.normalize  # 置信度 (3780, 80), 坐标 (3780, 4)
