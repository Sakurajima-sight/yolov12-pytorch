# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import gc
import math
import os
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import thop
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
    LOGGER,
    NUM_THREADS,
    PYTHON_VERSION,
    TORCHVISION_VERSION,
    WINDOWS,
    __version__,
    colorstr,
)
from ultralytics.utils.checks import check_version

# 版本检查（默认检查是否大于等于最小版本）
TORCH_1_9 = check_version(torch.__version__, "1.9.0")
TORCH_1_13 = check_version(torch.__version__, "1.13.0")
TORCH_2_0 = check_version(torch.__version__, "2.0.0")
TORCH_2_4 = check_version(torch.__version__, "2.4.0")
TORCHVISION_0_10 = check_version(TORCHVISION_VERSION, "0.10.0")
TORCHVISION_0_11 = check_version(TORCHVISION_VERSION, "0.11.0")
TORCHVISION_0_13 = check_version(TORCHVISION_VERSION, "0.13.0")
TORCHVISION_0_18 = check_version(TORCHVISION_VERSION, "0.18.0")
if WINDOWS and check_version(torch.__version__, "==2.4.0"):  # 在 Windows 上拒绝使用版本 2.4.0
    LOGGER.warning(
        "警告 ⚠️ Windows 上的 torch==2.4.0 存在已知问题，建议升级到 torch>=2.4.1 以解决 "
        "https://github.com/ultralytics/ultralytics/issues/15049"
    )


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """确保分布式训练中的所有进程等待本地主节点（rank 0）先完成任务。"""
    initialized = dist.is_available() and dist.is_initialized()

    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[local_rank])


def smart_inference_mode():
    """如果 torch>=1.9.0 则应用 torch.inference_mode() 装饰器，否则应用 torch.no_grad() 装饰器。"""

    def decorate(fn):
        """根据 PyTorch 版本应用适当的推理模式装饰器。"""
        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn  # 已在 inference_mode 中，直接返回函数
        else:
            return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)

    return decorate


def autocast(enabled: bool, device: str = "cuda"):
    """
    获取适合 PyTorch 版本和 AMP 设置的 autocast 上下文管理器。

    该函数返回一个适合自动混合精度（AMP）训练的上下文管理器，兼容较旧和较新的 PyTorch 版本。它处理 PyTorch 版本之间 autocast API 的差异。

    参数：
        enabled (bool): 是否启用自动混合精度。
        device (str, 可选): 用于自动混合精度的设备。默认为 'cuda'。

    返回：
        (torch.amp.autocast): 适当的 autocast 上下文管理器。

    注意：
        - 对于 PyTorch 版本 1.13 及更高版本，使用 `torch.amp.autocast`。
        - 对于较旧版本，使用 `torch.cuda.autocast`。

    示例：
        ```python
        with autocast(amp=True):
            # 在此处进行混合精度操作
            pass
        ```
    """
    if TORCH_1_13:
        return torch.amp.autocast(device, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled)


def get_cpu_info():
    """返回系统的 CPU 信息字符串，例如 'Apple M2'。"""
    from ultralytics.utils import PERSISTENT_CACHE  # 避免循环导入错误

    if "cpu_info" not in PERSISTENT_CACHE:
        try:
            import cpuinfo  # pip install py-cpuinfo

            k = "brand_raw", "hardware_raw", "arch_string_raw"  # 按优先级排序的键
            info = cpuinfo.get_cpu_info()  # 信息字典
            string = info.get(k[0] if k[0] in info else k[1] if k[1] in info else k[2], "unknown")
            PERSISTENT_CACHE["cpu_info"] = string.replace("(R)", "").replace("CPU ", "").replace("@ ", "")
        except Exception:
            pass
    return PERSISTENT_CACHE.get("cpu_info", "unknown")


def get_gpu_info(index):
    """返回系统的 GPU 信息字符串，例如 'Tesla T4, 15102MiB'。"""
    properties = torch.cuda.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"


def select_device(device="", batch=0, newline=False, verbose=True):
    """
    根据提供的参数选择适当的PyTorch设备。

    该函数接受一个指定设备的字符串或torch.device对象，并返回一个表示所选设备的torch.device对象。
    该函数还会验证可用设备的数量，如果请求的设备不可用，则会抛出异常。

    参数:
        device (str | torch.device, 可选): 设备字符串或torch.device对象。
            选项包括 'None', 'cpu', 或 'cuda', 或 '0' 或 '0,1,2,3'。默认为空字符串，自动选择
            第一个可用的GPU，如果没有GPU则选择CPU。
        batch (int, 可选): 模型使用的批次大小。默认为0。
        newline (bool, 可选): 如果为True，在日志字符串末尾添加换行符。默认为False。
        verbose (bool, 可选): 如果为True，则记录设备信息。默认为True。

    返回:
        (torch.device): 选择的设备。

    异常:
        ValueError: 如果指定的设备不可用，或者在使用多个GPU时批次大小不是设备数量的倍数。

    示例:
        >>> select_device("cuda:0")
        device(type='cuda', index=0)

        >>> select_device("cpu")
        device(type='cpu')

    注意:
        设置 'CUDA_VISIBLE_DEVICES' 环境变量来指定使用哪些GPU。
    """
    if isinstance(device, torch.device) or str(device).startswith("tpu"):
        return device

    s = f"Ultralytics {__version__} 🚀 Python-{PYTHON_VERSION} torch-{torch.__version__} "
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")  # 转为字符串，'cuda:0' -> '0' 和 '(0, 1)' -> '0,1'
    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制 torch.cuda.is_available() = False
    elif device:  # 请求非CPU设备
        if device == "cuda":
            device = "0"
        if "," in device:
            device = ",".join([x for x in device.split(",") if x])  # 删除多余的逗号，比如 "0,,1" -> "0,1"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # 设置环境变量 - 必须在验证是否可用之前
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            LOGGER.info(s)
            install = (
                "如果torch没有看到CUDA设备，请访问 https://pytorch.org/get-started/locally/ 获取最新的torch安装说明。\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"请求的CUDA 'device={device}' 无效。"
                f" 使用 'device=cpu' 或者提供有效的CUDA设备，例如 'device=0' 或 'device=0,1,2,3' 进行多GPU训练。\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                f"{install}"
            )

    if not cpu and not mps and torch.cuda.is_available():  # 如果GPU可用，优先选择GPU
        devices = device.split(",") if device else "0"  # 例如 "0,1" -> ["0", "1"]
        n = len(devices)  # 设备数量
        if n > 1:  # 多GPU
            if batch < 1:
                raise ValueError(
                    "Multi-GPU训练不支持批次小于1，请指定一个有效的批次大小，例如 batch=16。"
                )
            if batch >= 0 and batch % n != 0:  # 检查批次大小是否能被设备数量整除
                raise ValueError(
                    f"'batch={batch}' 必须是GPU数量 {n} 的倍数。请尝试 'batch={batch // n * n}' 或 "
                    f"'batch={batch // n * n + n}'，这两个批次大小能被 {n} 整除。"
                )
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            s += f"{'' if i == 0 else space}CUDA:{d} ({get_gpu_info(i)})\n"  # 字节到MB的转换
        arg = "cuda:0"
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():
        # 如果MPS可用，优先使用MPS
        s += f"MPS ({get_cpu_info()})\n"
        arg = "mps"
    else:  # 否则退回到CPU
        s += f"CPU ({get_cpu_info()})\n"
        arg = "cpu"

    if arg in {"cpu", "mps"}:
        torch.set_num_threads(NUM_THREADS)  # 为CPU训练重置OMP_NUM_THREADS
    if verbose:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)


def time_sync():
    """PyTorch精确时间。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def fuse_conv_and_bn(conv, bn):
    """融合Conv2d()和BatchNorm2d()层 https://tehnokv.com/posts/fusing-batchnorm-and-conv/。"""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # 准备卷积滤波器
    w_conv = conv.weight.view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备空间偏置
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_deconv_and_bn(deconv, bn):
    """融合 ConvTranspose2d() 和 BatchNorm2d() 层。"""
    fuseddconv = (
        nn.ConvTranspose2d(
            deconv.in_channels,
            deconv.out_channels,
            kernel_size=deconv.kernel_size,
            stride=deconv.stride,
            padding=deconv.padding,
            output_padding=deconv.output_padding,
            dilation=deconv.dilation,
            groups=deconv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(deconv.weight.device)
    )

    # 准备卷积滤波器
    w_deconv = deconv.weight.view(deconv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fuseddconv.weight.copy_(torch.mm(w_bn, w_deconv).view(fuseddconv.weight.shape))

    # 准备空间偏置
    b_conv = torch.zeros(deconv.weight.shape[1], device=deconv.weight.device) if deconv.bias is None else deconv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fuseddconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fuseddconv


def model_info(model, detailed=False, verbose=True, imgsz=640):
    """逐层打印和返回详细的模型信息。"""
    if not verbose:
        return
    n_p = get_num_params(model)  # 参数总数
    n_g = get_num_gradients(model)  # 梯度总数
    n_l = len(list(model.modules()))  # 层数
    if detailed:
        LOGGER.info(f"{'layer':>5}{'name':>40}{'gradient':>10}{'parameters':>12}{'shape':>20}{'mu':>10}{'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            LOGGER.info(
                f"{i:>5g}{name:>40s}{p.requires_grad!r:>10}{p.numel():>12g}{str(list(p.shape)):>20s}"
                f"{p.mean():>10.3g}{p.std():>10.3g}{str(p.dtype):>15s}"
            )

    flops = get_flops(model, imgsz)  # imgsz 可能是 int 或 list，例如 imgsz=640 或 imgsz=[640, 320]
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    LOGGER.info(f"{model_name} 摘要{fused}: {n_l:,} 层, {n_p:,} 参数, {n_g:,} 梯度{fs}")
    return n_l, n_p, n_g, flops


def get_num_params(model):
    """返回 YOLO 模型中的参数总数。"""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """返回 YOLO 模型中具有梯度的参数总数。"""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def model_info_for_loggers(trainer):
    """
    返回包含有用模型信息的字典，用于日志记录。

    示例：
        YOLOv8n 日志信息
        ```python
        results = {
            "model/parameters": 3151904,
            "model/GFLOPs": 8.746,
            "model/speed_ONNX(ms)": 41.244,
            "model/speed_TensorRT(ms)": 3.211,
            "model/speed_PyTorch(ms)": 18.755,
        }
        ```
    """
    if trainer.args.profile:  # 为 ONNX 和 TensorRT 计时
        from ultralytics.utils.benchmarks import ProfileModels

        results = ProfileModels([trainer.last], device=trainer.device).profile()[0]
        results.pop("model/name")
    else:  # 只返回最近验证的 PyTorch 时间
        results = {
            "model/parameters": get_num_params(trainer.model),
            "model/GFLOPs": round(get_flops(trainer.model), 3),
        }
    results["model/speed_PyTorch(ms)"] = round(trainer.validator.speed["inference"], 3)
    return results


def get_flops(model, imgsz=640):
    """返回 YOLO 模型的 FLOPs（浮点运算次数）。"""
    try:
        model = de_parallel(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]  # 如果是 int/float，则扩展为 list
        try:
            # 使用步幅大小来计算输入张量
            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  # 最大步幅
            im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # 输入图像 BCHW 格式
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # 使用步幅计算 GFLOPs
            return flops * imgsz[0] / stride * imgsz[1] / stride  # 使用图像大小计算 GFLOPs
        except Exception:
            # 使用实际的图像大小来计算输入张量（例如 RTDETR 模型需要此操作）
            im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # 输入图像 BCHW 格式
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # 使用图像大小计算 GFLOPs
    except Exception:
        return 0.0


def get_flops_with_torch_profiler(model, imgsz=640):
    """计算模型的 FLOPs（使用 torch.profiler 代替 thop 包，但速度慢 2-10 倍）。"""
    if not TORCH_2_0:  # 仅在 torch>=2.0 实现了 torch.profiler
        return 0.0
    model = de_parallel(model)
    p = next(model.parameters())
    if not isinstance(imgsz, list):
        imgsz = [imgsz, imgsz]  # 如果是 int/float，则扩展为 list
    try:
        # 使用步幅大小来计算输入张量
        stride = (max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32) * 2  # 最大步幅
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # 输入图像 BCHW 格式
        with torch.profiler.profile(with_flops=True) as prof:
            model(im)
        flops = sum(x.flops for x in prof.key_averages()) / 1e9
        flops = flops * imgsz[0] / stride * imgsz[1] / stride  # 640x640 GFLOPs
    except Exception:
        # 使用实际的图像大小来计算输入张量（例如 RTDETR 模型需要此操作）
        im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # 输入图像 BCHW 格式
        with torch.profiler.profile(with_flops=True) as prof:
            model(im)
        flops = sum(x.flops for x in prof.key_averages()) / 1e9
    return flops


def initialize_weights(model):
    """将模型权重初始化为随机值。"""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """缩放和填充图像张量，选择性地保持长宽比并填充到gs的倍数。"""
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # 新的尺寸
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # 重设大小
    if not same_shape:  # 填充/裁剪图像
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    """从对象'b'复制属性到对象'a'，可以选择包含/排除某些属性。"""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def get_latest_opset():
    """返回该版本PyTorch支持的第二新ONNX操作集版本，经过成熟度调整。"""
    if TORCH_1_13:
        # 如果是PyTorch>=1.13，动态计算最新操作集减去一个使用'symbolic_opset'
        return max(int(k[14:]) for k in vars(torch.onnx) if "symbolic_opset" in k) - 1
    # 否则，对于PyTorch<=1.12，返回相应的预定义操作集
    version = torch.onnx.producer_version.rsplit(".", 1)[0]  # 即 '2.3'
    return {"1.12": 15, "1.11": 14, "1.10": 13, "1.9": 12, "1.8": 12}.get(version, 12)


def intersect_dicts(da, db, exclude=()):
    """返回具有匹配形状的交集键的字典，排除'exclude'键，使用da的值。"""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def is_parallel(model):
    """如果模型是类型为DP或DDP，则返回True。"""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):
    """去并行化模型：如果模型是DP或DDP类型，返回单GPU模型。"""
    return model.module if is_parallel(model) else model


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """返回一个从y1到y2的正弦坡度的lambda函数 https://arxiv.org/pdf/1812.01187.pdf。"""
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1


def init_seeds(seed=0, deterministic=False):
    """初始化随机数生成器（RNG）种子 https://pytorch.org/docs/stable/notes/randomness.html。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 用于多GPU，保证异常安全
    # torch.backends.cudnn.benchmark = True  # AutoBatch问题 https://github.com/ultralytics/yolov5/issues/9287
    if deterministic:
        if TORCH_2_0:
            torch.use_deterministic_algorithms(True, warn_only=True)  # 如果不支持确定性算法则警告
            torch.backends.cudnn.deterministic = True
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["PYTHONHASHSEED"] = str(seed)
        else:
            LOGGER.warning("WARNING ⚠️ 升级到torch>=2.0.0以进行确定性训练。")
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False


class ModelEMA:
    """
    更新的指数移动平均（EMA）来自 https://github.com/rwightman/pytorch-image-models。保持模型状态字典（参数和缓冲区）中的所有内容的移动平均。

    有关EMA的详细信息，请参见 https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    禁用EMA时，设置'enabled'属性为'False'。
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """为'model'初始化EMA，使用给定的参数。"""
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # EMA更新次数
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # 衰减指数坡度（帮助早期epochs）
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """更新EMA参数。"""
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # 模型的state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # 对于FP16和FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """更新属性并保存剥离后的模型，移除优化器。"""
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


def strip_optimizer(f: Union[str, Path] = "best.pt", s: str = "", updates: dict = None) -> dict:
    """
    从 'f' 中剥离优化器以完成训练，选择性地保存为 's'。

    参数：
        f (str): 从中剥离优化器的模型文件路径。默认为 'best.pt'。
        s (str): 保存剥离优化器后的模型的文件路径。如果未提供，将覆盖 'f'。
        updates (dict): 在保存之前将更新应用到检查点的字典。

    返回：
        (dict): 合并后的检查点字典。

    示例：
        ```python
        from pathlib import Path
        from ultralytics.utils.torch_utils import strip_optimizer

        for f in Path("path/to/model/checkpoints").rglob("*.pt"):
            strip_optimizer(f)
        ```

    注意：
        使用 `ultralytics.nn.torch_safe_load` 来加载缺失的模块，例如 `x = torch_safe_load(f)[0]`
    """
    try:
        x = torch.load(f, map_location=torch.device("cpu"))
        assert isinstance(x, dict), "检查点不是一个 Python 字典"
        assert "model" in x, "'model' 在检查点中缺失"
    except Exception as e:
        LOGGER.warning(f"警告 ⚠️ 跳过 {f}，不是一个有效的 Ultralytics 模型: {e}")
        return {}

    metadata = {
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }

    # 更新模型
    if x.get("ema"):
        x["model"] = x["ema"]  # 使用 EMA 替换模型
    if hasattr(x["model"], "args"):
        x["model"].args = dict(x["model"].args)  # 将 IterableSimpleNamespace 转换为字典
    if hasattr(x["model"], "criterion"):
        x["model"].criterion = None  # 移除损失函数
    x["model"].half()  # 转为 FP16
    for p in x["model"].parameters():
        p.requires_grad = False

    # 更新其他键
    args = {**DEFAULT_CFG_DICT, **x.get("train_args", {})}  # 合并参数
    for k in "optimizer", "best_fitness", "ema", "updates":  # 需要清除的键
        x[k] = None
    x["epoch"] = -1
    x["train_args"] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # 仅保留默认键
    # x['model'].args = x['train_args']

    # 保存
    combined = {**metadata, **x, **(updates or {})}
    torch.save(combined, s or f)  # 合并字典（优先使用右侧）
    mb = os.path.getsize(s or f) / 1e6  # 文件大小
    LOGGER.info(f"优化器已从 {f} 中移除,{f' 已保存为 {s},' if s else ''} {mb:.1f}MB")
    return combined


def convert_optimizer_state_dict_to_fp16(state_dict):
    """
    将给定优化器的 state_dict 转换为 FP16，重点是将 'state' 键中的张量转换为 FP16。

    该方法旨在减少存储大小，而不改变 'param_groups'，因为它们包含非张量数据。
    """
    for state in state_dict["state"].values():
        for k, v in state.items():
            if k != "step" and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                state[k] = v.half()

    return state_dict


@contextmanager
def cuda_memory_usage(device=None):
    """
    监控和管理 CUDA 内存使用情况。

    该函数检查是否可用 CUDA，如果可用，则清空 CUDA 缓存以释放未使用的内存。
    然后它会返回一个字典，包含内存使用信息，调用者可以更新该字典。
    最后，它会更新字典，显示指定设备上 CUDA 已预留的内存量。

    参数：
        device (torch.device, 可选): 要查询内存使用情况的 CUDA 设备。默认为 None。

    返回：
        (dict): 包含键 'memory' 的字典，初始值为 0，调用后会更新为已预留的内存。
    """
    cuda_info = dict(memory=0)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            yield cuda_info
        finally:
            cuda_info["memory"] = torch.cuda.memory_reserved(device)
    else:
        yield cuda_info


def profile(input, ops, n=10, device=None, max_num_obj=0):
    """
    Ultralytics速度、内存和FLOPs分析器。

    示例:
        ```python
        from ultralytics.utils.torch_utils import profile

        input = torch.randn(16, 3, 640, 640)
        m1 = lambda x: x * torch.sigmoid(x)
        m2 = nn.SiLU()
        profile(input, [m1, m2], n=100)  # 在100次迭代中进行性能分析
        ```
    """
    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    LOGGER.info(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )
    gc.collect()  # 尝试释放未使用的内存
    torch.cuda.empty_cache()
    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m  # 设置设备
            m = m.half() if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # 正向传播时间，反向传播时间
            try:
                flops = thop.profile(m, inputs=[x], verbose=False)[0] / 1e9 * 2  # GFLOPs
            except Exception:
                flops = 0

            try:
                mem = 0
                for _ in range(n):
                    with cuda_memory_usage(device) as cuda_info:
                        t[0] = time_sync()
                        y = m(x)
                        t[1] = time_sync()
                        try:
                            (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                            t[2] = time_sync()
                        except Exception:  # 没有反向传播方法
                            # print(e)  # 用于调试
                            t[2] = float("nan")
                    mem += cuda_info["memory"] / 1e9  # (GB)
                    tf += (t[1] - t[0]) * 1000 / n  # 每次正向传播时间 (ms)
                    tb += (t[2] - t[1]) * 1000 / n  # 每次反向传播时间 (ms)
                    if max_num_obj:  # 模拟训练时每张图片的预测目标数（用于AutoBatch）
                        with cuda_memory_usage(device) as cuda_info:
                            torch.randn(
                                x.shape[0],
                                max_num_obj,
                                int(sum((x.shape[-1] / s) * (x.shape[-2] / s) for s in m.stride.tolist())),
                                device=device,
                                dtype=torch.float32,
                            )
                        mem += cuda_info["memory"] / 1e9  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else "list" for x in (x, y))  # 输入输出形状
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # 模型参数数目
                LOGGER.info(f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}")
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                LOGGER.info(e)
                results.append(None)
            finally:
                gc.collect()  # 尝试释放未使用的内存
                torch.cuda.empty_cache()
    return results


class EarlyStopping:
    """早停类，当指定的轮数内没有改善时停止训练。"""

    def __init__(self, patience=50):
        """
        初始化早停对象。

        参数:
            patience (int, 可选): 在训练的适应度停止改善后等待的轮数，直到停止训练。
        """
        self.best_fitness = 0.0  # 即 mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # 等待的轮数，如果适应度停止改善则停止训练
        self.possible_stop = False  # 可能在下一轮停止

    def __call__(self, epoch, fitness):
        """
        检查是否停止训练。

        参数:
            epoch (int): 当前训练轮次
            fitness (float): 当前轮次的适应度值

        返回:
            (bool): 如果训练应该停止，则返回True，否则返回False
        """
        if fitness is None:  # 检查fitness是否为None（当val=False时会发生）
            return False

        if fitness >= self.best_fitness:  # >= 0 允许训练初期适应度为零的情况
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # 没有改善的轮数
        self.possible_stop = delta >= (self.patience - 1)  # 如果delta大于等于patience，则可能会停止
        stop = delta >= self.patience  # 如果超出patience则停止训练
        if stop:
            prefix = colorstr("EarlyStopping: ")
            LOGGER.info(
                f"{prefix}训练提前停止，因为在最后 {self.patience} 轮中没有观察到改善。"
                f"最佳结果出现在第 {self.best_epoch} 轮，最佳模型保存为 best.pt。\n"
                f"要更新 EarlyStopping(patience={self.patience})，请传递一个新的 patience 值，"
                f"例如 `patience=300` 或使用 `patience=0` 来禁用早停。"
            )
        return stop


class FXModel(nn.Module):
    """
    一个用于torch.fx兼容性的自定义模型类。

    该类扩展了`torch.nn.Module`，旨在确保与torch.fx兼容，以进行追踪和图形操作。
    它复制了现有模型的属性，并显式设置模型属性以确保正确复制。

    参数:
        model (torch.nn.Module): 要包装以便与torch.fx兼容的原始模型。
    """

    def __init__(self, model):
        """
        初始化FXModel。

        参数:
            model (torch.nn.Module): 要包装以便与torch.fx兼容的原始模型。
        """
        super().__init__()
        copy_attr(self, model)
        # 显式设置`model`，因为`copy_attr`在某些情况下没有复制它。
        self.model = model.model

    def forward(self, x):
        """
        通过模型进行前向传播。

        该方法执行模型的前向传播，处理层之间的依赖关系并保存中间输出。

        参数:
            x (torch.Tensor): 输入到模型的张量。

        返回:
            (torch.Tensor): 模型的输出张量。
        """
        y = []  # 输出列表
        for m in self.model:
            if m.f != -1:  # 如果不是来自前一层
                # 来自早期层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)  # 执行
            y.append(x)  # 保存输出
        return x
