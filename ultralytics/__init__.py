# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import importlib.metadata
import inspect
import json
import logging.config
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Union
from urllib.parse import unquote

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm as tqdm_original

from ultralytics import __version__

# PyTorch 多GPU DDP 常量
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html

# 其他常量
ARGV = sys.argv or ["", ""]  # 有时 sys.argv 为空
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
ASSETS = ROOT / "assets"  # 默认图像
ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"  # assets GitHub URL
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"
DEFAULT_SOL_CFG_PATH = ROOT / "cfg/solutions/default.yaml"  # Ultralytics 解决方案 yaml 路径
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # YOLO 多进程线程数
AUTOINSTALL = str(os.getenv("YOLO_AUTOINSTALL", True)).lower() == "true"  # 全局自动安装模式
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  # 全局详细模式
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm 进度条格式
LOGGING_NAME = "ultralytics"
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # 环境布尔值
ARM64 = platform.machine() in {"arm64", "aarch64"}  # ARM64 布尔值
PYTHON_VERSION = platform.python_version()
TORCH_VERSION = torch.__version__
TORCHVISION_VERSION = importlib.metadata.version("torchvision")  # 比导入 torchvision 更快
IS_VSCODE = os.environ.get("TERM_PROGRAM", False) == "vscode"
HELP_MSG = """
    Ultralytics 使用示例：

    1. 安装 ultralytics 包：

        pip install ultralytics

    2. 使用 Python SDK：

        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.yaml")  # 从头构建新模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 使用模型
        results = model.train(data="coco8.yaml", epochs=3)  # 训练模型
        results = model.val()  # 在验证集上评估模型性能
        results = model("https://ultralytics.com/images/bus.jpg")  # 在图像上进行预测
        success = model.export(format="onnx")  # 将模型导出为 ONNX 格式

    3. 使用命令行界面（CLI）：

        Ultralytics 'yolo' CLI 命令使用以下语法：

            yolo TASK MODE ARGS

            其中 TASK（可选）是以下之一 [detect, segment, classify, pose, obb]
                    MODE（必需）是以下之一 [train, val, predict, export, track, benchmark]
                    ARGS（可选）是任何数量的自定义“arg=value”对，如“imgsz=320”，可以覆盖默认值。
                        查看所有 ARGS：https://docs.ultralytics.com/usage/cfg 或使用“yolo cfg”

        - 用初始学习率 0.01 训练一个检测模型 10 轮
            yolo detect train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01

        - 使用预训练的分割模型预测 YouTube 视频，图像大小为 320：
            yolo segment predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

        - 用批量大小为 1、图像大小为 640 的验证集对预训练的检测模型进行验证：
            yolo detect val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640

        - 将 YOLO11n 分类模型导出为 ONNX 格式，图像大小为 224x128（不需要 TASK）
            yolo export model=yolo11n-cls.pt format=onnx imgsz=224,128

        - 执行特殊命令：
            yolo help
            yolo checks
            yolo version
            yolo settings
            yolo copy-cfg
            yolo cfg

    文档: https://docs.ultralytics.com
    社区: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

# 设置和环境变量
torch.set_printoptions(linewidth=320, precision=4, profile="default")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # 格式化为短格式 g，精度为5
cv2.setNumThreads(0)  # 防止 OpenCV 多线程（与 PyTorch DataLoader 不兼容）
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  # NumExpr 最大线程数
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 用于确定性训练，避免 CUDA 警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 在 Colab 中压制冗长的 TF 编译器警告
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # 压制“NNPACK.cpp 无法初始化 NNPACK”警告
os.environ["KINETO_LOG_LEVEL"] = "5"  # 压制 PyTorch 性能分析器计算 FLOPs 时的冗长输出


class TQDM(tqdm_original):
    """
    自定义 TQDM 进度条类，继承原始 tqdm 功能。

    此类根据全局设置修改原始 tqdm 进度条的行为，并提供额外的自定义选项。

    属性：
        disable (bool): 是否禁用进度条。由全局 VERBOSE 设置和传入的 'disable' 参数决定。
        bar_format (str): 进度条的格式字符串。如果没有显式设置，则使用全局 TQDM_BAR_FORMAT。

    方法：
        __init__: 使用自定义设置初始化 TQDM 对象。

    示例：
        >>> from ultralytics.utils import TQDM
        >>> for i in TQDM(range(100)):
        ...     # 在此处编写处理代码
        ...     pass
    """

class TQDM:
    def __init__(self, *args, **kwargs):
        """
        初始化自定义的 TQDM 进度条。

        该类扩展了原始的 tqdm 类，为 Ultralytics 项目提供了定制的行为。

        参数:
            *args (任何类型): 传递给原始 tqdm 构造函数的可变参数列表。
            **kwargs (任何类型): 传递给原始 tqdm 构造函数的任意关键字参数。

        备注:
            - 如果 VERBOSE 为 False 或者在 kwargs 中显式设置 'disable' 为 True，进度条将被禁用。
            - 默认的进度条格式被设置为 TQDM_BAR_FORMAT，除非在 kwargs 中覆盖。

        示例:
            >>> from ultralytics.utils import TQDM
            >>> for i in TQDM(range(100)):
            ...     # 你的代码
            ...     pass
        """
        kwargs["disable"] = not VERBOSE or kwargs.get("disable", False)  # 逻辑与默认值进行判断
        kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)  # 如果传入了，覆盖默认值
        super().__init__(*args, **kwargs)


class SimpleClass:
    """
    一个简单的基类，用于创建对象，并以字符串形式表示它们的属性。

    该类提供了一个基础类，用于创建可以轻松打印或以字符串形式表示的对象，显示其所有非可调用属性。
    它对于调试和检查对象的状态非常有用。

    方法:
        __str__: 返回对象的可读字符串表示。
        __repr__: 返回对象的机器可读字符串表示。
        __getattr__: 提供自定义的属性访问错误消息，包含有用的信息。

    示例:
        >>> class MyClass(SimpleClass):
        ...     def __init__(self):
        ...         self.x = 10
        ...         self.y = "hello"
        >>> obj = MyClass()
        >>> print(obj)
        __main__.MyClass 对象的属性：

        x: 10
        y: 'hello'

    备注:
        - 该类设计为可被继承。它提供了一种方便的方式来检查对象的属性。
        - 字符串表示包括对象的模块和类名。
        - 排除了可调用属性和以下划线开头的属性。
    """

    def __str__(self):
        """返回对象的可读字符串表示。"""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    # 仅显示子类的模块和类名
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} 对象"
                else:
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} 对象的属性:\n\n" + "\n".join(attr)

    def __repr__(self):
        """返回对象的机器可读字符串表示。"""
        return self.__str__()

    def __getattr__(self, attr):
        """提供自定义的属性访问错误消息，包含有用的信息。"""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' 对象没有属性 '{attr}'。请参阅以下有效属性。\n{self.__doc__}")


class IterableSimpleNamespace(SimpleNamespace):
    """
    一个可迭代的 SimpleNamespace 类，提供增强的属性访问和迭代功能。

    该类扩展了 SimpleNamespace 类，增加了迭代、字符串表示和属性访问等方法。
    它设计为作为一个方便的容器，用于存储和访问配置参数。

    方法:
        __iter__: 返回命名空间属性的键值对迭代器。
        __str__: 返回对象的可读字符串表示。
        __getattr__: 提供自定义的属性访问错误消息，包含有用的信息。
        get: 获取指定键的值，如果键不存在，返回默认值。

    示例:
        >>> cfg = IterableSimpleNamespace(a=1, b=2, c=3)
        >>> for k, v in cfg:
        ...     print(f"{k}: {v}")
        a: 1
        b: 2
        c: 3
        >>> print(cfg)
        a=1
        b=2
        c=3
        >>> cfg.get("b")
        2
        >>> cfg.get("d", "default")
        'default'

    备注:
        该类特别适用于以更易访问和可迭代的格式存储配置参数，优于标准字典。
    """

    def __iter__(self):
        """返回命名空间属性的键值对迭代器。"""
        return iter(vars(self).items())

    def __str__(self):
        """返回对象的可读字符串表示。"""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """提供自定义的属性访问错误消息，包含有用的信息。"""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' 对象没有属性 '{attr}'。这可能是由于修改或过时的 ultralytics
            'default.yaml' 文件引起的。
            请更新您的代码，使用 'pip install -U ultralytics'，如有必要，
            使用最新版本的 {DEFAULT_CFG_PATH}，更新路径为
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
            """
        )

    def get(self, key, default=None):
        """如果指定的键存在，则返回其值；否则，返回默认值。"""
        return getattr(self, key, default)


def plt_settings(rcparams=None, backend="Agg"):
    """
    装饰器，用于临时设置绘图函数的 rc 参数和后端。

    示例：
        装饰器：@plt_settings({"font.size": 12})
        上下文管理器：with plt_settings({"font.size": 12}):

    参数：
        rcparams (dict): 要设置的 rc 参数字典。
        backend (str, 可选): 要使用的后端名称。默认值为 'Agg'。

    返回：
        (Callable): 使用临时设置的 rc 参数和后端的装饰函数。此装饰器可应用于任何需要特定 matplotlib rc 参数和后端的函数。
    """
    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """装饰器，用于将临时的 rc 参数和后端应用于一个函数。"""

        def wrapper(*args, **kwargs):
            """设置 rc 参数和后端，调用原始函数，并恢复设置。"""
            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close("all")  # 后端切换时自动关闭图形（从3.8开始弃用）
                plt.switch_backend(backend)

            # 使用后端绘图并始终恢复原始后端
            try:
                with plt.rc_context(rcparams):
                    result = func(*args, **kwargs)
            finally:
                if switch:
                    plt.close("all")
                    plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


def set_logging(name="LOGGING_NAME", verbose=True):
    """
    设置带有 UTF-8 编码和可配置的详细程度的日志记录。

    该函数为 Ultralytics 库配置日志记录，基于详细程度标志和当前进程排名设置适当的日志级别和格式化程序。它还处理 Windows 环境中的特殊情况，确保 UTF-8 编码是默认的。

    参数：
        name (str): 日志记录器的名称。默认为 "LOGGING_NAME"。
        verbose (bool): 设置日志级别为 INFO 如果为 True，否则为 ERROR。默认为 True。

    示例：
        >>> set_logging(name="ultralytics", verbose=True)
        >>> logger = logging.getLogger("ultralytics")
        >>> logger.info("This is an info message")

    备注：
        - 在 Windows 上，该函数会尝试重新配置 stdout 以使用 UTF-8 编码（如果可能）。
        - 如果重新配置不可用，它会回退到自定义格式化程序，处理非 UTF-8 环境。
        - 该函数设置一个 StreamHandler，使用适当的格式化程序和级别。
        - 日志记录器的 propagate 标志被设置为 False，以防止父日志记录器中的重复日志。
    """
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR  # 多 GPU 训练时的进程排名

    # 配置控制台（stdout）编码为 UTF-8，检查兼容性
    formatter = logging.Formatter("%(message)s")  # 默认格式化程序
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":

        class CustomFormatter(logging.Formatter):
            def format(self, record):
                """设置带有 UTF-8 编码和可配置详细程度的日志记录。"""
                return emojis(super().format(record))

        try:
            # 尝试重新配置 stdout 以使用 UTF-8 编码（如果可能）
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # 对于无法使用 reconfigure 的环境，将 stdout 包装在 TextIOWrapper 中
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            else:
                formatter = CustomFormatter("%(message)s")
        except Exception as e:
            print(f"由于 {e} 创建自定义格式化程序用于非 UTF-8 环境")
            formatter = CustomFormatter("%(message)s")

    # 创建并配置具有适当格式化程序和级别的 StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    # 设置日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


# 设置日志记录器
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)  # 在全局范围内定义（在 train.py、val.py、predict.py 等中使用）
for logger in "sentry_sdk", "urllib3.connectionpool":
    logging.getLogger(logger).setLevel(logging.CRITICAL + 1)


def emojis(string=""):
    """返回平台相关的表情符号安全版本的字符串。"""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


class ThreadingLocked:
    """
    确保线程安全执行函数或方法的装饰器类。此类可以作为装饰器使用，以确保如果从多个线程调用被装饰的函数时，只有一个线程能够执行该函数。

    属性：
        lock (threading.Lock): 用于管理对装饰函数的访问的锁对象。

    示例：
        ```python
        from ultralytics.utils import ThreadingLocked

        @ThreadingLocked()
        def my_function():
            # 这里写你的代码
        ```
    """

    def __init__(self):
        """初始化装饰器类以实现函数或方法的线程安全执行。"""
        self.lock = threading.Lock()

    def __call__(self, f):
        """执行函数或方法的线程安全操作。"""
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            """应用线程安全性到被装饰的函数或方法。"""
            with self.lock:
                return f(*args, **kwargs)

        return decorated


def yaml_save(file="data.yaml", data=None, header=""):
    """
    将 YAML 数据保存到文件中。

    参数：
        file (str, 可选): 文件名。默认值为 'data.yaml'。
        data (dict): 要以 YAML 格式保存的数据。
        header (str, 可选): 要添加的 YAML 文件头。

    返回：
        (None): 数据已保存到指定的文件中。
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        # 如果父目录不存在，则创建父目录
        file.parent.mkdir(parents=True, exist_ok=True)

    # 将 Path 对象转换为字符串
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    # 将数据以 YAML 格式写入文件
    with open(file, "w", errors="ignore", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def yaml_load(file="data.yaml", append_filename=False):
    """
    从文件加载YAML数据。

    参数:
        file (str, 可选): 文件名。默认是 'data.yaml'。
        append_filename (bool): 是否将YAML文件名添加到YAML字典中。默认是False。

    返回:
        (dict): YAML数据和文件名。
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"尝试用yaml_load()加载非YAML文件 {file}"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # 字符串

        # 移除特殊字符
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # 将YAML文件名添加到字典并返回
        data = yaml.safe_load(s) or {}  # 始终返回一个字典（yaml.safe_load()可能会返回空文件时的None）
        if append_filename:
            data["yaml_file"] = str(file)
        return data


def yaml_print(yaml_file: Union[str, Path, dict]) -> None:
    """
    以美观的格式打印YAML文件或YAML格式的字典。

    参数:
        yaml_file: YAML文件的路径或YAML格式的字典。

    返回:
        (None)
    """
    yaml_dict = yaml_load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
    dump = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True, width=float("inf"))
    LOGGER.info(f"打印 '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")


# 默认配置
DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
DEFAULT_SOL_DICT = yaml_load(DEFAULT_SOL_CFG_PATH)  # Ultralytics解决方案配置
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == "none":
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


def read_device_model() -> str:
    """
    从系统读取设备型号信息，并缓存以便快速访问。用于is_jetson()和is_raspberrypi()。

    返回:
        (str): 内核发布信息。
    """
    return platform.release().lower()


def is_ubuntu() -> bool:
    """
    检查操作系统是否是Ubuntu。

    返回:
        (bool): 如果操作系统是Ubuntu，则返回True，否则返回False。
    """
    try:
        with open("/etc/os-release") as f:
            return "ID=ubuntu" in f.read()
    except FileNotFoundError:
        return False


def is_colab():
    """
    检查当前脚本是否在Google Colab笔记本中运行。

    返回:
        (bool): 如果在Colab笔记本中运行，则返回True，否则返回False。
    """
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ


def is_kaggle():
    """
    检查当前脚本是否在Kaggle内核中运行。

    返回:
        (bool): 如果在Kaggle内核中运行，则返回True，否则返回False。
    """
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"


def is_jupyter():
    """
    检查当前脚本是否在Jupyter Notebook中运行。

    返回:
        (bool): 如果在Jupyter Notebook中运行，则返回True，否则返回False。

    注意:
        - 仅适用于Colab和Kaggle，其他环境如Jupyterlab和Paperspace无法可靠地检测。
        - "get_ipython"在全局方法中会出现误报，特别是当手动安装了IPython包时。
    """
    return IS_COLAB or IS_KAGGLE


def is_runpod():
    """
    检查当前脚本是否在RunPod容器中运行。

    返回:
        (bool): 如果在RunPod中运行，则返回True，否则返回False。
    """
    return "RUNPOD_POD_ID" in os.environ


def is_docker() -> bool:
    """
    判断脚本是否在Docker容器中运行。

    返回:
        (bool): 如果脚本在Docker容器中运行，则返回True，否则返回False。
    """
    try:
        with open("/proc/self/cgroup") as f:
            return "docker" in f.read()
    except Exception:
        return False


def is_raspberrypi() -> bool:
    """
    判断Python环境是否运行在Raspberry Pi上，通过检查设备型号信息。

    返回:
        (bool): 如果运行在Raspberry Pi上，则返回True，否则返回False。
    """
    return "rpi" in DEVICE_MODEL


def is_jetson() -> bool:
    """
    判断Python环境是否运行在NVIDIA Jetson设备上，通过检查设备型号信息。

    返回:
        (bool): 如果运行在NVIDIA Jetson设备上，则返回True，否则返回False。
    """
    return "tegra" in DEVICE_MODEL


def is_online() -> bool:
    """
    通过尝试连接已知的在线主机来检查互联网连接。

    返回值:
        (bool): 如果连接成功，则返回True，否则返回False。
    """
    try:
        assert str(os.getenv("YOLO_OFFLINE", "")).lower() != "true"  # 检查环境变量YOLO_OFFLINE是否为"True"
        import socket

        for dns in ("1.1.1.1", "8.8.8.8"):  # 检查Cloudflare和Google DNS
            socket.create_connection(address=(dns, 80), timeout=2.0).close()
            return True
    except Exception:
        return False


def is_pip_package(filepath: str = __name__) -> bool:
    """
    判断给定文件路径是否是pip包的一部分。

    参数:
        filepath (str): 要检查的文件路径。

    返回值:
        (bool): 如果该文件是pip包的一部分，则返回True，否则返回False。
    """
    import importlib.util

    # 获取模块的规格
    spec = importlib.util.find_spec(filepath)

    # 返回规格是否不为None且源路径不为None（表示它是一个包）
    return spec is not None and spec.origin is not None


def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    检查目录是否可写。

    参数:
        dir_path (str | Path): 目录路径。

    返回值:
        (bool): 如果目录可写，则返回True，否则返回False。
    """
    return os.access(str(dir_path), os.W_OK)


def is_pytest_running():
    """
    判断是否正在运行pytest。

    返回值:
        (bool): 如果正在运行pytest，则返回True，否则返回False。
    """
    return ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules) or ("pytest" in Path(ARGV[0]).stem)


def is_github_action_running() -> bool:
    """
    判断当前环境是否为GitHub Actions运行器。

    返回值:
        (bool): 如果当前环境是GitHub Actions运行器，则返回True，否则返回False。
    """
    return "GITHUB_ACTIONS" in os.environ and "GITHUB_WORKFLOW" in os.environ and "RUNNER_OS" in os.environ


def get_git_dir():
    """
    判断当前文件是否是git仓库的一部分，如果是，返回仓库根目录。如果当前文件不在git仓库中，返回None。

    返回值:
        (Path | None): 如果找到git根目录，返回其路径，否则返回None。
    """
    for d in Path(__file__).parents:
        if (d / ".git").is_dir():
            return d


def is_git_dir():
    """
    判断当前文件是否是git仓库的一部分。如果当前文件不在git仓库中，返回None。

    返回值:
        (bool): 如果当前文件是git仓库的一部分，则返回True。
    """
    return GIT_DIR is not None


def get_git_origin_url():
    """
    获取git仓库的origin URL。

    返回值:
        (str | None): git仓库的origin URL，如果不是git目录则返回None。
    """
    if IS_GIT_DIR:
        try:
            origin = subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            return origin.decode().strip()
        except subprocess.CalledProcessError:
            return None


def get_git_branch():
    """
    返回当前的git分支名称。如果不在git仓库中，返回None。

    返回值:
        (str | None): 当前的git分支名称，如果不是git目录则返回None。
    """
    if IS_GIT_DIR:
        try:
            origin = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            return origin.decode().strip()
        except subprocess.CalledProcessError:
            return None


def get_default_args(func):
    """
    返回一个字典，包含函数的默认参数。

    参数:
        func (callable): 要检查的函数。

    返回值:
        (dict): 一个字典，键为参数名称，值为该参数的默认值。
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_ubuntu_version():
    """
    如果操作系统是 Ubuntu，获取 Ubuntu 版本。

    返回:
        (str): Ubuntu 版本，若不是 Ubuntu 操作系统则返回 None。
    """
    if is_ubuntu():
        try:
            with open("/etc/os-release") as f:
                return re.search(r'VERSION_ID="(\d+\.\d+)"', f.read())[1]
        except (FileNotFoundError, AttributeError):
            return None


def get_user_config_dir(sub_dir="Ultralytics"):
    """
    根据操作系统返回适当的配置目录。

    参数:
        sub_dir (str): 要创建的子目录的名称。

    返回:
        (Path): 用户配置目录的路径。
    """
    if WINDOWS:
        path = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:  # macOS
        path = Path.home() / "Library" / "Application Support" / sub_dir
    elif LINUX:
        path = Path.home() / ".config" / sub_dir
    else:
        raise ValueError(f"不支持的操作系统: {platform.system()}")

    # GCP 和 AWS lambda 修复，只有 /tmp 目录是可写的
    if not is_dir_writeable(path.parent):
        LOGGER.warning(
            f"警告 ⚠️ 用户配置目录 '{path}' 不可写，默认使用 '/tmp' 或当前工作目录."
            "或者您可以定义 YOLO_CONFIG_DIR 环境变量来指定此路径。"
        )
        path = Path("/tmp") / sub_dir if is_dir_writeable("/tmp") else Path().cwd() / sub_dir

    # 如果子目录不存在，则创建
    path.mkdir(parents=True, exist_ok=True)

    return path


# 定义常量（在下面需要使用）
DEVICE_MODEL = read_device_model()  # is_jetson() 和 is_raspberrypi() 依赖这个常量
ONLINE = is_online()
IS_COLAB = is_colab()
IS_KAGGLE = is_kaggle()
IS_DOCKER = is_docker()
IS_JETSON = is_jetson()
IS_JUPYTER = is_jupyter()
IS_PIP_PACKAGE = is_pip_package()
IS_RASPBERRYPI = is_raspberrypi()
GIT_DIR = get_git_dir()
IS_GIT_DIR = is_git_dir()
USER_CONFIG_DIR = Path(os.getenv("YOLO_CONFIG_DIR") or get_user_config_dir())  # Ultralytics 设置目录
SETTINGS_FILE = USER_CONFIG_DIR / "settings.json"


def colorstr(*input):
    r"""
    根据提供的颜色和样式参数为字符串上色。使用 ANSI 转义码。
    更多细节请参见：https://en.wikipedia.org/wiki/ANSI_escape_code

    该函数可以通过两种方式调用：
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    第二种形式下，默认会应用 'blue' 和 'bold'。

    参数:
        *input (str | Path): 一个字符串序列，其中前 n-1 个字符串为颜色和样式参数，
                      最后一个字符串是要上色的文本。

    支持的颜色和样式：
        基本颜色: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        明亮颜色: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                  'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        其他: 'end', 'bold', 'underline'

    返回:
        (str): 使用指定颜色和样式的 ANSI 转义码包装的输入字符串。

    示例:
        >>> colorstr("blue", "bold", "hello world")
        >>> "\033[34m\033[1mhello world\033[0m"
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # 颜色参数，字符串
    colors = {
        "black": "\033[30m",  # 基本颜色
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # 明亮颜色
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # 其他
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def remove_colorstr(input_string):
    """
    移除字符串中的 ANSI 转义码，从而取消颜色。

    参数:
        input_string (str): 要移除颜色和样式的字符串。

    返回:
        (str): 移除所有 ANSI 转义码后的新字符串。

    示例:
        >>> remove_colorstr(colorstr("blue", "bold", "hello world"))
        >>> "hello world"
    """
    ansi_escape = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    return ansi_escape.sub("", input_string)


class TryExcept(contextlib.ContextDecorator):
    """
    Ultralytics 的 TryExcept 类。可以用作 @TryExcept() 装饰器或 'with TryExcept():' 上下文管理器。

    示例：
        作为装饰器：
        >>> @TryExcept(msg="函数出错", verbose=True)
        >>> def func():
        >>> # 函数逻辑
        >>>     pass

        作为上下文管理器：
        >>> with TryExcept(msg="代码块出错", verbose=True):
        >>> # 代码块
        >>>     pass
    """

    def __init__(self, msg="", verbose=True):
        """初始化 TryExcept 类，支持可选的消息和详细输出设置。"""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """进入 TryExcept 上下文时执行，初始化实例。"""
        pass

    def __exit__(self, exc_type, value, traceback):
        """退出 'with' 块时定义行为，如果有必要，打印错误信息。"""
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


class Retry(contextlib.ContextDecorator):
    """
    用于函数执行的重试类，带有指数退避机制。

    可以作为装饰器使用，在发生异常时对函数进行重试，重试的次数和重试之间的延迟会逐步增加。

    示例:
        作为装饰器使用的示例:
        >>> @Retry(times=3, delay=2)
        >>> def test_func():
        >>> # 替换为可能引发异常的函数逻辑
        >>>     return True
    """

    def __init__(self, times=3, delay=2):
        """初始化 Retry 类，指定重试次数和延迟时间。"""
        self.times = times
        self.delay = delay
        self._attempts = 0

    def __call__(self, func):
        """重试装饰器实现，带有指数退避机制。"""

        def wrapped_func(*args, **kwargs):
            """对装饰的函数或方法应用重试机制。"""
            self._attempts = 0
            while self._attempts < self.times:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._attempts += 1
                    print(f"重试 {self._attempts}/{self.times} 失败: {e}")
                    if self._attempts >= self.times:
                        raise e
                    time.sleep(self.delay * (2**self._attempts))  # 指数退避延迟

        return wrapped_func


def threaded(func):
    """
    默认情况下使目标函数多线程化，并返回线程或函数的结果。

    使用时作为 @threaded 装饰器。函数将在单独的线程中运行，除非传递 'threaded=False' 参数。
    """

    def wrapper(*args, **kwargs):
        """根据 'threaded' 参数多线程化给定函数，并返回线程或函数的结果。"""
        if kwargs.pop("threaded", True):  # 在线程中运行
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)

    return wrapper


def set_sentry():
    """
    初始化 Sentry SDK，用于错误追踪和报告。仅在安装了 sentry_sdk 包并且 settings 中 sync=True 时使用。
    运行 'yolo settings' 查看和更新设置。

    发送错误所需的条件（必须满足所有条件，否则不会报告错误）:
        - 安装了 sentry_sdk 包
        - YOLO 设置中 sync=True
        - pytest 未运行
        - 在 pip 包安装中运行
        - 在非 git 目录中运行
        - 运行时 rank 为 -1 或 0
        - 在线环境
        - 使用 CLI 运行包（通过 'yolo' 作为主要 CLI 命令检查）

    此函数还会配置 Sentry SDK 忽略 KeyboardInterrupt 和 FileNotFoundError 异常，并排除异常消息中包含 'out of memory' 的事件。

    此外，函数会为 Sentry 事件设置自定义标签和用户信息。
    """
    if (
        not SETTINGS["sync"]
        or RANK not in {-1, 0}
        or Path(ARGV[0]).name != "yolo"
        or TESTS_RUNNING
        or not ONLINE
        or not IS_PIP_PACKAGE
        or IS_GIT_DIR
    ):
        return
    # 如果未安装 sentry_sdk 包，则返回并不使用 Sentry
    try:
        import sentry_sdk  # noqa
    except ImportError:
        return

    def before_send(event, hint):
        """
        在发送事件到 Sentry 之前，修改事件信息，基于特定的异常类型和消息。

        参数:
            event (dict): 包含错误信息的事件字典。
            hint (dict): 包含关于错误的额外信息的字典。

        返回:
            dict: 修改后的事件字典，如果事件不应该发送到 Sentry，则返回 None。
        """
        if "exc_info" in hint:
            exc_type, exc_value, _ = hint["exc_info"]
            if exc_type in {KeyboardInterrupt, FileNotFoundError} or "out of memory" in str(exc_value):
                return None  # 不发送该事件

        event["tags"] = {
            "sys_argv": ARGV[0],
            "sys_argv_name": Path(ARGV[0]).name,
            "install": "git" if IS_GIT_DIR else "pip" if IS_PIP_PACKAGE else "other",
            "os": ENVIRONMENT,
        }
        return event

    sentry_sdk.init(
        dsn="https://888e5a0778212e1d0314c37d4b9aae5d@o4504521589325824.ingest.us.sentry.io/4504521592406016",
        debug=False,
        auto_enabling_integrations=False,
        traces_sample_rate=1.0,
        release=__version__,
        environment="runpod" if is_runpod() else "production",
        before_send=before_send,
        ignore_errors=[KeyboardInterrupt, FileNotFoundError],
    )
    sentry_sdk.set_user({"id": SETTINGS["uuid"]})  # SHA-256 匿名化的 UUID 哈希


class JSONDict(dict):
    """
    一个类似字典的类，提供 JSON 持久化功能。

    该类扩展了内建的字典类，每当字典内容发生更改时，自动将其保存到 JSON 文件中。
    它通过锁机制确保线程安全的操作。

    属性:
        file_path (Path): 用于持久化的 JSON 文件路径。
        lock (threading.Lock): 用于确保线程安全操作的锁对象。

    方法:
        _load: 从 JSON 文件中加载数据到字典中。
        _save: 将当前字典的状态保存到 JSON 文件中。
        __setitem__: 存储键值对并将其持久化到磁盘。
        __delitem__: 删除项目并更新持久化存储。
        update: 更新字典并持久化更改。
        clear: 清除所有条目并更新持久化存储。

    示例:
        >>> json_dict = JSONDict("data.json")
        >>> json_dict["key"] = "value"
        >>> print(json_dict["key"])
        value
        >>> del json_dict["key"]
        >>> json_dict.update({"new_key": "new_value"})
        >>> json_dict.clear()
    """

    def __init__(self, file_path: Union[str, Path] = "data.json"):
        """初始化 JSONDict 对象，指定用于 JSON 持久化的文件路径。"""
        super().__init__()
        self.file_path = Path(file_path)
        self.lock = Lock()
        self._load()

    def _load(self):
        """从 JSON 文件加载数据到字典中。"""
        try:
            if self.file_path.exists():
                with open(self.file_path) as f:
                    self.update(json.load(f))
        except json.JSONDecodeError:
            print(f"解码 {self.file_path} 中的 JSON 时出错。将以空字典开始。")
        except Exception as e:
            print(f"读取 {self.file_path} 时出错: {e}")

    def _save(self):
        """将当前字典的状态保存到 JSON 文件中。"""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w") as f:
                json.dump(dict(self), f, indent=2, default=self._json_default)
        except Exception as e:
            print(f"写入 {self.file_path} 时出错: {e}")

    @staticmethod
    def _json_default(obj):
        """处理 Path 对象的 JSON 序列化。"""
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"类型为 {type(obj).__name__} 的对象不能被 JSON 序列化")

    def __setitem__(self, key, value):
        """存储键值对并将其持久化到磁盘。"""
        with self.lock:
            super().__setitem__(key, value)
            self._save()

    def __delitem__(self, key):
        """删除一个项并更新持久化存储。"""
        with self.lock:
            super().__delitem__(key)
            self._save()

    def __str__(self):
        """返回字典的漂亮打印 JSON 字符串表示。"""
        contents = json.dumps(dict(self), indent=2, ensure_ascii=False, default=self._json_default)
        return f'JSONDict("{self.file_path}"):\n{contents}'

    def update(self, *args, **kwargs):
        """更新字典并持久化更改。"""
        with self.lock:
            super().update(*args, **kwargs)
            self._save()

    def clear(self):
        """清除所有条目并更新持久化存储。"""
        with self.lock:
            super().clear()
            self._save()


class SettingsManager(JSONDict):
    """
    SettingsManager 类用于管理和持久化 Ultralytics 设置。

    该类扩展了 JSONDict，以提供设置的 JSON 持久化，确保线程安全的操作和默认值。它在初始化时验证设置，并提供更新或重置设置的方法。

    属性：
        file (Path): 用于持久化的 JSON 文件路径。
        version (str): 设置模式的版本。
        defaults (Dict): 包含默认设置的字典。
        help_msg (str): 提供有关如何查看和更新设置的帮助消息。

    方法：
        _validate_settings: 验证当前设置并在必要时重置。
        update: 更新设置，验证键和值类型。
        reset: 重置设置为默认值并保存。

    示例：
        初始化并更新设置：
        >>> settings = SettingsManager()
        >>> settings.update(runs_dir="/new/runs/dir")
        >>> print(settings["runs_dir"])
        /new/runs/dir
    """

    def __init__(self, file=SETTINGS_FILE, version="0.0.6"):
        """使用默认设置初始化 SettingsManager 并加载用户设置。"""
        import hashlib

        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        root = GIT_DIR or Path()
        datasets_root = (root.parent if GIT_DIR and is_dir_writeable(root.parent) else root).resolve()

        self.file = Path(file)
        self.version = version
        self.defaults = {
            "settings_version": version,  # 设置模式版本
            "datasets_dir": str(datasets_root / "datasets"),  # 数据集目录
            "weights_dir": str(root / "weights"),  # 模型权重目录
            "runs_dir": str(root / "runs"),  # 实验运行目录
            "uuid": hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),  # SHA-256 匿名 UUID 哈希值
            "sync": True,  # 启用同步
            "api_key": "",  # Ultralytics API 密钥
            "openai_api_key": "",  # OpenAI API 密钥
            "clearml": True,  # ClearML 集成
            "comet": True,  # Comet 集成
            "dvc": True,  # DVC 集成
            "hub": True,  # Ultralytics HUB 集成
            "mlflow": True,  # MLflow 集成
            "neptune": True,  # Neptune 集成
            "raytune": True,  # Ray Tune 集成
            "tensorboard": True,  # TensorBoard 日志记录
            "wandb": False,  # Weights & Biases 日志记录
            "vscode_msg": True,  # VSCode 消息
        }

        self.help_msg = (
            f"\n通过 'yolo settings' 或 '{self.file}' 查看 Ultralytics 设置"
            "\n通过 'yolo settings key=value' 更新设置，例如 'yolo settings runs_dir=path/to/dir'。"
            "有关帮助，请参阅 https://docs.ultralytics.com/quickstart/#ultralytics-settings。"
        )

        with torch_distributed_zero_first(RANK):
            super().__init__(self.file)

            if not self.file.exists() or not self:  # 检查文件是否不存在或为空
                LOGGER.info(f"创建新的 Ultralytics 设置 v{version} 文件 ✅ {self.help_msg}")
                self.reset()

            self._validate_settings()

    def _validate_settings(self):
        """验证当前设置，并在必要时重置设置。"""
        correct_keys = set(self.keys()) == set(self.defaults.keys())
        correct_types = all(isinstance(self.get(k), type(v)) for k, v in self.defaults.items())
        correct_version = self.get("settings_version", "") == self.version

        if not (correct_keys and correct_types and correct_version):
            LOGGER.warning(
                "警告 ⚠️ Ultralytics 设置已重置为默认值。这可能是由于设置存在问题或最近的 ultralytics 包更新所致。"
                f"{self.help_msg}"
            )
            self.reset()

        if self.get("datasets_dir") == self.get("runs_dir"):
            LOGGER.warning(
                f"警告 ⚠️ Ultralytics 设置 'datasets_dir: {self.get('datasets_dir')}' "
                f"必须与 'runs_dir: {self.get('runs_dir')}' 不同。"
                f"请更改其中一个，以避免在训练过程中可能出现的问题。 {self.help_msg}"
            )

    def __setitem__(self, key, value):
        """更新一个键值对。"""
        self.update({key: value})

def update(self, *args, **kwargs):
    """更新设置，验证键和值的类型。"""
    for arg in args:
        if isinstance(arg, dict):
            kwargs.update(arg)
    for k, v in kwargs.items():
        if k not in self.defaults:
            raise KeyError(f"没有 '{k}' 的Ultralytics设置。 {self.help_msg}")
        t = type(self.defaults[k])
        if not isinstance(v, t):
            raise TypeError(
                f"Ultralytics设置 '{k}' 必须是 '{t.__name__}' 类型，而不是 '{type(v).__name__}' 类型。 {self.help_msg}"
            )
    super().update(*args, **kwargs)

def reset(self):
    """重置设置为默认值并保存。"""
    self.clear()
    self.update(self.defaults)


def deprecation_warn(arg, new_arg=None):
    """当使用已弃用的参数时，发出弃用警告，并建议使用更新后的参数。"""
    msg = f"警告 ⚠️ '{arg}' 已弃用，将来会被移除。"
    if new_arg is not None:
        msg += f" 请改用 '{new_arg}'。"
    LOGGER.warning(msg)


def clean_url(url):
    """去除URL中的认证信息，例如 https://url.com/file.txt?auth -> https://url.com/file.txt。"""
    url = Path(url).as_posix().replace(":/", "://")  # Pathlib 将 :// 转为 :/，使用 as_posix() 来处理Windows
    return unquote(url).split("?")[0]  # '%2F' 转为 '/'，分割 https://url.com/file.txt?auth


def url2file(url):
    """将URL转换为文件名，例如 https://url.com/file.txt?auth -> file.txt。"""
    return Path(clean_url(url)).name


def vscode_msg(ext="ultralytics.ultralytics-snippets") -> str:
    """如果没有安装Ultralytics-Snippets，显示提示信息，提示安装VS Code扩展。"""
    path = (USER_CONFIG_DIR.parents[2] if WINDOWS else USER_CONFIG_DIR.parents[1]) / ".vscode/extensions"
    obs_file = path / ".obsolete"  # 文件跟踪未安装的扩展，而源目录保持不变
    installed = any(path.glob(f"{ext}*")) and ext not in (obs_file.read_text("utf-8") if obs_file.exists() else "")
    url = "https://docs.ultralytics.com/integrations/vscode"
    return "" if installed else f"{colorstr('VS Code:')} 查看 Ultralytics VS Code 扩展 ⚡ 在 {url}"


# 以下代码在工具初始化时运行 ------------------------------------------------------------------------------------

# 检查首次安装步骤
PREFIX = colorstr("Ultralytics: ")
SETTINGS = SettingsManager()  # 初始化设置
PERSISTENT_CACHE = JSONDict(USER_CONFIG_DIR / "persistent_cache.json")  # 初始化持久化缓存
DATASETS_DIR = Path(SETTINGS["datasets_dir"])  # 全局数据集目录
WEIGHTS_DIR = Path(SETTINGS["weights_dir"])  # 全局权重目录
RUNS_DIR = Path(SETTINGS["runs_dir"])  # 全局运行目录
ENVIRONMENT = (
    "Colab"
    if IS_COLAB
    else "Kaggle"
    if IS_KAGGLE
    else "Jupyter"
    if IS_JUPYTER
    else "Docker"
    if IS_DOCKER
    else platform.system()
)
TESTS_RUNNING = is_pytest_running() or is_github_action_running()
set_sentry()

# 应用猴子补丁
from ultralytics.utils.patches import imread, imshow, imwrite, torch_load, torch_save

torch.load = torch_load
torch.save = torch_save
if WINDOWS:
    # 对于非ASCII和非UTF字符的图像路径，应用cv2补丁
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow
