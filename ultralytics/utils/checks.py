# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import inspect
import math
import os
import platform
import re
import shutil
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import torch

from ultralytics.utils import (
    ASSETS,
    AUTOINSTALL,
    IS_COLAB,
    IS_GIT_DIR,
    IS_KAGGLE,
    IS_PIP_PACKAGE,
    LINUX,
    LOGGER,
    MACOS,
    ONLINE,
    PYTHON_VERSION,
    ROOT,
    TORCHVISION_VERSION,
    USER_CONFIG_DIR,
    WINDOWS,
    Retry,
    SimpleNamespace,
    ThreadingLocked,
    TryExcept,
    clean_url,
    colorstr,
    downloads,
    emojis,
    is_github_action_running,
    url2file,
)


def parse_requirements(file_path=ROOT.parent / "requirements.txt", package=""):
    """
    解析 requirements.txt 文件，忽略以 '#' 开头的行以及 '#' 后的文本。

    参数：
        file_path (Path): requirements.txt 文件的路径。
        package (str, 可选): 用于替代 requirements.txt 文件的 Python 包，例如：package='ultralytics'。

    返回：
        (List[Dict[str, str]]): 解析后的要求列表，每个要求是一个包含 `name` 和 `specifier` 键的字典。

    示例：
        ```python
        from ultralytics.utils.checks import parse_requirements

        parse_requirements(package="ultralytics")
        ```
    """
    if package:
        requires = [x for x in metadata.distribution(package).requires if "extra == " not in x]
    else:
        requires = Path(file_path).read_text().splitlines()

    requirements = []
    for line in requires:
        line = line.strip()
        if line and not line.startswith("#"):
            line = line.split("#")[0].strip()  # 忽略行内注释
            if match := re.match(r"([a-zA-Z0-9-_]+)\s*([<>!=~]+.*)?", line):
                requirements.append(SimpleNamespace(name=match[1], specifier=match[2].strip() if match[2] else ""))

    return requirements


def parse_version(version="0.0.0") -> tuple:
    """
    将版本字符串转换为一个整数元组，忽略附加的任何非数字字符串。此函数替代了已弃用的 'pkg_resources.parse_version(v)'。

    参数：
        version (str): 版本字符串，例如 '2.0.1+cpu'

    返回：
        (tuple): 表示版本数字部分的整数元组和附加的字符串，例如 (2, 0, 1)
    """
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        LOGGER.warning(f"警告 ⚠️ 解析 parse_version({version}) 失败，返回 (0, 0, 0): {e}")
        return 0, 0, 0


def is_ascii(s) -> bool:
    """
    检查字符串是否只由 ASCII 字符组成。

    参数：
        s (str): 要检查的字符串。

    返回：
        (bool): 如果字符串只由 ASCII 字符组成，则返回 True，否则返回 False。
    """
    # 将列表、元组、None 等转换为字符串
    s = str(s)

    # 检查字符串是否只由 ASCII 字符组成
    return all(ord(c) < 128 for c in s)


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """
    验证图像大小是否是给定步长的倍数。如果图像大小不是步长的倍数，将其更新为大于或等于给定最小值的最接近的倍数。

    参数：
        imgsz (int | List[int]): 图像大小。
        stride (int): 步长值。
        min_dim (int): 最小维度数。
        max_dim (int): 最大维度数。
        floor (int): 图像大小的最小允许值。

    返回：
        (List[int]): 更新后的图像大小。
    """
    # 如果步长是张量，将其转换为整数
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # 如果图像大小是整数，则转换为列表
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):  # 即 '640' 或 '[640,640]'
        imgsz = [int(imgsz)] if imgsz.isnumeric() else eval(imgsz)
    else:
        raise TypeError(
            f"'imgsz={imgsz}' 是无效类型 {type(imgsz).__name__}。 "
            f"有效的 imgsz 类型是 int，例如 'imgsz=640' 或 list，例如 'imgsz=[640,640]'"
        )

    # 应用 max_dim
    if len(imgsz) > max_dim:
        msg = (
            "'train' 和 'val' imgsz 必须是整数，而 'predict' 和 'export' imgsz 可以是 [h, w] 列表 "
            "或整数，例如 'yolo export imgsz=640,480' 或 'yolo export imgsz=640'"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} 不是有效的图像大小。 {msg}")
        LOGGER.warning(f"警告 ⚠️ 更新为 'imgsz={max(imgsz)}'。 {msg}")
        imgsz = [max(imgsz)]
    # 将图像大小调整为步长的倍数
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # 如果图像大小已更新，打印警告消息
    if sz != imgsz:
        LOGGER.warning(f"警告 ⚠️ imgsz={imgsz} 必须是最大步长 {stride} 的倍数，已更新为 {sz}")

    # 如果需要，添加缺失的维度
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz


def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    """
    检查当前版本是否符合所需版本或版本范围。

    参数：
        current (str): 当前版本或包名来获取版本。
        required (str): 所需版本或范围（以 pip 风格格式表示）。
        name (str, 可选): 用于警告消息中的名称。
        hard (bool, 可选): 如果为 True，当不满足要求时引发 AssertionError。
        verbose (bool, 可选): 如果为 True，当要求不满足时打印警告消息。
        msg (str, 可选): 如果 verbose 为 True，则显示额外的消息。

    返回：
        (bool): 如果满足要求，则返回 True，否则返回 False。

    示例：
        ```python
        # 检查当前版本是否为 22.04
        check_version(current="22.04", required="==22.04")

        # 检查当前版本是否大于或等于 22.04
        check_version(current="22.10", required="22.04")  # 默认为 '>=' 不等式，如果没有传递

        # 检查当前版本是否小于或等于 22.04
        check_version(current="22.04", required="<=22.04")

        # 检查当前版本是否介于 20.04（包含）和 22.04（不包含）之间
        check_version(current="21.10", required=">20.04,<22.04")
        ```
    """
    if not current:  # 如果 current 是 '' 或 None
        LOGGER.warning(f"警告 ⚠️ 请求的 check_version({current}, {required}) 无效，请检查值。")
        return True
    elif not current[0].isdigit():  # 如果 current 是包名而不是版本字符串，例如 current='ultralytics'
        try:
            name = current  # 将包名分配给 'name' 参数
            current = metadata.version(current)  # 从包名获取版本字符串
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(emojis(f"警告 ⚠️ {current} 包是必需的，但未安装")) from e
            else:
                return False

    if not required:  # 如果 required 是 '' 或 None
        return True

    if "sys_platform" in required and (  # 即 required='<2.4.0,>=1.8.0; sys_platform == "win32"'
        (WINDOWS and "win32" not in required)
        or (LINUX and "linux" not in required)
        or (MACOS and "macos" not in required and "darwin" not in required)
    ):
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # 将 '>=22.04' 拆分为 ('>=', '22.04')
        if not op:
            op = ">="  # 如果没有传递操作符，默认认为是 >=
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op == ">=" and not (c >= v):
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"警告 ⚠️ {name}{op}{version} 是必需的，但当前安装的版本是 {name}=={current} {msg}"
        if hard:
            raise ModuleNotFoundError(emojis(warning))  # 如果未满足版本要求，抛出异常
        if verbose:
            LOGGER.warning(warning)
    return result


def check_latest_pypi_version(package_name="ultralytics"):
    """
    返回PyPI包的最新版本，不需要下载或安装它。

    参数:
        package_name (str): 要查找最新版本的包的名称。

    返回:
        (str): 包的最新版本。
    """
    try:
        requests.packages.urllib3.disable_warnings()  # 禁用 InsecureRequestWarning
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=3)
        if response.status_code == 200:
            return response.json()["info"]["version"]
    except Exception:
        return None


def check_pip_update_available():
    """
    检查PyPI上是否有ultralytics包的新版本可用。

    返回:
        (bool): 如果有更新可用，则返回True，否则返回False。
    """
    if ONLINE and IS_PIP_PACKAGE:
        try:
            from ultralytics import __version__

            latest = check_latest_pypi_version()
            if check_version(__version__, f"<{latest}"):  # 检查当前版本是否小于最新版本
                LOGGER.info(
                    f"新版本 https://pypi.org/project/ultralytics/{latest} 可用 😃 "
                    f"通过 'pip install -U ultralytics' 更新"
                )
                return True
        except Exception:
            pass
    return False


@ThreadingLocked()
def check_font(font="Arial.ttf"):
    """
    在本地查找字体，若不存在则下载到用户配置目录。

    参数:
        font (str): 字体的路径或名称。

    返回:
        file (Path): 解析后的字体文件路径。
    """
    from matplotlib import font_manager

    # 检查 USER_CONFIG_DIR
    name = Path(font).name
    file = USER_CONFIG_DIR / name
    if file.exists():
        return file

    # 检查系统字体
    matches = [s for s in font_manager.findSystemFonts() if font in s]
    if any(matches):
        return matches[0]

    # 如果缺少字体，则下载到 USER_CONFIG_DIR
    url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{name}"
    if downloads.is_url(url, check=True):
        downloads.safe_download(url=url, file=file)
        return file


def check_python(minimum: str = "3.8.0", hard: bool = True, verbose: bool = False) -> bool:
    """
    检查当前Python版本是否符合最低版本要求。

    参数:
        minimum (str): 所需的最低Python版本。
        hard (bool, optional): 如果为True，且未满足要求则抛出AssertionError。
        verbose (bool, optional): 如果为True，在不满足要求时打印警告信息。

    返回:
        (bool): 安装的Python版本是否符合最低要求。
    """
    return check_version(PYTHON_VERSION, minimum, name="Python", hard=hard, verbose=verbose)


@TryExcept()
def check_requirements(requirements=ROOT.parent / "requirements.txt", exclude=(), install=True, cmds=""):
    """
    检查已安装的依赖是否满足YOLOv8的要求，如果需要，尝试自动更新。

    参数:
        requirements (Union[Path, str, List[str]]): requirements.txt文件的路径、单个包要求的字符串，或包要求的字符串列表。
        exclude (Tuple[str]): 排除检查的包名元组。
        install (bool): 如果为True，尝试自动更新不满足要求的包。
        cmds (str): 安装包时传递给pip安装命令的附加命令。

    示例:
        ```python
        from ultralytics.utils.checks import check_requirements

        # 检查requirements.txt文件
        check_requirements("path/to/requirements.txt")

        # 检查单个包
        check_requirements("ultralytics>=8.0.0")

        # 检查多个包
        check_requirements(["numpy", "ultralytics>=8.0.0"])
        ```
    """
    prefix = colorstr("red", "bold", "requirements:")
    if isinstance(requirements, Path):  # requirements.txt文件
        file = requirements.resolve()
        assert file.exists(), f"{prefix} {file} 未找到，检查失败。"
        requirements = [f"{x.name}{x.specifier}" for x in parse_requirements(file) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    pkgs = []
    for r in requirements:
        r_stripped = r.split("/")[-1].replace(".git", "")  # 替换git+https://org/repo.git -> 'repo'
        match = re.match(r"([a-zA-Z0-9-_]+)([<>!=~]+.*)?", r_stripped)
        name, required = match[1], match[2].strip() if match[2] else ""
        try:
            assert check_version(metadata.version(name), required)  # 如果不满足要求则抛出异常
        except (AssertionError, metadata.PackageNotFoundError):
            pkgs.append(r)

    @Retry(times=2, delay=1)
    def attempt_install(packages, commands):
        """尝试使用重试机制安装pip命令。"""
        return subprocess.check_output(f"pip install --no-cache-dir {packages} {commands}", shell=True).decode()

    s = " ".join(f'"{x}"' for x in pkgs)  # 控制台字符串
    if s:
        if install and AUTOINSTALL:  # 检查环境变量
            n = len(pkgs)  # 包更新数量
            LOGGER.info(f"{prefix} Ultralytics要求的{'s' * (n > 1)} {pkgs} 未找到，正在尝试自动更新...")
            try:
                t = time.time()
                assert ONLINE, "由于离线，跳过自动更新"
                LOGGER.info(attempt_install(s, cmds))
                dt = time.time() - t
                LOGGER.info(
                    f"{prefix} 自动更新成功 ✅ {dt:.1f}s，已安装 {n} 个包{'s' * (n > 1)}: {pkgs}\n"
                    f"{prefix} ⚠️ {colorstr('bold', '重新启动运行时或重新运行命令以使更新生效')}\n"
                )
            except Exception as e:
                LOGGER.warning(f"{prefix} ❌ {e}")
                return False
        else:
            return False

    return True


def check_torchvision():
    """
    检查已安装的PyTorch和Torchvision版本，以确保它们是兼容的。

    该函数检查已安装的PyTorch和Torchvision版本，并警告如果它们根据以下兼容性表格不兼容：
    https://github.com/pytorch/vision#installation。

    兼容性表格是一个字典，键是PyTorch版本，值是兼容的Torchvision版本列表。
    """
    # 兼容性表格
    compatibility_table = {
        "2.5": ["0.20"],
        "2.4": ["0.19"],
        "2.3": ["0.18"],
        "2.2": ["0.17"],
        "2.1": ["0.16"],
        "2.0": ["0.15"],
        "1.13": ["0.14"],
        "1.12": ["0.13"],
    }

    # 提取主要和次要版本
    v_torch = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    if v_torch in compatibility_table:
        compatible_versions = compatibility_table[v_torch]
        v_torchvision = ".".join(TORCHVISION_VERSION.split("+")[0].split(".")[:2])
        if all(v_torchvision != v for v in compatible_versions):
            print(
                f"警告 ⚠️ torchvision=={v_torchvision} 与 torch=={v_torch} 不兼容。\n"
                f"运行 'pip install torchvision=={compatible_versions[0]}' 来修复torchvision，或 "
                "'pip install -U torch torchvision' 来更新两者。\n"
                "完整的兼容性表格请参见 https://github.com/pytorch/vision#installation"
            )


def check_suffix(file="yolo11n.pt", suffix=".pt", msg=""):
    """检查文件是否具有可接受的后缀。"""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # 文件后缀
            if len(s):
                assert s in suffix, f"{msg}{f} 的可接受后缀是 {suffix}，而不是 {s}"


def check_yolov5u_filename(file: str, verbose: bool = True):
    """将旧版YOLOv5文件名替换为更新的YOLOv5u文件名。"""
    if "yolov3" in file or "yolov5" in file:
        if "u.yaml" in file:
            file = file.replace("u.yaml", ".yaml")  # 例如 yolov5nu.yaml -> yolov5n.yaml
        elif ".pt" in file and "u" not in file:
            original_file = file
            file = re.sub(r"(.*yolov5([nsmlx]))\.pt", "\\1u.pt", file)  # 例如 yolov5n.pt -> yolov5nu.pt
            file = re.sub(r"(.*yolov5([nsmlx])6)\.pt", "\\1u.pt", file)  # 例如 yolov5n6.pt -> yolov5n6u.pt
            file = re.sub(r"(.*yolov3(|-tiny|-spp))\.pt", "\\1u.pt", file)  # 例如 yolov3-spp.pt -> yolov3-sppu.pt
            if file != original_file and verbose:
                LOGGER.info(
                    f"提示 💡 将 'model={original_file}' 替换为新的 'model={file}'。\nYOLOv5 'u' 模型是使用 "
                    f"https://github.com/ultralytics/ultralytics 训练的，具有比标准YOLOv5模型更好的性能，"
                    f"后者是使用 https://github.com/ultralytics/yolov5 训练的。\n"
                )
    return file


def check_model_file_from_stem(model="yolov8n"):
    """根据有效的模型前缀返回模型文件名。"""
    if model and not Path(model).suffix and Path(model).stem in downloads.GITHUB_ASSETS_STEMS:
        return Path(model).with_suffix(".pt")  # 添加后缀，例如 yolov8n -> yolov8n.pt
    else:
        return model


def check_file(file, suffix="", download=True, download_dir=".", hard=True):
    """搜索/下载文件（如果需要）并返回路径。"""
    check_suffix(file, suffix)  # 可选
    file = str(file).strip()  # 转换为字符串并去除空格
    file = check_yolov5u_filename(file)  # yolov5n -> yolov5nu
    if (
        not file
        or ("://" not in file and Path(file).exists())  # '://' 检查在 Windows Python<3.10 中需要
        or file.lower().startswith("grpc://")
    ):  # 文件存在或 gRPC Triton 图像
        return file
    elif download and file.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):  # 下载
        url = file  # 警告：Pathlib 会将 :// 转换为 :/
        file = Path(download_dir) / url2file(file)  # '%2F' 转换为 '/', 拆分 https://url.com/file.txt?auth
        if file.exists():
            LOGGER.info(f"已在 {file} 本地找到 {clean_url(url)}")  # 文件已存在
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return str(file)
    else:  # 搜索
        files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file))  # 查找文件
        if not files and hard:
            raise FileNotFoundError(f"'{file}' 不存在")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"多个文件匹配 '{file}'，请指定确切路径：{files}")
        return files[0] if len(files) else []  # 返回文件


def check_yaml(file, suffix=(".yaml", ".yml"), hard=True):
    """搜索/下载 YAML 文件（如有必要）并返回路径，检查后缀。"""
    return check_file(file, suffix, hard=hard)


def check_is_path_safe(basedir, path):
    """
    检查解析后的路径是否在目标目录下，以防止路径遍历。

    参数：
        basedir (Path | str): 目标目录。
        path (Path | str): 要检查的路径。

    返回：
        (bool): 如果路径安全则返回 True，否则返回 False。
    """
    base_dir_resolved = Path(basedir).resolve()
    path_resolved = Path(path).resolve()

    return path_resolved.exists() and path_resolved.parts[: len(base_dir_resolved.parts)] == base_dir_resolved.parts


def check_imshow(warn=False):
    """检查环境是否支持图像显示。"""
    try:
        if LINUX:
            assert not IS_COLAB and not IS_KAGGLE
            assert "DISPLAY" in os.environ, "DISPLAY 环境变量未设置。"
        cv2.imshow("test", np.zeros((8, 8, 3), dtype=np.uint8))  # 显示一个小的 8 像素图像
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f"警告 ⚠️ 环境不支持 cv2.imshow() 或 PIL Image.show()\n{e}")
        return False


def check_yolo(verbose=True, device=""):
    """返回一个人类可读的 YOLO 软件和硬件概述。"""
    import psutil

    from ultralytics.utils.torch_utils import select_device

    if IS_COLAB:
        shutil.rmtree("sample_data", ignore_errors=True)  # 删除 colab /sample_data 目录

    if verbose:
        # 系统信息
        gib = 1 << 30  # 每 GiB 的字节数
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage("/")
        s = f"({os.cpu_count()} 个 CPU, {ram / gib:.1f} GB RAM, {(total - free) / gib:.1f}/{total / gib:.1f} GB 硬盘)"
        try:
            from IPython import display

            display.clear_output()  # 如果是笔记本，清除显示
        except ImportError:
            pass
    else:
        s = ""

    select_device(device=device, newline=False)
    LOGGER.info(f"设置完成 ✅ {s}")


def collect_system_info():
    """收集并打印相关的系统信息，包括操作系统、Python、RAM、CPU 和 CUDA。"""
    import psutil

    from ultralytics.utils import ENVIRONMENT  # 使用作用域以避免循环导入
    from ultralytics.utils.torch_utils import get_cpu_info, get_gpu_info

    gib = 1 << 30  # 每 GiB 的字节数
    cuda = torch and torch.cuda.is_available()
    check_yolo()
    total, used, free = shutil.disk_usage("/")

    info_dict = {
        "操作系统": platform.platform(),
        "环境": ENVIRONMENT,
        "Python": PYTHON_VERSION,
        "安装方式": "git" if IS_GIT_DIR else "pip" if IS_PIP_PACKAGE else "other",
        "内存": f"{psutil.virtual_memory().total / gib:.2f} GB",
        "磁盘": f"{(total - free) / gib:.1f}/{total / gib:.1f} GB",
        "CPU": get_cpu_info(),
        "CPU 数量": os.cpu_count(),
        "GPU": get_gpu_info(index=0) if cuda else None,
        "GPU 数量": torch.cuda.device_count() if cuda else None,
        "CUDA": torch.version.cuda if cuda else None,
    }
    LOGGER.info("\n" + "\n".join(f"{k:<20}{v}" for k, v in info_dict.items()) + "\n")

    package_info = {}
    for r in parse_requirements(package="ultralytics"):
        try:
            current = metadata.version(r.name)
            is_met = "✅ " if check_version(current, str(r.specifier), name=r.name, hard=True) else "❌ "
        except metadata.PackageNotFoundError:
            current = "(未安装)"
            is_met = "❌ "
        package_info[r.name] = f"{is_met}{current}{r.specifier}"
        LOGGER.info(f"{r.name:<20}{package_info[r.name]}")

    info_dict["软件包信息"] = package_info

    if is_github_action_running():
        github_info = {
            "RUNNER_OS": os.getenv("RUNNER_OS"),
            "GITHUB_EVENT_NAME": os.getenv("GITHUB_EVENT_NAME"),
            "GITHUB_WORKFLOW": os.getenv("GITHUB_WORKFLOW"),
            "GITHUB_ACTOR": os.getenv("GITHUB_ACTOR"),
            "GITHUB_REPOSITORY": os.getenv("GITHUB_REPOSITORY"),
            "GITHUB_REPOSITORY_OWNER": os.getenv("GITHUB_REPOSITORY_OWNER"),
        }
        LOGGER.info("\n" + "\n".join(f"{k}: {v}" for k, v in github_info.items()))
        info_dict["GitHub 信息"] = github_info

    return info_dict


def check_amp(model):
    """
    检查 YOLO11 模型的 PyTorch 自动混合精度（AMP）功能。如果检查失败，意味着系统中存在 AMP 异常，可能会导致 NaN 损失或零 mAP 结果，因此在训练过程中将禁用 AMP。

    参数：
        model (nn.Module): 一个 YOLO11 模型实例。

    示例：
        ```python
        from ultralytics import YOLO
        from ultralytics.utils.checks import check_amp

        model = YOLO("yolo11n.pt").model.cuda()
        check_amp(model)
        ```

    返回：
        (bool): 如果 AMP 功能正常工作，返回 True，否则返回 False。
    """
    from ultralytics.utils.torch_utils import autocast

    device = next(model.parameters()).device  # 获取模型所在设备
    prefix = colorstr("AMP: ")
    if device.type in {"cpu", "mps"}:
        return False  # AMP 仅在 CUDA 设备上使用
    else:
        # 有 AMP 问题的 GPU
        pattern = re.compile(
            r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)", re.IGNORECASE
        )

        gpu = torch.cuda.get_device_name(device)
        if bool(pattern.search(gpu)):
            LOGGER.warning(
                f"{prefix}检查失败 ❌。在 {gpu} GPU 上进行 AMP 训练可能会导致 NaN 损失或零 mAP 结果，"
                f"因此训练过程中将禁用 AMP。"
            )
            return False

    def amp_allclose(m, im):
        """比较 FP32 与 AMP 结果是否接近。"""
        batch = [im] * 8
        imgsz = max(256, int(model.stride.max() * 4))  # 最大 stride P5-32 和 P6-64
        a = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data  # FP32 推理
        with autocast(enabled=True):
            b = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data  # AMP 推理
        del m
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)  # 绝对容差为 0.5 的相似性检查

    im = ASSETS / "bus.jpg"  # 用于检查的图片
    LOGGER.info(f"{prefix}正在运行自动混合精度（AMP）检查...")
    warning_msg = "设置 'amp=True'。如果遇到零 mAP 或 NaN 损失，可以通过 amp=False 禁用 AMP。"
    try:
        from ultralytics import YOLO

        assert amp_allclose(YOLO("yolo11n.pt"), im)
        LOGGER.info(f"{prefix}检查通过 ✅")
    except ConnectionError:
        LOGGER.warning(
            f"{prefix}检查跳过 ⚠️。离线，无法下载 YOLO11n 进行 AMP 检查。{warning_msg}"
        )
    except (AttributeError, ModuleNotFoundError):
        LOGGER.warning(
            f"{prefix}检查跳过 ⚠️。由于可能的 Ultralytics 软件包修改，无法加载 YOLO11n 进行 AMP 检查。{warning_msg}"
        )
    except AssertionError:
        LOGGER.warning(
            f"{prefix}检查失败 ❌。检测到系统中存在 AMP 异常，可能导致 NaN 损失或零 mAP 结果，因此训练过程中将禁用 AMP。"
        )
        return False
    return True


def git_describe(path=ROOT):  # path 必须是一个目录
    """返回人类可读的 git 描述，例如 v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe。"""
    try:
        return subprocess.check_output(f"git -C {path} describe --tags --long --always", shell=True).decode()[:-1]
    except Exception:
        return ""


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    """打印函数参数（可选的 args 字典）。"""

    def strip_auth(v):
        """清理较长的 Ultralytics HUB URL，去除可能的认证信息。"""
        return clean_url(v) if (isinstance(v, str) and v.startswith("http") and len(v) > 100) else v

    x = inspect.currentframe().f_back  # 上一帧
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # 自动获取参数
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={strip_auth(v)}" for k, v in args.items()))


def cuda_device_count() -> int:
    """
    获取环境中可用的 NVIDIA GPU 数量。

    返回：
        (int): 环境中可用的 NVIDIA GPU 数量。
    """
    try:
        # 运行 nvidia-smi 命令并捕获其输出
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"], encoding="utf-8"
        )

        # 取第一行并去除任何前后空白
        first_line = output.strip().split("\n")[0]

        return int(first_line)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        # 如果命令失败，nvidia-smi 未找到，或输出不是整数，假定没有 GPU 可用
        return 0


def cuda_is_available() -> bool:
    """
    检查环境中是否可用 CUDA。

    返回：
        (bool): 如果有一个或多个 NVIDIA GPU 可用，则返回 True，否则返回 False。
    """
    return cuda_device_count() > 0


def is_sudo_available() -> bool:
    """
    检查环境中是否可用 sudo 命令。

    返回：
        (bool): 如果 sudo 命令可用，则返回 True，否则返回 False。
    """
    if WINDOWS:
        return False
    cmd = "sudo --version"
    return subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


# 执行检查并定义常量
check_python("3.8", hard=False, verbose=True)  # 检查 Python 版本
check_torchvision()  # 检查 torch-torchvision 兼容性
IS_PYTHON_MINIMUM_3_10 = check_python("3.10", hard=False)
IS_PYTHON_3_12 = PYTHON_VERSION.startswith("3.12")
