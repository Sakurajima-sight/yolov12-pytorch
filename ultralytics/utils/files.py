# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import glob
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class WorkingDirectory(contextlib.ContextDecorator):
    """
    一个用于临时更改工作目录的上下文管理器和装饰器。

    这个类允许通过上下文管理器或装饰器临时更改工作目录。确保在上下文或装饰的函数完成后，原始工作目录会被恢复。

    属性:
        dir (Path): 要切换到的新目录。
        cwd (Path): 切换前的原始当前工作目录。

    方法:
        __enter__: 在进入上下文时将当前目录更改为指定目录。
        __exit__: 在退出上下文时恢复原始工作目录。

    示例:
        作为上下文管理器使用：
        >>> with WorkingDirectory('/path/to/new/dir'):
        >>> # 在新目录中执行操作
        >>>     pass

        作为装饰器使用：
        >>> @WorkingDirectory('/path/to/new/dir')
        >>> def some_function():
        >>> # 在新目录中执行操作
        >>>     pass
    """

    def __init__(self, new_dir):
        """在实例化时设置工作目录为 'new_dir'，以便用于上下文管理器或装饰器。"""
        self.dir = new_dir  # 新目录
        self.cwd = Path.cwd().resolve()  # 当前目录

    def __enter__(self):
        """进入上下文时将当前工作目录更改为指定的目录。"""
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa
        """退出上下文时恢复原始工作目录。"""
        os.chdir(self.cwd)


@contextmanager
def spaces_in_path(path):
    """
    用于处理路径中包含空格的上下文管理器。如果路径中包含空格，会将空格替换为
    下划线，将文件/目录复制到新路径，执行上下文代码块，然后将文件/目录复制回原位置。

    参数:
        path (str | Path): 可能包含空格的原始路径。

    返回:
        (Path): 如果路径中包含空格，则使用下划线替换空格后的临时路径，否则返回原始路径。

    示例:
        使用上下文管理器处理包含空格的路径：
        >>> from ultralytics.utils.files import spaces_in_path
        >>> with spaces_in_path('/path/with spaces') as new_path:
        >>> # 在这里编写代码
    """
    # 如果路径中有空格，则用下划线替换
    if " " in str(path):
        string = isinstance(path, str)  # 输入类型
        path = Path(path)

        # 创建临时目录并构建新路径
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / path.name.replace(" ", "_")

            # 复制文件/目录
            if path.is_dir():
                # tmp_path.mkdir(parents=True, exist_ok=True)
                shutil.copytree(path, tmp_path)
            elif path.is_file():
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, tmp_path)

            try:
                # 返回临时路径
                yield str(tmp_path) if string else tmp_path

            finally:
                # 将文件/目录复制回原位置
                if tmp_path.is_dir():
                    shutil.copytree(tmp_path, path, dirs_exist_ok=True)
                elif tmp_path.is_file():
                    shutil.copy2(tmp_path, path)  # 复制回文件

    else:
        # 如果没有空格，直接返回原始路径
        yield path


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    增加文件或目录路径，即将 runs/exp 增加为 runs/exp{sep}2、runs/exp{sep}3 等等。

    如果路径存在且 `exist_ok` 为 False，则路径会通过在路径末尾追加一个数字和 `sep` 来增加。如果路径是文件，则文件扩展名会被保留。如果路径是目录，则数字会直接附加在路径的末尾。如果设置了 `mkdir` 为 True，且路径不存在，则会创建该路径作为目录。

    参数:
        path (str | pathlib.Path): 要增加的路径。
        exist_ok (bool): 如果为 True，则路径不会被增加，而是直接返回原路径。
        sep (str): 用于路径与增加数字之间的分隔符。
        mkdir (bool): 如果目录不存在，则创建该目录。

    返回:
        (pathlib.Path): 增加后的路径。

    示例:
        增加目录路径:
        >>> from pathlib import Path
        >>> path = Path("runs/exp")
        >>> new_path = increment_path(path)
        >>> print(new_path)
        runs/exp2

        增加文件路径:
        >>> path = Path("runs/exp/results.txt")
        >>> new_path = increment_path(path)
        >>> print(new_path)
        runs/exp/results2.txt
    """
    path = Path(path)  # 跨平台
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # 方法 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # 增加路径
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # 创建目录

    return path


def file_age(path=__file__):
    """返回指定文件自上次修改以来的天数。"""
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)  # 时间差
    return dt.days  # + dt.seconds / 86400  # 小数天数


def file_date(path=__file__):
    """返回文件的修改日期，格式为 'YYYY-M-D'。"""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"


def file_size(path):
    """返回文件或目录的大小（以兆字节 MB 为单位）。"""
    if isinstance(path, (str, Path)):
        mb = 1 << 20  # 字节转 MiB (1024 ** 2)
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / mb
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / mb
    return 0.0


def get_latest_run(search_dir="."):
    """返回指定目录中最新的 'last.pt' 文件路径，用于恢复训练。"""
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""


def update_models(model_names=("yolo11n.pt",), source_dir=Path("."), update_names=False):
    """
    更新并重新保存指定的 YOLO 模型到 'updated_models' 子目录中。

    参数:
        model_names (Tuple[str, ...]): 要更新的模型文件名。
        source_dir (Path): 包含模型文件和目标子目录的目录。
        update_names (bool): 是否根据数据集 YAML 文件更新模型名称。

    示例:
        更新指定的 YOLO 模型并保存到 'updated_models' 子目录中:
        >>> from ultralytics.utils.files import update_models
        >>> model_names = ("yolo11n.pt", "yolov8s.pt")
        >>> update_models(model_names, source_dir=Path("/models"), update_names=True)
    """
    from ultralytics import YOLO
    from ultralytics.nn.autobackend import default_class_names

    target_dir = source_dir / "updated_models"
    target_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在

    for model_name in model_names:
        model_path = source_dir / model_name
        print(f"从 {model_path} 加载模型")

        # 加载模型
        model = YOLO(model_path)
        model.half()
        if update_names:  # 从数据集 YAML 更新模型名称
            model.model.names = default_class_names("coco8.yaml")

        # 定义新的保存路径
        save_path = target_dir / model_name

        # 使用 model.save() 保存模型
        print(f"将 {model_name} 模型重新保存到 {save_path}")
        model.save(save_path)
