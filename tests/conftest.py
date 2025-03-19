# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

import shutil
from pathlib import Path

from tests import TMP


def pytest_addoption(parser):
    """
    向 pytest 添加自定义命令行选项。

    参数:
        parser (pytest.config.Parser): pytest 解析器对象，用于添加自定义命令行选项。

    返回:
        (None)
    """
    parser.addoption("--slow", action="store_true", default=False, help="运行耗时较长的测试")


def pytest_collection_modifyitems(config, items):
    """
    修改测试项列表，以排除被标记为 'slow' 的测试，除非指定了 --slow 选项。

    参数:
        config (pytest.config.Config): pytest 配置对象，提供访问命令行选项的功能。
        items (list): 收集到的 pytest 测试项列表，需根据 --slow 选项进行修改。

    返回:
        (None) 该函数直接修改 'items' 列表，而不会返回任何值。
    """
    if not config.getoption("--slow"):
        # 如果测试项被标记为 'slow'，则从测试项列表中移除
        items[:] = [item for item in items if "slow" not in item.keywords]


def pytest_sessionstart(session):
    """
    初始化 pytest 测试会话的配置。

    该函数在 pytest 创建 'Session' 对象后、执行测试收集之前自动调用。
    它设置初始随机种子，并准备用于测试的临时目录。

    参数:
        session (pytest.Session): pytest 测试会话对象。

    返回:
        (None)
    """
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds()
    shutil.rmtree(TMP, ignore_errors=True)  # 删除已存在的 tests/tmp 目录
    TMP.mkdir(parents=True, exist_ok=True)  # 创建一个新的空目录


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    pytest 测试会话结束后的清理操作。

    该函数在整个 pytest 测试会话结束时自动调用。它会删除测试过程中使用的某些文件和目录。

    参数:
        terminalreporter (pytest.terminal.TerminalReporter): 终端报告对象，用于控制台输出。
        exitstatus (int): pytest 运行的退出状态码。
        config (pytest.config.Config): pytest 配置对象。

    返回:
        (None)
    """
    from ultralytics.utils import WEIGHTS_DIR

    # 删除文件
    models = [path for x in ["*.onnx", "*.torchscript"] for path in WEIGHTS_DIR.rglob(x)]
    for file in ["decelera_portrait_min.mov", "bus.jpg", "yolo11n.onnx", "yolo11n.torchscript"] + models:
        Path(file).unlink(missing_ok=True)

    # 删除目录
    models = [path for x in ["*.mlpackage", "*_openvino_model"] for path in WEIGHTS_DIR.rglob(x)]
    for directory in [WEIGHTS_DIR / "path with spaces", TMP.parents[1] / ".pytest_cache", TMP] + models:
        shutil.rmtree(directory, ignore_errors=True)
