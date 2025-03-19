# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

from ultralytics.utils import ASSETS, ROOT, WEIGHTS_DIR, checks

# 测试中使用的常量
MODEL = WEIGHTS_DIR / "path with spaces" / "yolo11n.pt"  # 测试路径中包含空格的情况
CFG = "yolo11n.yaml"
SOURCE = ASSETS / "bus.jpg"
SOURCES_LIST = [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]  # 不同来源的测试输入
TMP = (ROOT / "../tests/tmp").resolve()  # 用于存放测试文件的临时目录
CUDA_IS_AVAILABLE = checks.cuda_is_available()  # 检查 CUDA 是否可用
CUDA_DEVICE_COUNT = checks.cuda_device_count()  # 获取可用的 CUDA 设备数量

__all__ = (
    "MODEL",
    "CFG",
    "SOURCE",
    "SOURCES_LIST",
    "TMP",
    "CUDA_IS_AVAILABLE",
    "CUDA_DEVICE_COUNT",
)
