#!/bin/bash
# Ultralytics YOLO 🚀, AGPL-3.0 许可证
# 从 https://github.com/ultralytics/assets/releases 下载最新的模型
# 示例用法: bash ultralytics/data/scripts/download_weights.sh
# 父目录
# └── weights
#     ├── yolov8n.pt  ← 下载到这里
#     ├── yolov8s.pt
#     └── ...

python - <<EOF
from ultralytics.utils.downloads import attempt_download_asset

assets = [f"yolov8{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose")]
for x in assets:
    attempt_download_asset(f"weights/{x}")

EOF
