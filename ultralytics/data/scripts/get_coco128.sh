#!/bin/bash
# Ultralytics YOLO 🚀, AGPL-3.0 许可证
# 下载 COCO128 数据集 https://www.kaggle.com/ultralytics/coco128 （来自 COCO train2017 的前 128 张图片）
# 示例用法: bash data/scripts/get_coco128.sh
# 父目录
# ├── ultralytics
# └── datasets
#     └── coco128  ← 下载到此处

# 下载并解压图片和标签
d='../datasets' # 解压目录
url=https://github.com/ultralytics/assets/releases/download/v0.0.0/
f='coco128.zip' # 或 'coco128-segments.zip', 68 MB
echo '正在下载' $url$f ' ...'
curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &

wait # 等待后台任务完成
