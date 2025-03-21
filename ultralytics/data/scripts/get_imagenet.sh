#!/bin/bash
# Ultralytics YOLO 🚀, AGPL-3.0 许可证
# 下载 ILSVRC2012 ImageNet 数据集 https://image-net.org
# 示例用法: bash data/scripts/get_imagenet.sh
# 父目录
# ├── ultralytics
# └── datasets
#     └── imagenet  ← 下载到此处

# 参数（可选） 用法: bash data/scripts/get_imagenet.sh --train --val
if [ "$#" -gt 0 ]; then
  for opt in "$@"; do
    case "${opt}" in
    --train) train=true ;;
    --val) val=true ;;
    esac
  done
else
  train=true
  val=true
fi

# 创建目录
d='../datasets/imagenet' # 解压目录
mkdir -p $d && cd $d

# 下载并解压训练集
if [ "$train" == "true" ]; then
  wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar # 下载 138G, 1281167 张图片
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME; do
    mkdir -p "${NAME%.tar}"
    tar -xf "${NAME}" -C "${NAME%.tar}"
    rm -f "${NAME}"
  done
  cd ..
fi

# 下载并解压验证集
if [ "$val" == "true" ]; then
  wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar # 下载 6.3G, 50000 张图片
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash # 移动到子目录
fi

# 删除损坏的图片（可选：PNG 文件以 JPEG 名称存在，可能导致数据加载器失败）
# rm train/n04266014/n04266014_10835.JPEG

# TFRecords（可选）
# wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt
