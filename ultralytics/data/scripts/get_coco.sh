#!/bin/bash
# Ultralytics YOLO 🚀, AGPL-3.0 许可证
# 下载 COCO 2017 数据集 https://cocodataset.org
# 示例用法: bash data/scripts/get_coco.sh
# 父目录
# ├── ultralytics
# └── datasets
#     └── coco  ← 下载到这里

# 参数（可选）用法: bash data/scripts/get_coco.sh --train --val --test --segments
if [ "$#" -gt 0 ]; then
  for opt in "$@"; do
    case "${opt}" in
    --train) train=true ;;  # 下载训练集
    --val) val=true ;;      # 下载验证集
    --test) test=true ;;    # 下载测试集
    --segments) segments=true ;;  # 下载分割标注
    --sama) sama=true ;;    # 下载Sama的COCO数据集
    esac
  done
else
  train=true           # 默认下载训练集
  val=true             # 默认下载验证集
  test=false           # 默认不下载测试集
  segments=false       # 默认不下载分割标注
  sama=false           # 默认不下载Sama数据集
fi

# 下载/解压标签
d='../datasets' # 解压目录
url=https://github.com/ultralytics/assets/releases/download/v0.0.0/
if [ "$segments" == "true" ]; then
  f='coco2017labels-segments.zip' # 169 MB
elif [ "$sama" == "true" ]; then
  f='coco2017labels-segments-sama.zip' # 199 MB https://www.sama.com/sama-coco-dataset/
else
  f='coco2017labels.zip' # 46 MB
fi
echo '下载' $url$f ' ...'
curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &

# 下载/解压图片
d='../datasets/coco/images' # 解压目录
url=http://images.cocodataset.org/zips/
if [ "$train" == "true" ]; then
  f='train2017.zip' # 19G, 118k 图片
  echo '下载' $url$f '...'
  curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &
fi
if [ "$val" == "true" ]; then
  f='val2017.zip' # 1G, 5k 图片
  echo '下载' $url$f '...'
  curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &
fi
if [ "$test" == "true" ]; then
  f='test2017.zip' # 7G, 41k 图片（可选）
  echo '下载' $url$f '...'
  curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &
fi
wait # 等待后台任务完成
