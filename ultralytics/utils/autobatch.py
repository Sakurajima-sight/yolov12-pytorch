# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""用于估算最佳YOLO批次大小，以使用PyTorch中可用CUDA内存的一个部分。"""

import os
from copy import deepcopy

import numpy as np
import torch

from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import autocast, profile


def check_train_batch_size(model, imgsz=640, amp=True, batch=-1, max_num_obj=1):
    """
    使用autobatch()函数计算最佳YOLO训练批次大小。

    参数：
        model (torch.nn.Module): 用于检查批次大小的YOLO模型。
        imgsz (int, 可选): 用于训练的图像大小。
        amp (bool, 可选): 如果为True，使用自动混合精度。
        batch (float, 可选): 要使用的GPU内存的比例。如果为-1，则使用默认值。
        max_num_obj (int, 可选): 数据集中的最大对象数量。

    返回：
        (int): 使用autobatch()函数计算出的最佳批次大小。

    注意：
        如果0.0 < batch < 1.0，表示要使用的GPU内存的比例。
        否则，使用默认比例0.6。
    """
    with autocast(enabled=amp):
        return autobatch(
            deepcopy(model).train(), imgsz, fraction=batch if 0.0 < batch < 1.0 else 0.6, max_num_obj=max_num_obj
        )


def autobatch(model, imgsz=640, fraction=0.60, batch_size=DEFAULT_CFG.batch, max_num_obj=1):
    """
    自动估算最佳YOLO批次大小，以使用可用CUDA内存的一个部分。

    参数：
        model (torch.nn.module): 用于计算批次大小的YOLO模型。
        imgsz (int, 可选): 用作YOLO模型输入的图像大小。默认为640。
        fraction (float, 可选): 要使用的可用CUDA内存的比例。默认为0.60。
        batch_size (int, 可选): 如果检测到错误，使用的默认批次大小。默认为16。
        max_num_obj (int, 可选): 数据集中的最大对象数量。

    返回：
        (int): 最佳批次大小。
    """
    # 检查设备
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}计算imgsz={imgsz}的最佳批次大小，使用{fraction * 100}%的CUDA内存。")
    device = next(model.parameters()).device  # 获取模型设备
    if device.type in {"cpu", "mps"}:
        LOGGER.info(f"{prefix} ⚠️ 仅适用于CUDA设备，使用默认批次大小{batch_size}")
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.info(f"{prefix} ⚠️ 需要torch.backends.cudnn.benchmark=False，使用默认批次大小{batch_size}")
        return batch_size

    # 检查CUDA内存
    gb = 1 << 30  # 字节转GiB (1024 ** 3)
    d = f"CUDA:{os.getenv('CUDA_VISIBLE_DEVICES', '0').strip()[0]}"  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # 设备属性
    t = properties.total_memory / gb  # GiB 总内存
    r = torch.cuda.memory_reserved(device) / gb  # GiB 已保留内存
    a = torch.cuda.memory_allocated(device) / gb  # GiB 已分配内存
    f = t - (r + a)  # GiB 剩余内存
    LOGGER.info(f"{prefix}{d} ({properties.name}) {t:.2f}G 总计, {r:.2f}G 保留, {a:.2f}G 已分配, {f:.2f}G 剩余")

    # 分析批次大小
    batch_sizes = [1, 2, 4, 8, 16] if t < 16 else [1, 2, 4, 8, 16, 32, 64]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=1, device=device, max_num_obj=max_num_obj)

        # 拟合解决方案
        xy = [
            [x, y[2]]
            for i, (x, y) in enumerate(zip(batch_sizes, results))
            if y  # 有效结果
            and isinstance(y[2], (int, float))  # 数字类型
            and 0 < y[2] < t  # 在GPU限制范围内
            and (i == 0 or not results[i - 1] or y[2] > results[i - 1][2])  # 第一个元素或内存递增
        ]
        fit_x, fit_y = zip(*xy) if xy else ([], [])
        p = np.polyfit(np.log(fit_x), np.log(fit_y), deg=1)  # 对数空间中的一次多项式拟合
        b = int(round(np.exp((np.log(f * fraction) - p[1]) / p[0])))  # y截距（最佳批次大小）
        if None in results:  # 一些大小失败
            i = results.index(None)  # 第一个失败的索引
            if b >= batch_sizes[i]:  # 截距高于失败点
                b = batch_sizes[max(i - 1, 0)]  # 选择之前的安全点
        if b < 1 or b > 1024:  # b超出安全范围
            LOGGER.info(f"{prefix}警告 ⚠️ 批次={b}超出安全范围，使用默认批次大小{batch_size}。")
            b = batch_size

        fraction = (np.exp(np.polyval(p, np.log(b))) + r + a) / t  # 预测的比例
        LOGGER.info(f"{prefix}使用批次大小{b}，CUDA设备{d}内存使用{t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")
        return b
    except Exception as e:
        LOGGER.warning(f"{prefix}警告 ⚠️ 检测到错误: {e}, 使用默认批次大小{batch_size}。")
        return batch_size
    finally:
        torch.cuda.empty_cache()
