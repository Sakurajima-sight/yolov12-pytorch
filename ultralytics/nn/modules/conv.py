# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""卷积模块（Convolution modules）"""

import math
import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
)


def autopad(k, p=None, d=1):  # 卷积核大小，填充，膨胀系数
    """自动计算 'same' 输出大小所需的填充。"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # 实际卷积核大小
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动填充
    return p


class Conv(nn.Module):
    """标准卷积层，参数包括 (输入通道, 输出通道, 卷积核大小, 步长, 填充, 分组, 膨胀系数, 激活函数)。"""

    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, bias=False, p=None, g=1, d=1, act=True):
        """使用指定参数初始化 Conv 卷积层，可选择是否使用激活函数。"""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """对输入张量进行卷积、归一化和激活操作。"""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """在没有归一化的情况下执行卷积和激活操作（用于融合模型）。"""
        return self.act(self.conv(x))


class Conv2(Conv):
    """简化版 RepConv 模块，具备卷积融合功能。"""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """使用给定参数初始化卷积层，并添加 1x1 卷积进行结构优化。"""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # 添加 1x1 卷积

    def forward(self, x):
        """对输入执行主干卷积与 1x1 卷积的加法融合，再进行归一化和激活。"""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """融合后的前向传播：不再使用 cv2，仅使用主卷积层。"""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """将平行卷积融合到一个卷积中。"""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    轻量级卷积，参数为 (输入通道, 输出通道, 卷积核大小)。

    引用实现：
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """使用给定参数和激活函数初始化轻量卷积结构。"""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """对输入依次执行两个卷积操作。"""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """深度可分离卷积（Depth-wise Convolution）。"""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """使用指定参数初始化深度可分离卷积。"""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """深度可分离的转置卷积（上采样卷积）。"""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """使用指定参数初始化 DWConvTranspose2d 类。"""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """二维反卷积层（ConvTranspose2d）封装。"""

    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """初始化 ConvTranspose2d 层，带有可选的归一化与激活函数。"""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """对输入执行反卷积、归一化和激活操作。"""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """融合版本：仅执行反卷积和激活函数，不使用 BatchNorm。"""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """将空间信息聚焦到通道空间。"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """初始化 Focus 对象，用户可以定义通道数、卷积核、步长、填充、分组和激活值。"""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        对拼接后的张量应用卷积，并返回输出。

        输入形状为 (b,c,w,h)，输出形状为 (b,4c,w/2,h/2)。
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost 卷积 https://github.com/huawei-noah/ghostnet。"""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """初始化 Ghost 卷积模块，使用主卷积操作和廉价操作进行高效的特征学习。"""
        super().__init__()
        c_ = c2 // 2  # 隐藏通道数
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """通过 Ghost Bottleneck 层进行前向传播，带有跳跃连接。"""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv 是一个基础的 Rep 风格块，包含训练和推理状态。

    该模块用于 RT-DETR。
    基于 https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """初始化轻量级卷积层，包括输入、输出和可选的激活函数。"""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """前向传播过程。"""
        return self.act(self.conv(x))

    def forward(self, x):
        """前向传播过程。"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """通过将 3x3 卷积核、1x1 卷积核和恒等卷积核及其偏置相加，返回等效的卷积核和偏置。"""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """将 1x1 卷积核填充为 3x3 卷积核。"""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """通过融合神经网络的分支，生成适当的卷积核和偏置。"""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """将两个卷积层合并为一个单独的层，并移除类中未使用的属性。"""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """通道注意力模块，参考 https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet。"""

    def __init__(self, channels: int) -> None:
        """初始化类并设置所需的基本配置和实例变量。"""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过对输入进行卷积并应用激活函数来实现前向传播，通常会使用批归一化。"""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """空间注意力模块。"""

    def __init__(self, kernel_size=7):
        """初始化空间注意力模块，带有内核大小参数。"""
        super().__init__()
        assert kernel_size in {3, 7}, "内核大小必须是3或7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """对输入应用通道和空间注意力进行特征重新校准。"""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """卷积块注意力模块 (CBAM)。"""

    def __init__(self, c1, kernel_size=7):
        """用给定的输入通道（c1）和内核大小初始化CBAM。"""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """通过C1模块应用前向传播。"""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """在指定维度上连接一组张量。"""

    def __init__(self, dimension=1):
        """在指定维度上连接一组张量。"""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """YOLOv8掩模原型模块的前向传播。"""
        return torch.cat(x, self.d)


class Index(nn.Module):
    """返回输入的特定索引。"""

    def __init__(self, c1, c2, index=0):
        """返回输入的特定索引。"""
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        前向传播。

        期望输入为一个张量列表。
        """
        return x[self.index]
