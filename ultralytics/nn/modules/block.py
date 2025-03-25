# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license
"""模块块（Block）定义。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
)


class DFL(nn.Module):
    """
    分布式焦点损失（Distribution Focal Loss, DFL）中的积分模块。

    该模块提出于《Generalized Focal Loss》论文中：https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """用指定通道数初始化一个 1x1 卷积层（不参与训练，仅用于分布回归）。"""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """对输入张量 x 应用 softmax 后进行加权平均，实现浮点回归输出。"""
        b, _, a = x.shape  # batch, 通道, anchor 数
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 的 mask 原型模块（Proto），用于分割模型。"""

    def __init__(self, c1, c_=256, c2=32):
        """
        初始化 YOLOv8 的 mask 原型模块，包含原型数量和掩膜通道数。

        输入参数包括：输入通道数，原型通道数，输出掩膜通道数。
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # 上采样，替代 nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """对输入特征图执行前向传播并上采样用于生成 mask 原型图。"""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    PPHGNetV2 的主干起始模块（StemBlock），包含 5 个卷积层和 1 个最大池化层。

    参考 PaddleDetection 实现：
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """初始化 SPP 模块的输入输出通道，以及用于最大池化的核大小。"""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())  # 初始下采样
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())  # 分支路径 1
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())  # 分支路径 2
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())  # 融合后下采样
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())  # 输出调整
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)  # 最大池化分支

    def forward(self, x):
        """执行 PPHGNetV2 主干网络起始模块的前向传播。"""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])  # 填充为偶数尺寸
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)  # 拼接两个分支
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    PPHGNetV2 的 HG_Block，由两层卷积和若干 LightConv 组成。

    参考 PaddleDetection 实现：
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """使用指定输入输出通道数初始化一个包含多个卷积的块结构（支持轻量化卷积）。"""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))  # 多层轻量/标准卷积
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv，压缩通道
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv，恢复通道
        self.add = shortcut and c1 == c2  # 判断是否使用残差连接

    def forward(self, x):
        """执行 PPHGNetV2 主干 HGBlock 的前向传播逻辑。"""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)  # 累积每层输出
        y = self.ec(self.sc(torch.cat(y, 1)))  # 拼接 + 通道压缩与激发
        return y + x if self.add else y  # 如果开启 shortcut 且通道一致，则残差连接


class SPP(nn.Module):
    """空间金字塔池化（Spatial Pyramid Pooling, SPP）层，参考论文：https://arxiv.org/abs/1406.4729。"""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """使用输入/输出通道和不同池化核大小初始化 SPP 层。"""
        super().__init__()
        c_ = c1 // 2  # 中间通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """执行 SPP 层的前向传播，进行空间金字塔池化。"""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """快速空间金字塔池化（SPPF）层，应用于 YOLOv5，由 Glenn Jocher 提出。"""

    def __init__(self, c1, c2, k=5):
        """
        使用指定的输入/输出通道和卷积核大小初始化 SPPF 层。

        此模块等价于 SPP(k=(5, 9, 13))。
        """
        super().__init__()
        c_ = c1 // 2  # 中间通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """执行 Ghost 卷积块的前向传播过程。"""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """只包含一个卷积的 CSP Bottleneck 模块。"""

    def __init__(self, c1, c2, n=1):
        """使用输入通道、输出通道和模块数量初始化 CSP Bottleneck（单卷积版本）。"""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """在 C3 模块中对输入进行交叉卷积处理。"""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """包含两个卷积的 CSP Bottleneck 模块。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """初始化带两个卷积和可选 shortcut 的 CSP Bottleneck。"""
        super().__init__()
        self.c = int(c2 * e)  # 中间通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # 可选激活函数 FReLU
        # self.attention = ChannelAttention(2 * self.c)  # 或 SpatialAttention
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """通过两个卷积的 CSP Bottleneck 执行前向传播。"""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """使用两个卷积实现的快速 CSP Bottleneck。"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """初始化包含 n 个 Bottleneck 的快速 CSP 模块。"""
        super().__init__()
        self.c = int(c2 * e)  # 中间通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 可选使用 FReLU 激活
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """执行 C2f 层的前向传播。"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """使用 split() 替代 chunk() 执行前向传播。"""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """包含三个卷积的标准 CSP Bottleneck 模块。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """根据输入通道、输出通道、层数、shortcut、组数、扩展系数初始化 CSP Bottleneck。"""
        super().__init__()
        c_ = int(c2 * e)  # 中间通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # 可选激活 FReLU
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """通过三个卷积的 CSP Bottleneck 执行前向传播。"""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """使用交叉卷积的 C3 模块。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """初始化 C3x 模块并设置默认参数。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """使用 RepConv 重构的 C3 模块。"""

    def __init__(self, c1, c2, n=3, e=1.0):
        """使用输入通道、输出通道和层数初始化单卷积版本的 CSP Bottleneck。"""
        super().__init__()
        c_ = int(c2 * e)  # 中间通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """RT-DETR 颈部模块的前向传播过程。"""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """集成 TransformerBlock 的 C3 模块。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """初始化 C3TR 模块，内部使用 TransformerBlock。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """使用 GhostBottleneck() 的 C3 模块。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """初始化 C3Ghost 模块，内部包含多个 GhostBottleneck 模块。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # 隐藏通道数
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """GhostBottleneck 模块，来自 Huawei GhostNet: https://github.com/huawei-noah/ghostnet"""

    def __init__(self, c1, c2, k=3, s=1):
        """初始化 GhostBottleneck 模块，设置输入/输出通道、卷积核大小和步长。"""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pointwise 卷积
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # depthwise 卷积（仅在下采样时）
            GhostConv(c_, c2, 1, 1, act=False),  # linear 变换
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """对输入执行主路径卷积 + shortcut 分支，进行求和。"""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """标准瓶颈结构。"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """初始化标准 Bottleneck 模块，支持 shortcut 和分组卷积。"""
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """执行 YOLO 特征融合路径（FPN）的前向传播。"""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP（Cross Stage Partial）瓶颈结构，参考：https://github.com/WongKinYiu/CrossStagePartialNetworks"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """初始化 CSP Bottleneck，设置输入输出通道、重复次数、是否使用 shortcut、组数、扩展比例。"""
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # 对 cat(cv2, cv3) 应用 BN
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """通过三个卷积层构建 CSP 结构前向传播路径。"""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """包含标准卷积层的 ResNet 基础块。"""

    def __init__(self, c1, c2, s=1, e=4):
        """初始化带残差连接的卷积块。"""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """执行 ResNet 块的前向传播。"""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """由多个 ResNetBlock 组成的 ResNet 层。"""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """初始化 ResNet 层，支持初始卷积和多个残差块。"""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """执行 ResNet 层的前向传播。"""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """最大-激活 Sigmoid 注意力模块。"""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """使用指定参数初始化 MaxSigmoidAttnBlock。"""
        super().__init__()
        self.nh = nh  # 多头数量
        self.hc = c2 // nh  # 每个头的通道数
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)  # 文本特征引导线性层
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """执行注意力机制的前向传播。"""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """带有注意力模块的 C2f 模块。"""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """初始化带注意力机制的 C2f 模块，用于增强特征提取与处理能力。"""
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # 可选激活函数 act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """C2f 层的前向传播。"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """使用 split() 而非 chunk() 的前向传播方式。"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn：用于将图像感知信息增强到文本嵌入中。"""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """使用指定参数初始化 ImagePoolingAttn。"""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """对输入张量 x 和引导张量 text 执行注意力机制。"""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """实现用于视觉-语言模型中区域-文本相似性的对比学习头模块。"""

    def __init__(self):
        """使用指定的区域-文本相似性参数初始化 ContrastiveHead。"""
        super().__init__()
        # 注意：使用 -10.0 可保持该 loss 初始化与其他损失一致
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """对比学习的前向函数。"""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    YOLO-World 中使用批归一化的对比学习头模块，替代 L2 归一化。

    参数：
        embed_dims (int): 文本和图像特征的嵌入维度。
    """

    def __init__(self, embed_dims: int):
        """使用区域-文本相似性参数初始化 ContrastiveHead。"""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # 注意：使用 -10.0 可保持初始化分类损失的一致性
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # 使用 -1.0 更加稳定
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """对比学习的前向函数。"""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """可重参数化的瓶颈模块（RepBottleneck）。"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """使用可配置的输入/输出通道、shortcut、分组数、扩展率初始化 RepBottleneck 模块。"""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """用于高效特征提取的可重复 CSP 模块（RepCSP）。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """使用指定通道数、重复次数、shortcut、分组数和扩展比例初始化 RepCSP。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # 隐藏通道数
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """基于 CSP-ELAN 的结构改进模块。"""

    def __init__(self, c1, c2, c3, c4, n=1):
        """使用指定的通道数、重复次数和卷积结构初始化 CSP-ELAN 层。"""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """RepNCSPELAN4 层的前向传播函数。"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """使用 split() 而非 chunk() 的前向传播实现。"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 模块，包含 4 层卷积结构。"""

    def __init__(self, c1, c2, c3, c4):
        """使用指定通道数量初始化 ELAN1 层结构。"""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)              # 第一步：压缩输入通道
        self.cv2 = Conv(c3 // 2, c4, 3, 1)         # 第二步：3x3 卷积
        self.cv3 = Conv(c4, c4, 3, 1)              # 第三步：继续卷积提取特征
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)   # 最终通道融合


class AConv(nn.Module):
    """AConv 模块。"""

    def __init__(self, c1, c2):
        """初始化 AConv 模块，包含一个标准卷积层。"""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)  # 下采样卷积

    def forward(self, x):
        """执行 AConv 层的前向传播。"""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)  # 平滑平均池化
        return self.cv1(x)


class ADown(nn.Module):
    """ADown 模块，用于空间下采样。"""

    def __init__(self, c1, c2):
        """初始化 ADown 模块，使用不同路径卷积实现通道压缩与空间缩减。"""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)  # 3x3 卷积路径
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)  # 1x1 卷积路径

    def forward(self, x):
        """执行 ADown 层的前向传播。"""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)  # 沿通道分割为两部分
        x1 = self.cv1(x1)       # 第一路径：卷积下采样
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)  # 第二路径：最大池化
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)  # 拼接两条路径


class SPPELAN(nn.Module):
    """SPP-ELAN 模块，用于空间金字塔池化与多尺度融合。"""

    def __init__(self, c1, c2, c3, k=5):
        """初始化 SPP-ELAN 模块，包含多重最大池化结构与卷积。"""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)                       # 输入通道压缩
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)                   # 融合卷积

    def forward(self, x):
        """执行 SPPELAN 层的前向传播。"""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])  # 多级池化后拼接
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear 模块，用于通道分组线性输出。"""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """初始化 CBLinear 模块，实现输出通道按组划分。"""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """执行 CBLinear 层的前向传播，按预设通道列表切分输出。"""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse 模块，用于选择性地融合多个特征图。"""

    def __init__(self, idx):
        """初始化 CBFuse 模块，使用指定的索引从不同特征图中取值进行融合。"""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """执行 CBFuse 层的前向传播，使用最近邻插值对齐后进行加和融合。"""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """简化快速版 CSP Bottleneck，包含 2 层卷积结构。"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        初始化 CSP bottleneck 模块。
        
        参数包括：
        - 输入通道 c1
        - 输出通道 c2
        - 堆叠层数 n
        - 是否使用残差 shortcut
        - 卷积组数 g
        - 扩展比 e
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # 输出通道融合卷积
        self.m = nn.ModuleList(
            Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        """执行 C3f 层的前向传播过程。"""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """更快速的 CSP Bottleneck 实现，包含 2 个主卷积结构。"""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """初始化 C3k2 模块，可选择使用 C3k 或标准 Bottleneck 结构。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k(C3):
    """C3k 模块是带有自定义卷积核大小的 CSP Bottleneck，用于特征提取。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """使用自定义卷积核大小初始化 C3k 模块，支持多层堆叠结构。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # 隐藏通道
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class RepVGGDW(torch.nn.Module):
    """RepVGGDW 模块，代表 RepVGG 架构中的深度可分离卷积块。"""

    def __init__(self, ed) -> None:
        """使用深度可分离卷积结构初始化 RepVGGDW 模块，以提升推理效率。"""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)  # 7x7 深度卷积
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False) # 3x3 深度卷积
        self.dim = ed
        self.act = nn.SiLU()  # 激活函数使用 SiLU

    def forward(self, x):
        """
        执行 RepVGGDW 模块的前向传播。

        参数：
            x (torch.Tensor): 输入张量。

        返回：
            (torch.Tensor): 经过深度可分离卷积处理后的输出张量。
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        执行未融合状态下的 RepVGGDW 模块前向传播。

        参数：
            x (torch.Tensor): 输入张量。

        返回：
            (torch.Tensor): 经过深度可分离卷积处理后的输出张量。
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        融合 RepVGGDW 模块中的卷积层。

        此方法将多个卷积层进行融合，并相应更新权重与偏置参数。
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    条件身份块（Conditional Identity Block，CIB）模块。

    参数：
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        shortcut (bool, 可选): 是否使用残差连接，默认为 True。
        e (float, 可选): 中间通道扩展比例，默认为 0.5。
        lk (bool, 可选): 是否在第三个卷积中使用 RepVGGDW，默认为 False。
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """使用可选的残差连接、通道缩放比例和 RepVGGDW 层初始化 CIB 模块。"""
        super().__init__()
        c_ = int(c2 * e)  # 中间通道数
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        执行 CIB 模块的前向传播。

        参数：
            x (torch.Tensor): 输入张量。

        返回：
            (torch.Tensor): 输出张量。
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB 类：结合 C2f 与 CIB 模块的卷积结构。

    参数：
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        n (int, 可选): 堆叠的 CIB 模块数量，默认值为 1。
        shortcut (bool, 可选): 是否使用残差连接，默认值为 False。
        lk (bool, 可选): 是否使用 local key（RepVGGDW），默认值为 False。
        g (int, 可选): 分组卷积组数，默认值为 1。
        e (float, 可选): CIB 模块的通道扩展系数，默认值为 0.5。
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """根据通道数、残差连接、local key、分组数和扩展系数初始化模块。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    注意力模块，对输入张量执行自注意力（self-attention）。

    参数：
        dim (int): 输入张量的维度。
        num_heads (int): 注意力头数量。
        attn_ratio (float): 注意力 key 的维度与每个 head 的维度之间的比例。

    属性：
        num_heads (int): 注意力头数。
        head_dim (int): 每个注意力头的维度。
        key_dim (int): attention key 的维度。
        scale (float): 注意力分数的缩放因子。
        qkv (Conv): 生成 query、key、value 的卷积层。
        proj (Conv): 对注意力输出进行投影的卷积层。
        pe (Conv): 位置编码的卷积层。
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """使用 query、key、value 卷积和位置编码初始化多头注意力模块。"""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Attention 模块的前向传播。

        参数：
            x (torch.Tensor): 输入张量。

        返回：
            (torch.Tensor): 自注意力处理后的输出张量。
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock：位置敏感注意力块（Position-Sensitive Attention Block）。

    此模块结合多头注意力机制和前馈神经网络，并可选择是否使用残差连接以增强特征提取能力。

    属性：
        attn (Attention): 多头注意力模块。
        ffn (nn.Sequential): 前馈神经网络。
        add (bool): 是否使用残差连接的标志。

    示例：
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """初始化 PSABlock，包括注意力和前馈层。"""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """对输入张量执行注意力和前馈处理。"""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA：位置敏感注意力模块（Position-Sensitive Attention）。

    用于在网络中引入空间位置敏感的注意力机制，提升特征表达能力。

    属性：
        c (int): 中间隐藏通道数。
        cv1 (Conv): 初始 1x1 卷积层，通道压缩。
        cv2 (Conv): 输出 1x1 卷积层，恢复通道数。
        attn (Attention): 自注意力机制。
        ffn (nn.Sequential): 前馈神经网络模块。

    示例：
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """使用指定输入/输出通道和扩展因子初始化 PSA 模块。"""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """对输入张量进行 PSA 注意力和前馈网络处理。"""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA 模块：带有堆叠的 PSABlock 的注意力机制卷积模块。

    该模块结合了多个位置敏感注意力块与卷积，提升特征表达能力。

    属性：
        c (int): 中间通道数。
        cv1 (Conv): 初始 1x1 卷积层。
        cv2 (Conv): 输出 1x1 卷积层。
        m (nn.Sequential): 多个 PSABlock 堆叠。

    示例：
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """初始化 C2PSA 模块，包括输入输出通道数、PSABlock 个数和扩展因子。"""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """对输入张量依次进行多个 PSA 块处理并拼接输出。"""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA 模块：集成 PSA 块的增强版 C2f 模块，用于提升注意力特征提取能力。

    属性：
        c (int): 中间通道数。
        cv1 (Conv): 初始卷积。
        cv2 (Conv): 输出卷积。
        m (nn.ModuleList): 多个 PSA 注意力块组成的列表。

    示例：
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """初始化 C2fPSA 模块，融合 PSA 注意力机制与标准 C2f 模块结构。"""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown 模块：使用可分离卷积进行下采样。

    本模块通过逐点卷积（pointwise）与深度卷积（depthwise）的组合方式实现高效的下采样，
    在保持通道信息的同时，显著减少空间分辨率。

    属性:
        cv1 (Conv): 逐点卷积层，用于调整通道数；
        cv2 (Conv): 深度卷积层，用于进行空间下采样。

    方法:
        forward: 对输入张量应用 SCDown 模块。

    示例:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """初始化 SCDown 模块，指定输入输出通道数、卷积核大小与步幅。"""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)               # 逐点卷积
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)  # 深度卷积

    def forward(self, x):
        """对输入张量进行卷积与下采样。"""
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision 模块：用于加载任意 torchvision 模型。

    该类允许从 torchvision 库中加载指定模型，并支持加载预训练权重、解包层结构、裁剪输出等自定义行为。

    属性:
        m (nn.Module): 加载并处理后的 torchvision 模型。

    参数:
        c1 (int): 输入通道数。
        c2 (): 输出通道数（未使用）。
        model (str): 要加载的 torchvision 模型名称。
        weights (str, 可选): 是否加载预训练权重。默认为 "DEFAULT"。
        unwrap (bool, 可选): 若为 True，则去除模型末尾的若干层并转为 Sequential。默认为 True。
        truncate (int, 可选): 当 unwrap 为 True 时，从模型末尾裁剪的层数。默认为 2。
        split (bool, 可选): 若为 True，则返回中间每一层的输出列表。默认为 False。
    """

    def __init__(self, c1, c2, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """从 torchvision 加载模型和权重。"""
        import torchvision  # 放置在局部作用域以加快 ultralytics 导入速度

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())[:-truncate]
            if isinstance(layers[0], nn.Sequential):  # 针对 EfficientNet、Swin 等结构中的嵌套层
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*layers)
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()  # 移除分类头

    def forward(self, x):
        """模型前向传播。"""
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


from flash_attn.flash_attn_interface import flash_attn_func
from timm.models.layers import drop_path, trunc_normal_


class AAttn(nn.Module):
    """
    Area-Attention 模块，依赖 flash attention 的高效注意力机制。

    属性:
        dim (int): 输入的隐藏通道数；
        num_heads (int): 注意力划分的头数；
        area (int, 可选): 将特征图划分为多少个区域。默认为 1。

    方法:
        forward: 对输入张量执行注意力机制处理，并输出结果。

    示例:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=3, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)

    注意：
        建议 dim // num_heads 是 32 或 64 的倍数以获得最佳性能。
    """

    def __init__(self, dim, num_heads, area=1):
        """初始化 area-attention 模块，为 YOLO 提供轻量而高效的注意力机制。"""
        super().__init__()

        self.area = area
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)  # QKV 融合卷积
        self.proj = Conv(all_head_dim, dim, 1, act=False)     # 输出映射
        self.pe = Conv(all_head_dim, dim, 9, 1, 4, g=dim, act=False)  # 位置编码

    def forward(self, x):
        """将输入张量 x 应用于 area-attention 或全局注意力处理。"""
        B, C, H, W = x.shape
        N = H * W
        if x.is_cuda:
            qkv = self.qkv(x).flatten(2).transpose(1, 2)
            if self.area > 1:
                qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
                B, N, _ = qkv.shape
            q, k, v = qkv.view(B, N, self.num_heads, self.head_dim * 3).split(
                [self.head_dim, self.head_dim, self.head_dim], dim=3
            )
            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
            if self.area > 1:
                x = x.reshape(B // self.area, N * self.area, C)
                v = v.reshape(B // self.area, N * self.area, C)
                B, N, _ = x.shape
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            v = v.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            qkv = self.qkv(x).flatten(2)
            if self.area > 1:
                qkv = qkv.reshape(B * self.area, C * 3, N // self.area)
                B, _, N = qkv.shape
            q, k, v = qkv.view(B, self.num_heads, self.head_dim * 3, N).split(
                [self.head_dim, self.head_dim, self.head_dim], dim=2
            )
            attn = (q.transpose(-2, -1) @ k) * (self.num_heads ** -0.5)  # 缩放点积注意力
            max_attn = attn.max(dim=-1, keepdim=True).values             # 数值稳定处理
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))
            if self.area > 1:
                x = x.reshape(B // self.area, C, N * self.area)
                v = v.reshape(B // self.area, C, N * self.area)
                B, _, N = x.shape
            x = x.reshape(B, C, H, W)
            v = v.reshape(B, C, H, W)

        x = x + self.pe(v)  # 加上位置编码
        x = self.proj(x)    # 映射回原始通道
        return x
  

class ABlock(nn.Module):
    """
    ABlock 类，实现了带有区域划分的多头注意力机制，用于加速特征提取。

    该类封装了将特征图划分为多个区域后执行多头注意力机制，以及前馈神经网络的功能。

    属性：
        dim (int): 隐藏通道数；
        num_heads (int): 注意力机制中头的数量；
        mlp_ratio (float, 可选): MLP 扩展比例（或隐藏层维度比例），默认为 1.2；
        area (int, 可选): 特征图划分的区域数，默认为 1。

    方法：
        forward: 对输入执行前向传播，依次进行区域注意力与前馈操作。

    示例：
        创建 ABlock 并执行前向传播
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)

    注意：
        推荐 dim//num_heads 为 32 或 64 的倍数。
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """使用区域注意力机制与前馈网络初始化 ABlock 模块。"""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化卷积层权重。"""
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """执行 ABlock 的前向传播，应用区域注意力与前馈网络。"""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):  
    """
    A2C2f 模块，也称为 R-ELAN，是一种集成 ABlock 区域注意力机制的残差增强特征提取结构。

    该类基于 C2f 模块拓展，加入 ABlock 区域注意力机制，实现更快的注意力与特征提取过程。

    属性：
        c1 (int): 输入通道数；
        c2 (int): 输出通道数；
        n (int, 可选): 堆叠的 2×ABlock 模块数量，默认为 1；
        a2 (bool, 可选): 是否使用区域注意力，默认为 True；
        area (int, 可选): 特征图划分的区域数，默认为 1；
        align (bool, 可选): 是否对通道数进行对齐，默认为 False；
        residual (bool, 可选): 是否使用残差（带 layer scale），默认为 False；
        e (float, 可选): 通道扩展比例，默认为 0.5；
        mlp_ratio (float, 可选): MLP 扩展比例，默认为 1.2；
        g (int, 可选): 分组卷积的组数，默认为 1；
        shortcut (bool, 可选): 是否使用残差连接，默认为 True。

    方法：
        forward: 执行 A2C2f 模块的前向传播。
        forward_split: 使用 split() 替代 chunk() 执行前向传播。

    示例：
        >>> import torch
        >>> from ultralytics.nn.modules import A2C2f
        >>> model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, align=False, residual=True, e=0.5)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, align=False, residual=False, e=0.5, mlp_ratio=1.2, g=1, shortcut=True):
        super().__init__()

        self.a2 = a2
        self.residual = residual
        c_ = int(c2 * e)  # 中间通道数
        assert c_ % 32 == 0, "ABlock 的维度必须为 32 的倍数。"

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        if self.a2:
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv((1 + n) * c_, c2, 1)  # 可选使用 FReLU 激活

            if residual:
                self.align = Conv(c1, c2, 1, 1) if align else nn.Identity()
                init_values = 0.01  # 初始化缩放因子
                self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True)
            else:
                self.align, self.gamma = None, None
        else:
            self.cv1 = Conv(c1, 2 * c_, 1, 1)
            self.cv2 = Conv((2 + n) * c_, c2, 1)  # 可选使用 FReLU 激活

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2
            else Bottleneck(c_, c_, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """执行 R-ELAN 模块的前向传播。"""
        if self.a2:
            y = [self.cv1(x)]
            y.extend(m(y[-1]) for m in self.m)
            if self.residual:
                return self.align(x) + (self.gamma * self.cv2(torch.cat(y, 1)).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                return self.cv2(torch.cat(y, 1))
        else:
            y = list(self.cv1(x).chunk(2, 1))
            y.extend(m(y[-1]) for m in self.m)
            return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """使用 split() 替代 chunk() 执行前向传播。"""
        if self.a2:
            y = [self.cv1(x)]
            y.extend(m(y[-1]) for m in self.m)
            if self.residual:
                return self.align(x) + (self.gamma * self.cv2(torch.cat(y, 1)).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                return self.cv2(torch.cat(y, 1))
        else:
            y = list(self.cv1(x).chunk(2, 1))
            y.extend(m(y[-1]) for m in self.m)
            return self.cv2(torch.cat(y, 1))
