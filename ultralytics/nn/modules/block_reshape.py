# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""模块定义：Block 模块。"""

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
    分布式焦点损失（DFL）的积分模块。

    提出自 Generalized Focal Loss：https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """使用给定输入通道数量初始化一个卷积层。"""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """对输入张量 x 应用变换层并返回输出张量。"""
        b, _, a = x.shape  # 批量大小，通道数，anchor数
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 分割模型使用的 mask 原型（Proto）模块。"""

    def __init__(self, c1, c_=256, c2=32):
        """
        初始化 YOLOv8 的 mask 原型模块，指定原型数量和输出 mask 数量。

        输入参数：ch_in 输入通道数、原型数量、输出 mask 通道数。
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # 可替代为 nn.Upsample(scale_factor=2)
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """将输入图像上采样后，依次通过各层处理。"""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    PPHGNetV2 的 StemBlock，包含 5 个卷积层和 1 个最大池化层。

    参考：https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """初始化 SPP 层，设置输入/输出通道数和最大池化的核大小。"""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """执行 PPHGNetV2 主干网络中 Stem 部分的前向传播。"""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    PPHGNetV2 的 HG_Block，包含 2 个卷积层和一个 LightConv。

    参考：https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """初始化 CSP 瓶颈模块，包含指定输入输出通道的卷积操作。"""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # 压缩卷积
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # 激励卷积
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """执行 PPHGNetV2 主干网络中 HGBlock 部分的前向传播。"""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y

class SPP(nn.Module):
    """空间金字塔池化（SPP）层，参考：https://arxiv.org/abs/1406.4729。"""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """使用输入/输出通道数与池化核大小初始化 SPP 层。"""
        super().__init__()
        c_ = c1 // 2  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """SPP 层的前向传播，执行空间金字塔池化操作。"""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """SPPF（快速空间金字塔池化）层，YOLOv5 中由 Glenn Jocher 引入。"""

    def __init__(self, c1, c2, k=5):
        """
        使用给定输入/输出通道和卷积核大小初始化 SPPF 层。

        此模块等效于 SPP(k=(5, 9, 13))。
        """
        super().__init__()
        c_ = c1 // 2  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """通过 Ghost 卷积块执行前向传播。"""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """使用一个卷积的 CSP（Cross Stage Partial）瓶颈结构。"""

    def __init__(self, c1, c2, n=1):
        """初始化包含一个卷积的 CSP 模块，参数为输入通道、输出通道和模块数量。"""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """在 C3 模块中应用交叉卷积。"""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """包含两个卷积的 CSP 瓶颈结构。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """初始化 CSP 模块，包含两个卷积以及可选的 shortcut 连接。"""
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # 可选激活函数 act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # 或 SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """通过两个卷积构成的 CSP 瓶颈执行前向传播。"""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """包含两个卷积的 CSP 瓶颈结构的快速实现。"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """初始化快速版本的 CSP 瓶颈结构，使用 n 个 Bottleneck 模块。"""
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 可选激活函数 act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """C2f 层的标准前向传播。"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """使用 split() 而非 chunk() 的前向传播方式。"""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """包含三个卷积的 CSP 瓶颈结构。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """使用指定通道数、模块数量、shortcut、组数和扩展率初始化 CSP 瓶颈。"""
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # 可选激活函数 act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """通过 C3 模块中三个卷积构建的 CSP 结构执行前向传播。"""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """带有交叉卷积（cross-convolutions）的 C3 模块。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """初始化带交叉卷积的 C3TR 实例并设置默认参数。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """RepC3 模块：使用可重参数化结构的 C3 模块。"""

    def __init__(self, c1, c2, n=3, e=1.0):
        """使用输入通道数、输出通道数和层数初始化 CSP 模块（使用 RepConv）。"""
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """RT-DETR 结构中的 neck 层前向传播实现。"""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """使用 TransformerBlock() 的 C3 模块。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """初始化包含 GhostBottleneck() 的 C3Ghost 模块。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """使用 GhostBottleneck() 的 C3 模块。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """使用不同的池化大小初始化 'SPP' 模块，用于空间金字塔池化。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # 中间通道数
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck 模块，来源：https://github.com/huawei-noah/ghostnet。"""

    def __init__(self, c1, c2, k=3, s=1):
        """使用输入通道、输出通道、卷积核大小、步幅等参数初始化 GhostBottleneck 模块。"""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # 点卷积
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # 深度卷积
            GhostConv(c_, c2, 1, 1, act=False),  # 点卷积线性层
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """对输入张量应用跳跃连接和结果相加。"""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """标准的瓶颈结构（Bottleneck）。"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """初始化标准 Bottleneck 模块，可选跳跃连接，并支持可配置参数。"""
        super().__init__()
        c_ = int(c2 * e)  # 中间通道数
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """对输入数据应用 YOLO 特征金字塔网络结构（FPN）。"""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP 瓶颈结构，来源：https://github.com/WongKinYiu/CrossStagePartialNetworks。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """初始化 CSP Bottleneck，使用输入通道、输出通道、层数、是否使用跳跃连接、分组卷积数、扩展系数等参数。"""
        super().__init__()
        c_ = int(c2 * e)  # 中间通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # 作用于 cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """应用包含三个卷积的 CSP Bottleneck。"""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """具有标准卷积层的 ResNet 模块。"""

    def __init__(self, c1, c2, s=1, e=4):
        """使用给定参数初始化卷积结构。"""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """通过 ResNet 模块执行前向传播。"""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """包含多个 ResNetBlock 的 ResNet 层。"""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """使用指定参数初始化 ResNetLayer。"""
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
    """最大值- Sigmoid 注意力模块（Max-Sigmoid Attention Block）。"""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """使用指定参数初始化 MaxSigmoidAttnBlock。"""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """执行前向处理。"""
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
    """带注意力机制的 C2f 模块。"""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """初始化带注意力机制的 C2f 模块，用于增强特征提取与处理能力。"""
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # 可选激活函数：FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """C2f 层的前向传播。"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """使用 split()（而非 chunk()）进行前向传播。"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn：通过图像感知增强文本特征的注意力模块。"""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """初始化 ImagePoolingAttn，配置关键参数。"""
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
        """对输入图像特征 x 和文本引导特征 text 执行注意力机制。"""
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
    """用于视觉-语言模型中的区域-文本对比学习头模块。"""

    def __init__(self):
        """初始化对比学习头模块，包括区域与文本相似度计算参数。"""
        super().__init__()
        # 使用 -10.0 初始化偏置以保持与其他损失一致
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """执行对比学习的前向传播。"""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    YOLO-World 中使用的 BatchNorm 对比学习头，用于替代 L2 归一化。

    参数：
        embed_dims (int): 文本与图像特征的嵌入维度。
    """

    def __init__(self, embed_dims: int):
        """初始化带有区域-文本相似度计算的对比学习头模块。"""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # 使用 -1.0 初始化可提高稳定性
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """执行对比学习的前向传播。"""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep 风格的瓶颈结构。"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """初始化 RepBottleneck 模块，可配置输入输出通道、残差连接、分组和扩展比例。"""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """可重复的跨阶段部分结构（RepCSP），用于高效特征提取。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """初始化 RepCSP 层，设置通道数、重复次数、残差连接、分组及扩展比例。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # 隐藏通道数
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN 结构模块。"""

    def __init__(self, c1, c2, c3, c4, n=1):
        """使用指定的通道数量、重复次数和卷积层初始化 CSP-ELAN 层。"""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """执行 RepNCSPELAN4 层的前向传播。"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """使用 split()（而非 chunk()）进行前向传播。"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 模块，包含 4 个卷积操作。"""

    def __init__(self, c1, c2, c3, c4):
        """使用指定的通道数初始化 ELAN1 层。"""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv 模块。"""

    def __init__(self, c1, c2):
        """使用卷积层初始化 AConv 模块。"""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """AConv 层的前向传播。"""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown 模块。"""

    def __init__(self, c1, c2):
        """初始化 ADown 模块，用于将输入从 c1 通道下采样到 c2 通道。"""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """ADown 层的前向传播。"""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN 模块。"""

    def __init__(self, c1, c2, c3, k=5):
        """使用卷积层和最大池化层初始化 SPP-ELAN 模块，用于空间金字塔池化。"""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """SPPELAN 层的前向传播。"""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear 模块。"""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """初始化 CBLinear 模块，对输入进行不变换的卷积操作。"""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """CBLinear 层的前向传播。"""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse 模块。"""

    def __init__(self, idx):
        """使用给定索引初始化 CBFuse 模块，用于选择性特征融合。"""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """CBFuse 层的前向传播。"""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """更快实现的含两个卷积的 CSP 瓶颈结构。"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        使用输入通道、输出通道、模块数量、shortcut、分组数和扩展比例初始化 CSP 瓶颈层。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # 可选激活函数 FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """C2f 层的前向传播实现。"""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """更快实现的含两个卷积的 CSP 瓶颈结构。"""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """初始化 C3k2 模块，带有可选 C3k 块的快速 CSP 瓶颈结构。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k 是一种 CSP 瓶颈模块，支持自定义卷积核大小，用于神经网络中的特征提取。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """使用指定通道数、层数和配置参数初始化 C3k 模块。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # 隐藏通道数
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW 是 RepVGG 架构中的一个深度可分离卷积模块。"""

    def __init__(self, ed) -> None:
        """使用深度可分离卷积层初始化 RepVGGDW，以实现高效计算。"""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)  # 7x7 深度卷积
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)  # 3x3 深度卷积
        self.dim = ed
        self.act = nn.SiLU()  # 使用 SiLU 激活函数

    def forward(self, x):
        """
        执行 RepVGGDW 模块的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            (torch.Tensor): 应用深度可分离卷积后的输出张量。
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        执行未融合卷积时的前向传播（只使用主分支）。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            (torch.Tensor): 只经过主分支卷积的输出张量。
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        融合 RepVGGDW 模块中的两个卷积层。

        该方法将主分支和辅助分支的卷积层进行权重和偏置融合，
        提高推理速度，适用于部署阶段。
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        # 将 3x3 的权重 pad 成 7x7，以便和主分支形状匹配
        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1  # 删除辅助分支（部署时不再需要）


class CIB(nn.Module):
    """
    条件恒等模块（Conditional Identity Block，简称 CIB）。

    参数:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        shortcut (bool, 可选): 是否添加残差连接。默认为 True。
        e (float, 可选): 隐藏层通道扩展比例。默认为 0.5。
        lk (bool, 可选): 是否在第三层使用 RepVGGDW。默认为 False。
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """根据是否使用残差、扩展因子、RepVGGDW 初始化模块结构。"""
        super().__init__()
        c_ = int(c2 * e)  # 中间隐藏通道
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),  # 深度卷积
            Conv(c1, 2 * c_, 1),  # 1x1 卷积压缩 + 扩展
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),  # 是否使用轻量模块
            Conv(2 * c_, c2, 1),  # 通道恢复
            Conv(c2, c2, 3, g=c2),  # 最后深度卷积
        )

        self.add = shortcut and c1 == c2  # 是否添加残差（通道一致）

    def forward(self, x):
        """
        执行 CIB 模块的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            (torch.Tensor): 输出张量（是否加残差）。
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB 表示由多个 CIB 模块堆叠而成的改进卷积块。

    参数:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        n (int, 可选): 堆叠的 CIB 模块数量。默认 1。
        shortcut (bool, 可选): 是否使用残差连接。默认 False。
        lk (bool, 可选): 是否使用轻量化（local key）连接结构。默认 False。
        g (int, 可选): 分组卷积中的组数。默认 1。
        e (float, 可选): 通道扩展因子。默认 0.5。
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """初始化 C2fCIB 模块，内部由多个 CIB 模块组成。"""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention 模块，用于对输入张量执行自注意力机制。

    参数:
        dim (int): 输入特征的维度。
        num_heads (int): 注意力头的数量。
        attn_ratio (float): 每个注意力头中键向量维度与头部维度的比例。

    属性:
        num_heads (int): 注意力头数。
        head_dim (int): 每个注意力头的维度。
        key_dim (int): 注意力键的维度。
        scale (float): 用于缩放注意力分数的系数（防止梯度爆炸）。
        qkv (Conv): 用于生成查询、键、值的卷积层。
        proj (Conv): 用于注意力值投影的卷积层。
        pe (Conv): 用于位置编码的卷积层。
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """初始化多头注意力模块，包括 QKV 卷积和位置编码卷积。"""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个注意力头的维度
        self.key_dim = int(self.head_dim * attn_ratio)  # 键向量维度
        self.scale = self.key_dim**-0.5  # 缩放系数
        nh_kd = self.key_dim * num_heads  # 所有注意力头的键总维度
        h = dim + nh_kd * 2  # Q, K, V 的维度总和
        self.qkv = Conv(dim, h, 1, act=False)  # QKV 的线性层
        self.proj = Conv(dim, dim, 1, act=False)  # 输出映射
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)  # 深度卷积形式的位置编码

    def forward(self, x):
        """
        Attention 模块的前向传播过程。

        参数：
            x (torch.Tensor): 输入张量。

        返回：
            (torch.Tensor): 经过自注意力机制处理后的输出张量。
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
    实现位置敏感注意力（Position-Sensitive Attention）的 PSABlock 类。

    该类封装了多头注意力机制和前馈神经网络模块，并支持是否使用残差连接。

    属性：
        attn (Attention): 多头注意力模块。
        ffn (nn.Sequential): 前馈神经网络模块。
        add (bool): 是否使用残差连接的标志位。

    方法：
        forward: 对输入进行注意力和前馈操作的前向传播。

    示例：
        创建 PSABlock 并执行前向传播
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """使用注意力模块和前馈层初始化 PSABlock，以增强特征提取能力。"""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """执行 PSABlock 的前向传播，对输入张量应用注意力和前馈模块。"""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA 模块，用于在神经网络中实现位置敏感注意力机制。

    本模块封装了位置敏感注意力机制和前馈网络，对输入张量进行增强的特征提取和处理。

    属性：
        c (int): 初始卷积后中间通道数。
        cv1 (Conv): 用于将输入通道压缩为 2*c 的 1x1 卷积层。
        cv2 (Conv): 用于将输出通道还原为 c 的 1x1 卷积层。
        attn (Attention): 用于位置敏感注意力的注意力模块。
        ffn (nn.Sequential): 用于后续特征处理的前馈神经网络。

    方法：
        forward: 对输入张量应用位置敏感注意力和前馈网络。

    示例：
        创建 PSA 模块并应用于输入张量
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """使用输入/输出通道和注意力机制初始化 PSA 模块，用于特征提取。"""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """执行 PSA 模块的前向传播，对输入进行注意力机制和前馈网络的处理。"""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    带有注意力机制的 C2PSA 模块，用于增强特征提取与处理能力。

    该模块实现了卷积块与多层 PSABlock 的结合，用于进行自注意力和前馈操作。

    属性：
        c (int): 中间通道数。
        cv1 (Conv): 1x1 卷积层，将输入通道压缩为 2*c。
        cv2 (Conv): 1x1 卷积层，将通道数还原为原始维度。
        m (nn.Sequential): 多层 PSABlock 的顺序容器。

    方法：
        forward: 执行 C2PSA 模块的前向传播，应用注意力与前馈网络。

    说明：
        本模块与 PSA 类似，但结构上更适合堆叠多个 PSABlock。

    示例：
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """使用输入/输出通道数、层数、扩展比初始化 C2PSA 模块。"""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """将输入张量 x 通过一系列 PSA 块处理并返回转换后的结果。"""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    使用 PSA 块增强特征提取能力的 C2fPSA 模块。

    该类在原有 C2f 模块基础上，引入 PSA 注意力块，以提升注意力机制效果与特征表达能力。

    属性：
        c (int): 中间通道数。
        cv1 (Conv): 1x1 卷积层，将输入通道数压缩至 2*c。
        cv2 (Conv): 1x1 卷积层，将通道数还原为原始维度。
        m (nn.ModuleList): PSA 模块列表，用于堆叠特征提取结构。

    方法：
        forward: 执行 C2fPSA 模块的前向传播。
        forward_split: 使用 split() 而不是 chunk() 执行前向传播。

    示例：
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """初始化 C2fPSA 模块，是 C2f 的变体，引入 PSA 块用于增强特征提取。"""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    使用可分离卷积进行下采样的 SCDown 模块。

    该模块结合了逐点卷积（pointwise）和深度卷积（depthwise）来实现下采样，
    能在保留通道信息的同时高效地降低输入张量的空间尺寸。

    属性：
        cv1 (Conv): 逐点卷积层，用于通道数的转换。
        cv2 (Conv): 深度卷积层，用于空间下采样。

    方法：
        forward: 对输入张量应用下采样操作。

    示例：
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """使用指定的输入/输出通道数、卷积核大小和步长初始化 SCDown 模块。"""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """在 SCDown 模块中对输入张量执行卷积和下采样。"""
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision 模型模块，用于加载任意 torchvision 模型。

    该类支持从 torchvision 库中加载模型、加载预训练权重，并通过裁剪或解包层来进行自定义。

    属性：
        m (nn.Module): 加载的 torchvision 模型（可能被裁剪或解包）。

    参数：
        c1 (int): 输入通道数。
        c2 (): 输出通道数。
        model (str): 要加载的 torchvision 模型名称。
        weights (str, optional): 要加载的预训练权重，默认是 "DEFAULT"。
        unwrap (bool, optional): 如果为 True，则展开模型并去掉最后 `truncate` 层。默认是 True。
        truncate (int, optional): 如果 unwrap 为 True，表示要去除的末尾层数。默认是 2。
        split (bool, optional): 如果为 True，返回中间每个子模块的输出组成的列表。默认是 False。
    """

    def __init__(self, c1, c2, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """从 torchvision 加载模型及其权重。"""
        import torchvision  # 本地导入，加快 'import ultralytics' 的速度

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())[:-truncate]
            if isinstance(layers[0], nn.Sequential):  # 二级结构，如 EfficientNet、Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*layers)
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """模型的前向传播过程。"""
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


from flash_attn.flash_attn_interface import flash_attn_func
from timm.models.layers import drop_path, trunc_normal_


class DropPath(nn.Module):
    """DropPath（随机深度），用于残差块中按样本随机丢弃主路径。"""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    """多层感知机（MLP）模块，包含两层 1x1 卷积，用于通道维度的非线性变换。"""

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w = nn.Sequential(
            Conv(in_features, hidden_features, 1),
            Conv(hidden_features, out_features, 1, act=False)
        )

    def forward(self, x):
        return self.w(x)


class AAttn(nn.Module):
    def __init__(self, dim, num_heads=2, win=True, area=4, flip=False):
        super().__init__()

        self.flip = flip
        self.win = win
        self.area = self.area_set = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 9, 1, 4, g=dim, act=False)
        

    def forward(self, x):
        B, C, H, W = x.shape
        if self.win:
            try:
                x = x.reshape(B * self.area, C, H // self.area, W)
            except RuntimeError:
                self.area = 2
                x = x.reshape(B * self.area, C, H // self.area, W)
            H //= self.area
            B *= self.area
        
        N = H * W
        if x.is_cuda:
            qkv = self.qkv(x).flatten(2).transpose(1, 2)
            q, k, v = qkv.view(B, N, self.num_heads, self.head_dim * 3).split(
                [self.head_dim, self.head_dim, self.head_dim], dim=3
            )
            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            )

            if self.win:
                x = x.reshape(B, N, C).reshape(B // self.area, N * self.area, C).reshape(B // self.area, H * self.area, W, C).permute(0, 3, 1, 2).to(q.dtype)
                v = v.reshape(B, N, C).reshape(B // self.area, N * self.area, C).reshape(B // self.area, H * self.area, W, C).permute(0, 3, 1, 2)
            else:
                x = x.reshape(B, N, C).reshape(B, H, W, C).permute(0, 3, 1, 2).to(q.dtype)
                v = v.reshape(B, N, C).reshape(B, H, W, C).permute(0, 3, 1, 2).to(q.dtype)
            
            x = x + self.pe(v)
        
        else:
            qkv = self.qkv(x)
            q, k, v = qkv.view(B, self.num_heads, self.head_dim * 3, N).split(
                [self.head_dim, self.head_dim, self.head_dim], dim=2
            )
            attn = (q.transpose(-2, -1) @ k) * (self.num_heads ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values 
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))

            if self.win:
                x = x.reshape(B, C, H, W).reshape(B // self.area, C, H * self.area, W)
                v = v.reshape(B, C, H, W).reshape(B // self.area, C, H * self.area, W)
            else:
                x = x.reshape(B, C, H, W)
                v = v.reshape(B, C, H, W)

            x = x + self.pe(v)
        
        self.area = self.area_set
        
        x = self.proj(x)
        return x


class ABlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=1.5, area=4, flip=False, win=True, drop_path=0.0):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn = AAttn(dim, num_heads=num_heads, area=area, flip=flip, win=win)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        模块的前向传播过程。

        对输入应用注意力机制和 MLP，并添加 DropPath 以增强泛化能力。
        """
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class A2C2f(C2f):
    """使用两层卷积的更快版本的 CSP Bottleneck（Cross Stage Partial）模块。"""
    def __init__(self, c1, c2, n=1, a2=True, win=True, num_heads=2, drop_path=0.0, e=0.5, area=4, mlp_ratio=1.2, g=1, shortcut=True):
        """
        初始化 C3k2 模块，这是一个快速实现的 CSP Bottleneck，采用两层卷积，并可选集成 C3k 注意力模块。

        参数：
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): 层数
            a2 (bool): 是否启用 ABlock 注意力模块
            win (bool): 是否使用局部窗口注意力
            num_heads (int): 注意力头数
            drop_path (float): DropPath 概率
            e (float): 通道扩展比例
            area (int): 局部窗口面积
            mlp_ratio (float): MLP 层扩展比例
            g (int): 分组卷积的组数
            shortcut (bool): 是否使用残差连接
        """
        super().__init__(c1, c2, n, shortcut, g, e)

        self.c = int(c2 * e)  # 中间通道数
        self.attn_c = round(self.c / 64) * 64  # 对齐到 64 的倍数以适配注意力头数

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        # self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 可选使用 FReLU 激活
        self.cv2 = Conv(2 * self.c + n * self.attn_c, c2, 1)  # 输出融合层，可选 FReLU 激活

        self.m = nn.ModuleList(
            AC3(self.attn_c, self.attn_c, 2, 1., num_heads, area, mlp_ratio, win, drop_path) if a2 
            else Bottleneck(self.attn_c, self.attn_c, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """前向传播过程，使用 split() 而非 chunk() 对特征进行划分与处理。"""
        y = self.cv1(x).split((2 * self.c - self.attn_c, self.attn_c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class AC3(nn.Module):
    """C3k 是一种 CSP 瓶颈模块，可自定义卷积核大小，用于神经网络中的特征提取。"""

    def __init__(self, c1, c2, n=1, e=1., num_heads=2, area=4, mlp_ratio=1.5, win=True, drop_path=0.0):
        """
        使用指定通道数、层数及配置参数初始化 C3k 模块。

        参数：
            c1 (int): 输入通道
            c2 (int): 输出通道
            n (int): 层数
            e (float): 通道扩展系数
            num_heads (int): 多头注意力头数
            area (int): 注意力局部窗口面积
            mlp_ratio (float): 前馈网络扩展比例
            win (bool): 是否启用窗口机制
            drop_path (float): DropPath 概率
        """
        super().__init__()
        c_ = int(c2 * e)  # 中间通道数
        self.cv1 = Conv(c1, c_, 5, 1, 2, g=c_, act=False)  # 可理解为深度卷积
        # self.cv2 = Conv(2 * c_, c2, 1)  # 可选：使用 FReLU 激活
        self.m = nn.Sequential(*(
            ABlock(dim=c_, num_heads=num_heads, mlp_ratio=mlp_ratio, win=win, area=area, drop_path=drop_path)
            for i in range(n)
        ))

    def forward(self, x):
        """通过包含注意力机制的 CSP 瓶颈模块执行前向传播。"""
        # return self.m(x)
        return self.m(x) + self.cv1(x)
        # return self.cv2(torch.cat((self.m(x), self.cv1(x)), 1))
