# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license
"""
Ultralytics 模块。

示例：
    使用 Netron 可视化某个模块。
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f"{m._get_name()}.onnx"
    torch.onnx.export(m, x, f)
    os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
    ```
"""

from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
    A2C2f,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    Index,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect, v10Detect
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    # 卷积模块
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    
    # 注意力机制
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    
    # 拼接模块
    "Concat",
    
    # Transformer 相关模块
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    
    # 模块组件
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3k2",
    "SCDown",
    "C2fPSA",
    "C2PSA",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    
    # 检测与分割任务模块
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    
    # Transformer 编码器层
    "TransformerEncoderLayer",
    
    # 可重参数化结构
    "RepC3",
    
    # RT-DETR 解码器
    "RTDETRDecoder",
    
    # 注意力机制（高级）
    "AIFI",
    
    # 可变形 Transformer 解码器及其子结构
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    
    # 多层感知器
    "MLP",
    
    # ResNet 层
    "ResNetLayer",
    
    # 面向对象边界框检测
    "OBB",
    
    # 多尺度世界检测
    "WorldDetect",
    
    # YOLOv10 检测头
    "v10Detect",
    
    # 图像池化注意力机制
    "ImagePoolingAttn",
    
    # 对比学习头
    "ContrastiveHead",
    "BNContrastiveHead",
    
    # 可重参数化的 NCSPELAN 模块
    "RepNCSPELAN4",
    
    # 下采样与增强模块
    "ADown",
    
    # 可分支的 ELAN 模块
    "SPPELAN",
    
    # CB 模块组合结构
    "CBFuse",
    "CBLinear",
    
    # 注意力卷积与增强模块
    "AConv",
    "ELAN1",
    
    # 可重参数化 VGG 轻量模块
    "RepVGGDW",
    
    # 可交叉信息块结构
    "CIB",
    "C2fCIB",
    
    # 注意力模块（通用）
    "Attention",
    "PSA",
    
    # TorchVision 兼容模块
    "TorchVision",
    
    # 索引层
    "Index",
    
    # 可变维度的 C2f 结构
    "A2C2f"
)
