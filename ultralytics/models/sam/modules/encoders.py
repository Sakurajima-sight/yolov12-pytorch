# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import LayerNorm2d

from .blocks import (
    Block,
    CXBlock,
    Fuser,
    MaskDownSampler,
    MultiScaleBlock,
    PatchEmbed,
    PositionEmbeddingRandom,
    PositionEmbeddingSine,
)


class ImageEncoderViT(nn.Module):
    """
    使用视觉 Transformer (ViT) 架构的图像编码器，将图像编码为紧凑的潜在空间。

    该类通过将图像分割为补丁、应用 Transformer 块并通过颈部模块生成最终的编码表示来处理图像。

    属性:
        img_size (int): 输入图像的尺寸，假定为正方形。
        patch_embed (PatchEmbed): 用于补丁嵌入的模块。
        pos_embed (nn.Parameter | None): 用于补丁的绝对位置嵌入。
        blocks (nn.ModuleList): 用于处理补丁嵌入的 Transformer 块列表。
        neck (nn.Sequential): 用于进一步处理输出的颈部模块。

    方法:
        forward: 通过补丁嵌入、位置嵌入、Transformer 块和颈部模块处理输入。

    示例:
        >>> import torch
        >>> encoder = ImageEncoderViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
        >>> input_image = torch.randn(1, 3, 224, 224)
        >>> output = encoder(input_image)
        >>> print(output.shape)
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        使用视觉 Transformer 架构初始化 ImageEncoderViT 实例，用于编码图像。

        参数:
            img_size (int): 输入图像尺寸，假定为正方形。
            patch_size (int): 图像补丁的大小。
            in_chans (int): 输入图像的通道数。
            embed_dim (int): 补丁嵌入的维度。
            depth (int): Transformer 块的数量。
            num_heads (int): 每个块中的注意力头数。
            mlp_ratio (float): MLP 隐藏层维度与嵌入维度的比率。
            out_chans (int): 颈部模块的输出通道数。
            qkv_bias (bool): 如果为 True，则在查询、键、值投影中添加可学习的偏置。
            norm_layer (Type[nn.Module]): 要使用的归一化层类型。
            act_layer (Type[nn.Module]): 要使用的激活层类型。
            use_abs_pos (bool): 如果为 True，则使用绝对位置嵌入。
            use_rel_pos (bool): 如果为 True，则向注意力图添加相对位置嵌入。
            rel_pos_zero_init (bool): 如果为 True，则将相对位置参数初始化为零。
            window_size (int): 窗口化注意力块的注意力窗口大小。
            global_attn_indexes (Tuple[int, ...]): 使用全局注意力的块的索引。

        属性:
            img_size (int): 输入图像的尺寸。
            patch_embed (PatchEmbed): 用于补丁嵌入的模块。
            pos_embed (nn.Parameter | None): 用于补丁的绝对位置嵌入。
            blocks (nn.ModuleList): Transformer 块的列表。
            neck (nn.Sequential): 最终处理的颈部模块。

        示例:
            >>> encoder = ImageEncoderViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
            >>> input_image = torch.randn(1, 3, 224, 224)
            >>> output = encoder(input_image)
            >>> print(output.shape)
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # 初始化绝对位置嵌入，预设图像大小。
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过补丁嵌入、位置嵌入、Transformer 块和颈部模块处理输入数据。"""
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = (
                F.interpolate(self.pos_embed.permute(0, 3, 1, 2), scale_factor=self.img_size / 1024).permute(0, 2, 3, 1)
                if self.img_size != 1024
                else self.pos_embed
            )
            x = x + pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.neck(x.permute(0, 3, 1, 2))


class PromptEncoder(nn.Module):
    """
    对不同类型的提示进行编码，供 SAM 的掩码解码器使用，生成稀疏和密集的嵌入表示。

    属性说明:
        embed_dim (int): 嵌入的维度。
        input_image_size (Tuple[int, int]): 输入图像的大小，格式为 (H, W)。
        image_embedding_size (Tuple[int, int]): 图像嵌入的空间大小，格式为 (H, W)。
        pe_layer (PositionEmbeddingRandom): 随机位置嵌入的模块。
        num_point_embeddings (int): 用于不同类型点的点嵌入数量。
        point_embeddings (nn.ModuleList): 点嵌入的列表。
        not_a_point_embed (nn.Embedding): 用于表示非标签点的嵌入。
        mask_input_size (Tuple[int, int]): 输入掩码的大小。
        mask_downscaling (nn.Sequential): 用于下采样掩码的神经网络。
        no_mask_embed (nn.Embedding): 用于未提供掩码的情况的嵌入。

    方法说明:
        get_dense_pe: 返回用于编码点提示的位置信息编码。
        forward: 对不同类型的提示进行嵌入，返回稀疏和密集的嵌入表示。

    示例用法:
        >>> prompt_encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
        >>> points = (torch.rand(1, 5, 2), torch.randint(0, 4, (1, 5)))
        >>> boxes = torch.rand(1, 2, 2)
        >>> masks = torch.rand(1, 1, 256, 256)
        >>> sparse_embeddings, dense_embeddings = prompt_encoder(points, boxes, masks)
        >>> print(sparse_embeddings.shape, dense_embeddings.shape)
        torch.Size([1, 7, 256]) torch.Size([1, 256, 64, 64])
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        初始化 PromptEncoder 模块，用于编码各种类型的提示。

        该模块将不同类型的提示（点、框、掩码）编码为 SAM 的掩码解码器输入，
        生成稀疏和密集的嵌入表示。

        参数说明:
            embed_dim (int): 嵌入的维度。
            image_embedding_size (Tuple[int, int]): 图像嵌入的空间大小，格式为 (H, W)。
            input_image_size (Tuple[int, int]): 输入图像的填充大小，格式为 (H, W)。
            mask_in_chans (int): 用于编码输入掩码的隐藏通道数。
            activation (Type[nn.Module]): 用于编码输入掩码时的激活函数。

        属性说明:
            embed_dim (int): 嵌入的维度。
            input_image_size (Tuple[int, int]): 输入图像的大小，格式为 (H, W)。
            image_embedding_size (Tuple[int, int]): 图像嵌入的空间大小，格式为 (H, W)。
            pe_layer (PositionEmbeddingRandom): 随机位置嵌入的模块。
            num_point_embeddings (int): 用于不同类型点的点嵌入数量。
            point_embeddings (nn.ModuleList): 点嵌入的列表。
            not_a_point_embed (nn.Embedding): 用于表示非标签点的嵌入。
            mask_input_size (Tuple[int, int]): 输入掩码的大小。
            mask_downscaling (nn.Sequential): 用于下采样掩码的神经网络。

        示例用法:
            >>> prompt_encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
            >>> points = (torch.rand(1, 5, 2), torch.randint(0, 4, (1, 5)))
            >>> boxes = torch.rand(1, 2, 2)
            >>> masks = torch.rand(1, 1, 256, 256)
            >>> sparse_embeddings, dense_embeddings = prompt_encoder(points, boxes, masks)
            >>> print(sparse_embeddings.shape, dense_embeddings.shape)
            torch.Size([1, 7, 256]) torch.Size([1, 256, 64, 64])
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg 点 + 2 个框角点
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        返回用于编码点提示的密集位置编码。

        该方法生成一个与图像编码形状匹配的密集位置编码。此编码用于在处理点提示时，向模型提供空间信息。

        返回:
            (torch.Tensor): 位置编码张量，形状为 (1, embed_dim, H, W)，其中 H 和 W 是图像嵌入大小的高度和宽度。

        示例:
            >>> prompt_encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
            >>> dense_pe = prompt_encoder.get_dense_pe()
            >>> print(dense_pe.shape)
            torch.Size([1, 256, 64, 64])
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """通过应用位置编码和标签特定的嵌入，嵌入点提示。"""
        points = points + 0.5  # 将点移动到像素中心
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        point_embedding[labels == 2] += self.point_embeddings[2].weight
        point_embedding[labels == 3] += self.point_embeddings[3].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """通过应用位置编码并添加角落嵌入，嵌入框提示。"""
        boxes = boxes + 0.5  # 将框移动到像素中心
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """通过下采样并通过卷积层处理掩码输入，嵌入掩码。"""
        return self.mask_downscaling(masks)

    @staticmethod
    def _get_batch_size(
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """根据输入提示的批量大小获取输出的批量大小。"""
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        """返回第一个点嵌入的权重张量的设备。"""
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        嵌入不同类型的提示，返回稀疏和密集的嵌入。

        参数:
            points (Tuple[torch.Tensor, torch.Tensor] | None): 要嵌入的点坐标和标签。第一个张量包含坐标，形状为 (B, N, 2)，第二个张量包含标签，形状为 (B, N)。
            boxes (torch.Tensor | None): 要嵌入的框，形状为 (B, M, 2, 2)，其中 M 是框的数量。
            masks (torch.Tensor | None): 要嵌入的掩码，形状为 (B, 1, H, W)。

        返回:
            (Tuple[torch.Tensor, torch.Tensor]): 一个包含以下内容的元组:
                - sparse_embeddings (torch.Tensor): 点和框的稀疏嵌入，形状为 (B, N, embed_dim)。
                - dense_embeddings (torch.Tensor): 掩码的密集嵌入，形状为 (B, embed_dim, embed_H, embed_W)。

        示例:
            >>> encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
            >>> points = (torch.rand(1, 5, 2), torch.randint(0, 4, (1, 5)))
            >>> boxes = torch.rand(1, 2, 2, 2)
            >>> masks = torch.rand(1, 1, 256, 256)
            >>> sparse_emb, dense_emb = encoder(points, boxes, masks)
            >>> print(sparse_emb.shape, dense_emb.shape)
            torch.Size([1, 7, 256]) torch.Size([1, 256, 64, 64])
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class MemoryEncoder(nn.Module):
    """
    将像素特征和掩码编码为内存表示，以实现高效的图像分割。

    该类处理像素级特征和掩码，将其融合生成适用于下游任务（如 SAM（Segment Anything Model））的编码内存表示。

    属性:
        mask_downsampler (MaskDownSampler): 用于下采样输入掩码的模块。
        pix_feat_proj (nn.Conv2d): 用于将像素特征投影到另一个空间的卷积层。
        fuser (Fuser): 用于融合像素特征和掩码的模块。
        position_encoding (PositionEmbeddingSine): 用于向特征添加位置编码的模块。
        out_proj (nn.Module): 输出投影层，可以是 nn.Identity 或 nn.Conv2d。

    方法:
        forward: 处理输入像素特征和掩码，生成编码的内存表示。

    示例:
        >>> import torch
        >>> encoder = MemoryEncoder(out_dim=256, in_dim=256)
        >>> pix_feat = torch.randn(1, 256, 64, 64)
        >>> masks = torch.randn(1, 1, 64, 64)
        >>> encoded_feat, pos = encoder(pix_feat, masks)
        >>> print(encoded_feat.shape, pos.shape)
        torch.Size([1, 256, 64, 64]) torch.Size([1, 128, 64, 64])
    """

    def __init__(
        self,
        out_dim,
        in_dim=256,  # pix_feats 的 in_dim
    ):
        """初始化 MemoryEncoder，用于将像素特征和掩码编码为内存表示。"""
        super().__init__()

        self.mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1)

        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = Fuser(CXBlock(dim=256), num_layers=2)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=64)
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """处理像素特征和掩码，生成用于分割的编码内存表示。"""
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        # 融合像素特征和下采样后的掩码，如果视觉特征在 CPU 上，将其转换为 CUDA
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}


class ImageEncoder(nn.Module):
    """
    使用 trunk-neck 架构编码图像，生成多尺度特征和位置编码。

    该类结合了用于特征提取的 trunk 网络和用于特征细化及位置编码生成的 neck 网络。
    它可以选择性地丢弃最低分辨率的特征。

    属性：
        trunk (nn.Module): 用于初步特征提取的 trunk 网络。
        neck (nn.Module): 用于特征细化和位置编码生成的 neck 网络。
        scalp (int): 丢弃的最低分辨率特征级别数。

    方法：
        forward: 通过 trunk 和 neck 网络处理输入图像。

    示例：
        >>> trunk = SomeTrunkNetwork()
        >>> neck = SomeNeckNetwork()
        >>> encoder = ImageEncoder(trunk, neck, scalp=1)
        >>> image = torch.randn(1, 3, 224, 224)
        >>> output = encoder(image)
        >>> print(output.keys())
        dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
    """

    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        """初始化 ImageEncoder，使用 trunk 和 neck 网络进行特征提取和细化。"""
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert self.trunk.channel_list == self.neck.backbone_channel_list, (
            f"trunk {self.trunk.channel_list} 和 neck {self.neck.backbone_channel_list} 的通道维度不匹配。"
        )

    def forward(self, sample: torch.Tensor):
        """通过补丁嵌入、位置编码、transformer 块和 neck 模块编码输入。"""
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # 丢弃最低分辨率的特征
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        return {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }


class FpnNeck(nn.Module):
    """
    用于目标检测模型中的多尺度特征融合的特征金字塔网络（FPN）变体。

    该 FPN 变体去除了输出卷积，并使用双三次插值进行特征调整，
    类似于 ViT 的位置编码插值。

    属性：
        position_encoding (PositionEmbeddingSine): 正弦位置编码模块。
        convs (nn.ModuleList): 每个骨干网络级别的卷积层列表。
        backbone_channel_list (List[int]): 来自骨干网络的通道维度列表。
        fpn_interp_model (str): FPN 特征调整的插值模式。
        fuse_type (str): 特征融合的类型，'sum' 或 'avg'。
        fpn_top_down_levels (List[int]): 输出中包含自上而下特征的层级。

    方法：
        forward: 执行 FPN neck 的前向传播。

    示例：
        >>> backbone_channels = [64, 128, 256, 512]
        >>> fpn_neck = FpnNeck(256, backbone_channels)
        >>> inputs = [torch.rand(1, c, 32, 32) for c in backbone_channels]
        >>> outputs, positions = fpn_neck(inputs)
        >>> print(len(outputs), len(positions))
        4 4
    """

    def __init__(
        self,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """
        初始化一个修改版的特征金字塔网络（FPN）颈部模块。

        这个 FPN 变体去除了输出卷积，采用了类似于 ViT 位置嵌入插值的双三次插值来进行特征大小调整。

        参数:
            d_model (int): 模型的维度。
            backbone_channel_list (List[int]): 来自主干网络的通道维度列表。
            kernel_size (int): 卷积层的核大小。
            stride (int): 卷积层的步幅。
            padding (int): 卷积层的填充。
            fpn_interp_model (str): FPN 特征调整的插值模式。
            fuse_type (str): 特征融合的类型，可以是 'sum' 或 'avg'。
            fpn_top_down_levels (Optional[List[int]]): 输出中具有自顶向下特征的级别。

        示例:
            >>> backbone_channels = [64, 128, 256, 512]
            >>> fpn_neck = FpnNeck(256, backbone_channels)
            >>> print(fpn_neck)
        """
        super().__init__()
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=256)
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in {"sum", "avg"}
        self.fuse_type = fuse_type

        # 输出中具有自顶向下特征的级别
        # 例如，如果 fpn_top_down_levels 是 [2, 3]，则只有级别 2 和 3 的输出
        # 才会进行自顶向下传播，而级别 0 和 级别 1 仅包含来自同一主干级别的横向特征。
        if fpn_top_down_levels is None:
            # 默认情况下，所有级别都会有自顶向下特征
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):
        """
        执行通过特征金字塔网络（FPN）颈部的前向传播。

        该方法通过 FPN 处理来自主干网络的输入张量列表，应用横向连接和自顶向下的特征融合。
        它生成输出特征图和相应的位置信息编码。

        参数:
            xs (List[torch.Tensor]): 来自主干网络的输入张量列表，每个张量的形状为 (B, C, H, W)。

        返回:
            (Tuple[List[torch.Tensor], List[torch.Tensor]]): 一个元组，包含：
                - out (List[torch.Tensor]): 经 FPN 处理后的输出特征图列表，每个张量的形状为
                  (B, d_model, H, W)。
                - pos (List[torch.Tensor]): 对应每个输出特征图的位置信息编码。

        示例:
            >>> fpn_neck = FpnNeck(d_model=256, backbone_channel_list=[64, 128, 256, 512])
            >>> inputs = [torch.rand(1, c, 32, 32) for c in [64, 128, 256, 512]]
            >>> outputs, positions = fpn_neck(inputs)
            >>> print(len(outputs), len(positions))
            4 4
        """
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn 前向传播
        # 参考 https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # 按照自顶向下的顺序进行前向传播（从低分辨率到高分辨率）
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(None if self.fpn_interp_model == "nearest" else False),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos
1

class Hiera(nn.Module):
    """
    用于图像处理任务的高效多尺度特征提取的分层视觉 Transformer。

    该类实现了 Hiera 模型，这是一种分层视觉 Transformer 架构，旨在高效地进行多尺度特征提取。
    它使用一系列 Transformer 块，并将这些块组织成多个阶段，同时支持可选的池化和全局注意力机制。

    属性说明:
        window_spec (Tuple[int, ...]): 每个阶段的窗口大小。
        q_stride (Tuple[int, int]): 阶段间的下采样步幅。
        stage_ends (List[int]): 每个阶段中最后一个块的索引。
        q_pool_blocks (List[int]): 进行池化操作的块的索引。
        return_interm_layers (bool): 是否返回每个阶段的中间层输出。
        patch_embed (PatchEmbed): 补丁嵌入模块。
        global_att_blocks (Tuple[int, ...]): 含有全局注意力的块的索引。
        window_pos_embed_bkg_spatial_size (Tuple[int, int]): 窗口位置嵌入背景的空间大小。
        pos_embed (nn.Parameter): 背景的位置信息嵌入。
        pos_embed_window (nn.Parameter): 窗口的位置信息嵌入。
        blocks (nn.ModuleList): MultiScaleBlock 模块的列表。
        channel_list (List[int]): 每个阶段的输出通道维度列表。

    方法说明:
        _get_pos_embed: 通过插值并组合窗口和背景嵌入来生成位置嵌入。
        forward: 通过 Hiera 模型执行前向传播。

    示例用法:
        >>> model = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3))
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output_features = model(input_tensor)
        >>> for feat in output_features:
        ...     print(feat.shape)
    """

    def __init__(
        self,
        embed_dim: int = 96,  # 初始嵌入维度
        num_heads: int = 1,  # 初始头数
        drop_path_rate: float = 0.0,  # 随机深度
        q_pool: int = 3,  # q_pool 阶段的数量
        q_stride: Tuple[int, int] = (2, 2),  # 阶段间的下采样步幅
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # 每个阶段的块数量
        dim_mul: float = 2.0,  # 阶段切换时的维度倍数
        head_mul: float = 2.0,  # 阶段切换时的头数倍数
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # 每个阶段的窗口大小，若不使用全局注意力时
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # 在这些块中使用全局注意力
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        return_interm_layers=True,  # 是否返回每个阶段的特征
    ):
        """初始化 Hiera 模型，配置其分层视觉 Transformer 架构。"""
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
            kernel_size=(7, 7),
            stride=(4, 4),
            padding=(3, 3),
        )
        # 哪些块使用全局注意力？
        self.global_att_blocks = global_att_blocks

        # 窗口位置嵌入（https://arxiv.org/abs/2311.05613）
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 随机深度衰减规则

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # 当前块比前一阶段滞后一个，因此下一阶段的第一个块使用前一阶段的初始窗口大小
            # 和当前阶段的最终窗口大小
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        """通过插值并组合窗口和背景嵌入来生成位置嵌入。"""
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile([x // y for x, y in zip(pos_embed.shape, window_embed.shape)])
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """通过 Hiera 模型执行前向传播，从输入图像中提取多尺度特征。"""
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # 添加位置嵌入
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs
