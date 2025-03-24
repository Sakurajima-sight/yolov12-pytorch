# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from ultralytics.nn.modules import MLP, LayerNorm2d


class MaskDecoder(nn.Module):
    """
    用于生成掩码及其相关质量评分的解码模块，采用 Transformer 架构。

    该类接收图像和提示的嵌入，通过 Transformer 进行处理，从而预测掩码和相应的质量分数。

    属性:
        transformer_dim (int): Transformer 模块的通道维度。
        transformer (nn.Module): 用于掩码预测的 Transformer 模块。
        num_multimask_outputs (int): 用于掩码歧义消解时需要预测的掩码数量。
        iou_token (nn.Embedding): 用于 IoU（交并比）评分的嵌入。
        num_mask_tokens (int): 掩码 token 的数量。
        mask_tokens (nn.Embedding): 掩码 token 的嵌入。
        output_upscaling (nn.Sequential): 用于上采样输出的神经网络模块。
        output_hypernetworks_mlps (nn.ModuleList): 用于生成掩码的超网络 MLP 列表。
        iou_prediction_head (nn.Module): 用于预测掩码质量（IoU 分数）的 MLP 模块。

    方法:
        forward: 给定图像和提示嵌入后进行掩码预测。
        predict_masks: 掩码预测的内部方法。

    示例:
        >>> decoder = MaskDecoder(transformer_dim=256, transformer=transformer_module)
        >>> masks, iou_pred = decoder(
        ...     image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output=True
        ... )
        >>> print(f"Predicted masks shape: {masks.shape}, IoU predictions shape: {iou_pred.shape}")
    """

    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        初始化 MaskDecoder 模块，用于生成掩码及其质量分数。

        参数:
            transformer_dim (int): Transformer 模块的通道维度。
            transformer (nn.Module): 用于掩码预测的 Transformer 模块。
            num_multimask_outputs (int): 用于掩码歧义处理时要输出的掩码数量。
            activation (Type[nn.Module]): 用于上采样阶段的激活函数类型。
            iou_head_depth (int): 用于预测掩码质量的 MLP 的深度。
            iou_head_hidden_dim (int): 用于预测掩码质量的 MLP 的隐藏层维度。

        示例:
            >>> transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=8), num_layers=6)
            >>> decoder = MaskDecoder(transformer_dim=256, transformer=transformer)
            >>> print(decoder)
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for _ in range(self.num_mask_tokens)]
        )

        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据图像和提示嵌入预测掩码。

        参数:
            image_embeddings (torch.Tensor): 来自图像编码器的图像嵌入。
            image_pe (torch.Tensor): 与 image_embeddings 形状相同的位置编码。
            sparse_prompt_embeddings (torch.Tensor): 点和框的提示嵌入。
            dense_prompt_embeddings (torch.Tensor): 掩码提示的稠密嵌入。
            multimask_output (bool): 是否输出多个掩码（用于掩码歧义消解）。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 返回一个元组，包含以下内容：
                - masks (torch.Tensor): 批处理的预测掩码。
                - iou_pred (torch.Tensor): 批处理的掩码质量预测结果（IoU 分数）。

        示例:
            >>> decoder = MaskDecoder(transformer_dim=256, transformer=transformer_module)
            >>> image_emb = torch.rand(1, 256, 64, 64)
            >>> image_pe = torch.rand(1, 256, 64, 64)
            >>> sparse_emb = torch.rand(1, 2, 256)
            >>> dense_emb = torch.rand(1, 256, 64, 64)
            >>> masks, iou_pred = decoder(image_emb, image_pe, sparse_emb, dense_emb, multimask_output=True)
            >>> print(f"Masks shape: {masks.shape}, IoU predictions shape: {iou_pred.shape}")
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # 根据是否启用多掩码输出，选择对应的掩码切片
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # 返回输出结果
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """通过 Transformer 架构，使用图像和提示嵌入预测掩码和质量得分。"""
        
        # 拼接输出 token
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.shape[0], -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 在 batch 维度上扩展每张图像的数据，以适配每个掩码的处理
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # 输入 Transformer 模型进行处理
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # 对掩码嵌入进行上采样，并使用掩码 token 预测掩码
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = [
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(self.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # 生成掩码质量的预测结果
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class SAM2MaskDecoder(nn.Module):
    """
    基于 Transformer 的解码器，用于从图像和提示嵌入中预测实例分割掩码。

    该类扩展了 MaskDecoder 的功能，集成了诸如高分辨率特征处理、动态多掩码输出、
    对象得分预测等附加特性。

    属性说明:
        transformer_dim (int): Transformer 的通道维度。
        transformer (nn.Module): 用于掩码预测的 Transformer 模块。
        num_multimask_outputs (int): 在多掩码判歧时所预测的掩码数量。
        iou_token (nn.Embedding): IOU token 的嵌入表示。
        num_mask_tokens (int): 掩码 token 的总数量。
        mask_tokens (nn.Embedding): 掩码 token 的嵌入表示。
        pred_obj_scores (bool): 是否预测对象得分。
        obj_score_token (nn.Embedding): 对象得分 token 的嵌入表示。
        use_multimask_token_for_obj_ptr (bool): 是否使用多掩码 token 作为对象指针。
        output_upscaling (nn.Sequential): 输出上采样层。
        use_high_res_features (bool): 是否使用高分辨率特征。
        conv_s0 (nn.Conv2d): 用于处理高分辨率特征（尺度 s0）的卷积层。
        conv_s1 (nn.Conv2d): 用于处理高分辨率特征（尺度 s1）的卷积层。
        output_hypernetworks_mlps (nn.ModuleList): 输出超网络使用的 MLP 列表。
        iou_prediction_head (MLP): IOU 预测用的多层感知机。
        pred_obj_score_head (nn.Linear | MLP): 对象得分预测的线性层或 MLP。
        dynamic_multimask_via_stability (bool): 是否基于稳定性动态选择多掩码。
        dynamic_multimask_stability_delta (float): 多掩码稳定性判断的 delta 值。
        dynamic_multimask_stability_thresh (float): 多掩码稳定性的阈值。

    方法说明:
        forward: 给定图像和提示嵌入，预测掩码。
        predict_masks: 从图像和提示嵌入中预测实例分割掩码。
        _get_stability_scores: 计算不同阈值间 IoU 得分以评估掩码稳定性。
        _dynamic_multimask_via_stability: 动态选择最稳定的掩码输出。

    示例用法:
        >>> image_embeddings = torch.rand(1, 256, 64, 64)
        >>> image_pe = torch.rand(1, 256, 64, 64)
        >>> sparse_prompt_embeddings = torch.rand(1, 2, 256)
        >>> dense_prompt_embeddings = torch.rand(1, 256, 64, 64)
        >>> decoder = SAM2MaskDecoder(256, transformer)
        >>> masks, iou_pred, sam_tokens_out, obj_score_logits = decoder.forward(
        ...     image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, True, False
        ... )
    """

def __init__(
    self,
    transformer_dim: int,
    transformer: nn.Module,
    num_multimask_outputs: int = 3,
    activation: Type[nn.Module] = nn.GELU,
    iou_head_depth: int = 3,
    iou_head_hidden_dim: int = 256,
    use_high_res_features: bool = False,
    iou_prediction_use_sigmoid=False,
    dynamic_multimask_via_stability=False,
    dynamic_multimask_stability_delta=0.05,
    dynamic_multimask_stability_thresh=0.98,
    pred_obj_scores: bool = False,
    pred_obj_scores_mlp: bool = False,
    use_multimask_token_for_obj_ptr: bool = False,
) -> None:
    """
    初始化 SAM2MaskDecoder 模块，用于预测实例分割掩码。

    该解码器在原始 MaskDecoder 的基础上进行了扩展，增加了高分辨率特征处理、
    动态多掩码输出和目标得分预测等功能。

    参数说明:
        transformer_dim (int): Transformer 的通道维度。
        transformer (nn.Module): 用于预测掩码的 Transformer 模块。
        num_multimask_outputs (int): 在掩码歧义时生成的掩码数量。
        activation (Type[nn.Module]): 在上采样掩码时所使用的激活函数类型。
        iou_head_depth (int): 用于预测掩码质量的 MLP 深度。
        iou_head_hidden_dim (int): 用于预测掩码质量的 MLP 的隐藏层维度。
        use_high_res_features (bool): 是否使用高分辨率特征。
        iou_prediction_use_sigmoid (bool): 是否对 IOU 预测使用 Sigmoid 激活。
        dynamic_multimask_via_stability (bool): 是否启用通过稳定性分数的动态多掩码机制。
        dynamic_multimask_stability_delta (float): 用于判断掩码稳定性的增量阈值。
        dynamic_multimask_stability_thresh (float): 掩码稳定性的判断阈值。
        pred_obj_scores (bool): 是否预测目标得分。
        pred_obj_scores_mlp (bool): 是否通过 MLP 结构来预测目标得分。
        use_multimask_token_for_obj_ptr (bool): 是否在目标指针中使用多掩码 token。

    使用示例:
        >>> transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=8), num_layers=6)
        >>> decoder = SAM2MaskDecoder(transformer_dim=256, transformer=transformer)
        >>> print(decoder)
    """
    super().__init__()
    self.transformer_dim = transformer_dim
    self.transformer = transformer

    self.num_multimask_outputs = num_multimask_outputs

    # 用于 IOU 预测的嵌入 token
    self.iou_token = nn.Embedding(1, transformer_dim)
    self.num_mask_tokens = num_multimask_outputs + 1
    self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

    self.pred_obj_scores = pred_obj_scores
    if self.pred_obj_scores:
        # 如果需要预测目标得分，则添加目标得分 token
        self.obj_score_token = nn.Embedding(1, transformer_dim)
    self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

    # 掩码特征的上采样模块
    self.output_upscaling = nn.Sequential(
        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
        LayerNorm2d(transformer_dim // 4),
        activation(),
        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        activation(),
    )

    self.use_high_res_features = use_high_res_features
    if use_high_res_features:
        # 当启用高分辨率特征时，添加额外卷积层将其对齐维度
        self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1, stride=1)
        self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, stride=1)

    # 每个掩码 token 对应一个 hypernetwork 的 MLP，用于从 token 生成掩码参数
    self.output_hypernetworks_mlps = nn.ModuleList(
        [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for _ in range(self.num_mask_tokens)]
    )

    # 用于预测掩码质量（如 IOU）的 MLP 头
    self.iou_prediction_head = MLP(
        transformer_dim,
        iou_head_hidden_dim,
        self.num_mask_tokens,
        iou_head_depth,
        sigmoid=iou_prediction_use_sigmoid,
    )

    if self.pred_obj_scores:
        # 可选目标得分预测模块（线性或 MLP）
        self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
        if pred_obj_scores_mlp:
            self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

    # 如果仅输出单一掩码，可以在稳定性分数过低时动态回退至最稳定的多掩码 token
    self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
    self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
    self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        给定图像和提示的嵌入，预测分割掩码。

        参数:
            image_embeddings (torch.Tensor): 来自图像编码器的图像嵌入，形状为 (B, C, H, W)。
            image_pe (torch.Tensor): 位置编码，与 image_embeddings 形状相同 (B, C, H, W)。
            sparse_prompt_embeddings (torch.Tensor): 稀疏提示（如点和框）的嵌入，形状为 (B, N, C)。
            dense_prompt_embeddings (torch.Tensor): 密集提示（如掩码）的嵌入，形状为 (B, C, H, W)。
            multimask_output (bool): 是否输出多个掩码。
            repeat_image (bool): 是否重复图像嵌入（用于批次维度扩展）。
            high_res_features (List[torch.Tensor] | None): 可选的高分辨率特征。

        返回:
            (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): 返回一个元组，包含：
                - masks (torch.Tensor): 批次预测的掩码，形状为 (B, N, H, W)。
                - iou_pred (torch.Tensor): 掩码质量的预测结果，形状为 (B, N)。
                - sam_tokens_out (torch.Tensor): 用于掩码输出的 SAM token，形状为 (B, N, C)。
                - object_score_logits (torch.Tensor): 对象得分的 logits，形状为 (B, 1)。

        示例:
            >>> image_embeddings = torch.rand(1, 256, 64, 64)
            >>> image_pe = torch.rand(1, 256, 64, 64)
            >>> sparse_prompt_embeddings = torch.rand(1, 2, 256)
            >>> dense_prompt_embeddings = torch.rand(1, 256, 64, 64)
            >>> decoder = SAM2MaskDecoder(256, transformer)
            >>> masks, iou_pred, sam_tokens_out, obj_score_logits = decoder.forward(
            ...     image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, True, False
            ... )
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # 根据 multimask_output 参数选择正确的掩码
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # 形状 [b, 3, c]
        else:
            # 选择掩码输出的 token。在这里我们总是使用单个掩码输出的 token。
            # 在测试阶段，即使启用了 multimask_output=True（例如单击后追踪），
            # 我们仍然取单个掩码 token。这是因为在训练时我们通常是多次点击的追踪训练，
            # 所以历史 token 始终是单掩码 token（我们会将其视为 object-memory token）。
            sam_tokens_out = mask_tokens_out[:, 0:1]  # 形状 [b, 1, c]

        # 返回最终输出
        return masks, iou_pred, sam_tokens_out, object_score_logits


    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用 Transformer 从图像和提示嵌入中预测实例分割掩码。"""
        
        # 拼接输出用的 token
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,  # 对象得分 token
                    self.iou_token.weight,        # IOU 预测 token
                    self.mask_tokens.weight,      # 掩码 token
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)

        # 将 token 扩展到 batch 维度
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 如果需要，在 batch 维度上复制图像嵌入（按 token 重复）
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings

        # 验证位置编码的 batch 维度
        assert image_pe.size(0) == 1, "image_pe 的 batch 维度应为 1（由 get_dense_pe() 提供）"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # 运行 Transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]  # IOU token 的输出
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]  # 掩码 token 的输出

        # 将嵌入上采样，并使用掩码 token 预测掩码
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # 使用超网络（hypernetwork）生成掩码预测权重
        hyper_in_list: List[torch.Tensor] = [
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(self.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # 生成掩码质量预测（IOU）
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # 对象得分 logits —— 默认设为 10.0，假设对象一定存在，sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """根据上限和下限阈值之间的 IoU 计算掩码稳定性评分。"""
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        return torch.where(area_u > 0, area_i / area_u, 1.0)

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        基于稳定性评分和 IoU 预测动态选择最稳定的掩码输出。

        该方法在输出单一掩码时使用。如果当前单一掩码输出的稳定性评分（基于输出 token 0）低于阈值，
        则从多掩码输出（基于输出 token 1-3）中选择具有最高预测 IoU 分数的掩码。这样可以确保在点击和追踪场景中获得有效的掩码。

        参数:
            all_mask_logits (torch.Tensor): 所有预测掩码的 logits，形状为 (B, N, H, W)，其中 B 是批量大小，N 是掩码数量（通常为 4），H 和 W 是掩码的维度。
            all_iou_scores (torch.Tensor): 所有掩码的预测 IoU 分数，形状为 (B, N)。

        返回:
            (Tuple[torch.Tensor, torch.Tensor]):
                - mask_logits_out (torch.Tensor): 选中的掩码 logits，形状为 (B, 1, H, W)。
                - iou_scores_out (torch.Tensor): 选中的 IoU 分数，形状为 (B, 1)。

        示例:
            >>> decoder = SAM2MaskDecoder(...)
            >>> all_mask_logits = torch.rand(2, 4, 256, 256)  # 2 张图像，每张图像 4 个掩码
            >>> all_iou_scores = torch.rand(2, 4)
            >>> mask_logits, iou_scores = decoder._dynamic_multimask_via_stability(all_mask_logits, all_iou_scores)
            >>> print(mask_logits.shape, iou_scores.shape)
            torch.Size([2, 1, 256, 256]) torch.Size([2, 1])
        """
        # 从多掩码输出 token（1~3）中选择最佳掩码
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(multimask_iou_scores.size(0), device=all_iou_scores.device)
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # 来自单一掩码输出 token 0 及其稳定性评分
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # 如果稳定性评分较低，则动态回退到最佳多掩码输出。
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
