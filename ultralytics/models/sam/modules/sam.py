# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# 版权所有 (c) Meta Platforms, Inc. and 其子公司。
# 保留所有权利。

# 本源代码依据根目录 LICENSE 文件中的许可证授权使用。

from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_

from ultralytics.nn.modules import MLP

from .blocks import SAM2TwoWayTransformer
from .decoders import MaskDecoder, SAM2MaskDecoder
from .encoders import ImageEncoderViT, PromptEncoder
from .utils import get_1d_sine_pe, select_closest_cond_frames

# 一个用于占位缺失目标的极小分数（负无穷近似值）
NO_OBJ_SCORE = -1024.0


class SAMModel(nn.Module):
    """
    Segment Anything Model (SAM) 用于目标分割任务。

    该类结合了图像编码器、提示编码器和掩码解码器，从图像和输入提示中预测目标掩码。

    属性:
        mask_threshold (float): 掩码预测的阈值。
        image_encoder (ImageEncoderViT): 用于将图像编码为嵌入的主干网络。
        prompt_encoder (PromptEncoder): 用于编码各种类型的输入提示。
        mask_decoder (MaskDecoder): 从图像和提示嵌入中预测目标掩码。

    方法:
        __init__: 初始化 SAMModel，包含编码器、解码器和归一化参数。

    示例:
        >>> image_encoder = ImageEncoderViT(...)
        >>> prompt_encoder = PromptEncoder(...)
        >>> mask_decoder = MaskDecoder(...)
        >>> sam_model = SAMModel(image_encoder, prompt_encoder, mask_decoder)
        >>> # 进一步的使用取决于 SAMPredictor 类

    备注:
        所有的 forward() 操作都在 SAMPredictor 类中实现。
    """

    mask_threshold: float = 0.0

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = (123.675, 116.28, 103.53),
        pixel_std: List[float] = (58.395, 57.12, 57.375),
    ) -> None:
        """
        初始化 SAMModel 类，从图像和输入提示中预测目标掩码。

        参数:
            image_encoder (ImageEncoderViT): 用于将图像编码为图像嵌入的主干网络。
            prompt_encoder (PromptEncoder): 编码各种类型的输入提示。
            mask_decoder (MaskDecoder): 从图像嵌入和编码的提示中预测掩码。
            pixel_mean (List[float]): 用于归一化输入图像的像素均值。
            pixel_std (List[float]): 用于归一化输入图像的像素标准差。

        示例:
            >>> image_encoder = ImageEncoderViT(...)
            >>> prompt_encoder = PromptEncoder(...)
            >>> mask_decoder = MaskDecoder(...)
            >>> sam_model = SAMModel(image_encoder, prompt_encoder, mask_decoder)
            >>> # 进一步的使用取决于 SAMPredictor 类

        备注:
            所有的 forward() 操作都移到 SAMPredictor 中。
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    def set_imgsz(self, imgsz):
        """
        设置图像尺寸，使模型能够兼容不同的图像尺寸。

        参数:
            imgsz (Tuple[int, int]): 输入图像的尺寸。
        """
        if hasattr(self.image_encoder, "set_imgsz"):
            self.image_encoder.set_imgsz(imgsz)
        self.prompt_encoder.input_image_size = imgsz
        self.prompt_encoder.image_embedding_size = [x // 16 for x in imgsz]  # 16 是 ViT 模型的固定补丁大小
        self.image_encoder.img_size = imgsz[0]


class SAM2Model(torch.nn.Module):
    """
    SAM2Model 类是 Segment Anything Model 2，具有基于记忆的视频目标分割能力。

    该类扩展了 SAM 的功能，处理视频序列，结合记忆机制以保证时间一致性，并有效追踪跨帧的目标。

    属性:
        mask_threshold (float): 掩码预测的阈值。
        image_encoder (ImageEncoderViT): 用于提取图像特征的视觉编码器。
        memory_attention (nn.Module): 用于处理记忆特征的模块。
        memory_encoder (nn.Module): 用于生成记忆表示的编码器。
        num_maskmem (int): 可访问的记忆帧的数量。
        image_size (int): 输入图像的尺寸。
        backbone_stride (int): 主干网络输出的步幅。
        sam_prompt_embed_dim (int): SAM 提示嵌入的维度。
        sam_image_embedding_size (int): SAM 图像嵌入的尺寸。
        sam_prompt_encoder (PromptEncoder): 用于处理输入提示的编码器。
        sam_mask_decoder (SAM2MaskDecoder): 用于生成目标掩码的解码器。
        obj_ptr_proj (nn.Module): 用于目标指针的投影层。
        obj_ptr_tpos_proj (nn.Module): 用于目标指针中时间位置编码的投影。

    方法:
        forward_image: 通过编码器处理图像批次，提取多层特征。
        track_step: 执行单步跟踪，更新目标掩码和记忆特征。

    示例:
        >>> model = SAM2Model(image_encoder, memory_attention, memory_encoder)
        >>> image_batch = torch.rand(1, 3, 512, 512)
        >>> features = model.forward_image(image_batch)
        >>> track_results = model.track_step(0, True, features, None, None, None, {})
    """

    mask_threshold: float = 0.0

    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,
        image_size=512,
        backbone_stride=16,
        sigmoid_scale_for_mem_enc=1.0,
        sigmoid_bias_for_mem_enc=0.0,
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,
        max_cond_frames_in_attn=-1,
        directly_add_no_mem_embed=False,
        use_high_res_features_in_sam=False,
        multimask_output_in_sam=False,
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=False,
        use_multimask_token_for_obj_ptr: bool = False,
        iou_prediction_use_sigmoid=False,
        memory_temporal_stride_for_eval=1,
        non_overlap_masks_for_mem_enc=False,
        use_obj_ptrs_in_encoder=False,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=False,
        use_signed_tpos_enc_to_obj_ptrs=False,
        only_obj_ptrs_in_the_past_for_eval=False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        fixed_no_obj_ptr: bool = False,
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        no_obj_embed_spatial: bool = False,
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
    ):
        """
        初始化 SAM2Model，用于视频目标分割和基于记忆的追踪。

        参数说明:
            image_encoder (nn.Module): 用于提取图像特征的视觉编码器。
            memory_attention (nn.Module): 用于对记忆特征进行注意力机制处理的模块。
            memory_encoder (nn.Module): 用于生成记忆表示的编码器。
            num_maskmem (int): 可访问的记忆帧数量，默认为7（1个输入帧 + 6个先前帧）。
            image_size (int): 输入图像的尺寸。
            backbone_stride (int): 图像骨干网络输出的步幅。
            sigmoid_scale_for_mem_enc (float): 用于掩码sigmoid概率的缩放因子。
            sigmoid_bias_for_mem_enc (float): 用于掩码sigmoid概率的偏置因子。
            binarize_mask_from_pts_for_mem_enc (bool): 是否在评估时对交互帧进行掩码二值化。
            use_mask_input_as_output_without_sam (bool): 是否在有掩码输入的帧上直接输出输入掩码，而不使用SAM提示编码器和掩码解码器。
            max_cond_frames_in_attn (int): 参与记忆注意力的最大条件帧数，-1表示无限制。
            directly_add_no_mem_embed (bool): 是否在第一帧上直接添加没有记忆的嵌入。
            use_high_res_features_in_sam (bool): 是否在SAM掩码解码器中使用高分辨率特征图。
            multimask_output_in_sam (bool): 是否为初始条件帧的第一次点击输出多个（3个）掩码。
            multimask_min_pt_num (int): 使用多掩码输出时最小的点击次数。
            multimask_max_pt_num (int): 使用多掩码输出时最大的点击次数。
            multimask_output_for_tracking (bool): 是否使用多掩码输出进行追踪。
            use_multimask_token_for_obj_ptr (bool): 是否使用多掩码token作为对象指针。
            iou_prediction_use_sigmoid (bool): 是否使用sigmoid限制IoU预测在[0, 1]范围内。
            memory_temporal_stride_for_eval (int): 在评估期间记忆库的时间步幅。
            non_overlap_masks_for_mem_enc (bool): 在评估时是否对记忆编码器中的对象掩码应用不重叠约束。
            use_obj_ptrs_in_encoder (bool): 是否在编码器中交叉关注来自其他帧的对象指针。
            max_obj_ptrs_in_encoder (int): 在编码器的交叉注意力中，来自其他帧的最大对象指针数量。
            add_tpos_enc_to_obj_ptrs (bool): 是否在编码器中为对象指针添加时间位置编码。
            proj_tpos_enc_in_obj_ptrs (bool): 是否为对象指针中的时间位置编码添加额外的线性投影层。
            use_signed_tpos_enc_to_obj_ptrs (bool): 是否在对象指针中的时间位置编码中使用符号距离（而不是无符号的绝对距离），仅在`use_obj_ptrs_in_encoder=True`和`add_tpos_enc_to_obj_ptrs=True`时相关。
            only_obj_ptrs_in_the_past_for_eval (bool): 在评估时是否只关注过去的对象指针。
            pred_obj_scores (bool): 是否预测帧中是否有对象。
            pred_obj_scores_mlp (bool): 是否使用MLP来预测对象得分。
            fixed_no_obj_ptr (bool): 是否在没有对象存在时使用固定的无对象指针。
            soft_no_obj_ptr (bool): 是否通过软方式混合无对象指针，以便更容易恢复和错误缓解。
            use_mlp_for_obj_ptr_proj (bool): 是否使用MLP进行对象指针的投影。
            no_obj_embed_spatial (bool): 是否在空间帧中添加无对象嵌入。
            sam_mask_decoder_extra_args (Dict | None): 用于构建SAM掩码解码器的额外参数。
            compile_image_encoder (bool): 是否编译图像编码器以提高推理速度。

        示例用法:
            >>> image_encoder = ImageEncoderViT(...)
            >>> memory_attention = SAM2TwoWayTransformer(...)
            >>> memory_encoder = nn.Sequential(...)
            >>> model = SAM2Model(image_encoder, memory_attention, memory_encoder)
            >>> image_batch = torch.rand(1, 3, 512, 512)
            >>> features = model.forward_image(image_batch)
            >>> track_results = model.track_step(0, True, features, None, None, None, {})
        """
        super().__init__()

        # 部分 1: 图像骨干网络
        self.image_encoder = image_encoder
        # 如果使用高分辨率设置，使用level 0, 1, 2，否则默认使用level 2
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # 一个卷积层，用于将掩码提示下采样到步幅4（与低分辨率SAM掩码logits相同），
            # 并将其尺度从0~1转换为SAM的logit尺度，以便输入SAM掩码解码器生成指针。
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # 这两个选项必须一起使用
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # 部分 2: 记忆注意力，用于将当前帧的视觉特征与来自先前帧的记忆（和对象指针）进行条件处理
        self.memory_attention = memory_attention
        self.hidden_dim = memory_attention.d_model

        # 部分 3: 记忆编码器，用于处理先前帧的输出
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(self.memory_encoder.out_proj, "weight"):
            # 如果记忆在通道维度上被压缩
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # 可访问的记忆数量
        # 记忆的时间编码
        self.maskmem_tpos_enc = torch.nn.Parameter(torch.zeros(num_maskmem, 1, 1, self.mem_dim))
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # 用于表示没有来自先前帧的记忆嵌入的单个token
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # 在将掩码传入记忆编码器之前，是否应用sigmoid将掩码的原始logits转化为（0, 1）范围
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # 在有掩码输入的帧上，是否直接输出输入掩码，而不使用SAM提示编码器+掩码解码器
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # 部分 4: SAM 风格的提示编码器（用于掩码和点输入），
        # 以及SAM风格的掩码解码器用于最终的掩码输出
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # 模型编译
        if compile_image_encoder:
            # 编译前向函数（不是完整模块），以允许加载检查点。
            print("图像编码器编译已启用。第一次前向传播将较慢。")
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    @property
    def device(self):
        """返回模型参数所在的设备。"""
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        """处理图像和提示输入，以在视频序列中生成目标掩码和得分。"""
        raise NotImplementedError(
            "请使用 SAM2VideoPredictor 中的相应方法进行推理。"
            "示例请参考 notebooks/video_predictor_example.ipynb。"
        )
1
def _build_sam_heads(self):
    """构建 SAM 风格的提示编码器和掩码解码器，用于图像分割任务。"""
    self.sam_prompt_embed_dim = self.hidden_dim
    self.sam_image_embedding_size = self.image_size // self.backbone_stride

    # 从 SAM 构建 PromptEncoder 和 MaskDecoder（像 `mask_in_chans=16` 等超参数来自 SAM 代码）
    self.sam_prompt_encoder = PromptEncoder(
        embed_dim=self.sam_prompt_embed_dim,
        image_embedding_size=(
            self.sam_image_embedding_size,
            self.sam_image_embedding_size,
        ),
        input_image_size=(self.image_size, self.image_size),
        mask_in_chans=16,
    )
    self.sam_mask_decoder = SAM2MaskDecoder(
        num_multimask_outputs=3,
        transformer=SAM2TwoWayTransformer(
            depth=2,
            embedding_dim=self.sam_prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=self.sam_prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_high_res_features=self.use_high_res_features_in_sam,
        iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
        pred_obj_scores=self.pred_obj_scores,
        pred_obj_scores_mlp=self.pred_obj_scores_mlp,
        use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
        **(self.sam_mask_decoder_extra_args or {}),
    )
    if self.use_obj_ptrs_in_encoder:
        # 在 SAM 输出 token 上进行线性投影，将其转换为对象指针
        self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        if self.use_mlp_for_obj_ptr_proj:
            self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
    else:
        self.obj_ptr_proj = torch.nn.Identity()
    if self.proj_tpos_enc_in_obj_ptrs:
        # 对对象指针中的时间位置编码进行线性投影，以避免与空间位置编码的潜在干扰
        self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
    else:
        self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        通过 SAM 提示编码器和掩码头进行前向传播。

        该方法处理图像特征和可选的点/掩码输入，以生成对象掩码和得分。

        参数:
            backbone_features (torch.Tensor): 图像特征，形状为 (B, C, H, W)。
            point_inputs (Dict[str, torch.Tensor] | None): 包含点提示的字典。
                'point_coords': 形状为 (B, P, 2) 的张量，数据类型为 float32，包含 P 个输入点的绝对像素坐标 (x, y)。
                'point_labels': 形状为 (B, P) 的张量，数据类型为 int32，1 表示正向点击，0 表示负向点击，-1 表示填充。
            mask_inputs (torch.Tensor | None): 掩码，形状为 (B, 1, H*16, W*16)，数据类型为 float 或 bool，空间大小与图像相同。
            high_res_features (List[torch.Tensor] | None): 包含两个特征图的列表，形状分别为 (B, C, 4*H, 4*W) 和 (B, C, 2*H, 2*W)，用于 SAM 解码器的高分辨率特征图。
            multimask_output (bool): 如果为 True，输出 3 个候选掩码及其 IoU 估计；如果为 False，只输出 1 个掩码及其 IoU 估计。

        返回:
            (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
                low_res_multimasks: 形状为 (B, M, H*4, W*4) 的张量，包含 SAM 输出的掩码 logits。
                high_res_multimasks: 形状为 (B, M, H*16, W*16) 的张量，包含上采样后的掩码 logits。
                ious: 形状为 (B, M) 的张量，包含每个输出掩码的估计 IoU。
                low_res_masks: 形状为 (B, 1, H*4, W*4) 的张量，包含最佳低分辨率掩码。
                high_res_masks: 形状为 (B, 1, H*16, W*16) 的张量，包含最佳高分辨率掩码。
                obj_ptr: 形状为 (B, C) 的张量，包含输出掩码的对象指针向量。
                object_score_logits: 形状为 (B) 的张量，包含对象得分的 logits。

                其中 M 为 3（如果 multimask_output=True），为 1（如果 multimask_output=False）。

        示例:
            >>> backbone_features = torch.rand(1, 256, 32, 32)
            >>> point_inputs = {"point_coords": torch.rand(1, 2, 2), "point_labels": torch.tensor([[1, 0]])}
            >>> mask_inputs = torch.rand(1, 1, 512, 512)
            >>> results = model._forward_sam_heads(backbone_features, point_inputs, mask_inputs)
            >>> (
            ...     low_res_multimasks,
            ...     high_res_multimasks,
            ...     ious,
            ...     low_res_masks,
            ...     high_res_masks,
            ...     obj_ptr,
            ...     object_score_logits,
            ... ) = results
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) 处理点提示
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # 如果没有提供点提示，则用空点进行填充（标签为 -1）
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) 处理掩码提示
        if mask_inputs is not None:
            # 如果提供了掩码输入，将其下采样为低分辨率掩码输入，必要时将其作为密集掩码提示输入到 SAM 掩码编码器
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # 使用抗锯齿进行下采样
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # 否则，直接输入 None（SAM 的提示编码器会添加一个学习到的 `no_mask_embed` 来表示没有掩码输入的情况）。
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # 图像已经是批量处理的
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # 空间记忆掩码是对象与非对象之间的硬选择，与实际掩码预测一致
            low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks, NO_OBJ_SCORE)

        # 将掩码从可能的 bfloat16（或 float16）转换为 float32
        # （旧版 PyTorch（2.1 之前）不支持 `interpolate` 的 bf16）
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # 选择最佳掩码预测（具有最高的 IoU 估计）
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # 从 SAM 输出 token 提取对象指针（带遮挡处理）
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # 允许 *软* 非对象指针，与掩码不同
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """直接将掩码输入作为输出，绕过 SAM 编码器/解码器。"""
        # 使用 -10/+10 作为负/正像素的 logits（在 sigmoid 后非常接近 0/1 的概率值）。
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # 使用抗锯齿进行下采样
        )
        # 一个假设的 IoU 预测，假设所有输入掩码的 IoU 都为 1
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # 使用全零的假对象指针（形状为 [B, C]）
            obj_ptr = torch.zeros(mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device)
        else:
            # 使用 SAM 解码器根据掩码输入生成对象指针
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # 在此方法中，我们将 mask_input 视为输出，例如直接使用它来创建空间记忆；
        # 下面，我们遵循相同的设计准则，使用 mask_input 来决定对象是否出现，而不是依赖
        # SAM 解码器中的 object_scores。
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, img_batch: torch.Tensor):
        """通过编码器处理图像批次，提取 SAM 模型所需的多层特征。"""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # 预计算 SAM 解码器中的第 0 层和第 1 层特征
            # 以避免在每次 SAM 点击时都重新计算
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """准备并展开图像骨干网络输出的视觉特征，以便进一步处理。"""
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # 将 NxCxHxW 展开为 HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # 是否按时间逆序跟踪（用于演示）
    ):
        """通过将当前帧的视觉特征与先前的记忆融合，准备基于记忆的条件特征。"""
        B = current_vision_feats[-1].size(1)  # 当前帧的批次大小
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # 最顶层（最低分辨率）的特征大小
        device = current_vision_feats[-1].device
        # 当 `self.num_maskmem == 0` 时，主要用于复现 SAM 在图像上的效果。
        # 在这种情况下，我们跳过与任何记忆的融合。
        if self.num_maskmem == 0:  # 禁用记忆并跳过融合
            return current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # 步骤 1: 将当前帧的视觉特征与先前的记忆进行条件化
        if not is_init_cond_frame:
            # 获取通过 maskmem 骨干网络编码的记忆
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # 首先添加当前条件帧的输出（所有条件帧的 t_pos = 0）
            assert len(output_dict["cond_frame_outputs"]) > 0
            # 选择最大数量的时间上最接近的条件帧进行交叉注意
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # 在当前帧之前添加最后 (self.num_maskmem - 1) 帧的非条件记忆
            # 最早的帧 t_pos=1，最新的帧 t_pos=self.num_maskmem-1
            # 允许不连续选择记忆帧（r>1），在这种情况下，
            # 我们选择每 r 帧中的 (self.num_maskmem - 2) 帧和最后一帧。
            r = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # 距离当前帧的帧数
                if t_rel == 1:
                    # 对于 t_rel == 1，我们选择最后一帧（不考虑 r）
                    prev_frame_idx = frame_idx + t_rel if track_in_reverse else frame_idx - t_rel
                elif not track_in_reverse:
                    # 首先选择每 r 帧之前的最接近帧
                    prev_frame_idx = ((frame_idx - 2) // r) * r
                    prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                else:
                    # 首先选择每 r 帧之后的最接近帧
                    prev_frame_idx = -(-(frame_idx + 2) // r) * r
                    prev_frame_idx = prev_frame_idx + (t_rel - 2) * r
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # 跳过填充帧
                feats = prev["maskmem_features"].to(device=device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device=device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                to_cat_memory_pos_embed.append(maskmem_enc)

            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(t, unselected_cond_outputs.get(t, None))
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                if pos_and_ptrs:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            if self.directly_add_no_mem_embed:
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # 步骤 2: 将记忆和位置编码连接起来，并通过 Transformer 编码器进行前向传播
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """将帧特征和掩码编码为新的记忆表示，用于视频分割。"""
        B = current_vision_feats[-1].size(1)  # 当前帧的批量大小
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # 顶层（最低分辨率）特征尺寸
        # 顶层特征，(HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # 可选：对掩码应用不重叠约束（仅在评估时使用，确保所有对象来自同一视频，且批量大小为 1）。
            pred_masks_high_res = self._apply_non_overlapping_constraints(pred_masks_high_res)
        # 在应用 sigmoid 之前对原始掩码 logits 进行温度缩放
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # 对原始掩码 logits 应用 sigmoid，将其转化到 (0, 1) 范围内
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # 对 sigmoid 概率应用缩放和偏置项
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)  # 已经应用 sigmoid
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # 向空间记忆中添加一个无对象嵌入，表示该帧预测为被遮挡（即帧中没有出现对象）
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (1 - is_obj_appearing[..., None, None]) * self.no_obj_embed_spatial[
                ..., None, None
            ].expand(*maskmem_features.shape)

        return maskmem_features, maskmem_pos_enc

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        """执行单步跟踪，根据当前帧输入更新目标掩码和记忆特征。"""
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # SAM 头的高分辨率特征图，重塑 (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # 当 use_mask_input_as_output_without_sam=True 时，我们直接输出掩码输入
            # （视其为 GT 掩码），而不使用 SAM 提示编码器和掩码解码器。
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(pix_feat, high_res_features, mask_inputs)
        else:
            # 将视觉特征与之前的记忆特征融合在记忆库中
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # 应用 SAM 风格的分割头
            # 这里可能会将之前预测的低分辨率 SAM 掩码 logits 输入到 SAM 掩码解码器，
            # 例如在演示中，这些 logits 来自早期的交互而不是修正采样
            # （在这种情况下，任何 `mask_inputs` 都不应该到达这里，因为它们被发送到 _use_mask_as_output）
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )
        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        """最终在预测掩码上运行记忆编码器，将其编码为新的记忆特征（可以用于未来的帧）。"""
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # 反向时间顺序跟踪（用于演示）
        # 是否在预测掩码上运行记忆编码器。有时我们可能希望
        # 跳过记忆编码器，设置 run_mem_encoder=False。例如，
        # 在演示中，我们可能会为每个用户点击多次调用 `track_step`，
        # 并且仅在用户确认点击后才编码记忆。而在如 SAM 静态图像训练等消融设置中，我们不需要记忆编码器。
        run_mem_encoder=True,
        # 先前预测的 SAM 掩码 logits（可以与新的点击一起输入用于演示）。
        prev_sam_mask_logits=None,
    ):
        """执行单步跟踪，根据当前帧输入更新目标掩码和记忆特征。"""
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )
        _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # 仅在推理时添加（避免在激活检查点中使用未使用的参数；
            # 主要用于演示中通过统一的掩码编码空间记忆）
            current_out["object_score_logits"] = object_score_logits

        # 在预测掩码上运行记忆编码器，将其编码为新的记忆特征（供未来帧使用）
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """根据配置和输入决定是否在SAM头中使用多个掩码输出。"""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        return (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )

    @staticmethod
    def _apply_non_overlapping_constraints(pred_masks):
        """对掩码应用不重叠约束，保持每个位置得分最高的对象。"""
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds"：每个位置得分最高的对象的索引
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds"：每个对象切片在`pred_masks`中的索引（沿第0维）
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # 对重叠区域的得分应用抑制，保持前景区域不重叠
        # 这里将分数小于-10.0的区域设为无效（sigmoid(-10.0)=4.5398e-05）
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

    def set_binarize(self, binarize=False):
        """为 VideoPredictor 设置二值化参数。"""
        self.binarize_mask_from_pts_for_mem_enc = binarize

    def set_imgsz(self, imgsz):
        """
        设置图像大小以使模型兼容不同的图像尺寸。

        参数说明:
            imgsz (Tuple[int, int]): 输入图像的尺寸。
        """
        self.image_size = imgsz[0]
        self.sam_prompt_encoder.input_image_size = imgsz
        self.sam_prompt_encoder.image_embedding_size = [x // 16 for x in imgsz]  # 固定的ViT补丁大小为16
