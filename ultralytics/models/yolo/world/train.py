# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import itertools

from ultralytics.data import build_yolo_dataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import WorldModel
from ultralytics.utils import DEFAULT_CFG, RANK, checks
from ultralytics.utils.torch_utils import de_parallel


def on_pretrain_routine_end(trainer):
    """回调函数。"""
    if RANK in {-1, 0}:  # 只在主进程或设备上运行
        # 注意：用于评估
        names = [name.split("/")[0] for name in list(trainer.test_loader.dataset.data["names"].values())]
        de_parallel(trainer.ema.ema).set_classes(names, cache_clip_model=False)
    device = next(trainer.model.parameters()).device
    trainer.text_model, _ = trainer.clip.load("ViT-B/32", device=device)
    for p in trainer.text_model.parameters():
        p.requires_grad_(False)  # 冻结文本模型的参数


class WorldTrainer(yolo.detect.DetectionTrainer):
    """
    一个用于在闭集数据集上微调世界模型的类。

    示例：
        ```python
        from ultralytics.models.yolo.world import WorldModel

        args = dict(model="yolov8s-world.pt", data="coco8.yaml", epochs=3)
        trainer = WorldTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """使用给定的参数初始化一个 WorldTrainer 对象。"""
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

        # 导入并分配clip
        try:
            import clip
        except ImportError:
            checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip
        self.clip = clip

    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回用指定配置和权重初始化的 WorldModel 模型。"""
        # 注意：这里的 `nc` 是指图像中最多不同文本样本的数量，而不是实际的 `nc`。
        # 注意：根据官方配置，`nc` 暂时硬编码为80。
        model = WorldModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=3,
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        构建YOLO数据集。

        参数：
            img_path (str): 包含图像的文件夹路径。
            mode (str): `train` 模式或 `val` 模式，用户可以为每个模式自定义不同的增强方法。
            batch (int, 可选): 批次大小，用于 `rect` 模式。默认值为 None。
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )

    def preprocess_batch(self, batch):
        """预处理一批图像数据，用于YOLOWorld训练，调整格式和尺寸。"""
        batch = super().preprocess_batch(batch)

        # 注意：添加文本特征
        texts = list(itertools.chain(*batch["texts"]))
        text_token = self.clip.tokenize(texts).to(batch["img"].device)
        txt_feats = self.text_model.encode_text(text_token).to(dtype=batch["img"].dtype)  # torch.float32
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)  # 归一化文本特征
        batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
        return batch
