# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.data import YOLOConcatDataset, build_grounding, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.world import WorldTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import de_parallel


class WorldTrainerFromScratch(WorldTrainer):
    """
    一个继承自 WorldTrainer 的训练器类，用于在开放集数据集上从零开始训练 YOLO-World 模型。

    示例：
        ```python
        from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
        from ultralytics import YOLOWorld

        data = dict(
            train=dict(
                yolo_data=["Objects365.yaml"],
                grounding_data=[
                    dict(
                        img_path="../datasets/flickr30k/images",
                        json_file="../datasets/flickr30k/final_flickr_separateGT_train.json",
                    ),
                    dict(
                        img_path="../datasets/GQA/images",
                        json_file="../datasets/GQA/final_mixed_train_no_coco.json",
                    ),
                ],
            ),
            val=dict(yolo_data=["lvis.yaml"]),
        )

        model = YOLOWorld("yolov8s-worldv2.yaml")
        model.train(data=data, trainer=WorldTrainerFromScratch)
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """使用指定参数初始化 WorldTrainer 实例。"""
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        构建 YOLO 数据集。

        参数：
            img_path (List[str] | str): 包含图像的路径。
            mode (str): 模式为 `train` 或 `val`，支持为不同模式定制不同的数据增强方式。
            batch (int, optional): 批大小，用于 `rect` 长宽比排列模式，默认值为 None。
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        dataset = [
            build_yolo_dataset(self.args, im_path, batch, self.data, stride=gs, multi_modal=True)
            if isinstance(im_path, str)
            else build_grounding(self.args, im_path["img_path"], im_path["json_file"], batch, stride=gs)
            for im_path in img_path
        ]
        return YOLOConcatDataset(dataset) if len(dataset) > 1 else dataset[0]

    def get_dataset(self):
        """
        从 data 字典中提取 train 和 val 路径信息（如果存在）。

        如果数据格式不符合预期，将返回 None。
        """
        final_data = {}
        data_yaml = self.args.data
        assert data_yaml.get("train", False), "未找到训练集配置（train dataset not found）"  # 如 object365.yaml
        assert data_yaml.get("val", False), "未找到验证集配置（validation dataset not found）"  # 如 lvis.yaml
        data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
        assert len(data["val"]) == 1, f"当前仅支持一个验证集，但接收到 {len(data['val'])} 个。"
        val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
        for d in data["val"]:
            if d.get("minival") is None:  # 针对 lvis 数据集
                continue
            d["minival"] = str(d["path"] / d["minival"])
        for s in ["train", "val"]:
            final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
            # 如果存在 grounding 数据则保存
            grounding_data = data_yaml[s].get("grounding_data")
            if grounding_data is None:
                continue
            grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]
            for g in grounding_data:
                assert isinstance(g, dict), f"Grounding 数据应为字典格式，但得到的是 {type(g)}"
            final_data[s] += grounding_data
        # 注意：为确保训练流程正常，必须设置类别数量 `nc` 和类别名称 `names`
        final_data["nc"] = data["val"][0]["nc"]
        final_data["names"] = data["val"][0]["names"]
        self.data = final_data
        return final_data["train"], final_data["val"][0]

    def plot_training_labels(self):
        """不绘制训练标签图（跳过该步骤）。"""
        pass

    def final_eval(self):
        """执行 YOLO-World 模型的最终评估和验证过程。"""
        val = self.args.data["val"]["yolo_data"][0]
        self.validator.args.data = val
        self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
        return super().final_eval()
