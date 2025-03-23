# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.metrics import SegmentMetrics


class FastSAMValidator(SegmentationValidator):
    """
    专为Fast SAM（Segment Anything Model）分割任务在Ultralytics YOLO框架中的自定义验证类。

    扩展了SegmentationValidator类，特别定制了Fast SAM的验证过程。该类将任务设置为‘segment’，并使用SegmentMetrics进行评估。此外，禁用了绘图功能，以避免在验证过程中出现错误。

    属性：
        dataloader: 用于验证的数据加载器对象。
        save_dir (str): 存储验证结果的目录。
        pbar: 进度条对象。
        args: 自定义的额外参数。
        _callbacks: 在验证过程中调用的回调函数列表。
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        初始化FastSAMValidator类，将任务设置为‘segment’，并将评估指标设置为SegmentMetrics。

        参数：
            dataloader (torch.utils.data.DataLoader): 用于验证的数据加载器。
            save_dir (Path, optional): 存储结果的目录。
            pbar (tqdm.tqdm): 用于显示进度的进度条。
            args (SimpleNamespace): 验证器的配置。
            _callbacks (dict): 存储各种回调函数的字典。

        注意：
            本类中禁用了ConfusionMatrix和其他相关指标的绘图，以避免出现错误。
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "segment"
        self.args.plots = False  # 禁用ConfusionMatrix和其他绘图，以避免错误
        self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
