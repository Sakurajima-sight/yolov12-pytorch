# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, colorstr

try:
    # 警告：由于protobuf bug https://github.com/ultralytics/ultralytics/pull/4674，请不要移动SummaryWriter的导入
    from torch.utils.tensorboard import SummaryWriter

    assert not TESTS_RUNNING  # 不记录pytest日志
    assert SETTINGS["tensorboard"] is True  # 确认集成已启用
    WRITER = None  # TensorBoard SummaryWriter 实例
    PREFIX = colorstr("TensorBoard: ")

    # 以下导入仅在启用TensorBoard时需要
    import warnings
    from copy import deepcopy

    from ultralytics.utils.torch_utils import de_parallel, torch

except (ImportError, AssertionError, TypeError, AttributeError):
    # 处理Windows中的'TypeError'，即“描述符不能直接创建”的protobuf错误
    # 处理'AttributeError: module 'tensorflow' has no attribute 'io'，如果未安装'tensorflow'
    SummaryWriter = None


def _log_scalars(scalars, step=0):
    """将标量值记录到TensorBoard中。"""
    if WRITER:
        for k, v in scalars.items():
            WRITER.add_scalar(k, v, step)


def _log_tensorboard_graph(trainer):
    """将模型图记录到TensorBoard中。"""
    # 输入图像
    imgsz = trainer.args.imgsz
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    p = next(trainer.model.parameters())  # 获取设备和类型
    im = torch.zeros((1, 3, *imgsz), device=p.device, dtype=p.dtype)  # 输入图像（必须是零而不是空）

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # 忽略jit trace警告
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)  # 忽略jit trace警告

        # 尝试简单方法（YOLO）
        try:
            trainer.model.eval()  # 切换到.eval()模式以避免BatchNorm统计量变化
            WRITER.add_graph(torch.jit.trace(de_parallel(trainer.model), im, strict=False), [])
            LOGGER.info(f"{PREFIX}模型图形可视化已添加 ✅")
            return

        except Exception:
            # 回退到TorchScript导出步骤（RTDETR）
            try:
                model = deepcopy(de_parallel(trainer.model))
                model.eval()
                model = model.fuse(verbose=False)
                for m in model.modules():
                    if hasattr(m, "export"):  # 检测，RTDETRDecoder（Segment和Pose使用Detect基类）
                        m.export = True
                        m.format = "torchscript"
                model(im)  # 干运行
                WRITER.add_graph(torch.jit.trace(model, im, strict=False), [])
                LOGGER.info(f"{PREFIX}模型图形可视化已添加 ✅")
            except Exception as e:
                LOGGER.warning(f"{PREFIX}警告 ⚠️ TensorBoard图形可视化失败 {e}")


def on_pretrain_routine_start(trainer):
    """在预训练例程开始时初始化TensorBoard日志记录器。"""
    if SummaryWriter:
        try:
            global WRITER
            WRITER = SummaryWriter(str(trainer.save_dir))
            LOGGER.info(f"{PREFIX}开始使用 'tensorboard --logdir {trainer.save_dir}'，可以在 http://localhost:6006/ 查看")
        except Exception as e:
            LOGGER.warning(f"{PREFIX}警告 ⚠️ TensorBoard未正确初始化，未记录此次运行。{e}")


def on_train_start(trainer):
    """记录TensorBoard模型图。"""
    if WRITER:
        _log_tensorboard_graph(trainer)


def on_train_epoch_end(trainer):
    """在每个训练周期结束时记录标量统计数据。"""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)
    _log_scalars(trainer.lr, trainer.epoch + 1)


def on_fit_epoch_end(trainer):
    """在训练周期结束时记录周期指标。"""
    _log_scalars(trainer.metrics, trainer.epoch + 1)


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_start": on_train_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_epoch_end": on_train_epoch_end,
    }
    if SummaryWriter
    else {}
)
