# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import shutil
import socket
import sys
import tempfile

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9


def find_free_network_port() -> int:
    """
    查找本地主机上的空闲端口。

    在单节点训练中非常有用，当我们不想连接到真实的主节点，但必须设置
    `MASTER_PORT` 环境变量时。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # 返回端口


def generate_ddp_file(trainer):
    """生成 DDP 文件并返回其文件名。"""
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)

    content = f"""
# Ultralytics 多GPU训练临时文件（使用后应自动删除）
overrides = {vars(trainer.args)}

if __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # 处理额外的 'save_dir' 键
    trainer = {name}(cfg=cfg, overrides=overrides)
    trainer.args.model = "{getattr(trainer.hub_session, "model_url", trainer.args.model)}"
    results = trainer.train()
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name


def generate_ddp_command(world_size, trainer):
    """生成并返回分布式训练的命令。"""
    import __main__  # noqa 本地导入以避免 https://github.com/Lightning-AI/lightning/issues/15218

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # 删除 save_dir
    file = generate_ddp_file(trainer)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [sys.executable, "-m", dist_cmd, "--nproc_per_node", f"{world_size}", "--master_port", f"{port}", file]
    return cmd, file


def ddp_cleanup(trainer, file):
    """如果创建了临时文件，则删除该文件。"""
    if f"{id(trainer)}.py" in file:  # 如果文件中包含临时文件的后缀
        os.remove(file)
