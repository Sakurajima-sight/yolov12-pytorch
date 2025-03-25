# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""激活函数模块（Activation modules）。"""

import torch
import torch.nn as nn


class AGLU(nn.Module):
    """来自 https://github.com/kostas1515/AGLU 的统一激活函数模块。"""

    def __init__(self, device=None, dtype=None) -> None:
        """初始化统一激活函数。"""
        super().__init__()
        self.act = nn.Softplus(beta=-1.0)
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # lambda 参数
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # kappa 参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算统一激活函数的前向传播。"""
        lam = torch.clamp(self.lambd, min=0.0001)
        return torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))
