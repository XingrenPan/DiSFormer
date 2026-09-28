import torch
import torch.nn as nn
import torch.nn.functional as F

from src.efficient_kan import KANLinear
from .SCM import StatisticalChannelModulator


class DirectionalStatisticalFusion(nn.Module):


    def __init__(self, channel: int):
        super().__init__()
        self.channel = channel

        self.register_buffer("kernel_h", torch.ones(channel, 1, 1, 3))
        self.register_buffer("kernel_v", torch.ones(channel, 1, 3, 1))

        kernel_ld = torch.zeros(channel, 1, 3, 3)
        kernel_ld[:, :, 0, 0] = 1.0
        kernel_ld[:, :, 1, 1] = 1.0
        kernel_ld[:, :, 2, 2] = 1.0
        self.register_buffer("kernel_ld", kernel_ld)

        kernel_rd = torch.zeros(channel, 1, 3, 3)
        kernel_rd[:, :, 0, 2] = 1.0
        kernel_rd[:, :, 1, 1] = 1.0
        kernel_rd[:, :, 2, 0] = 1.0
        self.register_buffer("kernel_rd", kernel_rd)

        self.fusion = nn.Conv2d(channel * 4, channel, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(channel)

        self.fusion_act = nn.ReLU(inplace=True)
        self.gate = KANLinear(channel, channel)

        self.scm = StatisticalChannelModulator(
            use_mean=True, use_std=True, use_skew=True
        )

        self.w_dga = nn.Parameter(torch.tensor(0.0))
        self.w_scm = nn.Parameter(torch.tensor(0.0))
        self.w_res = nn.Parameter(torch.tensor(0.0))

        self.enable_dga = True
        self.enable_scm = True
        self.enable_residual = True
        self.active_dirs = ["h", "v", "ld", "rd"]

    def _directional_responses(self, x: torch.Tensor):
        c = self.channel
        return {
            "h": F.conv2d(x, self.kernel_h, padding=(0, 1), groups=c),
            "v": F.conv2d(x, self.kernel_v, padding=(1, 0), groups=c),
            "ld": F.conv2d(x, self.kernel_ld, padding=1, groups=c),
            "rd": F.conv2d(x, self.kernel_rd, padding=1, groups=c),
        }

    def _apply_dga(self, x: torch.Tensor) -> torch.Tensor:
        responses = self._directional_responses(x)
        directional = [
            responses[name] if name in self.active_dirs else torch.zeros_like(x)
            for name in ("h", "v", "ld", "rd")
        ]
        fused = self.fusion(torch.cat(directional, dim=1))
        fused = self.fusion_bn(fused)
        fused = self.fusion_act(fused)

        b, c, h, w = fused.shape
        flat = fused.permute(0, 2, 3, 1).reshape(-1, c)
        mask = torch.sigmoid(self.gate(flat))
        mask = mask.view(b, h, w, c).permute(0, 3, 1, 2)
        return x * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = []
        if self.enable_dga:
            branches.append(("dga", self._apply_dga(x)))
        if self.enable_scm:
            branches.append(("scm", self.scm(x)))
        if self.enable_residual:
            branches.append(("residual", x))

        if not branches:
            return x
        if len(branches) == 1:
            return branches[0][1]

        weights = torch.softmax(torch.stack([self.w_dga, self.w_scm, self.w_res]), dim=0)
        weight_map = {"dga": weights[0], "scm": weights[1], "residual": weights[2]}
        denom = sum(weight_map[name] for name, _ in branches)
        out = torch.zeros_like(x)
        for name, tensor in branches:
            out = out + (weight_map[name] / denom) * tensor
        return out
