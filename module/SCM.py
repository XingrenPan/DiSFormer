import torch
import torch.nn as nn


class StatisticalDescriptor(nn.Module):


    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        x_flat = x.reshape(batch, channels, -1)
        mean = x_flat.mean(dim=-1)
        centered = x_flat - mean.unsqueeze(-1)
        variance = centered.square().mean(dim=-1)
        std = torch.sqrt(variance + self.eps)
        skew = centered.pow(3).mean(dim=-1) / (std.pow(3) + self.eps)
        return torch.stack([mean, std, skew], dim=1)


class StatisticalChannelModulator(nn.Module):


    def __init__(
        self,
        kernel_size: int = 5,
        use_mean: bool = True,
        use_std: bool = True,
        use_skew: bool = True,
    ):
        super().__init__()
        self.use_mean = use_mean
        self.use_std = use_std
        self.use_skew = use_skew

        padding = (kernel_size - 1) // 2
        self.descriptor = StatisticalDescriptor()
        self.conv1d = nn.Conv1d(
            in_channels=3,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = self.descriptor(x)


        mask = stats.new_tensor([
            float(self.use_mean),
            float(self.use_std),
            float(self.use_skew),
        ]).view(1, 3, 1)
        if mask.sum() == 0:
            mask[:, 0, :] = 1.0
        stats = stats * mask

        gate = self.sigmoid(self.conv1d(stats))
        gate = gate.transpose(1, 2).unsqueeze(-1)
        return x * gate
