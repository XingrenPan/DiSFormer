import torch
import torch.nn as nn

"""
InceptionNeXt (Base) - 无预训练权重
修正版：LayerNorm 在 [B,H,W,C] 维度上执行，和官方实现对齐
"""

class InceptionDWConv(nn.Module):
    """
    Inception-style depthwise convolution decomposition:
    分支：1x1, kx1, 1xk, kxk
    """
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.branch1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=dim, bias=False)
        self.branch2 = nn.Conv2d(dim, dim, (kernel_size, 1), 1, (pad, 0), groups=dim, bias=False)
        self.branch3 = nn.Conv2d(dim, dim, (1, kernel_size), 1, (0, pad), groups=dim, bias=False)
        self.branch4 = nn.Conv2d(dim, dim, kernel_size, 1, pad, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = self.branch1(x) + self.branch2(x) + self.branch3(x) + self.branch4(x)
        return self.bn(out)


class InceptionNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = InceptionDWConv(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)   # [B,C,H,W]
        # [B,C,H,W] -> [B,H,W,C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # [B,H,W,C] -> [B,C,H,W]
        x = x.permute(0, 3, 1, 2)
        return shortcut + x


class DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch, eps=1e-6)
        self.reduction = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        # [B,C,H,W] -> [B,H,W,C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # [B,H,W,C] -> [B,C,H,W]
        x = x.permute(0, 3, 1, 2)
        x = self.reduction(x)
        return x


class InceptionNeXt(nn.Module):
    def __init__(self, num_classes=1000, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]):
        """
        Base 配置:
        depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]
        """
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        # Stem
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)

        # 其余 3 个阶段的降采样
        for i in range(3):
            down = DownsampleLayer(dims[i], dims[i+1])
            self.downsample_layers.append(down)

        # 每个 stage 堆叠 block
        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(InceptionNeXtBlock(dims[i]))
            self.stages.append(nn.Sequential(*blocks))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x.mean([-2, -1])  # GAP

    def forward(self, x):
        x = self.forward_features(x)
        # [B,C] -> [B,C], LayerNorm 作用在最后一维
        x = self.norm(x)
        x = self.head(x)
        return x


def inceptionnext_base(num_classes=1000):
    return InceptionNeXt(num_classes=num_classes,
                         depths=[3, 3, 27, 3],
                         dims=[128, 256, 512, 1024])


# Debug
if __name__ == "__main__":
    model = inceptionnext_base(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)  # [1,10]
