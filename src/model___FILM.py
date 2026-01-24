import torch
import torch.nn as nn
from torch import Tensor

class FiLM(nn.Module):
    def __init__(self, channels: int, emb_dim: int):
        super().__init__()
        self.gamma = nn.Linear(emb_dim, channels)
        self.beta  = nn.Linear(emb_dim, channels)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        gamma = self.gamma(c).unsqueeze(-1).unsqueeze(-1)
        beta  = self.beta(c).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class DoubleConvCond(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=False)
        self.film1 = FiLM(out_ch, emb_dim)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=False)
        self.film2 = FiLM(out_ch, emb_dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x = self.relu(self.film1(self.norm1(self.conv1(x)), c))
        x = self.relu(self.film2(self.norm2(self.conv2(x)), c))
        return x

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvCond(in_ch, out_ch, emb_dim)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        return self.conv(self.pool(x), c)

class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConvCond(in_ch, out_ch, emb_dim)

    def forward(self, x_dec: Tensor, x_enc: Tensor, c: Tensor) -> Tensor:
        x = self.up(x_dec)
        x = torch.cat([x_enc, x], dim=1)
        return self.conv(x, c)

class ColorCorrectionModel(nn.Module):
    def __init__(
        self,
        conditions: int = 4,
        emb_dim: int = 64,
        base: int = 32
    ):
        super().__init__()

        self.embedding = nn.Embedding(conditions, emb_dim)

        self.inc   = DoubleConvCond(3, base, emb_dim)
        self.down1 = Down(base, base * 2, emb_dim)
        self.down2 = Down(base * 2, base * 4, emb_dim)
        self.down3 = Down(base * 4, base * 8, emb_dim)

        self.up1 = Up(base * 8 + base * 4, base * 4, emb_dim)
        self.up2 = Up(base * 4 + base * 2, base * 2, emb_dim)
        self.up3 = Up(base * 2 + base, base, emb_dim)

        self.outc = nn.Conv2d(base, 3, kernel_size=1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        c = self.embedding(condition)

        x1 = self.inc(x, c)
        x2 = self.down1(x1, c)
        x3 = self.down2(x2, c)
        x4 = self.down3(x3, c)

        x = self.up1(x4, x3, c)
        x = self.up2(x,  x2, c)
        x = self.up3(x,  x1, c)

        return torch.sigmoid(self.outc(x))
