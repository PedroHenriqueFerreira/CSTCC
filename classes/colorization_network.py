import torch
from torch import nn, Tensor
from torch.nn.functional import one_hot

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x_dec: Tensor, x_enc: Tensor):
        return self.conv(torch.cat([self.up(x_dec), x_enc], dim=1))

class ColorizationNetwork(nn.Module):
    """
    Residual conditional U-Net for image recoloring
    Condition: type of color vision deficiency
    """

    def __init__(self, conditions: int = 4):
        super().__init__()
        self.conditions = conditions

        # Encoder (note +conditions)
        self.enc1 = ConvBlock(3 + conditions, 32)
        self.enc2 = Down(32, 64)
        self.enc3 = Down(64, 128)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 128)

        # Decoder
        self.dec2 = Up(128 + 64, 64)
        self.dec1 = Up(64 + 32, 32)

        # RGB residual
        self.out = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """
        x: (B, 3, H, W)
        condition: (B,) integer labels
        """

        B, _, H, W = x.shape

        # One-hot encode condition
        one_hot_condition = one_hot(condition, self.conditions).float()
        condition = one_hot_condition.view(B, self.conditions, 1, 1).expand(-1, -1, H, W)

        # Concatenate condition map
        x_cond = torch.cat([x, condition], dim=1)

        # Encoder
        x1 = self.enc1(x_cond)
        
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # Bottleneck
        b = self.bottleneck(x3)

        # Decoder
        d2 = self.dec2(b, x2)
        d1 = self.dec1(d2, x1)

        # Residual correction
        delta = self.out(d1)

        return torch.clamp(x + delta, 0.0, 1.0)