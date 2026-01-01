import torch
from torch import nn, Tensor
from torch.nn.functional import one_hot

class ConvBlock(nn.Module):
    ''' Convolutional Block: Conv2d -> InstanceNorm2d -> ReLU '''
    
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    ''' Downsampling followed by a ConvBlock '''
    
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.MaxPool2d(2), 
            ConvBlock(in_size, out_size)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    ''' Upsampling followed by a ConvBlock with skip connections '''
    
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = ConvBlock(in_size, out_size)

    def forward(self, x_dec: Tensor, x_enc: Tensor):
        return self.conv(torch.cat([self.up(x_dec), x_enc], dim=1))

class ColorCorrectionModel(nn.Module):
    ''' Residual U-Net for Color Blindness Correction '''

    def __init__(self):
        super().__init__()

        # Encoder (note +types)
        self.enc1 = ConvBlock(3 + 4, 32)
        self.enc2 = Down(32, 64)
        self.enc3 = Down(64, 128)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 128)

        # Decoder
        self.dec2 = Up(128 + 64, 64)
        self.dec1 = Up(64 + 32, 32)

        # RGB residual
        self.out = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x: Tensor, type: Tensor) -> Tensor:
        '''Simulate recoloring for color blindness correction.
        
        x: (B, 3, H, W) in [0, 1]
        type: (B,) in [0, 3]
        '''

        B, _, H, W = x.shape

        one_hot_enc = one_hot(type, 4).float()
        type = one_hot_enc.view(B, 4, 1, 1).expand(-1, -1, H, W)

        # (B, 3, H, W) -> (B, 3 + 4, H, W)
        x_enc = torch.cat([x, type], dim=1)

        # Encoder
        x1 = self.enc1(x_enc)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # Bottleneck
        b = self.bottleneck(x3)

        # Decoder
        d2 = self.dec2(b, x2)
        d1 = self.dec1(d2, x1)

        # Residual correction
        delta: Tensor = self.out(d1)

        return torch.clamp(x + delta.tanh(), 0.0, 1.0)