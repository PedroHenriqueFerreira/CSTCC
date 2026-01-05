import torch
from torch import nn, Tensor
from torch.nn.functional import one_hot

class ConvBlock(nn.Module):
    ''' Conv -> InstanceNorm -> ReLU '''
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

class Down(nn.Module):
    ''' Downsampling + ConvBlock '''
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

class Up(nn.Module):
    ''' Upsampling + ConvBlock (with skip connection) '''
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x_dec: Tensor, x_enc: Tensor) -> Tensor:
        x = self.up(x_dec)
        return self.conv(torch.cat([x, x_enc], dim=1))

class ColorCorrectionModel(nn.Module):
    ''' Residual U-Net conditioned on color vision deficiency type '''

    def __init__(self):
        super().__init__()
        self.type_embedding = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
        )

        self.enc1 = ConvBlock(3 + 16, 64)
        self.enc2 = Down(64 + 16, 128)
        self.enc3 = Down(128 + 16, 256)
        self.enc4 = Down(256 + 16, 512)

        self.bottleneck = ConvBlock(512 + 16, 512)

        self.dec3 = Up(512 + 256, 256)
        self.dec2 = Up(256 + 128, 128)
        self.dec1 = Up(128 + 64, 64)

        self.out = nn.Conv2d(64, 3, kernel_size=1)
        
    def join(self, x: Tensor, t: Tensor) -> Tensor:
        ''' Concatenate image tensor with type embedding tensor '''
        
        B, _, H, W = x.shape
        
        return torch.cat([x, t.view(B, -1, 1, 1).expand(-1, -1, H, W)], dim=1)

    def forward(self, x: Tensor, type: Tensor) -> Tensor:
        ''' Color correct input images based on color vision deficiency type.
        
        x: (B, 3, 512, 512) in [0, 1]
        type: (B,) in [0, 3]
        '''

        # One-hot -> embedding
        t = self.type_embedding(one_hot(type, 4).float())

        # Encoder with conditioning
        x1 = self.enc1(self.join(x, t))
        x2 = self.enc2(self.join(x1, t))
        x3 = self.enc3(self.join(x2, t))
        x4 = self.enc4(self.join(x3, t))

        # Bottleneck
        b = self.bottleneck(self.join(x4, t))

        # Decoder
        d3 = self.dec3(b, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)

        # Residual recoloring
        delta = self.out(d1)

        return (x + delta).clamp(0.0, 1.0)