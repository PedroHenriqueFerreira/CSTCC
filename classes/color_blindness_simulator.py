import torch
from torch import nn, Tensor
from torch import tensor

class ColorBlindnessSimulator(nn.Module):
    ''' Color Vision Deficiency Simulator Based on Machado et al. (2009) '''

    def __init__(self):
        super().__init__()

        # Machado et al. (2009) RGB -> RGB matrices (sRGB domain)
        self.register_buffer('protanopia', tensor([
            [0.152286,  1.052583, -0.204868],
            [0.114503,  0.786281,  0.099216],
            [-0.003882, -0.048116,  1.051998],
        ], dtype=torch.float32))
        
        self.register_buffer('deuteranopia', tensor([
            [0.367322,  0.860646, -0.227968],
            [0.280085,  0.672501,  0.047413],
            [-0.011820,  0.042940,  0.968881],
        ], dtype=torch.float32))
        
        self.register_buffer('tritanopia', tensor([
            [1.255528, -0.076749, -0.178779],
            [-0.078411,  0.930809,  0.147602],
            [0.004733,  0.691367,  0.303900],
        ], dtype=torch.float32))
        
        # BT.601 luminance (achromatopsia)
        self.register_buffer('achromatopsia', tensor([
            [0.299, 0.587, 0.114],
            [0.299, 0.587, 0.114],
            [0.299, 0.587, 0.114],
        ], dtype=torch.float32))

    def forward(self, x: Tensor, type: int) -> Tensor:
        ''' x: (B,3,H,W) in [0,1] '''

        # (B,3,H,W) -> (B,H,W,3)
        x = x.permute(0, 2, 3, 1)

        if type == 0: # Protanopia
            x = x @ self.protanopia.T
        elif type == 1: # Deuteranopia
            x = x @ self.deuteranopia.T
        elif type == 2: # Tritanopia
            x = x @ self.tritanopia.T
        elif type == 3: # Achromatopsia
            x = x @ self.achromatopsia.T

        # (B,H,W,3) -> (B,3,H,W)
        x = x.permute(0, 3, 1, 2)

        return x.clamp(0.0, 1.0)
