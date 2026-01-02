import torch
from torch import nn, tensor, Tensor

class ColorBlindnessSimulator(nn.Module):
    ''' Color Vision Deficiency Simulator Based on Machado et al. (2009) '''

    def __init__(self):
        super().__init__()

        protan = tensor([
            [0.152286,  1.052583, -0.204868],
            [0.114503,  0.786281,  0.099216],
            [-0.003882, -0.048116,  1.051998],
        ], dtype=torch.float32).T

        deutan = tensor([
            [0.367322,  0.860646, -0.227968],
            [0.280085,  0.672501,  0.047413],
            [-0.011820,  0.042940,  0.968881],
        ], dtype=torch.float32).T 

        tritan = tensor([
            [1.255528, -0.076749, -0.178779],
            [-0.078411,  0.930809,  0.147602],
            [0.004733,  0.691367,  0.303900],
        ], dtype=torch.float32).T

        achrom = tensor([
            [0.299, 0.587, 0.114],
            [0.299, 0.587, 0.114],
            [0.299, 0.587, 0.114],
        ], dtype=torch.float32).T

        self.register_buffer('matrices', torch.stack([protan, deutan, tritan, achrom], dim=0))

    def rgb_to_linear(self, x: Tensor) -> Tensor:
        '''Convert sRGB to linear RGB'''
        
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    
    def linear_to_rgb(self, x: Tensor) -> Tensor:
        '''Convert linear RGB to sRGB'''
        
        x = x.clamp(0.0, 1.0)
        
        return torch.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)

    def forward(self, x: Tensor, type: Tensor) -> Tensor:
        '''Simulate color blindness on input images.
        
        x: (B, 3, H, W) in [0, 1]
        type: (B,) in [0, 3]
        '''
        
        B, _, H, W = x.shape
        
        # (B, 3, H, W) -> (B, H*W, 3)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, H * W, 3)
        
        x = self.rgb_to_linear(x)
        x = torch.bmm(x, self.matrices[type])
        x = self.linear_to_rgb(x)
        
        # (B, H*W, 3) -> (B, 3, H, W)
        x = x.reshape(B, H, W, 3)
        x = x.permute(0, 3, 1, 2)
        
        return x.clamp(0.0, 1.0)
