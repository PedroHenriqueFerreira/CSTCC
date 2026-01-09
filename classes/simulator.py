import torch
from torch import nn, tensor, Tensor

from kornia.color import rgb_to_linear_rgb,  linear_rgb_to_rgb

class ColorBlindnessSimulator(nn.Module):
    ''' Color Vision Deficiency Simulator Based on Machado et al. (2009) '''

    def __init__(self):
        super().__init__()

        protan = tensor([
            [0.152286,  1.052583, -0.204868],
            [0.114503,  0.786281,  0.099216],
            [-0.003882, -0.048116,  1.051998],
        ], dtype=torch.float32)

        deutan = tensor([
            [0.367322,  0.860646, -0.227968],
            [0.280085,  0.672501,  0.047413],
            [-0.011820,  0.042940,  0.968881],
        ], dtype=torch.float32) 

        tritan = tensor([
            [1.255528, -0.076749, -0.178779],
            [-0.078411,  0.930809,  0.147602],
            [0.004733,  0.691367,  0.303900],
        ], dtype=torch.float32)

        achrom = tensor([
            [0.299, 0.587, 0.114],
            [0.299, 0.587, 0.114],
            [0.299, 0.587, 0.114],
        ], dtype=torch.float32)

        self.register_buffer('matrices', torch.stack([protan, deutan, tritan, achrom], dim=0))

    def forward(self, x: Tensor, type: Tensor) -> Tensor:
        '''Simulate color blindness on input images.
        
        x: (B, 3, H, W) in [0, 1]
        type: (B,) in [0, 3]
        '''
        
        x = rgb_to_linear_rgb(x)
        
        x = torch.einsum('bchw,btc->bthw', x, self.matrices[type])
        
        x = linear_rgb_to_rgb(x)
        
        return x.clamp(0.0, 1.0)
