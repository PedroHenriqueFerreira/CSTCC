from torch import nn
from torch import Tensor
import torch
from torch.nn import L1Loss

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg19, VGG19_Weights
from kornia.color import rgb_to_lab
from kornia.losses import ssim_loss
from kornia.filters import sobel

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.l1_loss(x, y)

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        x = rgb_to_lab(x)[:, :1] / 100.0
        y = rgb_to_lab(y)[:, :1] / 100.0
        
        return self.l1_loss(sobel(x), sobel(y))

class ColorSpreadLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.l1_loss(x.var(dim=(2, 3)), y.var(dim=(2, 3)))
    
class SSIMLoss(nn.Module):
    ''' Perceptual Loss using Structural Similarity Index (SSIM) '''
    
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return ssim_loss(x, y, window_size=11)

class ContrastiveDeltaELoss(nn.Module):
    def __init__(self, margin: float = 5.0):
        super().__init__()
        self.margin = margin

    def forward(self, sim_x, sim_corrected):
        lab_x = rgb_to_lab(sim_x)
        lab_corr = rgb_to_lab(sim_corrected)

        delta_e = torch.norm(lab_corr - lab_x, dim=1)

        # hinge loss
        loss = torch.relu(self.margin - delta_e)
        return loss.mean()

class VGGLoss(nn.Module):
    ''' Perceptual Loss using pre-trained VGG19 features '''
    
    def __init__(self):
        super().__init__()

        self.l1_loss = L1Loss()

        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        print(self.vgg)
        
        self.blocks = nn.ModuleList([
            self.vgg[0:4],    # relu1_2 [0:4]
            # self.vgg[4:9],   # relu2_2 [4:9]
            # self.vgg[9:18],  # relu3_4 [9:18]
            # self.vgg[18:27], # relu4_4 [18:27]
        ])
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def transforms(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.transforms(x)
        y = self.transforms(y) 
    
        loss = 0.0
        
        for block in self.blocks:
            x = block(x)
            y = block(y)
            
            loss += self.l1_loss(x, y)

        return loss

class PerceptualLosses(nn.Module):
    def __init__(
        self,
        w_l1=1.0,
        w_vgg=0.2,
        w_contrast=1.0,
    ):
        super().__init__()

        # self.l1 = nn.L1Loss()
        self.vgg_loss = VGGLoss()
        # self.contrast = ContrastiveDeltaELoss(margin=5.0)

        self.w_l1 = w_l1
        self.w_vgg = w_vgg
        self.w_contrast = w_contrast

    def forward(self, x, corrected, sim_x, sim_corrected):
        # loss_l1 = self.l1(corrected, x)
        # loss_vgg = self.vgg(corrected, x)
        # loss_contrast = self.contrast(sim_x, sim_corrected)

        # return (
        #     self.w_l1 * loss_l1
        #     + self.w_vgg * loss_vgg
        #     + self.w_contrast * loss_contrast
        # )
        
        return self.vgg_loss(x, sim_corrected)