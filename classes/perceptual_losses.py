from torch import nn
from torch import Tensor
from torch.nn import L1Loss

from torchvision.models import resnet50, ResNet50_Weights
from kornia.color import rgb_to_lab
from kornia.losses import ssim_loss

class SSIMLoss(nn.Module):
    ''' Perceptual Loss using Structural Similarity Index (SSIM) '''
    
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return ssim_loss(x, y, 11)

class ResNetLoss(nn.Module):
    ''' Perceptual Loss using pre-trained ResNet50 features '''
    
    def __init__(self):
        super().__init__()

        self.l1_loss = L1Loss()

        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.transforms = ResNet50_Weights.DEFAULT.transforms()
        
        self.resnet.eval()
        for p in self.resnet.parameters():
            p.requires_grad = False

        self.blocks = nn.ModuleDict({
            "layer1": nn.Sequential(
                self.resnet.conv1, 
                self.resnet.bn1, 
                self.resnet.relu,
                self.resnet.maxpool, 
                self.resnet.layer1
            ),
            "layer2": self.resnet.layer2,
            "layer3": self.resnet.layer3,
        })
            
        # self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer('std', tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # def normalize(self, x: Tensor) -> Tensor:
    #     return (x - self.mean) / self.std

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.transforms(x)
        y = self.transforms(y)
    
        loss = 0.0
        
        x_feat = x
        y_feat = y
        
        for block in self.blocks.values():
            x_feat = block(x_feat)
            y_feat = block(y_feat)
            
            loss += self.l1_loss(x_feat, y_feat)
            
        return loss
    
class LabLoss(nn.Module):
    ''' Perceptual Loss using CIELAB color space '''
    
    def __init__(self):
        super().__init__()

        self.l1_loss = L1Loss()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_lab = rgb_to_lab(x)
        y_lab = rgb_to_lab(y)
        
        L_err = self.l1_loss(x_lab[:, 0:1] / 100.0, y_lab[:, 0:1] / 100.0)
        # ab_err = self.l1_loss(x_lab[:, 1:3] / 128.0, y_lab[:, 1:3] / 128.0)

        return L_err # + ab_err
    
class PerceptualLosses(nn.Module):
    ''' Combines multiple perceptual losses '''
    
    def __init__(self):
        super().__init__()
        
        self.ssim_loss = SSIMLoss()
        self.resnet_loss = ResNetLoss()
        self.lab_loss = LabLoss()
        
    def forward(self, x: Tensor, y: Tensor) -> dict:
        ''' Computes all perceptual losses between two batches of images.
        
        x: (B, 3, H, W) in [0, 1]
        y: (B, 3, H, W) in [0, 1]
        
        Returns a dictionary with the computed losses.
        '''
        
        # return self.ssim_loss(x, y) + self.resnet_loss(x, y) + self.lab_loss(x, y)
    
        return self.resnet_loss(x, y)
    