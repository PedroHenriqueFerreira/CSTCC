from torch import nn
from torch import tensor, Tensor
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

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.transforms = ResNet50_Weights.DEFAULT.transforms()
        
        resnet.eval()
        for p in resnet.parameters():
            p.requires_grad = False
           
        self.blocks = nn.ModuleList([
            nn.Sequential(
                resnet.conv1, 
                resnet.bn1, 
                resnet.relu,
                resnet.maxpool,
            ),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        ])
            
        self.register_buffer('mean', tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.normalize(x)
        y = self.normalize(y)
    
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            
            loss += self.l1_loss(x, y)

        return loss
          
        # x = self.transforms(x)
        # y = self.transforms(y)  
        # return self.l1_loss(self.resnet(x), self.resnet(y))
    
class LabLoss(nn.Module):
    ''' Perceptual Loss using CIELAB color space '''
    
    def __init__(self):
        super().__init__()

        self.l1_loss = L1Loss()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_lab = rgb_to_lab(x)
        y_lab = rgb_to_lab(y)
        
        L_err = self.l1_loss(x_lab[:, 0:1] / 100.0, y_lab[:, 0:1] / 100.0)
        ab_err = self.l1_loss(x_lab[:, 1:3] / 128.0, y_lab[:, 1:3] / 128.0)

        return 0.5 * L_err + ab_err
    
class PerceptualLosses(nn.Module):
    ''' Combines multiple perceptual losses '''
    
    def __init__(self):
        super().__init__()
        
        self.ssim_loss = SSIMLoss()
        self.resnet_loss = ResNetLoss()
        self.lab_loss = LabLoss()
        
    def forward(self, x: Tensor, y: Tensor) -> dict:
        ''' Computes all perceptual losses between two batches of images.
        
        x: (B, 3, 512, 512) in [0, 1]
        y: (B, 3, 512, 512) in [0, 1] '''
        
        # return self.ssim_loss(x, y) + self.resnet_loss(x, y) + self.lab_loss(x, y)
    
        return self.resnet_loss(x, y)
    