from torch import nn
from torch import Tensor
from torch.nn import L1Loss

from torchvision.models import resnet50, ResNet50_Weights
from kornia.color import rgb_to_lab
from kornia.losses import ssim_loss

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return ssim_loss(x, y, 11)

class ResNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1_loss = L1Loss()

        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.transforms = ResNet50_Weights.DEFAULT.transforms()
        
        self.resnet.eval()
        for p in self.resnet.parameters():
            p.requires_grad = False

        # self.blocks = nn.ModuleDict({
        #     "layer1": nn.Sequential(
        #         resnet.conv1, 
        #         resnet.bn1, 
        #         resnet.relu,
        #         resnet.maxpool, 
        #         resnet.layer1
        #     ),
        #     "layer2": resnet.layer2,
        #     "layer3": resnet.layer3,
        # })
            
        # self.register_buffer('mean', tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer('std', tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # def normalize(self, x: Tensor) -> Tensor:
    #     return (x - self.mean) / self.std

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.transforms(x)
        y = self.transforms(y)
    
        return self.l1_loss(self.resnet(x), self.resnet(y))
    
class LabLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1_loss = L1Loss()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_lab = rgb_to_lab(x)
        y_lab = rgb_to_lab(y)
        
        L_err = self.l1_loss(x_lab[:, 0:1] / 100.0, y_lab[:, 0:1] / 100.0)
        ab_err = self.l1_loss(x_lab[:, 1:3] / 128.0, y_lab[:, 1:3] / 128.0)

        return L_err + ab_err