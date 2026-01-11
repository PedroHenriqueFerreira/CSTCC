from torch import tensor, Tensor
from torch.nn import Module, ModuleList, L1Loss
from torchvision.models import vgg19, VGG19_Weights

class VGGLoss(Module):
    def __init__(self):
        super().__init__()

        self.register_buffer('mean', tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.blocks = ModuleList([
            self.vgg[0:4],
            self.vgg[4:9],
            self.vgg[9:18],
            self.vgg[18:27],
            self.vgg[27:36]
        ])
        
        self.l1_loss = L1Loss()
    
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
