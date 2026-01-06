from torch import Tensor, float32
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder

class ToSquare:
    def __init__(self, size=512, fill=255, padding_mode='constant'):
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img: Tensor) -> Tensor:
        h, w = v2.functional.get_size(img)
        
        if h == self.size and w == self.size:
            return img
        
        scale = self.size / max(h, w)
        
        new_h, new_w = int(h * scale), int(w * scale)
        pad_h, pad_w = self.size - new_h, self.size - new_w
        pad_left, pad_top = pad_w // 2, pad_h // 2
        pad_right, pad_bottom = pad_w - pad_left, pad_h - pad_top
        
        return v2.functional.pad(
            v2.functional.resize(img, (new_h, new_w)), 
            (pad_left, pad_top, pad_right, pad_bottom), 
            fill=self.fill,
            padding_mode=self.padding_mode
        )
    
image_transform = v2.Compose([
    v2.ToImage(),
    ToSquare(),
    v2.ToDtype(float32, scale=True),    
])
    
class ImageDataset(ImageFolder):
    def __init__(self, root: str):
        super().__init__(root=root, transform=image_transform)
    
    def __getitem__(self, index):
        return super().__getitem__(index)[0], self.samples[index][0]
