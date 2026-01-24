from torch import float32, tensor, Tensor
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder

class ToSquare:
    def __init__(self, size=512):
        self.size = size
        
        self.empty_image = tensor(0.0).expand(3, size, size)

    def __call__(self, img: Tensor) -> Tensor:
        h, w = v2.functional.get_size(img)
        
        if h == self.size and w == self.size:
            return img
        
        if not (0.5 <= (w / h) <= 2.0):
            return self.empty_image
        
        scale = self.size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        pad_h, pad_w = self.size - new_h, self.size - new_w
        pad_left, pad_top = pad_w // 2, pad_h // 2
        pad_right, pad_bottom = pad_w - pad_left, pad_h - pad_top
        
        return v2.functional.pad(
            v2.functional.resize(img, (new_h, new_w)), 
            (pad_left, pad_top, pad_right, pad_bottom), 
            padding_mode='reflect'
        )
    
image_transform = v2.Compose([v2.ToImage(), ToSquare(), v2.ToDtype(float32, scale=True)])
    
class ImageDataset(ImageFolder):
    def __init__(self, root: str):
        super().__init__(root=root, transform=image_transform)
    
    def __getitem__(self, index):
        return super().__getitem__(index)[0], self.samples[index][0]

