from pathlib import Path
from shutil import move

from tqdm import tqdm

from torch import Tensor, cuda
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from classes.dataset import ImageDataset

class Preprocessor:
    def __init__(
        self, 
        root_dir = 'datasets/raw', 
        processed_dir = 'datasets/processed_colorful', 
        batch_size = 128, 
        train_size = 0.7, 
        val_size = 0.2, 
        colorfulness = 45.0
    ):
        self.root_path = Path(root_dir)
        self.processed_path = Path(processed_dir)
        
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.colorfulness = colorfulness
    
    def preprocess(self):
        device = 'cuda' if cuda.is_available() else 'cpu'
        
        if self.processed_path.exists():
            print('Images have already been preprocessed!')
            return
        
        loader = DataLoader(ImageDataset(self.root_path), self.batch_size, num_workers=4, pin_memory=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        saved_images: set[Path] = set()        
        
        for images, paths in tqdm(loader, desc='Preprocessing Images'):
            images: Tensor = images.to(device, non_blocking=True)
            
            r = images[:, 0, :, :] * 255.0
            g = images[:, 1, :, :] * 255.0
            b = images[:, 2, :, :] * 255.0
            
            rg = (r - g).abs()
            yb = (0.5 * (r + g) - b).abs()

            stdRGYB = (rg.var(dim=(1, 2)) + yb.var(dim=(1, 2))).sqrt()
            meanRGYB = (rg.mean(dim=(1, 2)) ** 2 + yb.mean(dim=(1, 2)) ** 2).sqrt()
            
            colorfulness = stdRGYB + 0.3 * meanRGYB
            
            mask = colorfulness >= self.colorfulness
            
            images = images[mask].cpu()
            paths = [Path(paths[i]) for i, v in enumerate(mask.cpu().tolist()) if v]

            for image, path in zip(images, paths):
                save_image(image, self.processed_path / path.name)
                saved_images.add(self.processed_path / path.name)
        
        train_dir = self.processed_path / 'train' / 'images'
        val_dir = self.processed_path / 'val' / 'images'
        test_dir = self.processed_path / 'test' / 'images'
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        n_train = int(len(saved_images) * self.train_size)
        n_val = int(len(saved_images) * self.val_size)
            
        for i, image_path in enumerate(tqdm(saved_images, desc='Splitting Dataset')):
            if i < n_train:
                move(image_path, train_dir / image_path.name)
            elif i < n_train + n_val:
                move(image_path, val_dir / image_path.name)
            else:
                move(image_path, test_dir / image_path.name)
                    
        print('Successfully preprocessed images!')
                
if __name__ == '__main__':
    Preprocessor().preprocess()