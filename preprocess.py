from pathlib import Path
from shutil import move

from tqdm import tqdm

from torch import Tensor, cuda
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from classes.dataset import ImageDataset

def preprocess(
    root_dir = 'datasets/raw', 
    processed_dir = 'datasets/processed', 
    batch_size = 128,
    train_size = 0.7,
    val_size = 0.2,
    colorfulness_threshold = 40.0,
):
    device = 'cuda' if cuda.is_available() else 'cpu'
    
    root_path = Path(root_dir)
    processed_path = Path(processed_dir)
    
    if processed_path.exists():
        print('Images have already been preprocessed!')
        return
    
    loader = DataLoader(ImageDataset(root_path), batch_size, num_workers=4, pin_memory=True)

    saved_images: set[Path] = set()
            
    processed_path.mkdir(parents=True, exist_ok=True)
    
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
        
        mask = colorfulness >= colorfulness_threshold
        
        images = images[mask].cpu()
        paths = [Path(paths[i]) for i, v in enumerate(mask.cpu().tolist()) if v]

        for image, path in zip(images, paths):
            save_image(image, processed_path / path.name)
            saved_images.add(processed_path / path.name)
    
    train_dir = processed_path / 'train' / 'images'
    val_dir = processed_path / 'val' / 'images'
    test_dir = processed_path / 'test' / 'images'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    n_train = int(len(saved_images) * train_size)
    n_val = int(len(saved_images) * val_size)
        
    for i, image_path in enumerate(tqdm(saved_images, desc='Splitting Dataset')):
        if i < n_train:
            move(image_path, train_dir / image_path.name)
        elif i < n_train + n_val:
            move(image_path, val_dir / image_path.name)
        else:
            move(image_path, test_dir / image_path.name)
                
    print('Successfully preprocessed images!')
                
if __name__ == '__main__':
    preprocess()