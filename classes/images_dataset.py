from torch import Tensor
from torch.utils.data import Dataset

from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms.functional import resize, pad

from os import listdir, path, makedirs
from tqdm import tqdm

class ImagesDataset(Dataset):
    ''' A dataset for processing images from a directory '''
    
    def __init__(self, root_dir: str, processed_dir: str):
        self.root_dir = root_dir
        self.processed_dir = processed_dir
        
        self.images: list[str] = []

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return read_image(self.images[idx]).float() / 255.0
    
    @staticmethod
    def prepare_image(img: Tensor, size = 512, fill = 255) -> Tensor:
        ''' Resizes an image to the specified size with padding '''
        
        h, w = img.shape[1:3]
        scale = size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        img = resize(img, (new_h, new_w))
        
        pad_w, pad_h = size - new_w, size - new_h
        left, top = pad_w // 2, pad_h // 2
    
        return pad(img, (left, top, pad_w - left, pad_h - top), fill=fill)
    
    @staticmethod
    def check_image_aspect(img: Tensor, min_ratio = 0.5, max_ratio = 2.0):
        ''' Checks if the image aspect ratio is within the specified range. '''
        
        return min_ratio <= (img.shape[2] / img.shape[1]) <= max_ratio

    @staticmethod
    def check_image_colorfulness(img: Tensor, threshold = 45.0) -> bool:
        ''' Checks if the image colorfulness is above the specified threshold '''
        
        r, g, b = img
        
        rg = (r - g).abs()
        yb = (0.5 * (r + g) - b).abs()
        
        stdRGYB = (rg.var() + yb.var()).sqrt()
        meanRGYB = (rg.mean() ** 2 + yb.mean() ** 2).sqrt()
        
        return stdRGYB + 0.3 * meanRGYB >= threshold
    
    def preprocess(self, device: str):
        ''' Processes the root directory images and saves them to the processed directory '''
        
        print('Starting dataset preprocessing...')
        
        if path.exists(self.processed_dir):
            print('Dataset already processed...')
        else:
            if not path.exists(self.root_dir):
                raise FileNotFoundError(f'Root directory does not exist')
            
            makedirs(f'{self.processed_dir}', exist_ok=True)
            
            for folder in listdir(self.root_dir):
                if not path.isdir(f'{self.root_dir}/{folder}'):
                    continue
                
                for file in tqdm(listdir(f'{self.root_dir}/{folder}'), desc=f'Processing [{folder}]'):
                    img = read_image(f'{self.root_dir}/{folder}/{file}').float().to(device)
                              
                    if not self.check_image_aspect(img) or not self.check_image_colorfulness(img): 
                        continue
                    
                    save_image(self.prepare_image(img) / 255.0, f'{self.processed_dir}/{folder}_{file}')
                    
                    
        self.images = [f'{self.processed_dir}/{file}' for file in listdir(self.processed_dir)]

        print('Finished dataset preprocessing.\n')