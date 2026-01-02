import cv2
import torch

from torch.utils.data import Dataset
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
        path = self.images[idx]

        return self.image_to_tensor(self.read_image(path))
    
    def read_image(self, path: str) -> cv2.Mat:
        ''' Reads an image from the specified path and converts it from BGR to RGB '''
        
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        
        if img is None:
            return None
        
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def image_to_tensor(self, img: cv2.Mat):
        ''' Processes an image to a tensor by permuting dimensions and normalizing '''
        
        return torch.from_numpy(img.astype('float32') / 255.0).permute(2, 0, 1)
    
    def tensor_to_image(self, tensor: torch.Tensor) -> cv2.Mat:
        ''' Converts a tensor back to an image by permuting dimensions and denormalizing '''
        
        img = tensor.permute(1, 2, 0).cpu().numpy() * 255.0
        return img.astype('uint8')
    
    def save_image(self, img: cv2.Mat, path: str):
        ''' Saves an image to the specified path after converting it from RGB to BGR '''
        
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    def resize_image(self, img: cv2.Mat, size = 512, color = 256) -> cv2.Mat:
        ''' Resizes an image to the specified size with padding '''
        
        h, w = img.shape[:2]

        scale = size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_h = size - new_h
        pad_w = size - new_w

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        return cv2.copyMakeBorder(
            resized, 
            top, bottom, left, right, cv2.BORDER_CONSTANT, 
            value=[color, color, color]
        )
    
    def check_image_aspect(self, img: cv2.Mat, min_ratio: float = 0.5, max_ratio: float = 2.0) -> bool:
        ''' Checks if the image aspect ratio is within the specified range. '''
        
        aspect_ratio = img.shape[0] / img.shape[1]
        return min_ratio <= aspect_ratio <= max_ratio

    def check_image_saturation(self, s: cv2.Mat, min_mean: float = 30.0) -> bool:
        ''' Checks if the mean saturation is above the specified threshold. '''
        
        return s.mean() >= min_mean

    def check_image_hue(self, h: cv2.Mat, s: cv2.Mat, v: cv2.Mat, min_std: float = 10.0) -> bool:
        ''' Checks if the hue channel variance is above the specified threshold '''
        
        mask_sv = (s > 20) & (v > 20)
        
        if not mask_sv.any():
            return False
        
        return h[mask_sv].std() >= min_std

    def check_image_colorfulness(self, img: cv2.Mat, threshold: float = 45.0) -> bool:
        ''' Checks if the image colorfulness is above the specified threshold '''
        
        B, G, R = cv2.split(img.astype('float'))
        
        rg = abs(R - G)
        yb = abs(0.5 * (R + G) - B)
        
        stdRGYB = ((rg.std() ** 2 + yb.std() ** 2)) ** 0.5
        meanRGYB = ((rg.mean() ** 2 + yb.mean() ** 2)) ** 0.5
        
        return stdRGYB + 0.3 * meanRGYB >= threshold
    
    def preprocess(self):
        ''' Processes the root directory images and saves them to the processed directory '''
        
        print('Starting dataset preprocessing...')
        
        if path.exists(self.processed_dir):
            print('Dataset already processed...')
        else:
            if not path.exists(self.root_dir):
                raise FileNotFoundError(f'Root directory does not exist')
            
            makedirs(f'{self.processed_dir}', exist_ok=True)
            
            for folder in listdir(self.root_dir):
                if '.' in folder:
                    continue
                
                for file in tqdm(listdir(f'{self.root_dir}/{folder}'), desc=f'Processing {folder}'):
                    img = self.read_image(f'{self.root_dir}/{folder}/{file}')
                
                    if img is None:
                        continue
                    
                    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
                    
                    if not self.check_image_aspect(img): 
                        continue
                    
                    if not self.check_image_saturation(s):
                        continue
                    
                    if not self.check_image_hue(h, s, v):
                        continue
                    
                    if not self.check_image_colorfulness(img):
                        continue
            
                    self.save_image(self.resize_image(img), f'{self.processed_dir}/{file}')
                    
        self.images = [f'{self.processed_dir}/{file}' for file in listdir(self.processed_dir)]

        print('Finished dataset preprocessing.\n')