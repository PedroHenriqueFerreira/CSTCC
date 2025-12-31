from cv2.typing import MatLike
import cv2

import numpy as np
from tqdm import tqdm

from os import listdir, makedirs

class ProcessData:
    ''' Class to process a dataset of images by applying quality checks and resizing '''
    
    def check_aspect_ratio(self, img: MatLike, min_ratio: float = 0.5, max_ratio: float = 2.0) -> bool:
        ''' Checks if the image aspect ratio is within the specified range. '''
        
        aspect_ratio = img.shape[0] / img.shape[1]
        return min_ratio <= aspect_ratio <= max_ratio

    def check_saturation_mean(self, s: MatLike, min_mean: float = 30.0) -> bool:
        ''' Checks if the mean saturation is above the specified threshold. '''
        
        return s.mean() >= min_mean

    def check_hue_variance(self, h: MatLike, s: MatLike, v: MatLike, min_std: float = 10.0) -> bool:
        ''' Checks if the hue channel variance is above the specified threshold '''
        
        mask_sv = (s > 20) & (v > 20)
        
        if not mask_sv.any():
            return False
        
        return h[mask_sv].std() >= min_std

    def check_colorfulness(self, img: MatLike, threshold: float = 20.0) -> bool:
        ''' Checks if the image colorfulness is above the specified threshold '''
        
        B, G, R = cv2.split(img.astype('float'))
        
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)
        
        stdRGYB = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        meanRGYB = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
        
        return stdRGYB + 0.3 * meanRGYB >= threshold

    def resize(self, img: MatLike, size = 512) -> MatLike:
        ''' Resizes the image to the specified size with padding to maintain aspect ratio '''
        
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

        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    
    def run(self):
        ''' Processes the dataset by applying quality checks and resizing images '''
        
        raw_dir = 'data/raw/'
        processed_dir = 'data/processed/'
        
        makedirs(f'{processed_dir}', exist_ok=True)
        
        for folder in listdir(raw_dir):
            if '.' in folder:
                continue
            
            for file in tqdm(listdir(f'{raw_dir}/{folder}'), desc=f"Processando {folder}"):
                img = cv2.imread(f'{raw_dir}/{folder}/{file}')
                
                if img is None:
                    continue
                
                h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
                
                if not self.check_aspect_ratio(img): 
                    continue
                
                if not self.check_saturation_mean(s):
                    continue
                
                if not self.check_hue_variance(h, s, v):
                    continue
                
                if not self.check_colorfulness(img):
                    continue
                 
                cv2.imwrite(f'{processed_dir}/{file}', self.resize(img))
                