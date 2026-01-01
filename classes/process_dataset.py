import cv2
import torch

from torch.utils.data import Dataset
from os import listdir, path

class ProcessDataset(Dataset):
    ''' A dataset for processing images from a directory '''
    
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

        self.images = [path.join(root_dir, file) for file in sorted(listdir(root_dir))]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        path = self.images[idx]

        # BGR -> RGB
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # [0, 255] -> [0, 1]
        img = img.astype('float32') / 255.0

        # (H, W, 3) -> (3, H, W)
        return torch.from_numpy(img).permute(2, 0, 1)
