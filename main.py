from tqdm import tqdm
from pathlib import Path
from sys import argv
from shutil import move
from random import shuffle, seed

seed(42)

import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.utils import save_image

from src.dataset import ImageDataset, image_transform
from src.model import ColorCorrectionModel
from src.simulator import ColorBlindnessSimulator
from src.criterion import ResNetLoss
from src.utils import colorfulness

from kornia.metrics import ssim, psnr

class Main:
    def __init__(self, model_path = 'model.pth', lr = 1e-4):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        self.model_path = Path(model_path)
        self.lr = lr
        
        self.model = ColorCorrectionModel().to(self.device)
        self.simulator = ColorBlindnessSimulator().to(self.device).eval()
        self.criterion = ResNetLoss().to(self.device).eval()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        # model = torch.compile(model)
        # simulator = torch.compile(simulator)
        # criterion = torch.compile(criterion)
        
    def preprocess(
        self, 
        root_path = 'datasets/raw', 
        processed_path = 'datasets/processed', 
        batch_size = 64, 
        train_size = 0.8, 
        colorfulness_threshold = 50.0
    ):
        root_path = Path(root_path)
        processed_path = Path(processed_path)
        
        if processed_path.exists():
            print('Images have already been preprocessed!')
            return
        
        loader = DataLoader(ImageDataset(root_path), batch_size, num_workers=0, pin_memory=True)
        
        processed_path.mkdir(parents=True, exist_ok=True)
        
        saved_images: set[Path] = set()        
                
        for images, paths in tqdm(loader, desc='Preprocessing Images'):
            images: Tensor = images.to(self.device, non_blocking=True)
            
            mask = colorfulness(images) >= colorfulness_threshold
            
            images = images[mask].cpu()
            paths = [Path(paths[i]) for i, v in enumerate(mask.cpu().tolist()) if v]

            for image, path in zip(images, paths):
                image_path = processed_path / path.name
                
                save_image(image, image_path)
                saved_images.add(image_path)
        
        train_path = processed_path / 'train' / 'images'
        val_path = processed_path / 'val' / 'images'
        
        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)

        saved_images = list(saved_images)
        shuffle(saved_images)  
           
        n_train = int(len(saved_images) * train_size)
        for i, image_path in enumerate(tqdm(saved_images, desc='Splitting Dataset')):
            output_path = train_path if i < n_train else val_path
            move(image_path, output_path / image_path.name)
                    
        print('Successfully preprocessed images!')
            
    def epoch(self, loader: DataLoader, train: bool, desc: str) -> float:
        if train:
            self.model.train()
        else:
            self.model.eval()
            
        total_loss = 0.0
        
        with torch.set_grad_enabled(train):  
            progress_bar = tqdm(loader, desc=desc)
            
            for i, (x, _) in enumerate(progress_bar):
                if train:
                    self.model.train()
                
                x = x.to(self.device, non_blocking=True)
                type = torch.randint(0, 4, (x.size(0),), device=self.device)
                
                loss = self.criterion(x, self.simulator(self.model(x, type), type), type)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                
                progress_bar.set_postfix(loss=loss.item())
                
                if train and (i % 10 == 0):
                    self.preview()
            
        return total_loss
        
    def preview(self, input_path = 'input.jpg', output_path = 'output.jpg'):
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            print('Input file does not exist!')
            return
        
        image: Tensor = image_transform(read_image(input_path)).to(self.device)
        
        x = image.unsqueeze(0).repeat(4, 1, 1, 1)
        type = torch.arange(4, device=self.device)
        
        psnr_value: tuple[float, float, float, float] = ()
        ssim_value: tuple[float, float, float, float] = ()
        
        self.model.eval()
        with torch.inference_mode():
            r_x = self.model(x, type)
            s_x = self.simulator(x, type)
            s_r_x = self.simulator(r_x, type)
        
            data: list[Tensor] = []
        
            for i in range(4):
                psnr_value += (psnr(x[i:i+1], r_x[i:i+1], max_val=1.0).item(),)
                ssim_value += (ssim(x[i:i+1], r_x[i:i+1], window_size=11).mean().item(),)
                
                data.extend([x[i], r_x[i], s_x[i], s_r_x[i]])    
                
            save_image(data, output_path, nrow=4)
            
        return psnr_value, ssim_value
        
    def train(self, dir = 'datasets/processed', batch_size = 4, epochs = 5):
        dir = Path(dir)
        
        train_loader = DataLoader(
            ImageDataset(dir / 'train'), 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )
    
        for epoch in range(epochs):
            train_loss = self.epoch(train_loader, True, f'Training Epoch [{epoch + 1}/{epochs}]')
                    
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.5f}')
            torch.save(self.model.state_dict(), self.model_path)
        
        print('Successfully trained the model!')
           
    def val(self, input_dir = 'datasets/processed/val', output_dir = 'outputs'):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
    
        if output_dir.exists():
            print('Output directory already exists!')
            return
    
        output_dir.mkdir(parents=True, exist_ok=True)
        
        psnr_values: list[tuple[float, float, float, float]] = []
        ssim_values: list[tuple[float, float, float, float]] = []
        
        images = list(input_dir.glob('images/*'))
        
        for input_path in tqdm(images, desc='Validating Images'):
            psnr_value, ssim_value = self.preview(input_path, output_dir / input_path.name)
             
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)   
            
        psnr_tensor = torch.tensor(psnr_values, device=self.device)
        ssim_tensor = torch.tensor(ssim_values, device=self.device)
        
        psnr_avg = psnr_tensor.mean(dim=0).tolist()
        psnr_std = psnr_tensor.std(dim=0, unbiased=False).tolist()
        
        ssim_avg = ssim_tensor.mean(dim=0).tolist()
        ssim_std = ssim_tensor.std(dim=0, unbiased=False).tolist()   
        
        print('Successfully validated images!')
                
        for i, type in enumerate(['PROT', 'DEUT', 'TRIT', 'MONO']):
            print(f'{type} | PSNR {psnr_avg[i]:.6f} ± {psnr_std[i]:.6f} | SSIM {ssim_avg[i]:.6f} ± {ssim_std[i]:.6f}')
        

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    
    if len(argv) < 2:
        print('Please provide a command!')
        exit(1)
    
    command, args = argv[1], argv[2:]
    
    for i, arg in enumerate(args):
        try:
            args[i] = int(arg)
        except ValueError:
            try:
                args[i] = float(arg)
            except ValueError:
                pass
    
    main = Main()
    
    if command == 'preprocess':
        main.preprocess(*args)
    elif command == 'preview': 
        main.preview(*args)
    elif command == 'train':
        main.train(*args)
    elif command == 'val':
        main.val(*args)
    else:
        print('Unknown command!')