from tqdm import tqdm
from pathlib import Path
from sys import argv
from shutil import move

import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.utils import save_image

from classes.dataset import ImageDataset, image_transform
from classes.model import ColorCorrectionModel
from classes.simulator import ColorBlindnessSimulator
from classes.criterion import VGGLoss

class Main:
    def __init__(self, model_path = 'model.pth', lr = 1e-4):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        self.model_path = Path(model_path)
        self.lr = lr
        
        self.model = ColorCorrectionModel().to(self.device)
        
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        self.simulator = ColorBlindnessSimulator().to(self.device).eval()
        self.criterion = VGGLoss().to(self.device).eval()
        
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
            
    def epoch(self, loader: DataLoader, train: bool, desc: str) -> float:
        if train:
            self.model.train()
        else:
            self.model.eval()
            
        total_loss = 0.0
        
        with torch.set_grad_enabled(train):  
            progress_bar = tqdm(loader, desc=desc)
            
            for x, _ in progress_bar:
                x = x.to(self.device, non_blocking=True)
                type = torch.randint(0, 4, (x.size(0),), device=self.device)
                
                loss = self.criterion(x, self.simulator(self.model(x, type), type))

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                
                progress_bar.set_postfix(loss=loss.item())
            
        return total_loss
        
    def train(self, dir = 'datasets/processed', batch_size = 4, epochs = 20):
        dir = Path(dir)
        
        train_loader = DataLoader(
            ImageDataset(dir / 'train'), 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )
        
        val_loader = DataLoader(
            ImageDataset(dir / 'val'), 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        # model = torch.compile(model)
        # simulator = torch.compile(simulator)
        # criterion = torch.compile(criterion)
        
        best_val_loss = float('inf')
        for epoch in range(epochs):
            train_loss = self.epoch(train_loader, True, f'Training Epoch [{epoch + 1}/{epochs}]')
            val_loss = self.epoch(val_loader, False, f'Validating Epoch [{epoch + 1}/{epochs}]')
                    
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                print('Saving model...')
                torch.save(self.model.state_dict(), self.model_path)
        
        print('Successfully trained the model!')
        
    def test(self, batch_size = 4):
        test_loader = DataLoader(
            ImageDataset(self.in_dir / 'test'), 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        test_loss = self.epoch(test_loader, False, 'Testing')
        
        print(f'Test Loss: {test_loss:.5f}')

    def preview(self, input_path = 'input.jpg', output_path = 'output.jpg'):
        input_path = Path(input_path)
        output_path = Path(output_path)        
        
        if not input_path.exists():
            print('Input file does not exist!')
            return
        
        image = image_transform(read_image(input_path)).to(self.device)
        
        x = image.unsqueeze(0).repeat(4, 1, 1, 1)
        type = torch.arange(4, device=self.device)
        
        self.model.eval()
        with torch.inference_mode():
            r_x = self.model(x, type)
            s_x = self.simulator(x, type)
            s_r_x = self.simulator(r_x, type)
            
            data: list[Tensor] = []
            
            for i in range(4):
                data.append(x[i])
                data.append(r_x[i])
                data.append(s_x[i])
                data.append(s_r_x[i])
            
            save_image(data, output_path, nrow=4)
            
    def preprocess(
        self, 
        root_path = 'datasets/raw', 
        processed_path = 'datasets/processed', 
        batch_size = 128, 
        train_size = 0.8, 
        val_size = 0.1, 
        colorfulness_threshold = 45.0
    ):
        root_path = Path(root_path)
        processed_path = Path(processed_path)
        
        if processed_path.exists():
            print('Images have already been preprocessed!')
            return
        
        processed_path.mkdir(parents=True, exist_ok=True)
        
        loader = DataLoader(ImageDataset(root_path), batch_size, num_workers=4, pin_memory=True)
        
        saved_images: set[Path] = set()        
        
        for images, paths in tqdm(loader, desc='Preprocessing Images'):
            images: Tensor = images.to(self.device, non_blocking=True)
            
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
        
        train_path = processed_path / 'train' / 'images'
        val_path = processed_path / 'val' / 'images'
        test_path = processed_path / 'test' / 'images'
        
        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)
        test_path.mkdir(parents=True, exist_ok=True)

        n_train = int(len(saved_images) * train_size)
        n_val = int(len(saved_images) * val_size)
            
        for i, image_path in enumerate(tqdm(saved_images, desc='Splitting Dataset')):
            if i < n_train:
                move(image_path, train_path / image_path.name)
            elif i < n_train + n_val:
                move(image_path, val_path / image_path.name)
            else:
                move(image_path, test_path / image_path.name)
                    
        print('Successfully preprocessed images!')

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    
    main = Main()
    
    command, args = argv[1], argv[2:]
    
    for i, arg in enumerate(args):
        try:
            args[i] = int(arg)
        except ValueError:
            try:
                args[i] = float(arg)
            except ValueError:
                pass
    
    if command == 'preprocess':
        main.preprocess(*args)
    elif command == 'train':
        main.train(*args)
    elif command == 'test':
        main.test(*args)
    elif command == 'preview': 
        main.preview(*args)
    else:
        print('Unknown command!')