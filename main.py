from tqdm import tqdm
from pathlib import Path
from sys import argv

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.utils import save_image

from classes.dataset import ImageDataset, image_transform
from classes.model import ColorCorrectionModel
from classes.simulator import ColorBlindnessSimulator
from classes.criterion import VGGLoss

class Main:
    def __init__(
        self, 
        in_dir = 'datasets/processed', 
        out_file = 'model.pth',  
        lr = 1e-4, 
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.in_dir = Path(in_dir)
        self.out_file = Path(out_file)
        
        self.lr = lr
        
        self.model = ColorCorrectionModel().to(self.device)
        
        if self.out_file.exists():
            self.model.load_state_dict(torch.load(self.out_file, map_location=self.device))
        
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
        
    def train(self, batch_size = 4, epochs = 20):
        train_loader = DataLoader(
            ImageDataset(self.in_dir / 'train'), 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )
        
        val_loader = DataLoader(
            ImageDataset(self.in_dir / 'val'), 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # model = torch.compile(model)
        # model = torch.compile(model)
        # criterion = torch.compile(criterion)
        
        best_val_loss = float('inf')
        for epoch in range(epochs):
            train_loss = self.epoch(train_loader, True, f'Training Epoch [{epoch + 1}/{epochs}]')
            val_loss = self.epoch(val_loader, False, f'Validating Epoch [{epoch + 1}/{epochs}]')
                    
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                print('Saving model...')
                torch.save(self.model.state_dict(), self.out_file)
        
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

    def preview(self, input = 'input.jpg', output = 'output.jpg'):
        input = Path(input)
        output = Path(output)        
        
        if not input.exists():
            print('Input file does not exist!')
            return
        
        image = image_transform(read_image(input)).to(self.device)
        
        x = image.unsqueeze(0).repeat(4, 1, 1, 1)
        type = torch.arange(4, device=self.device)
        
        self.model.eval()
        with torch.inference_mode():
            r_x = self.model(x, type)
            s_x = self.simulator(x, type)
            s_r_x = self.simulator(r_x, type)
            
            data: list[torch.Tensor] = []
            
            for i in range(4):
                data.append(x[i])
                data.append(r_x[i])
                data.append(s_x[i])
                data.append(s_r_x[i])
            
            save_image(data, output, nrow=4)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    
    main = Main()
    
    command, args = argv[1], argv[2:]
    
    if command == 'train':
        main.train()
    elif command == 'test':
        main.test()
    elif command == 'preview': 
        main.preview(*args)
    else:
        print('Unknown command!')