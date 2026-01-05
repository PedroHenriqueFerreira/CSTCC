import torch

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm

from pathlib import Path

from classes.color_blindness_simulator import ColorBlindnessSimulator
from classes.color_correction_model import ColorCorrectionModel
from classes.perceptual_losses import PerceptualLosses
from classes.image_dataset import ImageDataset

def train(processed_dir = 'datasets/processed', batch_size: int = 16, epochs: int = 20):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    processed_path = Path(processed_dir)
    
    train_dataset = ImageDataset(processed_path / 'train')
    val_dataset = ImageDataset(processed_path / 'val')
    test_dataset = ImageDataset(processed_path / 'test')
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    color_blindness_simulator = ColorBlindnessSimulator().to(device)
    color_blindness_simulator.eval()
    
    color_correction_model = ColorCorrectionModel().to(device)
    
    perceptual_losses = PerceptualLosses().to(device)
    adam_optimizer = Adam(color_correction_model.parameters(), lr=1e-4)

    test_x = read_image('image.jpg').to(device)
    test_type = torch.tensor([0, 1, 2, 3]).to(device)
    
    test_x = torch.stack([test_x, test_x, test_x, test_x])
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]")
        
        for x in pbar:
            color_correction_model.train()
            x = x.to(device, non_blocking=True)
            type = torch.randint(0, 4, (x.size(0),), device=device)
        
            s_r_x = color_blindness_simulator(color_correction_model(x, type), type)
            
            loss = perceptual_losses(x, s_r_x)
            
            adam_optimizer.zero_grad()
            loss.backward()
            adam_optimizer.step()
            
            pbar.set_postfix(loss=loss.item())
            
            total_loss += loss.item()
            
            color_correction_model.eval()
            with torch.no_grad():
                r_test = color_correction_model(test_x, test_type)
                s_test = color_blindness_simulator(test_x, test_type)
                s_r_test = color_blindness_simulator(r_test, test_type)

                for i in range(4):
                    save_image(r_test[i], f'test_type_{i}_r.jpg')
                    save_image(s_test[i], f'test_type_{i}_s.jpg')
                    save_image(s_r_test[i], f'test_type_{i}_s_r.jpg')
                
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.5f}')
        
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    train()