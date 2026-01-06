import torch

from torch.nn import Module
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm

from torch.optim import Optimizer

from pathlib import Path

from classes.color_blindness_simulator import ColorBlindnessSimulator
from classes.color_correction_model import ColorCorrectionModel
from classes.perceptual_losses import PerceptualLosses
from classes.image_dataset import ImageDataset, image_transform

def run_epoch(
    loader: DataLoader, 
    model: Module, 
    simulator: Module,
    criterion: Module, 
    optimizer: Optimizer, 
    train: bool, 
    device: str
):
    if train:
        model.train()
    else:
        model.eval()
        
    total_loss = 0.0
  
    context_manager = torch.enable_grad() if train else torch.inference_mode()
  
    with context_manager:  
        progress_bar = tqdm(loader, desc='Training' if train else 'Validation', leave=False)
        
        for x, _ in progress_bar:
            if train:
                model.train()
            
            x = x.to(device, non_blocking=True)
            
            type = torch.randint(0, 4, (x.size(0),), device=device)

            corrected = model(x, type)
            simulated_corrected = simulator(corrected, type)
            loss = criterion(x, simulated_corrected)
    
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            progress_bar.set_postfix(loss=loss.item())
            
            preview(model, simulator, device)
        
    return total_loss / len(loader)

def preview(model: Module, simulator: Module, device: str):
    image_path = Path('preview/image.jpg')
    
    if not image_path.exists():
        return
    
    image = image_transform(read_image(image_path)).to(device)
    
    x = image.unsqueeze(0).repeat(4, 1, 1, 1)
    types = torch.tensor([0, 1, 2, 3], device=device)
    
    model.eval()
    with torch.inference_mode():
        corrected = model(x, types)
        simulated = simulator(x, types)
        
        simulated_corrected = simulator(corrected, types)
        
        for i in range(4):
            group = torch.stack([x[i], corrected[i], simulated[i], simulated_corrected[i]], dim=0)
            
            save_image(group, f'preview/preview_{i}.jpg', nrow=2)

def train(processed_dir = 'datasets/processed', batch_size: int = 16, epochs: int = 20):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    processed_path = Path(processed_dir)
    
    train_dataset = ImageDataset(processed_path / 'train')
    val_dataset = ImageDataset(processed_path / 'val')
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )

    color_blindness_simulator = ColorBlindnessSimulator().to(device).eval()
    perceptual_losses = PerceptualLosses().to(device).eval()
    color_correction_model = ColorCorrectionModel().to(device)
    
    # NUMBER OF PARAMETERS
    total_params = sum(p.numel() for p in color_correction_model.parameters() if p.requires_grad)
    
    print('Number of trainable parameters in Color Correction Model:', total_params)
    
    # color_blindness_simulator = torch.compile(color_blindness_simulator)
    # color_correction_model = torch.compile(color_correction_model)
    # perceptual_losses = torch.compile(perceptual_losses)
    
    adam_optimizer = Adam(color_correction_model.parameters(), lr=1e-4)

    # test_x = read_image('image.jpg').to(device)
    # test_type = torch.tensor([0, 1, 2, 3]).to(device)
    
    # test_x = torch.stack([test_x, test_x, test_x, test_x])

    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = run_epoch(
            train_loader,
            color_correction_model,
            color_blindness_simulator,
            perceptual_losses,
            adam_optimizer,
            train=True,
            device=device
        )
        
        val_loss = run_epoch(
            val_loader,
            color_correction_model,
            color_blindness_simulator,
            perceptual_losses,
            adam_optimizer,
            train=False,
            device=device
        )
                
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print('Saving Model...')
            torch.save(color_correction_model.state_dict(), 'model.pth')
    
    print('Successfully trained the model!')
    
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    train()