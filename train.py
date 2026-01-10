import torch

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision.io import read_image
from torchvision.utils import save_image

from tqdm import tqdm

from torch.optim import Optimizer

from pathlib import Path

from classes.simulator import ColorBlindnessSimulator
from classes.model import ColorCorrectionModel
from classes.criterion import PerceptualLosses
from classes.dataset import ImageDataset, image_transform

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
    
    with torch.set_grad_enabled(train):  
        progress_bar = tqdm(loader, desc='Training' if train else 'Validation')
        
        for x, _ in progress_bar:
            x = x.to(device, non_blocking=True)
            
            if train:
                model.train()
            
            type = torch.randint(0, 4, (x.size(0),), device=device)

            corrected = model(x, type)
            sim_corrected = simulator(corrected, type)
            sim = simulator(x, type)
            loss = criterion(x, corrected, sim, sim_corrected)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            progress_bar.set_postfix(loss=loss.item())
            
            preview(model, simulator, device)
        
    return total_loss

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

def train(processed_dir = 'datasets/processed', batch_size: int = 4, epochs: int = 20, preview_interval: int = 1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    processed_path = Path(processed_dir)
    
    train_dataset = ImageDataset(processed_path / 'train')
    # val_dataset = ImageDataset(processed_path / 'val')
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    # val_loader = DataLoader(
    #     dataset=val_dataset, 
    #     batch_size=batch_size, 
    #     shuffle=False, 
    #     num_workers=4, 
    #     pin_memory=True
    # )

    simulator = ColorBlindnessSimulator().to(device).eval()
    criterion = PerceptualLosses().to(device).eval()
    model = ColorCorrectionModel().to(device)
    
    # NUMBER OF PARAMETERS
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Trainable parameters: {total_params:,}')
    
    # model = torch.compile(model)
    
    # model = torch.compile(model)
    # criterion = torch.compile(criterion)
    
    optimizer = Adam(model.parameters(), lr=1e-4)

    # best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = run_epoch(
            train_loader,
            model,
            simulator,
            criterion,
            optimizer,
            train=True,
            device=device
        )
        
        # val_loss = run_epoch(
        #     val_loader,
        #     model,
        #     simulator,
        #     criterion,
        #     optimizer,
        #     train=False,
        #     device=device
        # )
                
        # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.5f}')
        
        # if val_loss < best_val_loss:
            # best_val_loss = val_loss
        print('Saving Model...')
        torch.save(model.state_dict(), 'model.pth')
    
    print('Successfully trained the model!')
    
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    train()