import torch

from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from classes.preprocess_dataset import PreprocessDataset
from classes.process_dataset import ProcessDataset

from classes.color_blindness_simulator import ColorBlindnessSimulator
from classes.color_correction_model import ColorCorrectionModel
from classes.perceptual_losses import PerceptualLosses

def train(batch_size: int = 10, epochs: int = 20):
    device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    
    preprocessor = PreprocessDataset('dataset/raw', 'dataset/processed')
    preprocessor.run()

    dataset = ProcessDataset('dataset/processed')
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    color_correction_model = ColorCorrectionModel().to(device)
    color_blindness_simulator = ColorBlindnessSimulator().to(device)
    perceptual_losses = PerceptualLosses().to(device)
    
    adam_optimizer = Adam(color_correction_model.parameters(), lr=1e-4)
    
    types = [
        torch.full((batch_size,), 0, device=device),
        torch.full((batch_size,), 1, device=device),
        torch.full((batch_size,), 2, device=device),
        torch.full((batch_size,), 3, device=device),
    ]

    for epoch in range(epochs):
        total_loss = 0.0
        
        for x in tqdm(loader, desc=f"Epoch {epoch:02d}/{epochs:02d}", leave=False):
            curr_loss = 0.0
            
            x = x.to(device)
            
            for i, type in enumerate(types):
                color_correction_model.train()
                
                r_x = color_correction_model(x, type)
                s_r_x = color_blindness_simulator(r_x, type)
                
                loss = perceptual_losses(x, s_r_x)
                
                adam_optimizer.zero_grad()
                loss.backward()
                adam_optimizer.step()
                
                print('LOSS', loss.item())
                
                total_loss += loss.item()
                curr_loss += loss.item()
                
            print(f'Batch Loss: {curr_loss/len(types):.4f}')
                
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}')
        
if __name__ == '__main__':
    train()