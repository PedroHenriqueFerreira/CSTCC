import torch

from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from classes.images_dataset import ImagesDataset

from classes.color_blindness_simulator import ColorBlindnessSimulator
from classes.color_correction_model import ColorCorrectionModel
from classes.perceptual_losses import PerceptualLosses

def main(batch_size: int = 10, epochs: int = 20):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    images_dataset = ImagesDataset('dataset/raw', 'dataset/processed')
    images_dataset.preprocess()

    loader = DataLoader(
        images_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    color_blindness_simulator = ColorBlindnessSimulator().to(device)
    color_blindness_simulator.eval()
    
    color_correction_model = ColorCorrectionModel().to(device)
    perceptual_losses = PerceptualLosses().to(device)
    
    adam_optimizer = Adam(color_correction_model.parameters(), lr=1e-4)

    test_x = images_dataset.image_to_tensor(images_dataset.read_image('image.jpg')).to(device)
    test_x = test_x.unsqueeze(0)
    
    test_x = torch.cat([test_x, test_x, test_x, test_x], dim=0)
    test_type = torch.tensor([0, 1, 2, 3], device=device)
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch [{epoch + 1}/{epochs}]")
        
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
            
            # Save test image simulation every epoch
            color_correction_model.eval()
            with torch.no_grad():
                r_test = color_correction_model(test_x, test_type)
                s_test = color_blindness_simulator(test_x, test_type)
                s_r_test = color_blindness_simulator(r_test, test_type)
                
                for i in range(4):
                    images_dataset.save_image(images_dataset.tensor_to_image(r_test[i]), f'test_type_{i}_corrected.jpg')
                    images_dataset.save_image(images_dataset.tensor_to_image(s_test[i]), f'test_type_{i}_simulated.jpg')
                    images_dataset.save_image(images_dataset.tensor_to_image(s_r_test[i]), f'test_type_{i}_corrected_simulated.jpg')
                
                
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(loader):.5f}')
        
if __name__ == '__main__':
    main()