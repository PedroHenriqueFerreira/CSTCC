import torch

from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from classes.preprocess_dataset import PreprocessDataset
from classes.images_dataset import ProcessDataset

from classes.color_blindness_simulator import ColorBlindnessSimulator
from classes.color_correction_model import ColorCorrectionModel
from classes.perceptual_losses import PerceptualLosses

import cv2

batch_size = 10

dataset = ProcessDataset('dataset/processed')
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

color_correction_model = ColorCorrectionModel().cuda()
color_blindness_simulator = ColorBlindnessSimulator().cuda()
perceptual_losses = PerceptualLosses().cuda()

adam_optimizer = Adam(color_correction_model.parameters(), lr=1e-4)

types = [
    torch.full((batch_size,), 0).cuda()
]

images_saved = cv2.cvtColor(cv2.imread('image.jpg', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype('float32') / 255.0
images_saved = torch.from_numpy(images_saved).permute(2, 0, 1).unsqueeze(0).cuda()

# Simulate the image saved
color_blindness_simulator.eval()
with torch.no_grad():
    simulated_image = color_blindness_simulator(images_saved, types[0][0:1])

    cv2.imwrite('simulated_image.jpg', cv2.cvtColor((simulated_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype('uint8'), cv2.COLOR_RGB2BGR))

for x in loader:
    x = x.cuda()
    
    color_correction_model.train()
    
    c_x = color_correction_model(x, types[0])
    s_c_x = color_blindness_simulator(c_x, types[0])
        
    loss = perceptual_losses(x, s_c_x)
        
    adam_optimizer.zero_grad()
    loss.backward()
    adam_optimizer.step()
    
    print('LOSS', loss)
    
    # Recolor and save the first image in the batch
    with torch.no_grad():
        recolored_image = color_correction_model(images_saved[0:1], types[0][0:1])
        
        sim_recolored_image = color_blindness_simulator(recolored_image, types[0][0:1])
        
        recolored_image = recolored_image.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
        
        recolored_image = (recolored_image * 255.0).astype('uint8')
        
        cv2.imwrite('test.jpg', cv2.cvtColor(recolored_image, cv2.COLOR_RGB2BGR))
        
        # save simulated recolored image
        sim_recolored_image = sim_recolored_image.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
        sim_recolored_image = (sim_recolored_image * 255.0).astype('uint8')
        cv2.imwrite('test_simulated.jpg', cv2.cvtColor(sim_recolored_image, cv2.COLOR_RGB2BGR))
    # break