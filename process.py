import torch
import cv2

from classes.colorization_network import ColorizationNetwork
from classes.losses import ResNetLoss, LabLoss, SSIMLoss

img1 = cv2.cvtColor(cv2.imread('out/original.jpg'), cv2.COLOR_BGR2RGB)
img1_tensor = (torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0)

resnet_loss = ResNetLoss()
lab_loss = LabLoss()
ssim_loss = SSIMLoss()

colorization = ColorizationNetwork()

for i in range(4):
    img1_tensor = colorization(img1_tensor, torch.tensor([i]))

    img1_image = (img1_tensor.squeeze(0).permute(1, 2, 0).detach().numpy() * 255.0).astype('uint8')

    cv2.imwrite(f'out/processed_original_{i}.jpg', cv2.cvtColor(img1_image, cv2.COLOR_RGB2BGR))