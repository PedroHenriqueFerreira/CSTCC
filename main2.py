from torch import tensor
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2RGB

from classes.color_blindness_simulator import ColorBlindness

simulator = ColorBlindness().cuda()

image = tensor(cvtColor(imread("out/original.jpg"), COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0

name = ['protanopia', 'deuteranopia', 'tritanopia', 'achromatopsia']

for i in range(4):
    print(image.shape)
    
    sim = simulator(image, i)

    print(sim.shape)

    imwrite(f"./{name[i]}.jpg", sim.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255)