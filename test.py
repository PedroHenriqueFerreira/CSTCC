from torchvision.io import read_image
from torchvision.utils import save_image

from classes.dataset import image_transform
from kornia.filters import spatial_gradient, sobel
from kornia.color import rgb_to_lab

for i in range(4):
    image = image_transform(read_image(f'preview/preview_{i}.jpg')).unsqueeze(0)
    image = rgb_to_lab(image)[:, :1] / 100.0

    g = sobel(image)

    g_norm = (g - g.min()) / (g.max() - g.min())

    save_image(g_norm, f'preview/preview_{i}_sobel.jpg')