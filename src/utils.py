from torch import Tensor

def colorfulness(images: Tensor):
    r = images[:, 0, :, :] * 255.0
    g = images[:, 1, :, :] * 255.0
    b = images[:, 2, :, :] * 255.0
    
    rg = (r - g).abs()
    yb = (0.5 * (r + g) - b).abs()

    stdRGYB = (rg.var(dim=(1, 2)) + yb.var(dim=(1, 2))).sqrt()
    meanRGYB = (rg.mean(dim=(1, 2)) ** 2 + yb.mean(dim=(1, 2)) ** 2).sqrt()
    
    return stdRGYB + 0.3 * meanRGYB