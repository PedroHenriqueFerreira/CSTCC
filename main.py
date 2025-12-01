import numpy as np

from PIL import Image

matrices = {
    'rgb_to_lms': np.array([
        [17.8824, 43.5161, 4.1194],
        [3.4557, 27.1554, 3.8671],
        [0.0300, 0.1843, 1.4671]
    ]).T,
    'lms_to_rgb': np.array([
        [0.0809, -0.1305, 0.1167],
        [-0.0102, 0.0540, -0.1136],
        [-0.0004, -0.0041, 0.6935]
    ]).T,   
    'lms_to_protanopia': np.array([
        [0, 2.0234, -2.5258],
        [0, 1, 0],
        [0, 0, 1]
    ]).T,
    'lms_to_deuteranopia': np.array([
        [1, 0, 0],
        [0.4942, 0, 1.2483],
        [0, 0, 1]
    ]).T,   
    'lms_to_tritanopia': np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-0.3960, 0.8011, 0]
    ]).T,
    'lms_to_monochromacy': np.array([
        [0.2990, 0.5870, 0.1140],
        [0.2990, 0.5870, 0.1140],
        [0.2990, 0.5870, 0.1140]
    ]).T
}
        
class Utils:
    @staticmethod
    def load_lms(path):
        img_rgb = np.array(Image.open(path)) / 255
        img_lms = np.dot(img_rgb[:,:,:3], matrices["rgb_to_lms"])

        return img_lms
    
if __name__ == "__main__":
    img_rgb = np.array(Image.open("input.jpg")) / 255
    
    img_lms = img_rgb @ matrices["rgb_to_lms"]
    img_lms_protanopia = img_lms @ matrices["lms_to_protanopia"]
    img_rgb_protanopia = img_lms_protanopia @ matrices["lms_to_rgb"]
    
    img_lms_deuteranopia = img_lms @ matrices["lms_to_deuteranopia"]
    img_rgb_deuteranopia = img_lms_deuteranopia @ matrices["lms_to_rgb"]
    
    img_lms_tritanopia = img_lms @ matrices["lms_to_tritanopia"]
    img_rgb_tritanopia = img_lms_tritanopia @ matrices["lms_to_rgb"]
    
    Image.fromarray(np.uint8(img_rgb_protanopia * 255)).save("output_protanopia.jpg")
    Image.fromarray(np.uint8(img_rgb_deuteranopia * 255)).save("output_deuteranopia.jpg")
    Image.fromarray(np.uint8(img_rgb_tritanopia * 255)).save("output_tritanopia.jpg")