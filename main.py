import numpy as np

from PIL import Image

class Matrices:
    '''
    Holds transformation matrices.
    All the details for transformation matrices can be found at: https://arxiv.org/pdf/1711.10662.pdf.
    '''
    
    @staticmethod
    def rgb_to_lms():
        ''' Matrix for RGB color-space to LMS color-space transformation. '''
        
        return np.array([
            [17.8824, 43.5161, 4.1194],
            [3.4557, 27.1554, 3.8671],
            [0.0300, 0.1843, 1.4671]
        ]).T

    @staticmethod
    def lms_to_rgb():
        ''' Matrix for LMS color-space to RGB color-space transformation. '''
        
        return np.array([
            [0.0809, -0.1305, 0.1167],
            [-0.0102, 0.0540, -0.1136],
            [-0.0004, -0.0041, 0.6935]
        ]).T

    @staticmethod
    def lms_to_protanopia():
        ''' Matrix for Simulating Protanopia color-blindness from LMS color-space. '''
        
        return np.array([
            [0, 2.0234, -2.5258],
            [0, 1, 0],
            [0, 0, 1]
        ]).T

    @staticmethod
    def lms_to_deuteranopia():
        ''' Matrix for Simulating Deuteranopia color-blindness from LMS color-space. '''

        return np.array([
            [1, 0, 0],
            [0.4942, 0, 1.2483],
            [0, 0, 1]
        ]).T

    # CHECK LATER

    @staticmethod
    def lms_to_tritanopia():
        ''' Matrix for Simulating Tritanopia color-blindness from LMS color-space. '''
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-0.3960, 0.8011, 0]
        ]).T
        
    @staticmethod
    def lms_to_monochromacy():
        ''' Matrix for Simulating Monochromacy color-blindness from LMS color-space. '''
        
        return np.array([
            [0.2990, 0.5870, 0.1140],
            [0.2990, 0.5870, 0.1140],
            [0.2990, 0.5870, 0.1140]
        ]).T
        
class Utils:
    @staticmethod
    def load_lms(path):
        img_rgb = np.array(Image.open(path)) / 255
        img_lms = np.dot(img_rgb[:,:,:3], Matrices.rgb_to_lms())

        return img_lms
    
if __name__ == "__main__":
    img_rgb = np.array(Image.open("input.jpg")) / 255
    
    img_lms = img_rgb @ Matrices.rgb_to_lms()
    img_lms_protanopia = img_lms @ Matrices.lms_to_protanopia()
    img_rgb_protanopia = img_lms_protanopia @ Matrices.lms_to_rgb()
    
    img_lms_deuteranopia = img_lms @ Matrices.lms_to_deuteranopia()
    img_rgb_deuteranopia = img_lms_deuteranopia @ Matrices.lms_to_rgb()
    
    img_lms_tritanopia = img_lms @ Matrices.lms_to_tritanopia()
    img_rgb_tritanopia = img_lms_tritanopia @ Matrices.lms_to_rgb()
    
    Image.fromarray(np.uint8(img_rgb_protanopia * 255)).save("output_protanopia.jpg")
    Image.fromarray(np.uint8(img_rgb_deuteranopia * 255)).save("output_deuteranopia.jpg")
    Image.fromarray(np.uint8(img_rgb_tritanopia * 255)).save("output_tritanopia.jpg")