import numpy as np

class Matrices:
    '''
    Holds transformation matrices.
    All the details for transformation matrices can be found at: https://arxiv.org/pdf/1711.10662.pdf.
    '''

    @staticmethod
    def rgb_to_lms():
        ''' Matrix for RGB color-space to LMS color-space transformation. '''
        
        return np.array([
            [17.8824, 43.5161, 4.1193],
            [3.4557, 27.1554, 3.8671],
            [0.02996, 0.18431, 1.4670]
        ]).T

    @staticmethod
    def lms_to_rgb():
        ''' Matrix for LMS color-space to RGB color-space transformation. '''
        
        return np.array([
            [0.0809, -0.1305, 0.1167],
            [-0.0102, 0.0540, -0.1136], # -0.0102 or 0.0102
            [-0.0004, -0.0041, 0.6935]
        ]).T

    @staticmethod
    def lms_protanopia_sim():
        ''' Matrix for Simulating Protanopia color-blindness from LMS color-space. '''
        
        return np.array([
            [0, 2.02344, -2.5281],
            [0, 1, 0],
            [0, 0, 1]
        ]).T

    @staticmethod
    def lms_deuteranopia_sim():
        ''' Matrix for Simulating Deuteranopia color-blindness from LMS color-space. '''

        return np.array([
            [1, 0, 0],
            [0.494207, 0, 1.24827],
            [0, 0, 1]
        ]).T

    @staticmethod
    def lms_tritanopia_sim():
        ''' Matrix for Simulating Tritanopia color-blindness from LMS color-space. '''
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-0.395913, 0.801109, 0]
        ]).T