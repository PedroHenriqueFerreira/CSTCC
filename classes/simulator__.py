from daltonlens.simulate import Simulator_Machado2009, Deficiency

from cv2.typing import MatLike
from cv2 import cvtColor, COLOR_RGB2GRAY

class Simulator:
    ''' Simulator for color vision deficiency (CVD) using DaltonLens. '''
     
    def __init__(self):
        self.simulator = Simulator_Machado2009()
        
    def simulate_cvd(self, image: MatLike) -> tuple[MatLike, MatLike, MatLike, MatLike]:
        ''' Simulate protanopia, deuteranopia, tritanopia, and monochromacy on the input image '''
        
        protan = self.simulator.simulate_cvd(image, Deficiency.PROTAN, 1)
        deutan = self.simulator.simulate_cvd(image, Deficiency.DEUTAN, 1)
        tritan = self.simulator.simulate_cvd(image, Deficiency.TRITAN, 1)
        
        print('A', COLOR_RGB2GRAY)
        
        mono = cvtColor(image, COLOR_RGB2GRAY)
        
        return protan, deutan, tritan, mono
    
__all__ = ["Simulator"]