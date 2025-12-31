from classes.simulator__ import Simulator

from cv2 import imread, imwrite, cvtColor, COLOR_BGR2RGB

image = cvtColor(imread("out/original.jpg"), COLOR_BGR2RGB)

print(image.shape)

simulator = Simulator()
protan, deutan, tritan, mono = simulator.simulate_cvd(image)

imwrite("./out/original.jpg", image[:, :, ::-1])
imwrite("./out/protan.jpg", protan[:, :, ::-1])
imwrite("./out/deutan.jpg", deutan[:, :, ::-1])
imwrite("./out/tritan.jpg", tritan[:, :, ::-1])
imwrite("./out/mono.jpg", mono)