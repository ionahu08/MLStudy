#step01: read the grayscale image using 
# OpenCV or Python-PIL
#step02: resize 
# step03: convert to numpy array
# step04: flatten the matrix

import os

import numpy as np
import pandas as pands
import matplotlib.pyplot as plt

if __name__ == "__main__":
    random_image = np.random.randint(0, 256, (256, 256))

    plt.figure(figsize=(7, 7))

    plt.imshow(random_image, cmap='gray', vmin=0, vmax=255)
    plt.show()
