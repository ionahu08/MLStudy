
import os
import csv
import numpy as np
from PIL import Image


if __name__ == "__main__":

    # Directory containing the images
    input_directory = "/Users/ionahu/sources/MLStudy/resources/siim_acr_jpeg/dataset/dummy_png"

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(".png"):  # Process only PNG files
            new_filename = filename[:len(filename)-4] + "_mask"

        # Define the dimensions of the mask
        width, height = 1024, 1024

        # Create a black mask (all values set to zero)
        mask = np.zeros((height, width), dtype=np.uint8)

        # Convert the array to a PIL image
        image = Image.fromarray(mask, mode='L')

        # Save the mask as a PNG file
        image.save(f"{input_directory}/{new_filename}.png")

        print("Grayscale mask saved as 'black_mask.png'")
