

import os
from PIL import Image


if __name__ == "__main__":

    # Directory containing the images
    input_directory = "/Users/ionahu/sources/MLStudy/resources/siim_acr_jpeg/dataset/dummy_jp2"
    output_directory = "/Users/ionahu/sources/MLStudy/resources/siim_acr_jpeg/dataset/dummy_png"

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Loop through all files in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".jp2"):  # Process only JP2 files
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.png")
            
            # Open and convert the image
            with Image.open(input_path) as img:
                img.save(output_path, format="PNG")
                print(f"Converted: {filename} -> {output_path}")

