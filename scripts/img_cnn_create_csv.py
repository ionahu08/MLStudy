

import os
import csv


if __name__ == "__main__":

    # Directory containing the images
    input_directory = "/Users/ionahu/sources/MLStudy/resources/siim_acr_jpeg/dataset/dummy_png"

    # Output CSV file
    output_csv = "/Users/ionahu/sources/MLStudy/resources/siim_acr_jpeg/dataset/train.csv"

    # Prepare the data
    data = []
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(".png"):  # Process only PNG files
            data.append({"ImageID": filename, "target": 0})


    # Write the data to a CSV file
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["ImageID", "target"])
        writer.writeheader()  # Write the header
        writer.writerows(data)  # Write the data

    print(f"CSV file created with PNG files: {output_csv}")
