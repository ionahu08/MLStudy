import os
import glob
import torch

import numpy as np
import pandas as pd

from PIL import Image, ImageFile

from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms

from albumentations import (
    Compose, 
    OneOf, 
    RandomBrightnessContrast,
    RandomGamma, 
    ShiftScaleRotate
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# design a dataset class which outputs
# image and mask images
class SIIMDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        image_ids,
        transform=True,
        preprocessing_fn=None        
    ):
        self.data = defaultdict(dict)

        self.transform = True
        self.preprocessing_fn = preprocessing_fn

        self.aug = Compose(
            [
                ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1, 
                    rotate_limit=10, p=0.8
                ),
                OneOf(
                    [
                        RandomGamma(
                            gamma_limit=(90, 110)
                        ),
                        RandomBrightnessContrast(
                            brightness_limit=0.1, 
                            contrast_limit=0.1
                        ),
                    ],
                    p=0.5
                )
            ]
        )

        TRAIN_PATH = "/Users/ionahu/sources/MLStudy/resources/siim_acr_jpeg/dataset/dummy_png/"

        counter = -1
        for imgid in image_ids:
            counter += 1
            
            # files = glob.glob.glob(os.path.join(TRAIN_PATH, imgid, "*.png"))
            self.data[counter] = {
                "img_path": os.path.join(
                    TRAIN_PATH, imgid
                ),
                "mask_path": os.path.join(
                    TRAIN_PATH, imgid + "_mask"
                )
            }
            

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, item):
        img_path = self.data[item]["img_path"]
        mask_path = self.data[item]["mask_path"]

        img = Image.open(img_path)
        img = img.convert("RGB")

        img = np.array(img)
        mask = Image.open(mask_path)

        mask = (mask>=1).astype("float32")

        if self.transfrom is True:
            augmented = self.aug(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        img = self.preprocessing_fn(img)

        return {
            "image": transforms.ToTensor()(img),
            "mask": transforms.ToTensor()(mask).float()
        }



