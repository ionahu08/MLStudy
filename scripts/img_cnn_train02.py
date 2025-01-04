
# updated based on img_cnn_train.py
# replace pretrained model with the one from img_cnn.py

import os

import pandas as pd
import numpy as np

import albumentations 
import torch

from sklearn import metrics
from sklearn.model_selection import train_test_split

import img_cnn_dataset
import img_cnn_engine
from img_cnn_model import get_model
import img_cnn #new

if __name__ == "__main__":
    data_path = "/Users/ionahu/sources/MLStudy/resources/siim_acr_jpeg/dataset/"

    device = "cpu"

    epochs = 10

    df = pd.read_csv(os.path.join(data_path, "train.csv"))

    images = df["ImageID"].values.tolist()

    images = [
        os.path.join(data_path, "dummy_png/", i) for i in images
    ]

    targets = df["target"].values

    # model = get_model(pretrained=True)
    model = img_cnn.AlexNet()  #new

    model.to(device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            )
        ]
    )


    train_images, val_images, train_targets, valid_targets = train_test_split(
        images, targets, stratify=targets, random_state=42
    )
    # print("size of train:", len(train_images))
    # print("size of valid:", len(val_images))


    train_dataset = img_cnn_dataset.ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize = (227, 227),
        augmentations=aug
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=6, shuffle=True, num_workers=4
    )
    # print("type1*,", type(train_loader))

    val_dataset = img_cnn_dataset.ClassificationDataset(
        image_paths = val_images,
        targets=valid_targets,
        resize = (227, 227),
        augmentations=aug
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2, shuffle=False, num_workers=4
    )
    # print("type2*,", type(val_loader))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(epochs):
        img_cnn_engine.train(train_loader, model, optimizer, device=device)

        predictions, valid_targets = img_cnn_engine.evaluation(
            val_loader, model, device=device
        )

        roc_auc = metrics.roc_auc_score(valid_targets, predictions)
        print(
            f"Epoch={epoch}, Valid ROC AUC={roc_auc}"
        )



