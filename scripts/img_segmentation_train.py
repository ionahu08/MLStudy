import os 
import sys
import torch

import numpy as np
import pandas as pd 
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim

# from apex import amp
from collections import OrderedDict
from sklearn import model_selection
from tqdm import tqdm
from torch.optim import lr_scheduler

from img_segmentation_dataset import SIIMDataset

TRAINING_CSV = "./resources/siim_acr_jpeg/dataset/train.csv"

TRAINING_BATCH_SIZE = 6
TEST_BATCH_SIZE = 2

EPOCHS = 10

ENCODER = "resnet18"

ENCODER_WEIGHTS = "imagenet"

DEVICE = "cpu"

def train(dataset, data_loader, model, criterion, optimizer):
    model.train()

    num_batches = int(len(dataset)/data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)
     
    for d in tk0:
        inputs = d["image"]
        targets = d["mask"]

        inputs = inputs.to(DEVICE, dtype=torch.float)
        targets = targets.to(DEVICE, dtype=torch.float)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()


    tk0. close()


def evaluate(dataset, data_loader, model):
    model.eval()

    final_loss = 0

    num_batches = int(len(dataset) / data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)

    with torch.no_grad():
        for d in tk0:
            inputs = d["image"]
            targets = d["mask"]

            inputs = inputs.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.float)
            output = model(inputs)
            loss = criterion(output, targets)
            final_loss += loss

    tk0.close()

    return final_loss / num_batches


if __name__ == "__main__":
    df = pd.read_csv(TRAINING_CSV)

    prep_fn = smp.encoders.get_preprocessing_fn(
        ENCODER,
        ENCODER_WEIGHTS
    )

    df_train, df_test = model_selection.train_test_split(
        df, random_state=42, test_size=0.25
    )

    training_images = df_train["ImageID"].values
    validation_images = df_test["ImageID"].values

    train_dataset = SIIMDataset(
        training_images, 
        transform = True,
        preprocessing_fn = prep_fn
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = TRAINING_BATCH_SIZE,
        shuffle = True,
        num_workers = 12
    )

    val_dataset = SIIMDataset(
        validation_images, 
        transform = False,
        preprocessing_fn = prep_fn
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = TEST_BATCH_SIZE,
        shuffle = True,
        num_workers = 4
    )


    model = smp.Unet(
        encoder_name = ENCODER,
        encoder_weights = ENCODER_WEIGHTS,
        classes = 1,
        activation=None
    )
    model.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, verbose=True
    )

    # some logging
    print(f"Training batch size: {TRAINING_BATCH_SIZE}")
    print(f"Test batch size: {TEST_BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    print(f"Encoder: {ENCODER}")

    for epoch in range(EPOCHS):
        print(f"Training Epoch: {epoch}")
        train(
            train_dataset,
            train_loader,
            model,
            criterion,
            optimizer
        )
        print(f"Validation Epoch: {epoch}")

        val_log = evaluate(
            val_dataset,
            val_loader,
            model
        )

        scheduler.step(val_log["loss"])
        print("\n")








