import os
import random
import time
import datetime
import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics
from model import DeepSegNet  # Assuming you have a TensorFlow model implementation
from metrics import DiceLoss, DiceBCELoss  # Assuming you have TensorFlow metric implementations
from utils import load_names
import albumentations as A

# Define DATASET class
class DATASET(tf.keras.utils.Sequence):
    def __init__(self, images_path, masks_path, size, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size

    def __len__(self):
        return int(np.ceil(len(self.images_path) / self.batch_size))

    def __getitem__(self, index):
        batch_x = self.images_path[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = self.masks_path[index * self.batch_size:(index + 1) * self.batch_size]

        X = []
        Y = []

        for i in range(len(batch_x)):
            image = cv2.imread(batch_x[i], cv2.IMREAD_COLOR)
            mask = cv2.imread(batch_y[i], cv2.IMREAD_GRAYSCALE)

            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

            image = cv2.resize(image, self.size)
            image = image / 255.0

            mask = cv2.resize(mask, self.size)
            mask = mask / 255.0
            mask = np.expand_dims(mask, axis=2)

            X.append(image)
            Y.append(mask)

        return np.array(X), np.array(Y)

def load_data(path):
    print("Current working directory:", os.getcwd())
    train_names_path = f"dataset/Kvasir-SEG/train.txt"
    valid_names_path = f"dataset/Kvasir-SEG/val.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)

# Rest of your code...
