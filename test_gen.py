import os
import time
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import tensorflow as tf
from utils import create_dir, calculate_metrics
from model import DeepSegNet  # Assuming your DeepSegNet model is in model.py

def evaluate(model, save_path, test_x, test_y, size):
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in enumerate(zip(test_x, test_y)):
        name = y.split("/")[-1].split(".")[0]

        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        image = image / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)

        with tf.device('/gpu:0'):  # Change this to your desired GPU
            """ FPS calculation """
            start_time = time.time()
            y_pred = model(image)
            end_time = time.time() - start_time
            time_taken.append(end_time)

            """ Evaluation metrics """
            score = calculate_metrics(mask, y_pred)
            metrics_score = [sum(scores) for scores in zip(metrics_score, score)]

            """ Predicted Mask """
            y_pred = np.squeeze(y_pred, axis=0)
            y_pred = y_pred > 0.5
            y_pred = y_pred.astype(np.int32) * 255
            y_pred = np.expand_dims(y_pred, axis=-1)
            y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)

        """ Save the image - mask - pred """
        line = np.ones((size[0], 10, 3)) * 255
        cat_images = np.concatenate([save_img, line, save_mask, line, y_pred], axis=1)
        cv2.imwrite(f"{save_path}/joint/{name}.jpg", cat_images)
        cv2.imwrite(f"{save_path}/mask/{name}.jpg", y_pred)

    jaccard = metrics_score[0] / len(test_x)
    f1 = metrics_score[1] / len(test_x)
    recall = metrics_score[2] / len(test_x)
    precision = metrics_score[3] / len(test_x)
    acc = metrics_score[4] / len(test_x)
    f2 = metrics_score[5] / len(test_x)

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1 / mean_time_taken
    print("Mean FPS: ", mean_fps)

if __name__ == "__main__":
    # Set your GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Change to your GPU ID

    """ Load the checkpoint """
    model = DeepSegNet()
    model.load_weights("files/checkpoint.h5")  # Adjust the path to your checkpoint file

    # Define your test dataset paths
    test_x = sorted(glob("/path/to/test/images/*.jpg"))
    test_y = sorted(glob("/path/to/test/masks/*.jpg"))
    print(f"Images: {len(test_x)} - Masks: {len(test_y)}")

    save_path = "results/test_results"
    for item in ["mask", "joint"]:
        create_dir(f"{save_path}/{item}")

    size = (256, 256)
    create_dir(save_path)
    evaluate(model, save_path, test_x, test_y, size)
