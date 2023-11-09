import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from model import DeepSegNet
from utils import create_dir, calculate_metrics
from train import load_data

def evaluate(model, save_path, test_x, test_y, size):
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = y.split("/")[-1].split(".")[0]

        # Load and preprocess the image
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        # Load and preprocess the mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask
        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate([mask, mask, mask], axis=2)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)

        # Make predictions
        start_time = time.time()
        y_pred = model.predict(image)
        end_time = time.time() - start_time
        time_taken.append(end_time)

        # Evaluation metrics
        jaccard, f1, recall, precision, acc, f2 = calculate_metrics(mask, y_pred)
        metrics_score[0] += jaccard
        metrics_score[1] += f1
        metrics_score[2] += recall
        metrics_score[3] += precision
        metrics_score[4] += acc
        metrics_score[5] += f2

        # Post-process the predicted mask
        y_pred = y_pred[0]
        y_pred = (y_pred > 0.5).astype(np.uint8) * 255

        # Save the image, mask, and predicted mask
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
    # Load the checkpoint
    model = DeepSegNet()
    model.build((1, 256, 256, 3))  # Use the appropriate input shape

    # Load the weights
    model.load_weights("files/checkpoint.h5")

    # Test dataset
    path = "/media/nikhil/Seagate Backup Plus Drive/ML_DATASET/Kvasir-SEG"
    (train_x, train_y), (test_x, test_y) = load_data(path)

    save_path = f"results/Kvasir-SEG"
    for item in ["mask", "joint"]:
        create_dir(f"{save_path}/{item}")

    size = (256, 256)
    create_dir(save_path)
    evaluate(model, save_path, test_x, test_y, size)
