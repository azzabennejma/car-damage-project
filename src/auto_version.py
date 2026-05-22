import os
import cv2
import shutil
import random
import subprocess
import numpy as np
import re

from pathlib import Path

# ==================================================
# CONFIG
# ==================================================

OUTPUT_IMAGES = "data/outputs/images"
OUTPUT_LABELS = "data/outputs/labels"
PROCESSED_DIR = "data/processed"

THRESHOLD = 10
TRAIN_RATIO = 0.8
IMG_SIZE = (640, 640)
VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# ==================================================
# COUNT OUTPUT IMAGES
# ==================================================

def count_images():
    if not os.path.exists(OUTPUT_IMAGES):
        return 0

    return len([
        f for f in os.listdir(OUTPUT_IMAGES)
        if Path(f).suffix.lower() in VALID_EXTENSIONS
    ])

# ==================================================
# GET NEXT VERSION (SAFE)
# ==================================================

def get_next_version():
    if not os.path.exists(PROCESSED_DIR):
        return 1

    versions = []

    for d in os.listdir(PROCESSED_DIR):
        match = re.match(r"retrain_v(\d+)$", d)
        if match:
            versions.append(int(match.group(1)))

    return max(versions) + 1 if versions else 1

# ==================================================
# SAFE IMAGE PREPROCESSING (FIXED CRASH)
# ==================================================

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    # ❗ IMPORTANT FIX: prevent crash
    if img is None:
        print(f"⚠️ Skipping unreadable image: {image_path}")
        return None

    img = cv2.resize(img, (1280, 1280))

    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    img = cv2.filter2D(img, -1, kernel)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, IMG_SIZE)

    return img

# ==================================================
# CREATE DATASET VERSION
# ==================================================

def create_dataset(version):

    base_output = f"{PROCESSED_DIR}/retrain_v{version}"

    for folder in [
        f"{base_output}/train/images",
        f"{base_output}/train/labels",
        f"{base_output}/val/images",
        f"{base_output}/val/labels",
    ]:
        os.makedirs(folder, exist_ok=True)

    all_images = []

    for file in os.listdir(OUTPUT_IMAGES):

        ext = Path(file).suffix.lower()
        if ext not in VALID_EXTENSIONS:
            continue

        image_path = os.path.join(OUTPUT_IMAGES, file)
        label_path = os.path.join(OUTPUT_LABELS, Path(file).stem + ".txt")

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(image_path)

        if img is None:
            print(f"⚠️ corrupted image skipped: {file}")
            continue

        all_images.append(file)

    if len(all_images) == 0:
        raise ValueError("No valid images found")

    random.shuffle(all_images)

    split_index = int(len(all_images) * TRAIN_RATIO)

    train_files = all_images[:split_index]
    val_files = all_images[split_index:]

    print(f"Train samples: {len(train_files)}")
    print(f"Val samples: {len(val_files)}")

    def process(files, split):

        for file in files:

            image_path = os.path.join(OUTPUT_IMAGES, file)
            label_path = os.path.join(OUTPUT_LABELS, Path(file).stem + ".txt")

            processed = preprocess_image(image_path)

            # ❗ skip broken images safely
            if processed is None:
                continue

            cv2.imwrite(
                os.path.join(base_output, split, "images", file),
                processed
            )

            shutil.copy2(
                label_path,
                os.path.join(base_output, split, "labels", Path(file).stem + ".txt")
            )

    process(train_files, "train")
    process(val_files, "val")

    return base_output

# ==================================================
# DVC PUSH (SAFE MINIMAL)
# ==================================================

def dvc_push(version):

    train_folder = f"data/processed/retrain_v{version}/train"

    subprocess.run(["dvc", "add", train_folder], check=True)

    subprocess.run(["git", "add", f"{train_folder}.dvc"], check=True)

    subprocess.run([
        "git", "commit",
        "-m", f"Dataset version v{version}"
    ], check=True)

    subprocess.run(["dvc", "push", "-j", "1"], check=True)

    subprocess.run(["git", "push"], check=True)

# ==================================================
# CLEAR OUTPUTS
# ==================================================

def clear_outputs():

    shutil.rmtree(OUTPUT_IMAGES, ignore_errors=True)
    shutil.rmtree(OUTPUT_LABELS, ignore_errors=True)

    os.makedirs(OUTPUT_IMAGES, exist_ok=True)
    os.makedirs(OUTPUT_LABELS, exist_ok=True)

# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    total = count_images()

    print(f"Found {total} output images")

    if total >= THRESHOLD:

        version = get_next_version()

        print(f"Creating retrain_v{version}")

        create_dataset(version)

        dvc_push(version)

        clear_outputs()

        print(f"Dataset v{version} pushed successfully")

    else:
        print("Threshold not reached")