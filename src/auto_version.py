import os
import cv2
import shutil
import random
import subprocess
import numpy as np
import re
import uuid

from pathlib import Path
from datetime import datetime

# ==================================================
# CONFIG
# ==================================================

OUTPUT_IMAGES = "data/outputs/images"
OUTPUT_LABELS = "data/outputs/labels"

PROCESSED_DIR = "data/processed"
SUCCESS_FILE = "data/LATEST_SUCCESS"
THRESHOLD = 10
TRAIN_RATIO = 0.8

IMG_SIZE = (640, 640)

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]

CLASSES = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat"
]

# ==================================================
# COMMAND RUNNER
# ==================================================

def run_cmd(cmd, step=""):

    print(f"\n>>> {step}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}"
        )

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
# SAFE VERSION DETECTION
# ==================================================

SUCCESS_FILE = "data/LATEST_SUCCESS"

def get_next_version():

    if not os.path.exists(SUCCESS_FILE):
        return 1

    with open(SUCCESS_FILE, "r") as f:
        last_success = f.read().strip()

    if not last_success.isdigit():
        return 1

    return int(last_success) + 1

# ==================================================
# IMAGE PREPROCESSING
# ==================================================

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print(f"⚠️ Corrupted image skipped: {image_path}")
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
# CREATE YOLO DATASET YAML
# ==================================================

def create_dataset_yaml(base_output):

    yaml_path = os.path.join(base_output, "dataset.yaml")

    content = f"""
path: {base_output}

train: train/images
val: val/images

nc: {len(CLASSES)}

names:
"""

    for cls in CLASSES:
        content += f"  - {cls}\n"

    with open(yaml_path, "w") as f:
        f.write(content.strip())

    print(f"✅ Created {yaml_path}")

# ==================================================
# CREATE DATASET VERSION
# ==================================================

def create_dataset(version):

    base_output = f"{PROCESSED_DIR}/retrain_v{version}"

    folders = [
        f"{base_output}/train/images",
        f"{base_output}/train/labels",
        f"{base_output}/val/images",
        f"{base_output}/val/labels",
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    all_images = []

    for file in os.listdir(OUTPUT_IMAGES):

        ext = Path(file).suffix.lower()

        if ext not in VALID_EXTENSIONS:
            continue

        image_path = os.path.join(OUTPUT_IMAGES, file)

        label_path = os.path.join(
            OUTPUT_LABELS,
            Path(file).stem + ".txt"
        )

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(image_path)

        if img is None:
            print(f"⚠️ Corrupted image skipped: {file}")
            continue

        all_images.append(file)

    if len(all_images) == 0:
        raise ValueError("No valid images found")

    random.shuffle(all_images)

    split_index = int(len(all_images) * TRAIN_RATIO)

    train_files = all_images[:split_index]
    val_files   = all_images[split_index:]

    print(f"Train samples: {len(train_files)}")
    print(f"Val samples: {len(val_files)}")

    def process(files, split):

        for file in files:

            image_path = os.path.join(OUTPUT_IMAGES, file)

            label_path = os.path.join(
                OUTPUT_LABELS,
                Path(file).stem + ".txt"
            )

            processed = preprocess_image(image_path)

            if processed is None:
                continue

            output_image = os.path.join(
                base_output,
                split,
                "images",
                file
            )

            cv2.imwrite(output_image, processed)

            shutil.copy2(
                label_path,
                os.path.join(
                    base_output,
                    split,
                    "labels",
                    Path(file).stem + ".txt"
                )
            )

    process(train_files, "train")
    process(val_files, "val")

    create_dataset_yaml(base_output)

    return base_output

# ==================================================
# DVC + GIT TAG
# ==================================================

def dvc_push(version):

    processed_folder = f"data/processed/retrain_v{version}"

    current_date = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    unique_id = uuid.uuid4().hex[:6]

    tag = f"data.v{version}_{current_date}_{unique_id}"

    # DVC track whole dataset version
    run_cmd(
        ["dvc", "add", processed_folder],
        f"DVC tracking {processed_folder}"
    )

    # Stage files
    run_cmd([

        "git", "add",
        f"{processed_folder}.dvc",
    ], "Staging DVC file")

    run_cmd([
         
        "git", "add", "-f",
        f"{processed_folder}/dataset.yaml",
    ], "Force staging dataset config")
 
    # Commit
    run_cmd([
        "git",
        "commit",
        "-m",
        f"data: retrain_v{version}"
    ], "Git commit")

    # Create tag
    run_cmd([
        "git",
        "tag",
        "-a",
        tag,
        "-m",
        f"Dataset version {version}"
    ], f"Creating tag {tag}")

    # Push DVC cache
    run_cmd(
        ["dvc", "push", "-j", "1"],
        "Pushing DVC cache"
    )

    # Push git commit
    run_cmd(
        ["git", "push"],
        "Pushing git commit"
    )

    # Push tag → triggers GitHub Actions
    run_cmd(
        ["git", "push", "origin", tag],
        "Pushing tag"
    )

    print(f"\n🚀 GitHub Actions triggered with tag: {tag}")

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

    print(f"Found {total} images in output folder")

    if total >= THRESHOLD:

        version = get_next_version()

        print(f"\nCreating retrain_v{version}")

        create_dataset(version)

        dvc_push(version)

        clear_outputs()

        print(f"\n✅ Dataset v{version} completed")

    else:
        print("\nThreshold not reached")