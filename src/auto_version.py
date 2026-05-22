import os
import cv2
import shutil
import random
import subprocess
import numpy as np
import yaml
import re

from pathlib import Path
from datetime import datetime

# ==================================================
# CONFIG
# ==================================================

OUTPUT_IMAGES    = "data/outputs/images"
OUTPUT_LABELS    = "data/outputs/labels"
PROCESSED_DIR    = "data/processed"
PARAMS_PATH      = "params.yaml"
THRESHOLD        = 10
TRAIN_RATIO      = 0.8
IMG_SIZE         = (640, 640)
VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]
CLASS_NAMES      = [
    "dent", "scratch", "crack",
    "glass breakage", "lamp breakage", "tire flat"
]

# ==================================================
# UTILS
# ==================================================

def run_cmd(cmd: list, description: str = ""):
    print(f"\n>>> {description or ' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr.strip()}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr.strip()}")
    print(result.stdout.strip())
    return result

def load_params() -> dict:
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)

def save_params(params: dict):
    with open(PARAMS_PATH, "w") as f:
        yaml.safe_dump(params, f)

# ==================================================
# COUNT OUTPUT IMAGES
# ==================================================

def count_images() -> int:
    if not os.path.exists(OUTPUT_IMAGES):
        return 0
    return len([
        f for f in os.listdir(OUTPUT_IMAGES)
        if Path(f).suffix.lower() in VALID_EXTENSIONS
    ])

# ==================================================
# GET NEXT VERSION
# ==================================================

def get_next_version() -> int:
    if not os.path.exists(PROCESSED_DIR):
        return 1

    versions = []

    for d in os.listdir(PROCESSED_DIR):
        match = re.match(r"retrain_v(\d+)$", d)
        if match:
            versions.append(int(match.group(1)))

    return max(versions) + 1 if versions else 1

# ==================================================
# PREPROCESSING
# ==================================================

def preprocess_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (1280, 1280))
    sharpen_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    img = cv2.filter2D(img, -1, sharpen_kernel)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, IMG_SIZE)
    return img

# ==================================================
# CREATE DATASET VERSION
# ==================================================

def create_dataset(version: int) -> str:
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
            print(f"  ⚠️  No label for {file} — skipping")
            continue
        if cv2.imread(image_path) is None:
            print(f"  ⚠️  Cannot read {file} — skipping")
            continue
        all_images.append(file)

    if not all_images:
        raise ValueError("No valid image+label pairs found")

    random.shuffle(all_images)
    split_index = int(len(all_images) * TRAIN_RATIO)
    train_files = all_images[:split_index]
    val_files   = all_images[split_index:]

    print(f"  Train : {len(train_files)} samples")
    print(f"  Val   : {len(val_files)} samples")

    def process_and_save(files, split):
        for file in files:
            image_path = os.path.join(OUTPUT_IMAGES, file)
            label_path = os.path.join(OUTPUT_LABELS, Path(file).stem + ".txt")
            processed  = preprocess_image(image_path)
            cv2.imwrite(
                os.path.join(base_output, split, "images", file),
                processed
            )
            shutil.copy2(
                label_path,
                os.path.join(base_output, split, "labels", Path(file).stem + ".txt")
            )

    process_and_save(train_files, "train")
    process_and_save(val_files,   "val")
    return base_output

# ==================================================
# GENERATE DATASET.YAML
# ==================================================

def generate_dataset_yaml(version: int) -> str:
    yaml_path = f"{PROCESSED_DIR}/retrain_v{version}/dataset.yaml"
    config = {
        "path":  f"data/processed/retrain_v{version}",
        "train": "train/images",
        "val":   "val/images",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  ✅ dataset.yaml → {yaml_path}")
    return yaml_path

# ==================================================
# UPDATE PARAMS.YAML
# ==================================================

def update_params(version: int):
    params = load_params()
    params["data_version"] = version
    params["train"]["data_yaml"] = \
        f"data/processed/retrain_v{version}/dataset.yaml"
    save_params(params)
    print(f"  ✅ params.yaml updated → data_version={version}")

# ==================================================
# DVC + GIT TAG
# ==================================================

def dvc_push(version: int) -> str:
    processed_folder = f"data/processed/retrain_v{version}/train"
    current_date     = datetime.now().strftime("%Y-%m-%d")
    tag              = f"data.v{version}_{current_date}"

    run_cmd(
        ["dvc", "add", f"{processed_folder}/train"],
        f"DVC tracking train set for retrain_v{version}"
    )
    run_cmd([
        "git", "add",
        f"{processed_folder}/train.dvc",
        PARAMS_PATH,
        f"{processed_folder}/dataset.yaml",
    ], "Staging files")
    run_cmd([
        "git", "commit", "-m",
        f"data: retrain_v{version} on {current_date}"
    ], "Committing")
    run_cmd([
        "git", "tag", "-a", tag, "-m",
        f"Dataset version {version} — triggers retraining"
    ], f"Creating tag {tag}")
    run_cmd(["dvc", "push", "-j", "1"], "Pushing to DVC remote")
    run_cmd(["git", "push", "origin", "main"], "Pushing commit")
    run_cmd(["git", "push", "origin", tag],    "Pushing tag → triggers GitHub Actions")

    print(f"\n🚀 Tag {tag} pushed — GitHub Actions will trigger retraining")
    return tag

# ==================================================
# CLEAR OUTPUTS
# ==================================================

def clear_outputs():
    shutil.rmtree(OUTPUT_IMAGES)
    shutil.rmtree(OUTPUT_LABELS)
    os.makedirs(OUTPUT_IMAGES, exist_ok=True)
    os.makedirs(OUTPUT_LABELS, exist_ok=True)
    print("  ✅ Output folder cleared")

# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    total = count_images()
    print(f"\n📂 Found {total} images in output folder")

    if total < THRESHOLD:
        print(f"⏳ Need {THRESHOLD - total} more images to trigger versioning")
    else:
        version = get_next_version()
        print(f"\n🔖 Creating retrain_v{version}...")
        try:
            create_dataset(version)
            generate_dataset_yaml(version)
            update_params(version)
            tag = dvc_push(version)
            clear_outputs()
            print(f"\n{'='*50}")
            print(f"  ✅ Dataset v{version} ready")
            print(f"  🏷  Tag     : {tag}")
            print(f"  🚀  GitHub Actions retraining triggered")
            print(f"{'='*50}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise