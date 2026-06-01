import os
import json
import shutil
import yaml
import torch
import gc

import mlflow
import mlflow.pytorch

from ultralytics import YOLO

# ==================================================
# CLEAN MEMORY
# ==================================================

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==================================================
# LOAD PARAMETERS
# ==================================================

with open("params.yaml") as f:
    p = yaml.safe_load(f)

train_p = p["train"]

# ==================================================
# CREATE FOLDERS
# ==================================================

os.makedirs("metrics", exist_ok=True)
os.makedirs("model_versions", exist_ok=True)

# ==================================================
# MLFLOW
# ==================================================

mlflow.set_experiment("Car Damage Detection")

# ==================================================
# DEVICE
# ==================================================

device = 0 if torch.cuda.is_available() else "cpu"

print(
    f"\n🖥 Running on: {'GPU' if device == 0 else 'CPU'}"
)

# ==================================================
# LOAD MODEL
# ==================================================

if os.path.exists("model/best.pt"):
    print("\nUsing existing trained model for fine-tuning")
    model = YOLO("model/best.pt")
else:
    print("\nUsing pretrained YOLOv8 base model")
    model = YOLO("yolov8s.pt")

# ==================================================
# START MLFLOW RUN
# ==================================================

with mlflow.start_run():

    # ----------------------------------------------
    # TAGS
    # ----------------------------------------------

    mlflow.set_tag(
        "project",
        "car_damage_detection"
    )

    mlflow.set_tag(
        "framework",
        "YOLOv8"
    )

    mlflow.set_tag(
        "data_version",
        f"v{p['data_version']}"
    )

    mlflow.set_tag(
        "dataset_version",
        f"retrain_v{p['data_version']}"
    )

    mlflow.set_tag(
        "device",
        "gpu" if device == 0 else "cpu"
    )

    # ----------------------------------------------
    # PARAMETERS
    # ----------------------------------------------

    mlflow.log_param(
        "epochs",
        train_p["epochs"]
    )

    mlflow.log_param(
        "batch_size",
        train_p["batch_size"]
    )

    mlflow.log_param(
        "imgsz",
        train_p["imgsz"]
    )

    mlflow.log_param(
        "lr0",
        train_p["lr0"]
    )

    mlflow.log_param(
        "lrf",
        train_p["lrf"]
    )

    mlflow.log_param(
        "optimizer",
        train_p["optimizer"]
    )

    # ----------------------------------------------
    # TRAIN
    # ----------------------------------------------

    results = model.train(
        data=train_p["data_yaml"],
        epochs=train_p["epochs"],
        batch=train_p["batch_size"],
        imgsz=train_p["imgsz"],
        lr0=train_p["lr0"],
        lrf=train_p["lrf"],
        optimizer=train_p["optimizer"],
        device=device,
        project="runs",
        name=f"retrain_v{p['data_version']}",
        patience=20,
        plots=False,
        augment=False,
        exist_ok=True,
    )

    # ----------------------------------------------
    # TRAINING DIRECTORY
    # ----------------------------------------------

    run_dir = str(model.trainer.save_dir)

    print(f"\nRun directory: {run_dir}")

    # ----------------------------------------------
    # METRICS
    # ----------------------------------------------

    precision = float(
        results.results_dict.get(
            "metrics/precision(B)",
            0
        )
    )

    recall = float(
        results.results_dict.get(
            "metrics/recall(B)",
            0
        )
    )

    mAP50 = float(
        results.results_dict.get(
            "metrics/mAP50(B)",
            0
        )
    )

    mAP50_95 = float(
        results.results_dict.get(
            "metrics/mAP50-95(B)",
            0
        )
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    metrics = {
        "mAP50": round(mAP50, 4),
        "mAP50_95": round(mAP50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "F1": round(f1, 4),
        "data_version": p["data_version"],
        "device": "gpu" if device == 0 else "cpu",
        "epochs": train_p["epochs"],
    }

    # ----------------------------------------------
    # SAVE METRICS JSON
    # ----------------------------------------------

    with open(
        "metrics/results.json",
        "w"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2
        )

    # ----------------------------------------------
    # LOG METRICS TO MLFLOW
    # ----------------------------------------------

    mlflow.log_metric(
        "mAP50",
        mAP50
    )

    mlflow.log_metric(
        "mAP50_95",
        mAP50_95
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "F1",
        f1
    )

    # ----------------------------------------------
    # LOG TRAINING ARTIFACTS
    # ----------------------------------------------

    mlflow.log_artifacts(run_dir)

    mlflow.log_artifact(
        "metrics/results.json"
    )

    # ----------------------------------------------
    # FIND BEST MODEL
    # ----------------------------------------------

    best_model_path = os.path.join(
        run_dir,
        "weights",
        "best.pt"
    )

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Model not found: {best_model_path}"
        )

    # ----------------------------------------------
    # SAVE VERSIONED MODEL
    # ----------------------------------------------

    version = p["data_version"]

    save_folder = (
        f"model_versions/v{version}"
    )

    os.makedirs(
        save_folder,
        exist_ok=True
    )

    new_model_path = os.path.join(
        save_folder,
        "best.pt"
    )

    shutil.copy(
        best_model_path,
        new_model_path
    )

    # ----------------------------------------------
    # LOG MODEL TO MLFLOW
    # ----------------------------------------------

    mlflow.log_artifact(
        new_model_path
    )

    print(
        f"\n✅ Saved new model to: {new_model_path}"
    )

print("\n✅ Training completed successfully")