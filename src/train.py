import os
import json
import shutil
import yaml
import torch

from ultralytics import YOLO

# Load params
with open("params.yaml") as f:
    p = yaml.safe_load(f)

train_p = p["train"]

os.makedirs("models",  exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# CPU or GPU — works on both
device = 0 if torch.cuda.is_available() else "cpu"
print(f"\n🖥  Running on: {'GPU' if device == 0 else 'CPU'}")

# Train
model   = YOLO(train_p["model"])
results = model.train(
    data      = train_p["data_yaml"],
    epochs    = train_p["epochs"],
    batch     = train_p["batch_size"],
    imgsz     = train_p["imgsz"],
    lr0       = train_p["lr0"],
    lrf       = train_p["lrf"],
    optimizer = train_p["optimizer"],
    device    = device,
    project   = "runs",
    name      = f"retrain_v{p['data_version']}",
    patience  = 20,
    plots     = True,
    exist_ok  = True,
)

# Metrics
precision = float(results.results_dict.get("metrics/precision(B)", 0))
recall    = float(results.results_dict.get("metrics/recall(B)",    0))
mAP50     = float(results.results_dict.get("metrics/mAP50(B)",     0))
mAP50_95  = float(results.results_dict.get("metrics/mAP50-95(B)",  0))
f1 = (2 * precision * recall / (precision + recall)
      if precision + recall > 0 else 0.0)

metrics = {
    "mAP50":        round(mAP50,     4),
    "mAP50_95":     round(mAP50_95,  4),
    "precision":    round(precision,  4),
    "recall":       round(recall,     4),
    "F1":           round(f1,         4),
    "data_version": p["data_version"],
    "device":       "gpu" if device == 0 else "cpu",
    "epochs":       train_p["epochs"],
}

with open("metrics/results.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save model
best_model_path = results.save_dir + "/weights/best.pt"

if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"Model not found at {best_model_path}")

shutil.copy(best_model_path, "model/best.pt")

print("\n✅ Training complete")
for k, v in metrics.items():
    print(f"   {k:<15}: {v}")