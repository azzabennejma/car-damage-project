import json
import yaml

from ultralytics import YOLO

with open("params.yaml") as f:
    p = yaml.safe_load(f)

model       = YOLO("models/best.pt")
val_results = model.val(data=p["train"]["data_yaml"], verbose=False)

metrics = {
    "mAP50":    round(float(val_results.results_dict.get("metrics/mAP50(B)",    0)), 4),
    "mAP50_95": round(float(val_results.results_dict.get("metrics/mAP50-95(B)", 0)), 4),
}

with open("metrics/eval.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Evaluation complete:", metrics)