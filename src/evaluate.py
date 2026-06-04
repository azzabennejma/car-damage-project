import json
import yaml

from ultralytics import YOLO

with open("params.yaml") as f:
    p = yaml.safe_load(f)

version = p["data_version"]

candidate_model = f"model_versions/v{version}/best.pt"

print(f"Evaluating: {candidate_model}")

model = YOLO(candidate_model)

results = model.val(
    data=p["train"]["data_yaml"],
    verbose=False
)

metrics = {
    "mAP50": round(
        float(
            results.results_dict.get(
                "metrics/mAP50(B)",
                0
            )
        ),
        4
    ),
    "mAP50_95": round(
        float(
            results.results_dict.get(
                "metrics/mAP50-95(B)",
                0
            )
        ),
        4
    ),
}

with open("metrics/eval.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Evaluation complete")
print(metrics)