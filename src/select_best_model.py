import os
import json
import shutil
import yaml

# ==================================================
# LOAD PARAMS
# ==================================================

with open("params.yaml") as f:
    p = yaml.safe_load(f)

version = p["data_version"]

# ==================================================
# PATHS
# ==================================================

BEST_MODEL = "model/best.pt"

NEW_MODEL = f"model_versions/v{version}/best.pt"

BEST_SCORE_FILE = "model/best_score.txt"

# ==================================================
# LOAD NEW METRICS
# ==================================================

with open("metrics/results.json") as f:
    metrics = json.load(f)

new_map = metrics["mAP50"]

# ==================================================
# LOAD CURRENT BEST SCORE
# ==================================================

if os.path.exists(BEST_SCORE_FILE):

    with open(BEST_SCORE_FILE) as f:
        best_map = float(f.read().strip())

else:
    best_map = 0.74

print(f"\nCurrent BEST mAP50 : {best_map}")
print(f"New model mAP50    : {new_map}")

# ==================================================
# REPLACE ONLY IF BETTER
# ==================================================

if new_map > best_map:

    shutil.copy(NEW_MODEL, BEST_MODEL)
    

    with open(BEST_SCORE_FILE, "w") as f:
        f.write(str(new_map))

    print("\n✅ New model became BEST model")

else:
    print("\n❌ New model rejected")
    
import mlflow

mlflow.set_experiment("Car Damage Detection")

with mlflow.start_run():

    mlflow.log_metric("best_mAP50", new_map)

    mlflow.log_artifact(BEST_MODEL)