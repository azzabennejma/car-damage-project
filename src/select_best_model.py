import os
import json
import shutil
import yaml

# ==================================================
# LOAD PARAMS
# ==================================================
with open("params.yaml") as f:
    p = yaml.safe_load(f)

version = p["data"]["version"]

# ==================================================
# PATHS
# ==================================================
PRODUCTION_MODEL = "model/best.pt"        # static, never deleted
NEW_MODEL        = f"model_versions/{version}/best.pt"
CANDIDATE_MODEL  = f"model_versions/{version}/candidate.pt"  # new model saved here
BEST_SCORE_FILE  = "model/best_score.txt"

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
    best_map = 0.73  # your Kaggle model's mAP

print(f"\nProduction model mAP50 : {best_map}")
print(f"New candidate mAP50    : {new_map}")

# ==================================================
# SAVE NEW MODEL AS CANDIDATE (always)
# ==================================================
os.makedirs(f"model_versions/{version}", exist_ok=True)
shutil.copy(NEW_MODEL, CANDIDATE_MODEL)
print(f"\n📦 New model saved as candidate: {CANDIDATE_MODEL}")

# ==================================================
# REPLACE PRODUCTION ONLY IF BETTER
# ==================================================
if new_map > best_map:
    # Backup current production model before replacing
    backup_path = f"model_versions/production_backup/best.pt"
    os.makedirs("model_versions/production_backup", exist_ok=True)
    shutil.copy(PRODUCTION_MODEL, backup_path)
    
    # Replace production model
    shutil.copy(NEW_MODEL, PRODUCTION_MODEL)

    with open(BEST_SCORE_FILE, "w") as f:
        f.write(str(new_map))

    print(f"\n✅ New model promoted to production (mAP50: {new_map:.4f})")
    print(f"   Previous model backed up to: {backup_path}")

else:
    print(f"\n❌ New model rejected — production model unchanged")
    print(f"   Candidate saved at: {CANDIDATE_MODEL}")
    
# Write promotion result for GitHub Actions
promoted = new_map > best_map
with open("model/promoted.txt", "w") as f:
    f.write("true" if promoted else "false")
# ==================================================
# LOG TO MLFLOW
# ==================================================
import mlflow

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("car-damage-detection")

with mlflow.start_run():
    mlflow.log_metric("new_mAP50", new_map)
    mlflow.log_metric("production_mAP50", best_map)
    mlflow.log_param("version", version)
    mlflow.log_param("promoted", new_map > best_map)
    mlflow.log_artifact(CANDIDATE_MODEL, artifact_path=f"candidate_{version}")
    if new_map > best_map:
        mlflow.log_artifact(PRODUCTION_MODEL, artifact_path="production_model")

print("\n✅ Results logged to MLflow")