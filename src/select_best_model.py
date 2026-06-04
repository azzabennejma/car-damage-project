import os
import json
import yaml
import mlflow

# ==================================================
# LOAD PARAMS
# ==================================================

with open("params.yaml") as f:
    p = yaml.safe_load(f)

version = p["data_version"]

# ==================================================
# FILES
# ==================================================

BEST_SCORE_FILE = "model/best_score.txt"

CANDIDATE_MODEL = (
    f"model_versions/v{version}/best.pt"
)

PROMOTED_FILE = "model/promoted.txt"

# ==================================================
# LOAD CANDIDATE SCORE
# ==================================================

with open("metrics/eval.json") as f:
    metrics = json.load(f)

new_map = metrics["mAP50"]

# ==================================================
# LOAD CURRENT PRODUCTION SCORE
# ==================================================

if os.path.exists(BEST_SCORE_FILE):

    with open(BEST_SCORE_FILE) as f:
        best_map = float(f.read().strip())

else:

    best_map = 0.73

print(f"Production mAP50 : {best_map}")
print(f"Candidate  mAP50 : {new_map}")

# ==================================================
# DECISION
# ==================================================

promoted = new_map > best_map

os.makedirs("model", exist_ok=True)

with open(PROMOTED_FILE, "w") as f:
    f.write(
        "true"
        if promoted
        else "false"
    )

if promoted:

    with open(BEST_SCORE_FILE, "w") as f:
        f.write(str(new_map))

    print(
        "\nCandidate accepted"
    )

else:

    print(
        "\nCandidate rejected"
    )

# ==================================================
# MLFLOW
# ==================================================

mlflow.set_tracking_uri(
    os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )
)

mlflow.set_experiment(
    "car-damage-detection"
)

with mlflow.start_run():

    mlflow.log_param(
        "version",
        version
    )

    mlflow.log_param(
        "promoted",
        promoted
    )

    mlflow.log_metric(
        "candidate_mAP50",
        new_map
    )

    mlflow.log_metric(
        "production_mAP50",
        best_map
    )

    mlflow.log_artifact(
        CANDIDATE_MODEL,
        artifact_path="candidate_model"
    )

print("\nSelection completed")