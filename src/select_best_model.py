import os
import json
import shutil

# Current production model
BEST_MODEL = "model/best.pt"

# Newly trained model
with open("params.yaml") as f:
    import yaml
    p = yaml.safe_load(f)

NEW_MODEL = f"model/model_v{p['data_version']}.pt"

# Load metrics from training
with open("metrics/results.json") as f:
    metrics = json.load(f)

new_map = metrics["mAP50"]

# Current best score file
BEST_SCORE_FILE = "model/best_score.txt"

if os.path.exists(BEST_SCORE_FILE):
    with open(BEST_SCORE_FILE) as f:
        best_map = float(f.read().strip())
else:
    best_map = 0.0

print(f"\nCurrent BEST mAP50 : {best_map}")
print(f"New model mAP50    : {new_map}")

# Replace only if better
if new_map > best_map:

    shutil.copy(NEW_MODEL, BEST_MODEL)

    with open(BEST_SCORE_FILE, "w") as f:
        f.write(str(new_map))

    print("\n✅ New model became BEST model")

else:
    print("\n❌ New model rejected")