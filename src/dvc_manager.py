import subprocess
import yaml
import json
from datetime import datetime
from pathlib import Path


# ==================================================
# UTILS
# ==================================================

def run_cmd(cmd: list, description: str = ""):
    """Run a shell command — raises immediately if it fails."""
    print(f"\n>>> {description or ' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr.strip()}")
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n{result.stderr.strip()}"
        )
    if result.stdout.strip():
        print(result.stdout.strip())
    return result


def load_params(params_path: str) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def save_params(params_path: str, params: dict):
    with open(params_path, "w") as f:
        yaml.safe_dump(params, f)


# ==================================================
# DATA VERSIONING
# ==================================================

def update_data_dvc(data_dir: str, params_path: str):
    """
    Version a new dataset with DVC.
    - Increments data_version in params.yaml
    - Tracks data with DVC
    - Creates a git tag → triggers GitHub Actions
    - Pushes data to remote storage
    """
    params   = load_params(params_path)
    data_ver = params.get("data_version", 0)
    new_ver  = data_ver + 1
    current_date = datetime.now().strftime("%Y-%m-%d")
    tag = f"data.v{new_ver}_{current_date}"

    # 1. Track data with DVC
    run_cmd(["dvc", "add", data_dir], f"Tracking {data_dir} with DVC")

    # 2. Update params.yaml
    params["data_version"] = new_ver
    save_params(params_path, params)

    # 3. Stage files
    run_cmd(
        ["git", "add", f"{data_dir}.dvc", params_path],
        "Staging DVC pointer and params.yaml"
    )

    # 4. Commit
    run_cmd(
        ["git", "commit", "-m",
         f"data: version {new_ver} created on {current_date}"],
        "Committing new data version"
    )

    # 5. Create git tag — GitHub Actions listens for this
    run_cmd(
        ["git", "tag", "-a", tag, "-m",
         f"Dataset version {new_ver}"],
        f"Creating tag {tag}"
    )

    # 6. Push data to DVC remote (Azure)
    run_cmd(["dvc", "push"], "Pushing data to Azure remote")

    # 7. Push commit + tag to GitHub
    run_cmd(["git", "push", "origin", "main"], "Pushing commit to GitHub")
    run_cmd(["git", "push", "origin", tag],    "Pushing tag → triggers GitHub Actions")

    print(f"\n✅ Dataset versioned as {tag}")
    print(f"   GitHub Actions will now trigger retraining")
    return tag


# ==================================================
# MODEL VERSIONING
# ==================================================

def update_model_dvc(model_path: str, params_path: str, metrics: dict = None):
    """
    Version a trained model with DVC.
    - Increments model_version in params.yaml
    - Tracks model with DVC
    - Creates a git tag
    - Pushes model to remote storage
    """
    params    = load_params(params_path)
    model_ver = params.get("model_version", 0)
    new_ver   = model_ver + 1
    current_date = datetime.now().strftime("%Y-%m-%d")
    tag = f"model.v{new_ver}_{current_date}"

    # 1. Track model with DVC
    run_cmd(["dvc", "add", model_path], f"Tracking {model_path} with DVC")

    # 2. Update params.yaml
    params["model_version"] = new_ver
    if metrics:
        params["last_metrics"] = metrics
    save_params(params_path, params)

    # 3. Stage files
    run_cmd(
        ["git", "add", f"{model_path}.dvc", params_path],
        "Staging model DVC pointer and params.yaml"
    )

    # 4. Commit
    msg = f"model: version {new_ver} trained on {current_date}"
    if metrics:
        msg += f" | mAP50={metrics.get('mAP50', 'N/A')}"
    run_cmd(["git", "commit", "-m", msg], "Committing new model version")

    # 5. Create git tag
    run_cmd(
        ["git", "tag", "-a", tag, "-m", f"Model version {new_ver}"],
        f"Creating tag {tag}"
    )

    # 6. Push model to DVC remote
    run_cmd(["dvc", "push"], "Pushing model to Azure remote")

    # 7. Push tag to GitHub
    run_cmd(["git", "push", "origin", tag], "Pushing model tag to GitHub")

    print(f"\n✅ Model versioned as {tag}")
    return tag


# ==================================================
# PIPELINE
# ==================================================

def run_pipeline():
    """Trigger the full DVC pipeline: train → evaluate."""
    print("\n>>> Running DVC pipeline...")
    run_cmd(["dvc", "repro"], "Reproducing pipeline: train → evaluate")
    print("\n✅ Pipeline complete.")


# ==================================================
# RESTORE VERSION
# ==================================================

def pull_version(tag: str):
    """Restore a specific dataset or model version by git tag."""
    run_cmd(["git", "checkout", tag], f"Checking out tag {tag}")
    run_cmd(["dvc", "pull"], "Pulling corresponding data/model from remote")
    print(f"\n✅ Restored version: {tag}")


# ==================================================
# STATUS
# ==================================================

def show_status(params_path: str):
    """Print current data and model versions."""
    params = load_params(params_path)
    print("\n📊 Current Status:")
    print(f"   Data version  : {params.get('data_version', 0)}")
    print(f"   Model version : {params.get('model_version', 0)}")
    print(f"   Data yaml     : {params.get('train', {}).get('data_yaml', 'N/A')}")
    if params.get("last_metrics"):
        print(f"\n📈 Last Metrics:")
        for k, v in params["last_metrics"].items():
            print(f"   {k:<15}: {v}")


# ==================================================
# MAIN — for quick testing
# ==================================================

if __name__ == "__main__":
    show_status("params.yaml")