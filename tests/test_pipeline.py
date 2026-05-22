import yaml
import os
import json

def test_params_exists():
    assert os.path.exists("params.yaml")

def test_params_structure():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)
    assert "train"   in p
    assert "epochs"  in p["train"]
    assert "data_yaml" in p["train"]

def test_requirements_exists():
    assert os.path.exists("backend/requirements.txt")

def test_dvc_yaml_exists():
    assert os.path.exists("dvc.yaml")

def test_src_scripts_exist():
    for script in ["train.py", "evaluate.py", "auto_version.py"]:
        assert os.path.exists(f"src/{script}"), f"Missing: src/{script}"

