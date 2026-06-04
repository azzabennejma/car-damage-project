from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from fastapi import HTTPException
from dotenv import load_dotenv
load_dotenv()

import cv2
import os
import uuid
import base64
import subprocess
import sys
import requests

app = FastAPI(title="Car Damage Detection API")

# ─────────────────────────────────────────────
# CORS (React frontend)
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
model = YOLO("model/best.pt")

# ─────────────────────────────────────────────
# Create folders
# ─────────────────────────────────────────────
os.makedirs("data/outputs/images", exist_ok=True)
os.makedirs("data/outputs/labels", exist_ok=True)
os.makedirs("data/outputs/annotated", exist_ok=True)

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "Car Damage Detection API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25):

    contents = await file.read()

    # Unique ID for this sample
    file_id = str(uuid.uuid4())

    # Paths
    input_path = f"temp_{file_id}.jpg"
    image_path = f"data/outputs/images/{file_id}.jpg"
    label_path = f"data/outputs/labels/{file_id}.txt"
    annotated_path = f"data/outputs/annotated/{file_id}.jpg"

    # Save temp + dataset image
    with open(input_path, "wb") as f:
        f.write(contents)

    with open(image_path, "wb") as f:
        f.write(contents)

    # ───────── YOLO inference ─────────
    results = model(input_path, conf=conf)[0]

    # Annotated image
    annotated = results.plot()
    cv2.imwrite(annotated_path, annotated)

    # Image size
    img = cv2.imread(input_path)
    h, w, _ = img.shape

    detections = []
    yolo_lines = []

    for box in results.boxes:
        cls_id = int(box.cls)
        conf_score = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # JSON detection (for frontend)
        detections.append({
            "class": model.names[cls_id],
            "confidence": conf_score,
            "bbox": [x1, y1, x2, y2]
        })

        # YOLO format (normalized)
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        yolo_lines.append(f"{cls_id} {x_center} {y_center} {width} {height}")

    # Save YOLO label file
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))
    

    # Convert annotated image → base64
    _, buffer = cv2.imencode(".jpg", annotated)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    # Cleanup temp file
    os.remove(input_path)
    subprocess.Popen([sys.executable, "src/auto_version.py"])
    # ───────── RESPONSE ─────────
    return JSONResponse({
        "image_base64": image_base64,
        "detections": detections,
        "infer_ms": round(results.speed["inference"], 1),
        "conf": conf
    })

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/retrain")
def retrain():
    token = os.getenv("GITHUB_TOKEN")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    payload = {
        "ref": "main"
    }

    r = requests.post(
        "https://api.github.com/repos/azzabennejma/car-damage-project/actions/workflows/train.yml/dispatches",
        headers=headers,
        json=payload
    )

    print("STATUS:", r.status_code)
    print("RESPONSE:", r.text)

    return {
        "status": r.status_code,
        "message": "Workflow triggered" if r.status_code == 204 else r.text
    }
@app.post("/outputs/")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"data/outputs/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename}