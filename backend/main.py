from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from fastapi import HTTPException
from dotenv import load_dotenv
load_dotenv()
from fastapi.responses import RedirectResponse
from mlflow.tracking import MlflowClient
from typing import List
from datetime import datetime

import sqlite3
from pydantic import BaseModel
from passlib.context import CryptContext
import time
import json


import cv2
import os
import uuid
import base64
import subprocess
import sys
import requests


print(os.path.abspath("users.db"))

print("DATABASE PATH:", os.path.abspath("users.db"))
def get_db():
    return sqlite3.connect("users.db", check_same_thread=False)

app = FastAPI(title="Car Damage Detection API")


HISTORY_FILE = "history.json"
def save_history(entry):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f)
# ─────────────────────────────────────────────
# SQLite Database
# ─────────────────────────────────────────────


pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

def init_db():
    conn = sqlite3.connect("users.db", check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
         role TEXT NOT NULL DEFAULT 'user'
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS login_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        login_time TEXT NOT NULL,
        login_date TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS system_stats (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        total_analyses INTEGER DEFAULT 0,
        total_detections INTEGER DEFAULT 0,
        last_inference TEXT
    )
    """)

    # ensure one row exists
    cursor.execute("""
    INSERT OR IGNORE INTO system_stats (id, total_analyses, total_detections, last_inference)
    VALUES (1, 0, 0, NULL)
    """)

    # Dataset queue table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dataset_queue(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        image_path TEXT,
        label_path TEXT,
        num_detections INTEGER,
        infer_ms REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dataset_versions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT NOT NULL,
        username TEXT NOT NULL,
        num_files INTEGER NOT NULL,
        upload_date TEXT NOT NULL,
        upload_time TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS notifications(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        read INTEGER DEFAULT 0
    )
    """)


    conn.commit()
    conn.close()

init_db()
# ─────────────────────────────────────────────
# Notifications helper
# ─────────────────────────────────────────────
def add_notification(message: str):
    conn = sqlite3.connect("users.db", check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO notifications(message) VALUES (?)",
        (message,)
    )

    conn.commit()
    conn.close()
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
model.overrides["verbose"] = False
# ─────────────────────────────────────────────
# Create folders
# ─────────────────────────────────────────────
os.makedirs("data/outputs/images", exist_ok=True)
os.makedirs("data/outputs/labels", exist_ok=True)
os.makedirs("data/outputs/annotated", exist_ok=True)

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "Car Damage Detection API running"}

@app.post("/login")
def login(user: LoginRequest):

    conn = sqlite3.connect("users.db", check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT username, password, role
        FROM users
        WHERE username=?
    """, (user.username,))

    db_user = cursor.fetchone()

    if not db_user:
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid username or password")

    username, hashed_password, role = db_user

    if not pwd_context.verify(user.password, hashed_password):
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Save login history
    from datetime import datetime

    now = datetime.now()

    cursor.execute("""
        INSERT INTO login_history (username, login_time, login_date)
        VALUES (?, ?, ?)
    """, (
        username,
        now.strftime("%H:%M:%S"),
        now.strftime("%Y-%m-%d")
    ))

    conn.commit()
    conn.close()

    return {
        "username": username,
        "role": role
    }

@app.post("/register")
def register(user: RegisterRequest):

    conn = sqlite3.connect("users.db", check_same_thread=False)
    cursor = conn.cursor()

    # check if user exists
    cursor.execute(
        "SELECT * FROM users WHERE username=? OR email=?",
        (user.username, user.email)
    )

    existing = cursor.fetchone()

    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_password = pwd_context.hash(user.password)

    cursor.execute("""
        INSERT INTO users (username, email, password, role)
        VALUES (?, ?, ?, ?)
    """, (
        user.username,
        user.email,
        hashed_password,
        "user"   # 👈 ALL new accounts are normal users
    ))

    conn.commit()
    conn.close()

    return {"message": "Account created successfully"}
@app.post("/upload-dataset")
async def upload_dataset(
    username: str = Form(...),
    version: str = Form(...),
    files: List[UploadFile] = File(...)
):

    version_folder = f"datasets/{version}"

    os.makedirs(version_folder, exist_ok=True)

    count = 0

    for file in files:

        filepath = os.path.join(
            version_folder,
            file.filename
        )

        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())

        count += 1

    now = datetime.now()

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO dataset_versions(
        version,
        username,
        num_files,
        upload_date,
        upload_time
    )
    VALUES (?, ?, ?, ?, ?)
    """, (
        version,
        username,
        count,
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S")
    ))

    conn.commit()
    conn.close()

    return {
        "message": "Dataset uploaded",
        "version": version,
        "files": count
    }
@app.get("/dataset-history")
def dataset_history():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        version,
        username,
        num_files,
        upload_date,
        upload_time
    FROM dataset_versions
    ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    return [
        {
            "version": r[0],
            "username": r[1],
            "num_files": r[2],
            "date": r[3],
            "time": r[4]
        }
        for r in rows
    ]



@app.post("/predict")
async def predict(
    username: str = Form(...),
    file: UploadFile = File(...),
    conf: float = Form(0.25)
):
    
    contents = await file.read()

    file_id = str(uuid.uuid4())

    input_path = f"temp_{file_id}.jpg"
    image_path = f"data/outputs/images/{file_id}.jpg"
    label_path = f"data/outputs/labels/{file_id}.txt"
    annotated_path = f"data/outputs/annotated/{file_id}.jpg"

    # SAVE INPUT IMAGE
    with open(input_path, "wb") as f:
        f.write(contents)

    with open(image_path, "wb") as f:
        f.write(contents)
    
    # ───────── YOLO FIRST (IMPORTANT) ─────────
    results = model.predict(input_path, conf=conf, verbose=False)[0]
   
    annotated = results.plot()
    cv2.imwrite(annotated_path, annotated)
   
    img = cv2.imread(input_path)
    h, w, _ = img.shape

    detections = []
    yolo_lines = []

    for box in results.boxes:
        cls_id = int(box.cls)
        conf_score = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class": model.names[cls_id],
            "confidence": conf_score,
            "bbox": [x1, y1, x2, y2]
        })

        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        yolo_lines.append(f"{cls_id} {x_center} {y_center} {width} {height}")

    # SAVE LABELS
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

    # BASE64 IMAGE
    _, buffer = cv2.imencode(".jpg", annotated)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    # CLEAN TEMP FILE
    os.remove(input_path)

    # AUTO VERSION (optional)
    subprocess.run([
        sys.executable,
        "src/auto_version.py",
        username
    ])
    

    # ───────── SAVE TO DATABASE (FIXED) ─────────
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO dataset_queue (
            username,
            image_path,
            label_path,
            num_detections,
            infer_ms
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        username,
        image_path,
        label_path,
        len(detections),
        round(results.speed["inference"], 1)
    ))
    cursor.execute("""
        UPDATE system_stats
        SET
            total_analyses = total_analyses + 1,
            total_detections = total_detections + ?,
            last_inference = ?
        WHERE id = 1
        """, (
            len(detections),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
    conn.commit()
    conn.close()

    return JSONResponse({
        "image_base64": image_base64,
        "detections": detections,
        "infer_ms": round(results.speed["inference"], 1),
        "conf": conf,
        "num_detections": len(detections),
        "model_version": "YOLOv8s",
        "username": username
    })
@app.get("/system/stats")
def system_stats():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM dataset_queue")
    total_analyses = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(num_detections) FROM dataset_queue")
    total_detections = cursor.fetchone()[0] or 0

    cursor.execute("""
        SELECT created_at
        FROM dataset_queue
        ORDER BY id DESC
        LIMIT 1
    """)
    last = cursor.fetchone()
    
    conn.close()
    # ✅ ADD THIS (important for retrain logic)
    threshold = 100
    can_retrain = total_analyses >= threshold
    return {
        "total_analyses": total_analyses,
        "total_detections": total_detections,
        "last_inference": last[0] if last else None,
        "threshold": threshold,
        "can_retrain": can_retrain
    }
@app.get("/stats")
def get_stats():
    return {
        "total_analyses": 1254,
        "avg_inference": 42.3,
        "datasets": 18,
        "model_version": "v2.1"
    }


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
        "https://api.github.com/repos/azzabennejma/car-damage-project/actions/workflows/ct.yml/dispatches",
        headers=headers,
        json=payload
    )

    print("STATUS:", r.status_code)
    print("RESPONSE:", r.text)

    if r.status_code == 204:
        add_notification(
            " GitHub Actions retraining workflow triggered."
        )
        return {
            "status": r.status_code,
            "message": "Workflow triggered" if r.status_code == 204 else r.text
        }
    if r.status_code == 204:
        add_notification(
            " GitHub Actions retraining workflow triggered!"
        )

        return {
            "status": 204,
            "message": "Workflow triggered"
        }

    else:
        add_notification(
            f"⚠️ Workflow trigger failed (HTTP {r.status_code})"
        )

        return {
            "status": r.status_code,
            "message": r.text
        }
@app.get("/mlflow/metrics")
def get_mlflow_metrics():
    client = MlflowClient()

    runs = client.search_runs(
        experiment_ids=["0"],
        order_by=["start_time desc"],
        max_results=1
    )

    if not runs:
        return {"error": "No runs found"}

    run = runs[0]

    return {
        "run_id": run.info.run_id,
        "metrics": run.data.metrics,
        "params": run.data.params
    }
@app.get("/mlflow/history/{run_id}")
def get_history(run_id: str):
    client = MlflowClient()

    data = client.get_metric_history(run_id, "loss")

    return [
        {"step": m.step, "value": m.value}
        for m in data
    ]
@app.get("/mlflow/latest")
def mlflow_latest():
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        runs = client.search_runs(
            experiment_ids=["0"],
            max_results=1,
            order_by=["start_time desc"]
        )

        if not runs:
            return {
                "status": "no_runs",
                "metrics": {}
            }

        run = runs[0]

        return {
            "status": "ok",
            "metrics": run.data.metrics
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "metrics": {}
        }

@app.post("/outputs/")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"data/outputs/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename}
@app.get("/history")
def get_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
        return data
    except:
        return []
    
@app.get("/users/logins")
def get_logins():
    conn = sqlite3.connect("users.db", check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT username, login_time, login_date
        FROM login_history
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "username": r[0],
            "time": r[1],
            "date": r[2]
        }
        for r in rows
    ]
@app.get("/debug/tables")
def debug_tables():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
    """)

    tables = cursor.fetchall()
    conn.close()

    return tables
@app.get("/test-queue")
def test_queue():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM dataset_queue")

    rows = cursor.fetchall()

    conn.close()

    return rows
@app.get("/dataset-queue")
def get_dataset_queue():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        id,
        username,
        image_path,
        label_path,
        num_detections,
        infer_ms,
        created_at
    FROM dataset_queue
    ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    return [
        {
            "id": r[0],
            "username": r[1],
            "image_path": r[2],
            "label_path": r[3],
            "num_detections": r[4],
            "infer_ms": r[5],
            "created_at": r[6]
        }
        for r in rows
    ]
@app.get("/user/stats/{username}")
def user_stats(username: str):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT COUNT(*)
    FROM dataset_queue
    WHERE username=?
    """, (username,))
    total_analyses = cursor.fetchone()[0]

    cursor.execute("""
    SELECT SUM(num_detections)
    FROM dataset_queue
    WHERE username=?
    """, (username,))
    total_detections = cursor.fetchone()[0] or 0

    cursor.execute("""
    SELECT created_at
    FROM dataset_queue
    WHERE username=?
    ORDER BY id DESC
    LIMIT 1
    """, (username,))

    row = cursor.fetchone()

    last_inference = row[0] if row else 0

    conn.close()
    # 🚨 THRESHOLD LOGIC (IMPORTANT)
    return {
        "total_analyses": total_analyses,
        "total_detections": total_detections,
        "last_inference": last_inference,
        "model": "YOLOv8s v2.1"
    }
@app.get("/notifications")
def get_notifications():

    conn = sqlite3.connect(
        "users.db",
        check_same_thread=False
    )

    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, message, created_at, read
        FROM notifications
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "message": r[1],
            "created_at": r[2],
            "read": r[3]
        }
        for r in rows
    ]

@app.get("/notifications")
def get_notifications():

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM notifications
        WHERE datetime(created_at) <
              datetime('now', '-1 day')
    """)

    conn.commit()

    cursor.execute("""
        SELECT id, message, created_at
        FROM notifications
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    return [
        {
            "id": r[0],
            "message": r[1],
            "created_at": r[2]
        }
        for r in rows
    ]