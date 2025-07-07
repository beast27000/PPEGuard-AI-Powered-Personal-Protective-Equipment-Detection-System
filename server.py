import cv2
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from socketio import AsyncServer, ASGIApp
import base64
import asyncio
from PIL import Image
import io
import os
import psycopg
import time
from datetime import datetime, timedelta
import random
import logging
import tempfile
from pydantic import BaseModel
import json
from difflib import SequenceMatcher
from contextlib import asynccontextmanager

from original_code import (
    EnhancedCustomCNN, ModifiedResNet, EfficientNetModel, load_model, transform, CUDATransform,
    PPE_ITEMS, PPE_COLORS, VALID_DEPTS, MAX_FRAMES_TO_PROCESS, DETECTION_CONFIDENCE_THRESHOLD,
    DB_PARAMS, connect_db, init_database, PPEHistory, face_cascade, detect_faces,
    device, BASE_DIR
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    init_database(state["conn"])
    logger.info("Application startup completed")
    yield
    if state["conn"]:
        state["conn"].close()
        logger.info("Database connection closed")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
sio = AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app.mount('/socket.io', ASGIApp(sio))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoginData(BaseModel):
    employee_id: str
    department: str

class SelectData(BaseModel):
    target_class: str
    model: str

state = {
    "user_id": None,
    "department": None,
    "target_class": None,
    "selected_model": None,
    "model": None,
    "num_classes": len(PPE_ITEMS),
    "ppe_history": None,
    "ppe_stats": {"Accepted": 0, "Flagged": 0},
    "cap": None,
    "is_running": False,
    "start_time": None,
    "frame_count": 0,
    "conn": connect_db(),
}

# Dataset paths for Windows C: drive
DATASET_PATHS = {
    "Hardhat": {
        "valid": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\HARDHAT ALONE\\valid",
        "annotations": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\HARDHAT ALONE\\valid\\_annotations.coco.json"
    },
    "NO-Hardhat": {
        "valid": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\HARDHAT ALONE\\valid",
        "annotations": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\HARDHAT ALONE\\valid\\_annotations.coco.json"
    },
    "Gloves": {
        "valid": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\GLOVES ALONE\\valid",
        "annotations": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\GLOVES ALONE\\valid\\_annotations.coco.json"
    },
    "NO-Gloves": {
        "valid": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\GLOVES ALONE\\valid",
        "annotations": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\GLOVES ALONE\\valid\\_annotations.coco.json"
    },
    "Goggles": {
        "valid": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\GOGGLES ALONE\\valid",
        "annotations": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\GOGGLES ALONE\\valid\\_annotations.coco.json"
    },
    "NO-Goggles": {
        "valid": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\GOGGLES ALONE\\valid",
        "annotations": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\GOGGLES ALONE\\valid\\_annotations.coco.json"
    },
    "Safety Vest": {
        "valid": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\SAFETY VEST ALONE\\valid",
        "annotations": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\SAFETY VEST ALONE\\valid\\_annotations.coco.json"
    },
    "NO-Safety Vest": {
        "valid": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\SAFETY VEST ALONE\\valid",
        "annotations": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\SAFETY VEST ALONE\\valid\\_annotations.coco.json"
    },
    "default": {
        "valid": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\44000 IMAGES\\valid",
        "annotations": "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web\\44000 IMAGES\\valid\\_annotations.coco.json"
    }
}

@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")

@app.post("/login")
async def login(data: LoginData):
    employee_id = data.employee_id.strip()
    department = data.department
    if not employee_id:
        logger.warning("Login attempted with empty employee ID")
        raise HTTPException(status_code=400, detail="Employee ID is required")
    if department not in VALID_DEPTS:
        logger.warning(f"Invalid department: {department}")
        raise HTTPException(status_code=400, detail="Invalid department")
    state["user_id"] = employee_id
    state["department"] = department
    try:
        with state["conn"].cursor() as cur:
            cur.execute("SELECT employee_id FROM employees WHERE employee_id = %s", (employee_id,))
            if not cur.fetchone():
                cur.execute("INSERT INTO employees (employee_id, department) VALUES (%s, %s)", (employee_id, department))
                state["conn"].commit()
                logger.info(f"New employee registered: {employee_id}, {department}")
    except psycopg.Error as e:
        logger.error(f"Database error during login: {e}")
        state["conn"].rollback()
        raise HTTPException(status_code=500, detail="Database error")
    return {"message": "Login successful"}

@app.post("/select")
async def select(data: SelectData):
    if data.target_class not in PPE_ITEMS:
        logger.warning(f"Invalid target class: {data.target_class}")
        raise HTTPException(status_code=400, detail="Invalid target class")
    if data.model not in ["CustomCNN", "ResNet18", "EfficientNet"]:
        logger.warning(f"Invalid model: {data.model}")
        raise HTTPException(status_code=400, detail="Invalid model")
    state["target_class"] = data.target_class
    state["selected_model"] = data.model
    try:
        state["model"], state["num_classes"] = load_model(state["selected_model"])
        state["ppe_history"] = PPEHistory(history_size=5)
        logger.info(f"Model selected: {data.model}, Target class: {data.target_class}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    return {"message": "Selection saved"}

@sio.on("start_camera")
async def start_camera(sid):
    if state["is_running"]:
        logger.warning("Camera already running")
        return
    state["cap"] = cv2.VideoCapture(0)
    if not state["cap"].isOpened():
        logger.error("Failed to open camera")
        await sio.emit("error", {"message": "Failed to open camera"}, to=sid)
        return
    state["is_running"] = True
    state["start_time"] = time.time()
    state["frame_count"] = 0
    state["ppe_stats"] = {"Accepted": 0, "Flagged": 0}
    logger.info("Camera started")
    asyncio.create_task(process_frames(sid))

@sio.on("stop_camera")
async def stop_camera(sid):
    state["is_running"] = False
    if state["cap"]:
        state["cap"].release()
        state["cap"] = None
        logger.info("Camera stopped")
    record_session_data()

async def process_frames(sid):
    cuda_transform = CUDATransform(transform, use_edge=True)
    while state["is_running"] and state["cap"]:
        ret, frame = state["cap"].read()
        if not ret:
            logger.error("Failed to capture frame")
            await sio.emit("error", {"message": "Failed to capture frame"}, to=sid)
            break
        frame, stats, status = process_frame(frame, cuda_transform)
        _, buffer = cv2.imencode(".jpg", frame)
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        session_duration = time.time() - state["start_time"]
        hours, remainder = divmod(int(session_duration), 3600)
        minutes, seconds = divmod(remainder, 60)
        session_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        await sio.emit("frame", {
            "frame": frame_b64,
            "stats": stats,
            "sessionTime": session_time,
            "status": status.capitalize()
        }, to=sid)
        state["frame_count"] += 1
        await asyncio.sleep(0.033)

def process_frame(frame, cuda_transform):
    face_rois, face_coords = detect_faces(frame)
    stats = state["ppe_stats"]
    status = "No status detected yet"
    ppe_map = {name: idx for idx, name in enumerate(PPE_ITEMS)}
    target_idx = ppe_map[state["target_class"]]
    for face_roi, (x, y, w, h) in zip(face_rois, face_coords):
        pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        img_tensor = cuda_transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = state["model"](img_tensor)
            probs = torch.sigmoid(logits).squeeze()
            detected_items = [PPE_ITEMS[i] for i in range(len(probs)) if probs[i] >= DETECTION_CONFIDENCE_THRESHOLD]
            confidence = probs[target_idx].item()
        if detected_items or confidence >= DETECTION_CONFIDENCE_THRESHOLD:
            state["ppe_history"].add_ppe(detected_items, confidence)
            status, acceptance_rate, flagged_items = state["ppe_history"].get_status(state["target_class"], state["num_classes"])
            if status == "Accepted":
                stats["Accepted"] += 1
                color = PPE_COLORS[state["target_class"]]
            else:
                stats["Flagged"] += 1
                color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            status_text = f"{status}: {acceptance_rate:.1f}%"
            if flagged_items:
                status_text += f" | Missing: {', '.join(flagged_items)}"
            cv2.putText(frame, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame, stats, status

def record_session_data():
    if not state["start_time"] or state["frame_count"] == 0:
        logger.warning("No session data to record: start_time or frame_count is invalid")
        return
    try:
        end_time = time.time()
        session_duration = end_time - state["start_time"]
        status, acceptance_rate, flagged_items = state["ppe_history"].get_status(state["target_class"], state["num_classes"])
        session_date = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H:%M:%S")
        with state["conn"].cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (employee_id, duration_seconds, status, target_class, session_date, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
                """,
                (state["user_id"], session_duration, status, state["target_class"], session_date, timestamp)
            )
            session_id = cur.fetchone()[0]
            logger.info(f"Session recorded: ID {session_id}")
            item = state["target_class"]
            count = 1 if item in flagged_items else 0
            percentage = 0.0 if count else 100.0
            cur.execute(
                """
                INSERT INTO ppe_details (session_id, ppe_item, count, percentage, flagged)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, item, count, percentage, bool(count))
            )
            logger.info(f"PPE details recorded for session ID {session_id}")
            state["conn"].commit()
    except psycopg.Error as e:
        logger.error(f"Database error in record_session_data: {e}")
        state["conn"].rollback()
        if "relation \"ppe_details\" does not exist" in str(e).lower():
            logger.error("ppe_details table does not exist. Please create it manually.")
    except Exception as e:
        logger.error(f"Unexpected error in record_session_data: {e}")
        state["conn"].rollback()

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    if not file.content_type.startswith('video/'):
        logger.warning("Invalid file type uploaded")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")
    max_file_size = 500 * 1024 * 1024
    if file.size > max_file_size:
        logger.warning(f"File size exceeds limit: {file.size} bytes")
        raise HTTPException(status_code=400, detail="File size exceeds 500 MB limit")
    if not state.get("model") or not state.get("target_class"):
        logger.error("Model or target class not initialized")
        raise HTTPException(status_code=400, detail="Model or target class not initialized")
    
    cuda_transform = CUDATransform(transform, use_edge=True)
    temp_file_path = None
    cap = None
    processed_frames = []
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file_path = temp_file.name
            logger.info(f"Writing video to temporary file: {temp_file_path}")
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
        
        logger.info("Opening video for processing")
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            raise HTTPException(status_code=400, detail="Failed to open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            logger.error("Video has no frames")
            raise HTTPException(status_code=400, detail="Video has no frames")
        
        frames_to_process = min(total_frames, MAX_FRAMES_TO_PROCESS)
        frame_indices = np.linspace(0, total_frames - 1, frames_to_process, dtype=int)
        state["ppe_stats"] = {"Accepted": 0, "Flagged": 0}
        state["ppe_history"] = PPEHistory(history_size=5)
        frame_count = 0
        state["start_time"] = time.time()
        ppe_map = {name: idx for idx, name in enumerate(PPE_ITEMS)}
        target_idx = ppe_map[state["target_class"]]
        logger.info(f"Processing video: {frames_to_process}/{total_frames} frames")
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame at index {frame_idx}")
                continue
            
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = cuda_transform(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, _ = state["model"](img_tensor)
                probs = torch.sigmoid(logits).squeeze()
                detected_items = [PPE_ITEMS[i] for i in range(len(probs)) if probs[i] >= DETECTION_CONFIDENCE_THRESHOLD]
                confidence = probs[target_idx].item()
            
            confidence_scores = {PPE_ITEMS[i]: probs[i].item() for i in range(len(probs))}
            logger.info(f"Frame {frame_idx}: Confidence scores: {confidence_scores}")
            logger.info(f"Frame {frame_idx}: Detected items: {detected_items}, Target confidence: {confidence:.4f}")
            
            state["ppe_history"].add_ppe(detected_items, confidence)
            status, acceptance_rate, flagged_items = state["ppe_history"].get_status(state["target_class"], state["num_classes"])
            logger.info(f"Frame {frame_idx}: Status: {status}, Acceptance rate: {acceptance_rate:.1f}%, Flagged items: {flagged_items}")
            state["ppe_stats"]["Accepted" if status == "Accepted" else "Flagged"] += 1
            
            logger.warning(f"No annotations available for frame {frame_idx}. Consider using a detection model (e.g., YOLO) for real-time bounding boxes.")
            
            status_text = f"Status: {status} ({acceptance_rate:.1f}%)"
            if flagged_items:
                status_text += f" | Missing: {', '.join(flagged_items)}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            _, buffer = cv2.imencode(".jpg", frame)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            processed_frames.append({
                "data": frame_b64,
                "frame_index": int(frame_idx),
                "status": status,
                "acceptance_rate": acceptance_rate,
                "confidence_scores": confidence_scores
            })
            
            frame_count += 1
        
        logger.info(f"Video processing complete: {frame_count} frames processed")
        if frame_count > 0:
            record_session_data()
        else:
            logger.warning("No frames processed; skipping session data recording")
        
        total = sum(state["ppe_stats"].values())
        message = "Video Analysis Results:\n"
        if total > 0:
            for status, count in state["ppe_stats"].items():
                percentage = (count / total) * 100
                message += f"{status}: {percentage:.1f}%\n"
        else:
            message += "No PPE status detected in the video."
        logger.info("Video analysis results prepared")
        
        return {
            "success": True,
            "message": "Video processed successfully",
            "stats": state["ppe_stats"],
            "processed_frames": processed_frames
        }
    
    except Exception as e:
        logger.error(f"Error processing video upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        if cap is not None:
            cap.release()
            logger.info("Video capture released")
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Temporary file removed: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file_path}: {e}")

def map_category_name(target_class, category_names):
    if target_class in category_names:
        return target_class
    target_lower = target_class.lower()
    for cat_name in category_names:
        if cat_name.lower() == target_lower:
            logger.info(f"Mapped target class '{target_class}' to '{cat_name}' (case-insensitive)")
            return cat_name
    best_match = None
    best_ratio = 0
    for cat_name in category_names:
        ratio = SequenceMatcher(None, target_class.lower(), cat_name.lower()).ratio()
        if ratio > best_ratio and ratio > 0.8:
            best_ratio = ratio
            best_match = cat_name
    if best_match:
        logger.info(f"Fuzzy matched target class '{target_class}' to '{best_match}' (similarity: {best_ratio:.2f})")
        return best_match
    logger.warning(f"Target class '{target_class}' not found in annotations")
    return target_class

def find_valid_directory(base_path, target_dir):
    logger.info(f"Searching for valid directory starting at {base_path}")
    base_path = os.path.normpath(base_path)
    target_dir = os.path.normpath(target_dir)
    
    if os.path.exists(target_dir):
        logger.info(f"Found valid directory at {target_dir}")
        return target_dir
    
    parent_dir = os.path.dirname(target_dir)
    target_name = os.path.basename(target_dir).lower()
    if os.path.exists(parent_dir):
        for dir_name in os.listdir(parent_dir):
            if dir_name.lower() == target_name and os.path.isdir(os.path.join(parent_dir, dir_name)):
                found_dir = os.path.join(parent_dir, dir_name)
                logger.info(f"Found case-variant valid directory at {found_dir}")
                return found_dir
    
    current_dir = base_path
    while current_dir != os.path.dirname(current_dir):
        for root, dirs, _ in os.walk(current_dir):
            for dir_name in dirs:
                if dir_name.lower() == target_name.lower():
                    found_dir = os.path.join(root, dir_name)
                    logger.info(f"Found valid directory at {found_dir} via directory walk")
                    return found_dir
        current_dir = os.path.dirname(current_dir)
    
    try:
        root_dir = os.path.dirname(base_path)
        logger.info(f"Listing directories in {root_dir} for debugging:")
        for dir_name in os.listdir(root_dir):
            full_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(full_path):
                logger.info(f" - {full_path}")
    except Exception as e:
        logger.warning(f"Could not list directories in {root_dir}: {e}")
    
    logger.error(f"No valid directory found for {target_dir}")
    return None

@app.get("/test_samples")
async def test_samples():
    try:
        target_class = state.get("target_class", "Hardhat")
        dataset_key = target_class if target_class in DATASET_PATHS else "default"
        valid_dir = DATASET_PATHS[dataset_key]["valid"]
        annotations_path = DATASET_PATHS[dataset_key]["annotations"]
        
        logger.info(f"Accessing valid directory: {valid_dir}")
        logger.info(f"Accessing annotations: {annotations_path}")
        
        base_project_dir = "C:\\PPE Detection Cuda\\FULL_RAPIDS_USE\\CLAUDE\\FINAL_MODEL\\ppe_detection_web"
        valid_dir = find_valid_directory(base_project_dir, valid_dir)
        if not valid_dir or not os.path.exists(valid_dir):
            logger.error(f"Valid directory not found: {valid_dir}")
            raise HTTPException(status_code=400, detail=f"Valid directory not found at {valid_dir}")
        
        if not os.path.exists(annotations_path):
            possible_annotations = os.path.join(os.path.dirname(valid_dir), "_annotations.coco.json")
            if os.path.exists(possible_annotations):
                annotations_path = possible_annotations
                logger.info(f"Found annotations at {annotations_path}")
            else:
                logger.error(f"Annotations file not found: {annotations_path}")
                raise HTTPException(status_code=400, detail=f"Annotations file not found at {annotations_path}")
        
        try:
            with open(annotations_path, 'r') as f:
                coco_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse annotations file: {e}")
            raise HTTPException(status_code=400, detail="Invalid annotations file format")
        
        image_id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}
        category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        category_names = list(category_id_to_name.values())
        logger.info(f"Available categories: {category_names}")
        mapped_target_class = map_category_name(target_class, category_names)
        category_name_to_id = {name: id for id, name in category_id_to_name.items()}
        
        image_id_to_annotations = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in image_id_to_annotations:
                image_id_to_annotations[img_id] = []
            image_id_to_annotations[img_id].append(ann)
        
        target_images = []
        other_images = []
        target_cat_id = category_name_to_id.get(mapped_target_class)
        if not target_cat_id:
            logger.warning(f"Mapped target class '{mapped_target_class}' not found in annotations")
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in image_id_to_file:
                logger.debug(f"Image ID {img_id} not found in images")
                continue
            img_file = image_id_to_file[img_id]
            cat_id = ann["category_id"]
            if cat_id == target_cat_id:
                target_images.append(img_file)
            else:
                other_images.append(img_file)
        
        target_images = list(set(target_images))
        other_images = list(set(other_images))
        logger.info(f"Target images found: {len(target_images)}")
        logger.info(f"Other images found: {len(other_images)}")
        
        num_samples = 10
        target_samples = random.sample(target_images, min(3, len(target_images))) if target_images else []
        remaining_slots = num_samples - len(target_samples)
        other_samples = random.sample(other_images, min(remaining_slots, len(other_images))) if other_images else []
        sample_files = target_samples + other_samples
        
        if not sample_files:
            logger.warning("No images found for sampling")
            raise HTTPException(status_code=400, detail="No images found in valid directory")
        
        logger.info(f"Selected {len(sample_files)} sample images: {sample_files}")
        
        if len(sample_files) < num_samples:
            logger.info(f"Using {len(sample_files)} images instead of {num_samples}")
        
        if not state.get("model"):
            logger.error("Model not initialized")
            raise HTTPException(status_code=500, detail="Model not initialized")

        state["ppe_stats"] = {"Accepted": 0, "Flagged": 0}
        state["ppe_history"] = PPEHistory(history_size=5)
        cuda_transform = CUDATransform(transform, use_edge=True)
        ppe_map = {name: idx for idx, name in enumerate(PPE_ITEMS)}
        target_idx = ppe_map.get(target_class, 0)
        sample_images = []

        for img_file in sample_files:
            try:
                img_path = os.path.join(valid_dir, img_file)
                logger.info(f"Processing image: {img_path}")
                if not os.path.exists(img_path):
                    logger.warning(f"Image not found: {img_path}")
                    continue
                img = Image.open(img_path).convert('RGB')
                img_tensor = cuda_transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = state["model"](img_tensor)
                    probs = torch.sigmoid(logits).squeeze()
                    detected_items = [PPE_ITEMS[i] for i in range(len(probs)) if probs[i] >= DETECTION_CONFIDENCE_THRESHOLD]
                    confidence = probs[target_idx].item()
                
                confidence_scores = {PPE_ITEMS[i]: probs[i].item() for i in range(len(probs))}
                logger.info(f"Image {img_file}: Confidence scores: {confidence_scores}")
                logger.info(f"Image {img_file}: Detected items: {detected_items}, Target confidence: {confidence:.4f}")
                
                state["ppe_history"].add_ppe(detected_items, confidence)
                status, acceptance_rate, flagged_items = state["ppe_history"].get_status(target_class, state.get("num_classes", len(PPE_ITEMS)))
                logger.info(f"Image {img_file}: Status: {status}, Acceptance rate: {acceptance_rate:.1f}%, Flagged items: {flagged_items}")
                state["ppe_stats"]["Accepted" if status == "Accepted" else "Flagged"] += 1
                
                img_array = np.array(img)
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                max_dimension = 800
                h, w = img_cv.shape[:2]
                scale = min(max_dimension / w, max_dimension / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                img_id = None
                for img_data in coco_data["images"]:
                    if img_data["file_name"] == img_file:
                        img_id = img_data["id"]
                        break
                
                if img_id in image_id_to_annotations:
                    for ann in image_id_to_annotations[img_id]:
                        cat_id = ann["category_id"]
                        category_name = category_id_to_name.get(cat_id, "Unknown")
                        bbox = ann["bbox"]
                        x, y, w, h = map(int, [coord * scale for coord in bbox])
                        color = (0, 255, 0) if cat_id == target_cat_id else (0, 0, 255)
                        cv2.rectangle(img_cv, (x, y), (x+w, y+h), color, 2)
                        label = f"{category_name}"
                        cv2.putText(img_cv, label, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                status_text = f"Status: {status} ({acceptance_rate:.1f}%)"
                if flagged_items:
                    status_text += f" | Missing: {', '.join(flagged_items)}"
                cv2.putText(img_cv, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                border_size = 100
                img_cv = cv2.copyMakeBorder(
                    img_cv, border_size, border_size, border_size, border_size,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )
                
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                _, buffer = cv2.imencode(".jpg", img_cv, encode_param)
                img_b64 = base64.b64encode(buffer).decode("utf-8")
                sample_images.append({
                    "data": img_b64,
                    "label": img_file,
                    "status": status,
                    "acceptance_rate": acceptance_rate,
                    "confidence_scores": confidence_scores
                })
            except Exception as e:
                logger.error(f"Error processing image {img_file}: {str(e)}")
                continue

        if not sample_images:
            logger.error("No valid images processed")
            raise HTTPException(status_code=400, detail="No valid images processed")

        total = sum(state["ppe_stats"].values())
        message = "Random Sample Results:\n"
        if total > 0:
            for status, count in state["ppe_stats"].items():
                percentage = (count / total) * 100
                message += f"{status}: {percentage:.1f}%\n"
        else:
            message += "No PPE status detected."
        logger.info(f"Test samples completed: {message}")
        return {
            "success": True,
            "message": message,
            "stats": state["ppe_stats"],
            "sample_images": sample_images
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in test_samples: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process test samples: {str(e)}")

@app.get("/fault_analysis")
async def fault_analysis(department: str = "All", date_range: str = "Last 7 Days"):
    try:
        if department != "All" and department not in VALID_DEPTS:
            logger.warning(f"Invalid department: {department}")
            raise HTTPException(status_code=400, detail="Invalid department")
        if date_range not in ["Last 7 Days", "Last 30 Days"]:
            logger.warning(f"Invalid date range: {date_range}")
            raise HTTPException(status_code=400, detail="Invalid date range")
        
        today = datetime.now().date()
        start_date = today - timedelta(days=7 if date_range == "Last 7 Days" else 30)
        dept_condition = "" if department == "All" else "AND e.department = %s"
        
        if not state.get("conn"):
            logger.error("Database connection not initialized")
            raise HTTPException(status_code=500, detail="Database connection not initialized")
        
        with state["conn"].cursor() as cur:
            query_params = [start_date.strftime("%Y-%m-%d")]
            if dept_condition:
                query_params.append(department)
            cur.execute(
                f"""
                SELECT pd.id, s.employee_id, e.department, pd.ppe_item, s.session_date, s.timestamp, s.duration_seconds
                FROM ppe_details pd
                JOIN sessions s ON pd.session_id = s.id
                JOIN employees e ON s.employee_id = e.employee_id
                WHERE pd.flagged = TRUE AND s.session_date >= %s
                {dept_condition}
                ORDER BY s.session_date DESC, s.timestamp DESC
                """,
                query_params
            )
            violations = cur.fetchall()
            logger.info(f"Fault analysis fetched: {len(violations)} violations")
            return {
                "violations": [
                    {
                        "id": v[0],
                        "employee_id": v[1],
                        "department": v[2],
                        "ppe_item": v[3],
                        "session_date": v[4].strftime("%Y-%m-%d"),
                        "timestamp": v[5].strftime("%H:%M:%S"),
                        "duration_seconds": float(v[6])
                    } for v in violations
                ]
            }
    except psycopg.Error as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/admin_stats")
async def admin_stats(department: str = "All", date_range: str = "Last 7 Days"):
    if department != "All" and department not in VALID_DEPTS:
        logger.warning(f"Invalid department for admin stats: {department}")
        raise HTTPException(status_code=400, detail="Invalid department")
    if date_range not in ["Last 7 Days", "Last 30 Days", "All Time"]:
        logger.warning(f"Invalid date range for admin stats: {date_range}")
        raise HTTPException(status_code=400, detail="Invalid date range")
    today = datetime.now().date()
    if date_range == "Last 7 Days":
        start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    elif date_range == "Last 30 Days":
        start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    else:
        start_date = "2000-01-01"
    dept_condition = "" if department == "All" else f"AND e.department = %s"
    try:
        with state["conn"].cursor() as cur:
            query_params = [start_date]
            if dept_condition:
                query_params.append(department)
            cur.execute(
                f"""
                SELECT COALESCE(s.status, 'No Data') AS status, COALESCE(COUNT(*), 0) AS total
                FROM sessions s
                JOIN employees e ON s.employee_id = e.employee_id
                WHERE s.session_date >= %s
                {dept_condition}
                GROUP BY s.status
                """,
                query_params
            )
            status_stats = cur.fetchall()
            cur.execute(
                f"""
                SELECT e.employee_id, e.department, COALESCE(COUNT(s.id), 0) AS session_count,
                       COALESCE(AVG(CASE WHEN s.status = 'Accepted' THEN 100.0 ELSE 0.0 END), 0.0) AS acceptance_rate,
                       COALESCE(STRING_AGG(pd.ppe_item, ', '), 'None') AS flagged_items
                FROM employees e
                LEFT JOIN sessions s ON e.employee_id = s.employee_id AND s.session_date >= %s
                LEFT JOIN ppe_details pd ON s.id = pd.session_id AND pd.flagged = TRUE
                {dept_condition}
                GROUP BY e.employee_id, e.department
                ORDER BY session_count DESC
                """,
                query_params
            )
            employee_stats = cur.fetchall()
        logger.info(f"Admin stats fetched: {len(status_stats)} status records, {len(employee_stats)} employee records")
        return {
            "compliance": [{"status": s[0], "total": s[1]} for s in status_stats] or [{"status": "No Data", "total": 0}],
            "employees": [
                {
                    "employee_id": e[0],
                    "department": e[1],
                    "session_count": e[2],
                    "acceptance_rate": float(e[3]),
                    "flagged_items": e[4]
                } for e in employee_stats
            ]
        }
    except psycopg.Error as e:
        logger.error(f"Database query error in admin_stats: {e}")
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in admin_stats: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/faults")
async def get_faults(employee_id: str, date_range: str = "Last 7 Days"):
    try:
        if date_range not in ["Last 7 Days", "Last 30 Days"]:
            raise HTTPException(status_code=400, detail="Invalid date range")
        
        today = datetime.now().date()
        start_date = today - timedelta(days=7 if date_range == "Last 7 Days" else 30)
        
        with state["conn"].cursor() as cur:
            cur.execute(
                """
                SELECT pd.ppe_item, s.session_date, s.timestamp, s.duration_seconds
                FROM ppe_details pd
                JOIN sessions s ON pd.session_id = s.id
                JOIN employees e ON s.employee_id = e.employee_id
                WHERE s.employee_id = %s AND s.session_date >= %s AND pd.flagged = TRUE
                ORDER BY s.session_date DESC, s.timestamp DESC
                """,
                (employee_id, start_date.strftime("%Y-%m-%d"))
            )
            faults = cur.fetchall()
            return {
                "success": True,
                "employee_id": employee_id,
                "faults": [
                    {
                        "ppe_item": f[0],
                        "session_date": f[1].strftime("%Y-%m-%d"),
                        "timestamp": f[2].strftime("%H:%M:%S"),
                        "duration_seconds": float(f[3])
                    } for f in faults
                ]
            }
    except psycopg.Error as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)