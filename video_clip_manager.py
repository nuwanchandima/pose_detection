"""
Video Clip Generator with Background Task Management
- Upload videos and process in background
- View task list and status
- Browse clips for each task
"""

from pathlib import Path
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
import subprocess
import shutil
import time
import uuid
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from insightface.app import FaceAnalysis
from app2 import llm_call

# Directories
UPLOAD_DIR = Path("uploads")
CLIPS_DIR = Path("clips")
ALL_CLIPS_DIR = Path("all_clips")  # Shared folder for all clips
TEMP_DIR = Path("temp")
TASKS_DIR = Path("tasks")

for dir_path in [UPLOAD_DIR, CLIPS_DIR, ALL_CLIPS_DIR, TEMP_DIR, TASKS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Face recognition model
face_app = None

# Task storage (in production, use database)
tasks: Dict[str, dict] = {}

# Configuration
SIMILARITY_THRESHOLD = 0.50  # Faces above this are considered "same person"

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=2)


def save_task_state(task_id: str):
    """Save task state to disk."""
    task_file = TASKS_DIR / f"{task_id}.json"
    with open(task_file, "w") as f:
        json.dump(tasks[task_id], f, indent=2)


def load_all_tasks():
    """Load all tasks from disk on startup."""
    global tasks
    for task_file in TASKS_DIR.glob("*.json"):
        try:
            with open(task_file, "r") as f:
                task_data = json.load(f)
                tasks[task_data["task_id"]] = task_data
        except Exception as e:
            print(f"[Error] Failed to load task {task_file}: {e}")


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two face embeddings."""
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1_norm, emb2_norm))


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract audio from video using ffmpeg."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "aac", "-b:a", "128k",
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0 and audio_path.exists()
    except Exception as e:
        print(f"[Error] Audio extraction failed: {e}")
        return False


def create_clip_with_audio(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    output_path: Path,
    audio_path: Optional[Path] = None
) -> bool:
    """Create video clip from frame range with audio."""
    try:
        start_time = start_frame / fps
        duration = (end_frame - start_frame + 1) / fps
        
        if audio_path and audio_path.exists():
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", str(video_path),
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", str(audio_path),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                str(output_path)
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", str(video_path),
                "-c:v", "libx264",
                str(output_path)
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0 and output_path.exists()
    except Exception as e:
        print(f"[Error] Clip creation failed: {e}")
        return False


def process_video_task(task_id: str, video_path: Path):
    """Process video in background and update task status."""
    global face_app, tasks
    
    try:
        # Update task status
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = "Starting..."
        save_task_state(task_id)
        
        print(f"[Task {task_id}] Starting video: {video_path}")
        
        # Extract audio
        audio_path = TEMP_DIR / f"{task_id}_audio.aac"
        has_audio = extract_audio(video_path, audio_path)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception("Cannot open video")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[Task {task_id}] FPS: {fps}, Total frames: {total_frames}")
        
        # Create task clips directory
        task_clips_dir = CLIPS_DIR / task_id
        task_clips_dir.mkdir(exist_ok=True)
        
        # Processing state
        current_clip_start = None
        current_clip_embedding = None
        clips_info = []
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = face_app.get(frame)
            
            # Only process frames with exactly 1 face
            if len(faces) == 1:
                face = faces[0]
                face_embedding = face.normed_embedding
                
                if current_clip_start is None:
                    current_clip_start = frame_idx
                    current_clip_embedding = face_embedding
                else:
                    similarity = compute_similarity(face_embedding, current_clip_embedding)
                    
                    if similarity < SIMILARITY_THRESHOLD:
                        # Face changed! Save previous clip
                        clip_length = frame_idx - current_clip_start
                        clip_duration = clip_length / fps
                        
                        # Skip clips shorter than 1 second
                        if clip_duration < 1.0:
                            print(f"[Task {task_id}] Skipping short clip: {clip_duration:.2f}s")
                            current_clip_start = frame_idx
                            current_clip_embedding = face_embedding
                            continue
                        
                        clip_num = len(clips_info) + 1
                        clip_filename = f"clip_{clip_num:03d}.mp4"
                        clip_path = task_clips_dir / clip_filename
                        
                        # Create unique filename for all_clips folder
                        all_clips_filename = f"{task_id}_{clip_filename}"
                        all_clips_path = ALL_CLIPS_DIR / all_clips_filename
                        
                        if create_clip_with_audio(video_path, current_clip_start, frame_idx - 1, 
                                                 fps, clip_path, audio_path if has_audio else None):
                            # Copy to all_clips folder
                            shutil.copy2(clip_path, all_clips_path)
                            
                            clips_info.append({
                                "clip_name": clip_filename,
                                "clip_path": f"clips/{task_id}/{clip_filename}",
                                "start_frame": current_clip_start,
                                "end_frame": frame_idx - 1,
                                "frames": clip_length,
                                "duration": f"{clip_duration:.2f}s"
                            })
                        
                        current_clip_start = frame_idx
                        current_clip_embedding = face_embedding
            else:
                # No face or multiple faces
                if current_clip_start is not None:
                    clip_length = frame_idx - current_clip_start
                    clip_duration = clip_length / fps
                    
                    # Skip clips shorter than 1 second
                    if clip_duration < 1.0:
                        print(f"[Task {task_id}] Skipping short clip: {clip_duration:.2f}s")
                        current_clip_start = None
                        current_clip_embedding = None
                        continue
                    
                    clip_num = len(clips_info) + 1
                    clip_filename = f"clip_{clip_num:03d}.mp4"
                    clip_path = task_clips_dir / clip_filename
                    
                    # Create unique filename for all_clips folder
                    all_clips_filename = f"{task_id}_{clip_filename}"
                    all_clips_path = ALL_CLIPS_DIR / all_clips_filename
                    
                    if create_clip_with_audio(video_path, current_clip_start, frame_idx - 1,
                                             fps, clip_path, audio_path if has_audio else None):
                        # Copy to all_clips folder
                        shutil.copy2(clip_path, all_clips_path)
                        
                        clips_info.append({
                            "clip_name": clip_filename,
                            "clip_path": f"clips/{task_id}/{clip_filename}",
                            "start_frame": current_clip_start,
                            "end_frame": frame_idx - 1,
                            "frames": clip_length,
                            "duration": f"{clip_duration:.2f}s"
                        })
                    
                    current_clip_start = None
                    current_clip_embedding = None
            
            frame_idx += 1
            
            # Update progress
            if frame_idx % 100 == 0:
                progress = f"Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.0f}%)"
                tasks[task_id]["progress"] = progress
                tasks[task_id]["clips_count"] = len(clips_info)
                save_task_state(task_id)
        
        # Handle last clip
        if current_clip_start is not None:
            clip_length = frame_idx - current_clip_start
            clip_duration = clip_length / fps
            
            # Only save if duration is at least 1 second
            if clip_duration >= 1.0:
                clip_num = len(clips_info) + 1
                clip_filename = f"clip_{clip_num:03d}.mp4"
                clip_path = task_clips_dir / clip_filename
                
                # Create unique filename for all_clips folder
                all_clips_filename = f"{task_id}_{clip_filename}"
                all_clips_path = ALL_CLIPS_DIR / all_clips_filename
                
                if create_clip_with_audio(video_path, current_clip_start, frame_idx - 1,
                                         fps, clip_path, audio_path if has_audio else None):
                    # Copy to all_clips folder
                    shutil.copy2(clip_path, all_clips_path)
                    
                    clips_info.append({
                        "clip_name": clip_filename,
                        "clip_path": f"clips/{task_id}/{clip_filename}",
                        "start_frame": current_clip_start,
                        "end_frame": frame_idx - 1,
                        "frames": clip_length,
                        "duration": f"{clip_duration:.2f}s"
                    })
            else:
                print(f"[Task {task_id}] Skipping short last clip: {clip_duration:.2f}s")
        
        cap.release()
        
        # Cleanup temp audio
        if audio_path.exists():
            audio_path.unlink()
        
        # Update task as completed
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = "Complete"
        tasks[task_id]["clips_count"] = len(clips_info)
        tasks[task_id]["clips"] = clips_info
        tasks[task_id]["completed_at"] = datetime.now().isoformat()
        save_task_state(task_id)
        
        print(f"[Task {task_id}] Complete - Created {len(clips_info)} clips")
        
    except Exception as e:
        print(f"[Task {task_id}] Error: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["progress"] = f"Error: {str(e)}"
        tasks[task_id]["error"] = str(e)
        save_task_state(task_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load face recognition model and tasks on startup."""
    global face_app
    print("[Startup] Loading InsightFace model on GPU...")
    face_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("[Startup] Model loaded successfully on GPU")
    
    # Load existing tasks
    print("[Startup] Loading existing tasks...")
    load_all_tasks()
    print(f"[Startup] Loaded {len(tasks)} tasks")
    
    yield
    
    print("[Shutdown] Cleaning up...")
    executor.shutdown(wait=False)


# FastAPI app
app = FastAPI(lifespan=lifespan)

# Mount static directories
app.mount("/clips", StaticFiles(directory=CLIPS_DIR), name="clips")
app.mount("/all_clips", StaticFiles(directory=ALL_CLIPS_DIR), name="all_clips")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page with task list."""
    return templates.TemplateResponse("task_manager.html", {"request": request})


@app.post("/upload_video")
async def upload_video(video: UploadFile = File(...)):
    """Upload video and create background task."""
    try:
        # Create task
        task_id = str(uuid.uuid4())
        video_filename = f"{task_id}_{video.filename}"
        video_path = UPLOAD_DIR / video_filename
        
        # Save uploaded video
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        
        # Create task record
        tasks[task_id] = {
            "task_id": task_id,
            "video_name": video.filename,
            "video_path": str(video_path),
            "status": "queued",
            "progress": "Waiting to start...",
            "clips_count": 0,
            "clips": [],
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        }
        save_task_state(task_id)
        
        # Start background processing
        executor.submit(process_video_task, task_id, video_path)
        
        print(f"[Upload] Created task {task_id} for video: {video.filename}")
        
        return JSONResponse({
            "success": True,
            "message": "Video uploaded successfully. Processing started in background.",
            "task_id": task_id
        })
    
    except Exception as e:
        print(f"[Error] {e}")
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)


@app.get("/tasks")
async def get_tasks():
    """Get all tasks."""
    # Sort by created_at (newest first)
    sorted_tasks = sorted(tasks.values(), key=lambda x: x["created_at"], reverse=True)
    return {"tasks": sorted_tasks}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get specific task details."""
    if task_id not in tasks:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return {"task": tasks[task_id]}


@app.get("/tasks/{task_id}/clips", response_class=HTMLResponse)
async def view_task_clips(request: Request, task_id: str):
    """View clips for a specific task."""
    if task_id not in tasks:
        return HTMLResponse("<h1>Task not found</h1>", status_code=404)
    
    return templates.TemplateResponse("task_clips.html", {
        "request": request,
        "task": tasks[task_id]
    })


@app.get("/all-clips", response_class=HTMLResponse)
async def view_all_clips(request: Request):
    """View all clips from all tasks in one page."""
    all_clips = []
    
    # Get all clip files from all_clips directory
    if ALL_CLIPS_DIR.exists():
        for clip_file in sorted(ALL_CLIPS_DIR.glob("*.mp4")):
            # Parse filename: task_id_clip_name.mp4
            filename = clip_file.name
            parts = filename.split("_", 1)
            
            if len(parts) == 2:
                task_id = parts[0]
                clip_name = parts[1]
                
                # Get task info if available
                task_name = "Unknown"
                if task_id in tasks:
                    task_name = tasks[task_id].get("video_name", "Unknown")
                
                all_clips.append({
                    "filename": filename,
                    "clip_path": f"all_clips/{filename}",
                    "task_id": task_id,
                    "task_name": task_name,
                    "clip_name": clip_name
                })
    
    return templates.TemplateResponse("all_clips.html", {
        "request": request,
        "clips": all_clips,
        "total_clips": len(all_clips)
    })


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and all its clips."""
    try:
        if task_id not in tasks:
            return JSONResponse({"error": "Task not found"}, status_code=404)
        
        task = tasks[task_id]
        
        # Delete video file
        video_path = Path(task["video_path"])
        if video_path.exists():
            video_path.unlink()
        
        # Delete clips directory
        task_clips_dir = CLIPS_DIR / task_id
        if task_clips_dir.exists():
            shutil.rmtree(task_clips_dir)
        
        # Delete task state file
        task_file = TASKS_DIR / f"{task_id}.json"
        if task_file.exists():
            task_file.unlink()
        
        # Remove from memory
        del tasks[task_id]
        
        print(f"[Delete] Task {task_id} deleted successfully")
        
        return JSONResponse({
            "success": True,
            "message": "Task and all clips deleted successfully"
        })
    
    except Exception as e:
        print(f"[Error] Delete task failed: {e}")
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)


@app.post("/tasks/{task_id}/clips/{clip_name}/process-llm")
async def process_clip_with_llm(task_id: str, clip_name: str):
    """Process a clip with LLM using Gemini API."""
    try:
        if task_id not in tasks:
            return JSONResponse({"error": "Task not found"}, status_code=404)
        
        task = tasks[task_id]
        
        # Find clip
        clip_info = None
        for clip in task["clips"]:
            if clip["clip_name"] == clip_name:
                clip_info = clip
                break
        
        if not clip_info:
            return JSONResponse({"error": "Clip not found"}, status_code=404)
        
        # Get the clip file path
        clip_path = CLIPS_DIR / task_id / clip_name
        if not clip_path.exists():
            return JSONResponse({"error": "Clip file not found"}, status_code=404)
        
        print(f"[LLM] Processing clip: {clip_path}")
        
        # Call the LLM function from app2.py
        llm_result = llm_call(str(clip_path))
        
        # Check if there was an error
        if isinstance(llm_result, dict) and llm_result.get("error") == True:
            return JSONResponse({
                "success": False,
                "task_id": task_id,
                "clip_name": clip_name,
                "error": llm_result.get("message", "Unknown error occurred"),
                "processed_at": datetime.now().isoformat()
            })
        
        # Success - return the LLM result with additional metadata
        response = {
            "success": True,
            "task_id": task_id,
            "clip_name": clip_name,
            "clip_info": {
                "duration": clip_info.get("duration", "unknown"),
                "frame_count": clip_info.get("frames", 0),
                "start_frame": clip_info.get("start_frame", 0),
                "end_frame": clip_info.get("end_frame", 0)
            },
            "llm_analysis": llm_result,
            "processed_at": datetime.now().isoformat()
        }
        
        return JSONResponse(response)
        
    except Exception as e:
        print(f"[Error] LLM processing failed: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "task_id": task_id,
            "clip_name": clip_name,
            "error": str(e),
            "processed_at": datetime.now().isoformat()
        }, status_code=500)


@app.delete("/tasks/{task_id}/clips/{clip_name}")
async def delete_clip(task_id: str, clip_name: str):
    """Delete a specific clip from a task."""
    try:
        if task_id not in tasks:
            return JSONResponse({"error": "Task not found"}, status_code=404)
        
        task = tasks[task_id]
        
        # Find and remove clip from task clips list
        clip_found = False
        for i, clip in enumerate(task["clips"]):
            if clip["clip_name"] == clip_name:
                clip_found = True
                task["clips"].pop(i)
                break
        
        if not clip_found:
            return JSONResponse({"error": "Clip not found"}, status_code=404)
        
        # Delete clip file
        clip_path = CLIPS_DIR / task_id / clip_name
        if clip_path.exists():
            clip_path.unlink()
        
        # Update clips count
        task["clips_count"] = len(task["clips"])
        
        # Save updated task state
        save_task_state(task_id)
        
        print(f"[Delete] Clip {clip_name} from task {task_id} deleted successfully")
        
        return JSONResponse({
            "success": True,
            "message": "Clip deleted successfully",
            "clips_count": task["clips_count"]
        })
    
    except Exception as e:
        print(f"[Error] Delete clip failed: {e}")
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("video_clip_manager:app", host="0.0.0.0", port=5000, reload=True)
