"""
Simple video clip generator based on face changes.
Upload video -> Detect face changes -> Create clips with audio
NO minimum clip length - saves at EVERY face change point
"""

from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
import subprocess
import shutil

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from insightface.app import FaceAnalysis

# Directories
UPLOAD_DIR = Path("uploads")
CLIPS_DIR = Path("clips")
TEMP_DIR = Path("temp")

for dir_path in [UPLOAD_DIR, CLIPS_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# Face recognition model
face_app = None

# Configuration
SIMILARITY_THRESHOLD = 0.50  # Faces above this are considered "same person"


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
            # With audio
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
            # Without audio
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


def process_video(video_path: Path) -> List[dict]:
    """
    Process video frame-by-frame and create clips based on face changes.
    Creates a clip at EVERY face change point (no minimum length).
    
    Returns list of clip info:
    [
        {"clip_path": "clips/clip_001.mp4", "start_frame": 0, "end_frame": 100, "frames": 101},
        ...
    ]
    """
    global face_app
    
    print(f"[Processing] Starting video: {video_path}")
    
    # Extract audio first
    audio_path = TEMP_DIR / f"{video_path.stem}_audio.aac"
    has_audio = extract_audio(video_path, audio_path)
    print(f"[Audio] Extracted: {has_audio}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Cannot open video")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[Video] FPS: {fps}, Total frames: {total_frames}")
    
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
                # Start new clip
                current_clip_start = frame_idx
                current_clip_embedding = face_embedding
                print(f"[Clip] Started new clip at frame {frame_idx}")
            else:
                # Check if face is similar to current clip's face
                similarity = compute_similarity(face_embedding, current_clip_embedding)
                
                if similarity < SIMILARITY_THRESHOLD:
                    # Face changed! Save previous clip
                    clip_length = frame_idx - current_clip_start
                    
                    # Create clip (no minimum length check)
                    clip_num = len(clips_info) + 1
                    clip_filename = f"clip_{clip_num:03d}.mp4"
                    clip_path = CLIPS_DIR / clip_filename
                    
                    print(f"[Clip] Saving clip {clip_num}: frames {current_clip_start}-{frame_idx-1} ({clip_length} frames)")
                    
                    success = create_clip_with_audio(
                        video_path,
                        current_clip_start,
                        frame_idx - 1,
                        fps,
                        clip_path,
                        audio_path if has_audio else None
                    )
                    
                    if success:
                        clips_info.append({
                            "clip_path": f"clips/{clip_filename}",
                            "clip_name": clip_filename,
                            "start_frame": current_clip_start,
                            "end_frame": frame_idx - 1,
                            "frames": clip_length,
                            "duration": f"{clip_length/fps:.2f}s"
                        })
                        print(f"[Clip] ✓ Saved successfully")
                    else:
                        print(f"[Clip] ✗ Failed to save")
                    
                    # Start new clip with current face
                    current_clip_start = frame_idx
                    current_clip_embedding = face_embedding
                    print(f"[Clip] Started new clip at frame {frame_idx}")
        else:
            # No face or multiple faces - end current clip if exists
            if current_clip_start is not None:
                clip_length = frame_idx - current_clip_start
                
                # Create clip (no minimum length check)
                clip_num = len(clips_info) + 1
                clip_filename = f"clip_{clip_num:03d}.mp4"
                clip_path = CLIPS_DIR / clip_filename
                
                print(f"[Clip] Saving clip {clip_num}: frames {current_clip_start}-{frame_idx-1} ({clip_length} frames)")
                
                success = create_clip_with_audio(
                    video_path,
                    current_clip_start,
                    frame_idx - 1,
                    fps,
                    clip_path,
                    audio_path if has_audio else None
                )
                
                if success:
                    clips_info.append({
                        "clip_path": f"clips/{clip_filename}",
                        "clip_name": clip_filename,
                        "start_frame": current_clip_start,
                        "end_frame": frame_idx - 1,
                        "frames": clip_length,
                        "duration": f"{clip_length/fps:.2f}s"
                    })
                    print(f"[Clip] ✓ Saved successfully")
                else:
                    print(f"[Clip] ✗ Failed to save")
                
                current_clip_start = None
                current_clip_embedding = None
        
        frame_idx += 1
        
        # Progress update
        if frame_idx % 100 == 0:
            print(f"[Progress] Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
    
    # Handle last clip if still recording
    if current_clip_start is not None:
        clip_length = frame_idx - current_clip_start
        
        clip_num = len(clips_info) + 1
        clip_filename = f"clip_{clip_num:03d}.mp4"
        clip_path = CLIPS_DIR / clip_filename
        
        print(f"[Clip] Saving final clip {clip_num}: frames {current_clip_start}-{frame_idx-1} ({clip_length} frames)")
        
        success = create_clip_with_audio(
            video_path,
            current_clip_start,
            frame_idx - 1,
            fps,
            clip_path,
            audio_path if has_audio else None
        )
        
        if success:
            clips_info.append({
                "clip_path": f"clips/{clip_filename}",
                "clip_name": clip_filename,
                "start_frame": current_clip_start,
                "end_frame": frame_idx - 1,
                "frames": clip_length,
                "duration": f"{clip_length/fps:.2f}s"
            })
            print(f"[Clip] ✓ Saved successfully")
    
    cap.release()
    
    # Cleanup temp audio
    if audio_path.exists():
        audio_path.unlink()
    
    print(f"[Complete] Created {len(clips_info)} clips")
    return clips_info


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load face recognition model on startup."""
    global face_app
    print("[Startup] Loading InsightFace model on GPU...")
    face_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for GPU
    print("[Startup] Model loaded successfully on GPU")
    yield
    print("[Shutdown] Cleaning up...")


# FastAPI app
app = FastAPI(lifespan=lifespan)

# Mount static directories
app.mount("/clips", StaticFiles(directory=CLIPS_DIR), name="clips")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page."""
    return templates.TemplateResponse("simple_clip_generator.html", {"request": request})


@app.post("/upload_and_process")
async def upload_and_process(video: UploadFile = File(...)):
    """Upload video and process to create clips."""
    try:
        # Save uploaded video
        video_path = UPLOAD_DIR / video.filename
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        
        print(f"[Upload] Saved video: {video_path}")
        
        # Process video
        clips_info = process_video(video_path)
        
        return JSONResponse({
            "success": True,
            "message": f"Created {len(clips_info)} clips",
            "clips": clips_info
        })
    
    except Exception as e:
        print(f"[Error] {e}")
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)


@app.get("/clips_list")
async def clips_list():
    """Get list of all clips."""
    clips = []
    for clip_path in sorted(CLIPS_DIR.glob("*.mp4")):
        clips.append({
            "clip_name": clip_path.name,
            "clip_path": f"clips/{clip_path.name}"
        })
    return {"clips": clips}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("simple_clip_generator_v2:app", host="0.0.0.0", port=5000, reload=True)
