from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
import subprocess
import shutil

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from insightface.app import FaceAnalysis

FACES_ROOT = Path("faces_db")
FACES_ROOT.mkdir(exist_ok=True)

CLIPS_ROOT = Path("video_clips")
CLIPS_ROOT.mkdir(exist_ok=True)

# --------- Face DB in memory ---------
# structure:
# person_db = {
#   "person_0001": {
#       "embeddings": [np.array(...), ...],
#       "folder": Path("faces_db/person_0001")
#   },
#   ...
# }
person_db: Dict[str, Dict] = {}

# --------- Face Tracker ---------
# Structure to track faces across frames
# track_id -> {"person_id": "person_0001", "bbox": [x1,y1,x2,y2], "frames_lost": 0}
active_tracks: Dict[int, Dict] = {}
next_track_id = 0
MAX_FRAMES_LOST = 15  # Max frames to keep track without detection

# --------- Face Analysis Model ---------
face_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global face_app
    
    # Startup
    print("[FaceApp] Loading InsightFace (RetinaFace + ArcFace)...")
    face_app = FaceAnalysis(
        name="buffalo_l",  # buffalo_l (accurate) or buffalo_s (faster on CPU)
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("[FaceApp] Model loaded.")
    
    # Rebuild person_db from existing folders
    print("[FaceApp] Rebuilding person_db from folders...")
    for person_folder in FACES_ROOT.iterdir():
        if not person_folder.is_dir() or not person_folder.name.startswith("person_"):
            continue
        images = list(person_folder.glob("*.jpg")) + list(person_folder.glob("*.png"))
        if not images:
            continue
        img_path = images[0]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        faces = face_app.get(img)
        if not faces:
            continue

        emb = faces[0].normed_embedding
        person_db[person_folder.name] = {
            "embeddings": [emb],
            "folder": person_folder,
        }
    print(f"[FaceApp] Loaded {len(person_db)} persons.")
    
    yield
    
    # Shutdown (cleanup if needed)
    print("[FaceApp] Shutting down...")


app = FastAPI(lifespan=lifespan)

# Mount faces folder as static files (so we can show images in HTML)
app.mount("/faces", StaticFiles(directory=FACES_ROOT), name="faces")
app.mount("/clips", StaticFiles(directory=CLIPS_ROOT), name="clips")

templates = Jinja2Templates(directory="templates")


def extract_audio_from_video(video_path: Path, audio_path: Path) -> bool:
    """Extract audio from video using ffmpeg."""
    try:
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "copy",
            str(audio_path), "-y"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"[Audio] Failed to extract audio: {e}")
        return False


def create_clip_with_audio(
    video_path: Path,
    audio_path: Optional[Path],
    start_frame: int,
    end_frame: int,
    fps: float,
    output_path: Path
) -> bool:
    """Create a video clip from start_frame to end_frame with audio."""
    try:
        start_time = start_frame / fps
        duration = (end_frame - start_frame + 1) / fps
        
        if audio_path and audio_path.exists():
            # With audio
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-i", str(audio_path),
                "-ss", str(start_time),
                "-t", str(duration),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-strict", "experimental",
                str(output_path), "-y"
            ]
        else:
            # Without audio
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-ss", str(start_time),
                "-t", str(duration),
                "-c:v", "libx264",
                str(output_path), "-y"
            ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"[Clip] Failed to create clip: {e}")
        return False


def iou(box1, box2) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def match_detection_to_track(bbox, iou_threshold=0.3) -> Optional[int]:
    """Match a detected bbox to an existing track using IoU."""
    best_track_id = None
    best_iou = iou_threshold
    
    for track_id, track_info in active_tracks.items():
        track_bbox = track_info["bbox"]
        overlap = iou(bbox, track_bbox)
        if overlap > best_iou:
            best_iou = overlap
            best_track_id = track_id
    
    return best_track_id


def update_tracks():
    """Increment frames_lost for all tracks and remove stale ones."""
    global active_tracks
    to_remove = []
    for track_id, track_info in active_tracks.items():
        track_info["frames_lost"] += 1
        if track_info["frames_lost"] > MAX_FRAMES_LOST:
            to_remove.append(track_id)
    
    for track_id in to_remove:
        del active_tracks[track_id]


def next_person_id() -> str:
    """Generate the next person_xxxx id based on existing folders."""
    existing = [p for p in FACES_ROOT.iterdir() if p.is_dir() and p.name.startswith("person_")]
    if not existing:
        return "person_0001"
    nums = []
    for p in existing:
        try:
            n = int(p.name.split("_")[1])
            nums.append(n)
        except Exception:
            continue
    return f"person_{(max(nums) + 1):04d}" if nums else "person_0001"


def is_good_quality_face(face_img: np.ndarray, bbox_area: float, det_score: float) -> bool:
    """Check if face image has good quality for registration."""
    # Minimum face size (pixels) - moderate
    if face_img.shape[0] < 60 or face_img.shape[1] < 60:
        return False
    
    # Detection score threshold - moderate
    if det_score < 0.7:
        return False
    
    # Check blur using Laplacian variance - balanced
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 80:  # Balanced blur threshold
        return False
    
    # Check brightness - wider range
    mean_brightness = gray.mean()
    if mean_brightness < 35 or mean_brightness > 230:
        return False
    
    # Check contrast - more lenient
    std_brightness = gray.std()
    if std_brightness < 15:  # Low contrast = poor quality
        return False
    
    return True


def register_new_person(embedding: np.ndarray, face_img: np.ndarray) -> str:
    person_id = next_person_id()
    person_folder = FACES_ROOT / person_id
    person_folder.mkdir(parents=True, exist_ok=True)

    # Save first face image
    filename = person_folder / "face_0001.jpg"
    cv2.imwrite(str(filename), face_img)

    person_db[person_id] = {
        "embeddings": [embedding],
        "folder": person_folder,
    }
    return person_id


def best_match(embedding: np.ndarray, threshold: float = 0.47):
    """
    Find best matching person for this embedding using cosine similarity.
    All embeddings are assumed L2-normalized (ArcFace).
    Returns (person_id or None, similarity).
    """
    best_id = None
    best_sim = -1.0
    for pid, info in person_db.items():
        # Check against all embeddings, not just average
        embs = info["embeddings"]
        for emb in embs:
            sim = float(np.dot(emb, embedding))
            if sim > best_sim:
                best_sim = sim
                best_id = pid

    if best_sim >= threshold:
        return best_id, best_sim
    return None, best_sim


def save_face_image(person_id: str, embedding: np.ndarray, face_img: np.ndarray):
    """Append new embedding and save image in person's folder."""
    info = person_db[person_id]
    info["embeddings"].append(embedding)

    folder = info["folder"]
    # find next face index
    existing = [f for f in folder.iterdir() if f.is_file() and f.name.startswith("face_")]
    idx = len(existing) + 1
    filename = folder / f"face_{idx:04d}.jpg"
    cv2.imwrite(str(filename), face_img)


# --------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Show all persons and sample face images.
    """
    persons_view = []
    for pid, info in person_db.items():
        folder: Path = info["folder"]
        images = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
        sample = images[0].name if images else None
        persons_view.append({
            "id": pid,
            "sample_image": sample,
            "count": len(images),
        })

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "persons": persons_view, "clips": []},
    )


@app.post("/process_video", response_class=HTMLResponse)
async def process_video(
    request: Request,
    video_file: UploadFile = File(...),
    frame_skip: int = Form(1),      # process every frame (set to 1)
    sim_threshold: float = Form(0.47),
    min_clip_frames: int = Form(15)  # minimum frames to save a clip
):
    """
    Upload a video, detect faces, track person changes, and save clips.
    Only processes frames with exactly 1 face.
    Saves clips when person changes, ignoring short appearances.
    """
    global active_tracks, next_track_id
    
    # Reset tracking state for new video
    active_tracks = {}
    next_track_id = 0
    
    # Save temp video file
    temp_video_path = Path("temp_video.mp4")
    with open(temp_video_path, "wb") as f:
        f.write(await video_file.read())

    cap = cv2.VideoCapture(str(temp_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract audio once
    temp_audio_path = Path("temp_audio.aac")
    has_audio = extract_audio_from_video(temp_video_path, temp_audio_path)
    
    frame_idx = 0
    detected_count = 0
    rejected_count = 0
    skipped_no_face = 0
    skipped_multiple_faces = 0
    
    # Track person changes
    # Structure: [(frame_num, person_id), ...]
    frame_person_map: List[Tuple[int, Optional[str]]] = []
    
    print(f"[Video] Processing {total_frames} frames at {fps} FPS...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # BGR image → faces (detect all faces in frame)
        faces = face_app.get(frame)
        
        # Skip frames with no faces
        if len(faces) == 0:
            skipped_no_face += 1
            frame_person_map.append((frame_idx, None))
            continue
        
        # Skip frames with 2 or more faces
        if len(faces) >= 2:
            skipped_multiple_faces += 1
            frame_person_map.append((frame_idx, None))
            continue
        
        # Process only if exactly 1 face
        face = faces[0]
        emb = face.normed_embedding
        det_score = float(face.det_score) if hasattr(face, 'det_score') else 1.0
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        
        # clamp bbox
        h, w, _ = frame.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        face_img = frame[y1:y2, x1:x2]

        if face_img.size == 0:
            frame_person_map.append((frame_idx, None))
            continue

        # Calculate bbox area
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # Quality check - skip low quality faces
        if not is_good_quality_face(face_img, bbox_area, det_score):
            rejected_count += 1
            frame_person_map.append((frame_idx, None))
            continue

        emb = np.array(emb, dtype=np.float32)

        # Try to match with existing persons by embedding
        person_id, sim = best_match(emb, threshold=sim_threshold)
        
        if person_id is None:
            # Completely new person
            person_id = register_new_person(emb, face_img)
            print(f"[FaceApp] Frame {frame_idx}: New person {person_id} (sim={sim:.3f})")
        else:
            # Matched to existing person by embedding
            save_face_image(person_id, emb, face_img)
            print(f"[FaceApp] Frame {frame_idx}: Matched {person_id} (sim={sim:.3f})")
        
        frame_person_map.append((frame_idx, person_id))
        detected_count += 1

    cap.release()
    
    print(f"[FaceApp] Detection complete:")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames with 0 faces: {skipped_no_face}")
    print(f"  Frames with 2+ faces: {skipped_multiple_faces}")
    print(f"  Frames with 1 face (processed): {detected_count + rejected_count}")
    print(f"  Good quality faces saved: {detected_count}")
    print(f"  Low quality rejected: {rejected_count}")
    
    # Now analyze frame_person_map to create clips
    print(f"\n[Clips] Analyzing person changes (min_clip_frames={min_clip_frames})...")
    clips_created = []
    
    # Group consecutive frames by person
    segments = []  # [(start_frame, end_frame, person_id), ...]
    current_person = None
    segment_start = None
    
    for frame_num, person_id in frame_person_map:
        if person_id is None:
            # No valid face in this frame
            if current_person is not None:
                # End current segment
                segments.append((segment_start, frame_num - 1, current_person))
                current_person = None
                segment_start = None
            continue
        
        if person_id != current_person:
            # Person changed
            if current_person is not None:
                # End previous segment
                segments.append((segment_start, frame_num - 1, current_person))
            # Start new segment
            current_person = person_id
            segment_start = frame_num
    
    # Don't forget the last segment
    if current_person is not None and segment_start is not None:
        segments.append((segment_start, frame_idx, current_person))
    
    print(f"[Clips] Found {len(segments)} segments before filtering")
    
    # Filter segments by minimum frames
    valid_segments = [
        (start, end, pid) for start, end, pid in segments
        if (end - start + 1) >= min_clip_frames
    ]
    
    print(f"[Clips] {len(valid_segments)} segments meet minimum frame threshold")
    
    # Create clips for valid segments
    clip_counter = 1
    for start_frame, end_frame, person_id in valid_segments:
        num_frames = end_frame - start_frame + 1
        duration = num_frames / fps
        
        output_filename = f"clip_{clip_counter:04d}_{person_id}_frames_{start_frame}-{end_frame}.mp4"
        output_path = CLIPS_ROOT / output_filename
        
        print(f"[Clips] Creating clip {clip_counter}: {person_id}, frames {start_frame}-{end_frame} ({num_frames} frames, {duration:.2f}s)")
        
        success = create_clip_with_audio(
            temp_video_path,
            temp_audio_path if has_audio else None,
            start_frame - 1,  # Convert to 0-indexed
            end_frame - 1,
            fps,
            output_path
        )
        
        if success:
            clips_created.append({
                "filename": output_filename,
                "person_id": person_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "num_frames": num_frames,
                "duration": duration
            })
            clip_counter += 1
        else:
            print(f"[Clips] Failed to create clip for frames {start_frame}-{end_frame}")
    
    # Cleanup temp files
    temp_video_path.unlink(missing_ok=True)
    temp_audio_path.unlink(missing_ok=True)
    
    print(f"\n[Clips] Created {len(clips_created)} video clips")
    
    # Reload persons view
    persons_view = []
    for pid, info in person_db.items():
        folder: Path = info["folder"]
        images = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
        sample = images[0].name if images else None
        persons_view.append({
            "id": pid,
            "sample_image": sample,
            "count": len(images),
        })

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "persons": persons_view,
            "clips": clips_created,
            "message": f"Processed {total_frames} frames. Found {detected_count} good faces ({skipped_no_face} no-face, {skipped_multiple_faces} multi-face, {rejected_count} low-quality skipped). Created {len(clips_created)} video clips from {len(valid_segments)} valid segments. Discovered {len(persons_view)} unique persons.",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
