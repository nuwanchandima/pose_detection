from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
import json
from datetime import timedelta

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from insightface.app import FaceAnalysis
import librosa
import soundfile as sf
from pyannote.audio import Pipeline
import torch

FACES_ROOT = Path("faces_db")
FACES_ROOT.mkdir(exist_ok=True)

AUDIO_ROOT = Path("audio_segments")
AUDIO_ROOT.mkdir(exist_ok=True)

CLIPS_ROOT = Path("person_clips")
CLIPS_ROOT.mkdir(exist_ok=True)

RESULTS_FILE = Path("analysis_results.json")

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
speaker_pipeline = None


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    millisecs = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"


def extract_audio_from_video(video_path: Path, output_path: Path) -> bool:
    """Extract audio from video file using librosa."""
    try:
        import subprocess
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            str(output_path), '-y'
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"[Audio] Failed to extract audio: {e}")
        return False


def extract_audio_segment(audio_path: Path, start_sec: float, end_sec: float, output_path: Path):
    """Extract audio segment for specific time range."""
    try:
        audio, sr = librosa.load(str(audio_path), sr=16000)
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        segment = audio[start_sample:end_sample]
        sf.write(str(output_path), segment, sr)
        return True
    except Exception as e:
        print(f"[Audio] Failed to extract segment: {e}")
        return False


def identify_speakers(audio_path: Path) -> List[Dict]:
    """Identify unique speakers using pyannote.audio."""
    global speaker_pipeline
    try:
        if speaker_pipeline is None:
            return []
        
        diarization = speaker_pipeline(str(audio_path))
        
        speakers = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
                "duration": turn.end - turn.start
            })
        
        return speakers
    except Exception as e:
        print(f"[Speaker] Diarization failed: {e}")
        return []


def map_speakers_to_persons(person_periods: Dict[str, List[Dict]]) -> Dict[str, str]:
    """
    Map speakers to persons based on who talks most during each person's screen time.
    Returns: {person_id: speaker_id}
    """
    person_to_speaker = {}
    
    for person_id, periods in person_periods.items():
        # Count speaking time for each speaker during this person's periods
        speaker_durations = {}
        
        for period in periods:
            if "speakers" in period:
                for speaker_info in period["speakers"]:
                    speaker_id = speaker_info["speaker"]
                    duration = speaker_info["duration"]
                    
                    if speaker_id not in speaker_durations:
                        speaker_durations[speaker_id] = 0.0
                    speaker_durations[speaker_id] += duration
        
        # Find the speaker who talks most during this person's screen time
        if speaker_durations:
            most_talkative_speaker = max(speaker_durations.items(), key=lambda x: x[1])
            person_to_speaker[person_id] = most_talkative_speaker[0]
            print(f"[Mapping] {person_id} → {most_talkative_speaker[0]} (spoke {most_talkative_speaker[1]:.1f}s)")
        else:
            person_to_speaker[person_id] = None
            print(f"[Mapping] {person_id} → No speaker detected")
    
    return person_to_speaker


def create_person_clips(video_path: Path, person_id: str, periods: List[Dict], 
                       speaker_id: str, output_dir: Path):
    """
    Create video clips for a person containing only frames where they appear
    and audio segments where their assigned speaker is talking.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, period in enumerate(periods):
        if "speakers" not in period or "audio_file" not in period:
            continue
        
        # Filter speakers to only include the assigned speaker
        person_speaking_segments = [
            s for s in period["speakers"] 
            if s["speaker"] == speaker_id
        ]
        
        if not person_speaking_segments:
            print(f"[Clip] Skipping {person_id} segment {idx+1} - assigned speaker not talking")
            continue
        
        # Create clip for each speaking segment
        for seg_idx, speaking in enumerate(person_speaking_segments):
            # Calculate absolute timestamps in the video
            segment_start = period["start_time_sec"]
            speaker_start = segment_start + speaking["start"]
            speaker_end = segment_start + speaking["end"]
            
            output_filename = f"{person_id}_clip_{idx+1:03d}_{seg_idx+1:02d}.mp4"
            output_path = output_dir / output_filename
            
            # Use ffmpeg to extract video clip with audio
            try:
                import subprocess
                cmd = [
                    'ffmpeg',
                    '-i', str(video_path),
                    '-ss', str(speaker_start),
                    '-to', str(speaker_end),
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-y',
                    str(output_path)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                print(f"[Clip] Created {output_filename} ({speaking['duration']:.1f}s)")
                
                # Add to period data
                if "clips" not in period:
                    period["clips"] = []
                period["clips"].append({
                    "filename": output_filename,
                    "start_time": format_timestamp(speaker_start),
                    "end_time": format_timestamp(speaker_end),
                    "duration": round(speaking["duration"], 2)
                })
                
            except Exception as e:
                print(f"[Clip] Failed to create clip: {e}")
                continue


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global face_app, speaker_pipeline
    
    # Startup
    print("[FaceApp] Loading InsightFace (RetinaFace + ArcFace)...")
    face_app = FaceAnalysis(
        name="buffalo_l",  # buffalo_l (accurate) or buffalo_s (faster on CPU)
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("[FaceApp] Model loaded.")
    
    # Load speaker diarization model
    print("[Speaker] Loading speaker diarization model...")
    try:
        speaker_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=None  # Add your HuggingFace token if needed
        )
        if torch.cuda.is_available():
            speaker_pipeline.to(torch.device("cuda"))
        print("[Speaker] Model loaded.")
    except Exception as e:
        print(f"[Speaker] Failed to load model: {e}")
        print("[Speaker] Speaker diarization will be disabled.")
        speaker_pipeline = None
    
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

templates = Jinja2Templates(directory="templates")


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
        {"request": request, "persons": persons_view},
    )


@app.post("/process_video", response_class=HTMLResponse)
async def process_video(
    request: Request,
    video_file: UploadFile = File(...),
    frame_skip: int = Form(1),      # process every frame (set to 1)
    sim_threshold: float = Form(0.47)
):
    """
    Upload a video, run face clustering with tracking, extract audio segments, and identify speakers.
    Only processes frames with exactly 1 face.
    """
    global active_tracks, next_track_id
    
    # Reset tracking state for new video
    active_tracks = {}
    next_track_id = 0
    
    # Save temp video file
    temp_video_path = Path("temp_video.mp4")
    with open(temp_video_path, "wb") as f:
        f.write(await video_file.read())

    # Extract full audio from video
    temp_audio_path = Path("temp_audio.wav")
    audio_extracted = extract_audio_from_video(temp_video_path, temp_audio_path)

    cap = cv2.VideoCapture(str(temp_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_idx = 0
    detected_count = 0
    rejected_count = 0
    skipped_no_face = 0
    skipped_multiple_faces = 0
    
    # Track face appearance periods: person_id -> list of (start_frame, end_frame, start_sec, end_sec)
    person_periods: Dict[str, List[Dict]] = {}
    current_person = None
    period_start_frame = None
    period_start_sec = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = frame_idx / fps

        # BGR image → faces (detect all faces in frame)
        faces = face_app.get(frame)
        
        # Determine if we have exactly 1 face
        has_single_face = len(faces) == 1
        person_in_frame = None
        
        if len(faces) == 0:
            skipped_no_face += 1
        elif len(faces) >= 2:
            skipped_multiple_faces += 1
        else:
            # Process the single face
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

            if face_img.size > 0:
                bbox_area = (x2 - x1) * (y2 - y1)
                
                # Quality check
                if is_good_quality_face(face_img, bbox_area, det_score):
                    emb = np.array(emb, dtype=np.float32)
                    
                    # Match with existing persons
                    person_id, sim = best_match(emb, threshold=sim_threshold)
                    
                    if person_id is None:
                        # New person
                        person_id = register_new_person(emb, face_img)
                        person_periods[person_id] = []
                        print(f"[FaceApp] Frame {frame_idx}: New person {person_id}")
                    else:
                        # Existing person
                        save_face_image(person_id, emb, face_img)
                    
                    person_in_frame = person_id
                    detected_count += 1
                else:
                    rejected_count += 1
        
        # Track continuous periods for each person
        if person_in_frame != current_person:
            # Person changed or disappeared
            if current_person is not None and period_start_frame is not None:
                # End the previous period
                period_end_frame = frame_idx - 1
                period_end_sec = period_end_frame / fps
                
                person_periods[current_person].append({
                    "start_frame": period_start_frame,
                    "end_frame": period_end_frame,
                    "start_time": period_start_sec,
                    "end_time": period_end_sec,
                    "duration": period_end_sec - period_start_sec
                })
                print(f"[Period] {current_person}: {format_timestamp(period_start_sec)} - {format_timestamp(period_end_sec)}")
            
            # Start new period if there's a person
            if person_in_frame is not None:
                current_person = person_in_frame
                period_start_frame = frame_idx
                period_start_sec = current_time_sec
            else:
                current_person = None
                period_start_frame = None
                period_start_sec = None
        
        frame_idx += 1

    # Close the last period if any
    if current_person is not None and period_start_frame is not None:
        period_end_frame = frame_idx - 1
        period_end_sec = period_end_frame / fps
        person_periods[current_person].append({
            "start_frame": period_start_frame,
            "end_frame": period_end_frame,
            "start_time": period_start_sec,
            "end_time": period_end_sec,
            "duration": period_end_sec - period_start_sec
        })

    cap.release()
    
    # Process audio segments and speaker identification
    results = {
        "video_info": {
            "total_frames": frame_idx,
            "fps": fps,
            "duration": frame_idx / fps
        },
        "statistics": {
            "frames_no_face": skipped_no_face,
            "frames_multiple_faces": skipped_multiple_faces,
            "frames_single_face": detected_count + rejected_count,
            "good_quality_faces": detected_count,
            "low_quality_rejected": rejected_count
        },
        "persons": []
    }
    
    # Extract audio segments for each person's periods
    person_periods_with_audio = {}
    
    if audio_extracted and temp_audio_path.exists():
        print("[Audio] Extracting audio segments for each person...")
        
        for person_id, periods in person_periods.items():
            person_periods_with_audio[person_id] = []
            
            for idx, period in enumerate(periods):
                # Extract audio segment
                audio_filename = f"{person_id}_segment_{idx+1:03d}.wav"
                audio_path = AUDIO_ROOT / audio_filename
                
                success = extract_audio_segment(
                    temp_audio_path,
                    period["start_time"],
                    period["end_time"],
                    audio_path
                )
                
                period_data = {
                    "segment_id": idx + 1,
                    "start_time": format_timestamp(period["start_time"]),
                    "end_time": format_timestamp(period["end_time"]),
                    "start_time_sec": period["start_time"],
                    "end_time_sec": period["end_time"],
                    "duration": round(period["duration"], 2),
                    "start_frame": period["start_frame"],
                    "end_frame": period["end_frame"]
                }
                
                if success:
                    period_data["audio_file"] = audio_filename
                    
                    # Identify speakers in this segment
                    speakers = identify_speakers(audio_path)
                    if speakers:
                        period_data["speakers"] = speakers
                
                person_periods_with_audio[person_id].append(period_data)
        
        # Map speakers to persons based on speaking time
        print("[Mapping] Mapping speakers to persons...")
        speaker_mapping = map_speakers_to_persons(person_periods_with_audio)
        
        # Create video clips with person's own voice only
        print("[Clips] Creating video clips with matched voice...")
        for person_id, periods in person_periods_with_audio.items():
            assigned_speaker = speaker_mapping.get(person_id)
            if assigned_speaker:
                person_clip_dir = CLIPS_ROOT / person_id
                create_person_clips(temp_video_path, person_id, periods, 
                                  assigned_speaker, person_clip_dir)
        
        # Build results with all information
        for person_id, periods in person_periods_with_audio.items():
            person_data = {
                "person_id": person_id,
                "assigned_speaker": speaker_mapping.get(person_id),
                "total_appearances": len(periods),
                "total_screen_time": sum(p["duration"] for p in periods),
                "periods": periods
            }
            results["persons"].append(person_data)
    
    # Save results to JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[Results] Saved to {RESULTS_FILE}")
    
    # Cleanup
    temp_video_path.unlink(missing_ok=True)
    temp_audio_path.unlink(missing_ok=True)
    
    total_frames = frame_idx
    print(f"[FaceApp] Processing complete:")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames with 0 faces: {skipped_no_face}")
    print(f"  Frames with 2+ faces: {skipped_multiple_faces}")
    print(f"  Frames with 1 face (processed): {detected_count + rejected_count}")
    print(f"  Good quality faces saved: {detected_count}")
    print(f"  Low quality rejected: {rejected_count}")

    # Reload persons view
    persons_view = []
    for pid, info in person_db.items():
        folder: Path = info["folder"]
        images = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
        sample = images[0].name if images else None
        
        # Get statistics from results
        person_stats = next((p for p in results["persons"] if p["person_id"] == pid), None)
        appearances = person_stats["total_appearances"] if person_stats else 0
        screen_time = person_stats["total_screen_time"] if person_stats else 0
        speaker = person_stats["assigned_speaker"] if person_stats else None
        
        # Count clips
        clip_count = 0
        if person_stats and "periods" in person_stats:
            for period in person_stats["periods"]:
                if "clips" in period:
                    clip_count += len(period["clips"])
        
        persons_view.append({
            "id": pid,
            "sample_image": sample,
            "count": len(images),
            "appearances": appearances,
            "screen_time": round(screen_time, 1),
            "speaker": speaker,
            "clips": clip_count
        })

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "persons": persons_view,
            "message": f"Processed {total_frames} frames. Found {detected_count} good faces ({skipped_no_face} no-face, {skipped_multiple_faces} multi-face, {rejected_count} low-quality skipped). Discovered {len(persons_view)} unique persons. Results saved to analysis_results.json",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
