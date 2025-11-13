from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from insightface.app import FaceAnalysis

FACES_ROOT = Path("faces_db")
FACES_ROOT.mkdir(exist_ok=True)

app = FastAPI()

# Mount faces folder as static files (so we can show images in HTML)
app.mount("/faces", StaticFiles(directory=FACES_ROOT), name="faces")

templates = Jinja2Templates(directory="templates")

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


def best_match(embedding: np.ndarray, threshold: float = 0.5):
    """
    Find best matching person for this embedding using cosine similarity.
    All embeddings are assumed L2-normalized (ArcFace).
    Returns (person_id or None, similarity).
    """
    best_id = None
    best_sim = -1.0
    for pid, info in person_db.items():
        # average embedding for this person
        embs = info["embeddings"]
        avg_emb = np.mean(embs, axis=0)
        sim = float(np.dot(avg_emb, embedding))  # normed embeddings → cosine similarity
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


# --------- Load InsightFace (ArcFace) ----------
print("[FaceApp] Loading InsightFace (RetinaFace + ArcFace)...")
face_app = FaceAnalysis(
    name="buffalo_l",  # includes ArcFace-based recognition model
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("[FaceApp] Model loaded.")


@app.on_event("startup")
def startup_event():
    """
    On startup, scan any existing person_xxxx folders and
    compute a single embedding for one image in each folder
    to rebuild person_db.
    """
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
    frame_skip: int = Form(5),      # process every Nth frame
    sim_threshold: float = Form(0.5)
):
    """
    Upload a video, run face clustering, then show updated persons page.
    """
    # Save temp video file
    temp_video_path = Path("temp_video.mp4")
    with open(temp_video_path, "wb") as f:
        f.write(await video_file.read())

    cap = cv2.VideoCapture(str(temp_video_path))
    frame_idx = 0
    detected_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to speed up
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # BGR image → faces
        faces = face_app.get(frame)
        for face in faces:
            emb = face.normed_embedding
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            # clamp bbox
            h, w, _ = frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            face_img = frame[y1:y2, x1:x2]

            if face_img.size == 0:
                continue

            emb = np.array(emb, dtype=np.float32)

            person_id, sim = best_match(emb, threshold=sim_threshold)
            if person_id is None:
                # new person
                person_id = register_new_person(emb, face_img)
                print(f"[FaceApp] New person: {person_id} (sim={sim:.3f})")
            else:
                # existing person
                save_face_image(person_id, emb, face_img)
                print(f"[FaceApp] Matched {person_id} (sim={sim:.3f})")

            detected_count += 1

        frame_idx += 1

    cap.release()
    temp_video_path.unlink(missing_ok=True)
    print(f"[FaceApp] Processed video. Detected faces: {detected_count}")

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
            "message": f"Processed video. Detected {detected_count} faces.",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
