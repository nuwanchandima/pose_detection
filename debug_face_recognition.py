"""
Debug script to visualize face recognition results on every frame.
This will help investigate why only short clips are being generated.
"""

from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import json

# Configuration
FACES_ROOT = Path("faces_db")
INPUT_VIDEO = "input_video.mp4"  # Change this to your video path
OUTPUT_VIDEO = "debug_output.mp4"
OUTPUT_FRAMES_DIR = Path("debug_frames")  # Directory to save individual frames
SIMILARITY_THRESHOLD = 0.35  # Adjust this threshold (lowered for better matching)

# Create output frames directory
OUTPUT_FRAMES_DIR.mkdir(exist_ok=True)

# Face DB in memory
person_db = {}

def load_face_database():
    """Load all faces from the faces_db folder."""
    global person_db, face_app
    
    if not FACES_ROOT.exists():
        print(f"[ERROR] Face database folder '{FACES_ROOT}' does not exist!")
        return
    
    person_folders = sorted([p for p in FACES_ROOT.iterdir() if p.is_dir()])
    
    if not person_folders:
        print(f"[WARNING] No person folders found in '{FACES_ROOT}'")
        return
    
    print(f"[INFO] Loading face database from {len(person_folders)} persons...")
    
    for person_folder in person_folders:
        person_id = person_folder.name
        embeddings = []
        
        # Debug: Show folder contents
        all_files = list(person_folder.glob("*"))
        npy_files = list(person_folder.glob("*.npy"))
        img_files = list(person_folder.glob("*.jpg")) + list(person_folder.glob("*.png"))
        print(f"  - Checking {person_id}: {len(all_files)} total files, {len(npy_files)} .npy, {len(img_files)} images")
        
        # Try loading from .npy files first
        for emb_file in npy_files:
            try:
                emb = np.load(emb_file)
                # Ensure embedding is normalized
                emb_norm = emb / np.linalg.norm(emb)
                embeddings.append(emb_norm)
                print(f"    * Loaded {emb_file.name}: shape {emb.shape}")
            except Exception as e:
                print(f"    * ERROR loading {emb_file.name}: {e}")
        
        # Fallback: extract embeddings from images if no .npy files loaded
        if not embeddings and img_files:
            print(f"    ! No .npy files, extracting from {len(img_files)} images...")
            for img_path in img_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    faces = face_app.get(img)
                    if faces:
                        emb = faces[0].normed_embedding
                        # Ensure normalization
                        emb_norm = emb / np.linalg.norm(emb)
                        embeddings.append(emb_norm)
                        # Save normalized embedding
                        npy_path = img_path.with_suffix('.npy')
                        np.save(str(npy_path), emb_norm)
                        print(f"    * Extracted and saved {npy_path.name}")
                except Exception as e:
                    print(f"    * ERROR processing {img_path.name}: {e}")
        
        if embeddings:
            person_db[person_id] = {
                "embeddings": embeddings,
                "folder": person_folder
            }
            print(f"  ✓ Loaded {person_id}: {len(embeddings)} embeddings")
        else:
            print(f"  ✗ No valid embeddings found for {person_id}")
    
    print(f"[INFO] Face database loaded with {len(person_db)} persons")
    
    if not person_db:
        print("\n" + "="*60)
        print("WARNING: No embeddings loaded!")
        print("="*60)
        print("Please ensure your faces_db folder has images or .npy files")
        print("="*60)


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1_norm, emb2_norm))


def recognize_face(face_embedding: np.ndarray, threshold: float = SIMILARITY_THRESHOLD):
    """
    Match a face embedding against the database.
    Returns: (person_id, best_similarity, all_similarities)
    """
    if not person_db:
        return None, 0.0, {}
    
    # Ensure input embedding is normalized
    face_emb_norm = face_embedding / np.linalg.norm(face_embedding)
    
    best_person = None
    best_sim = -1.0
    all_similarities = {}
    
    for person_id, data in person_db.items():
        person_embeddings = data["embeddings"]
        
        # Compute similarity with all embeddings of this person
        sims = [compute_similarity(face_emb_norm, ref_emb) for ref_emb in person_embeddings]
        max_sim = max(sims) if sims else 0.0
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        
        all_similarities[person_id] = {
            "max": max_sim,
            "avg": avg_sim,
            "count": len(sims)
        }
        
        # Use max similarity for matching
        if max_sim > best_sim:
            best_sim = max_sim
            best_person = person_id
    
    # Check if best similarity meets threshold
    if best_sim >= threshold:
        return best_person, best_sim, all_similarities
    else:
        return None, best_sim, all_similarities


def process_video_with_debug():
    """Process video and annotate every frame with face recognition results."""
    
    # Initialize Face Analysis
    print("[INFO] Loading InsightFace model...")
    face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    print("[INFO] InsightFace model loaded successfully")
    
    # Load face database
    load_face_database()
    
    if not person_db:
        print("\n[ERROR] No faces in database. Please add faces first!")
        print("\nTo add faces to the database:")
        print("1. Use the web interface at http://localhost:5000")
        print("2. Upload a video and use 'Add New Person' feature")
        print("3. Or manually create faces_db/person_XXXX/*.npy files")
        print("\nAlternatively, check if faces_db folder has the correct structure.")
        return
    
    # Open input video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {INPUT_VIDEO}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[INFO] Video properties:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - Total frames: {total_frames}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    frame_count = 0
    stats = {
        "total_frames": 0,
        "frames_with_faces": 0,
        "frames_with_0_faces": 0,
        "frames_with_1_face": 0,
        "frames_with_multiple_faces": 0,
        "recognized_faces": 0,
        "unknown_faces": 0
    }
    
    print(f"\n[INFO] Processing video frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        stats["total_frames"] += 1
        
        # Detect faces
        faces = face_app.get(frame)
        num_faces = len(faces)
        
        # Create a copy for annotation
        annotated_frame = frame.copy()
        
        # Update stats
        if num_faces == 0:
            stats["frames_with_0_faces"] += 1
        elif num_faces == 1:
            stats["frames_with_1_face"] += 1
            stats["frames_with_faces"] += 1
        else:
            stats["frames_with_multiple_faces"] += 1
            stats["frames_with_faces"] += 1
        
        # Draw frame info
        info_y = 30
        cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Faces detected: {num_faces}", 
                    (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Process each detected face
        for idx, face in enumerate(faces):
            # Get face bbox
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Get face embedding
            face_embedding = face.embedding
            
            # Recognize face
            person_id, best_sim, all_sims = recognize_face(face_embedding, SIMILARITY_THRESHOLD)
            
            # Update stats
            if person_id:
                stats["recognized_faces"] += 1
            else:
                stats["unknown_faces"] += 1
            
            # Draw bounding box
            if person_id:
                color = (0, 255, 0)  # Green for recognized
                label = f"{person_id}"
            else:
                color = (0, 0, 255)  # Red for unknown
                label = "UNKNOWN"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with similarity score
            label_with_score = f"{label} ({best_sim:.3f})"
            label_size, _ = cv2.getTextSize(label_with_score, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background for label
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label_with_score, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw detailed similarity scores for all persons (below the face)
            detail_y = y2 + 20
            for person_id_detail, sim_data in sorted(all_sims.items(), key=lambda x: x[1]["max"], reverse=True)[:3]:
                detail_text = f"{person_id_detail}: max={sim_data['max']:.3f}, avg={sim_data['avg']:.3f}"
                cv2.putText(annotated_frame, detail_text, (x1, detail_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                detail_y += 15
        
        # Write frame to video
        out.write(annotated_frame)
        
        # Save frame as image with simple filename
        frame_filename = f"frame_{frame_count:06d}.jpg"
        frame_path = OUTPUT_FRAMES_DIR / frame_filename
        cv2.imwrite(str(frame_path), annotated_frame)
        
        # Progress update
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
    
    # Release resources
    cap.release()
    out.release()
    
    # Print final statistics
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Output video saved: {OUTPUT_VIDEO}")
    print(f"Output frames saved: {OUTPUT_FRAMES_DIR}/ ({stats['total_frames']} images)")
    print(f"\nStatistics:")
    print(f"  Total frames processed: {stats['total_frames']}")
    print(f"  Frames with 0 faces: {stats['frames_with_0_faces']} ({stats['frames_with_0_faces']/stats['total_frames']*100:.1f}%)")
    print(f"  Frames with 1 face: {stats['frames_with_1_face']} ({stats['frames_with_1_face']/stats['total_frames']*100:.1f}%)")
    print(f"  Frames with 2+ faces: {stats['frames_with_multiple_faces']} ({stats['frames_with_multiple_faces']/stats['total_frames']*100:.1f}%)")
    print(f"  Recognized faces (total): {stats['recognized_faces']}")
    print(f"  Unknown faces (total): {stats['unknown_faces']}")
    print(f"\nThreshold used: {SIMILARITY_THRESHOLD}")
    print(f"{'='*60}")
    
    # Save statistics to JSON
    with open("debug_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: debug_statistics.json")


if __name__ == "__main__":
    print("="*60)
    print("FACE RECOGNITION DEBUG TOOL")
    print("="*60)
    print(f"Input video: {INPUT_VIDEO}")
    print(f"Output video: {OUTPUT_VIDEO}")
    print(f"Output frames directory: {OUTPUT_FRAMES_DIR}/")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"Face database: {FACES_ROOT}")
    print("="*60)
    
    # Check if input video exists
    if not Path(INPUT_VIDEO).exists():
        print(f"\n[ERROR] Input video '{INPUT_VIDEO}' not found!")
        print("Please update the INPUT_VIDEO variable in the script.")
    else:
        process_video_with_debug()
