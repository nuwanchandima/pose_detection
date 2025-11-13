# Video Clip Generation Guide

## Overview
This application now automatically creates separate video clips based on person changes detected in the uploaded video. Each clip contains only frames where a specific person appears, with audio preserved.

## How It Works

### 1. Frame Processing
- Processes **ALL frames** in the video
- Only keeps frames with **exactly 1 face**
- Skips frames with:
  - 0 faces (empty scenes)
  - 2+ faces (group scenes)

### 2. Person Tracking
- Tracks when each person appears and disappears
- Groups consecutive frames by person
- Example:
  ```
  Frames 1-100:   Person 1 appears
  Frame 101:      Person 2 appears (Person 1 disappears)
  Frames 102-120: Person 2 continues
  Frame 121:      Person 1 returns (Person 2 disappears)
  ```

### 3. Clip Creation Rules
- Creates a new clip when person changes
- **Ignores short appearances** based on `min_clip_frames` setting
- Includes audio from the original video
- Clips are saved as MP4 files

### 4. Example Scenario
Given these settings:
- `min_clip_frames = 15` (ignore appearances shorter than 15 frames)

**Timeline:**
```
Frames 1-100:    Person 1 → Save as Clip 1 (100 frames)
Frames 101-120:  Person 2 → Save as Clip 2 (20 frames)
Frames 121-125:  Person 1 → IGNORED (only 5 frames, < 15)
Frames 126-150:  Person 2 → Save as Clip 3 (25 frames)
```

**Result:** 3 clips created
- `clip_0001_person_0001_frames_1-100.mp4` (Person 1, 100 frames)
- `clip_0002_person_0002_frames_101-120.mp4` (Person 2, 20 frames)
- `clip_0003_person_0002_frames_126-150.mp4` (Person 2, 25 frames)

## Configuration Options

### 1. Similarity Threshold (sim_threshold)
- **Default:** 0.47
- **Range:** 0.35 - 0.65
- **Effect:**
  - Lower values → More persons detected (more clips)
  - Higher values → Fewer persons detected (fewer clips)

### 2. Minimum Frames Per Clip (min_clip_frames)
- **Default:** 15
- **Range:** 1 - 300
- **Effect:**
  - Higher values → Ignore more short appearances
  - Lower values → Keep even brief appearances
- **Guidelines:**
  - At 30 FPS: 15 frames = 0.5 seconds
  - At 25 FPS: 15 frames = 0.6 seconds
  - At 24 FPS: 15 frames = 0.625 seconds

## Usage Instructions

### 1. Upload Video
1. Open the web interface (http://localhost:8000)
2. Click "Choose File" and select your video
3. Adjust settings if needed:
   - Similarity threshold
   - Minimum frames per clip
4. Click "Process Video & Create Clips"

### 2. View Results
After processing, you'll see:
- **Message:** Summary of processing (frames analyzed, persons detected, clips created)
- **Created Video Clips:** List of all generated clips with:
  - Person ID
  - Frame range
  - Duration
  - Preview player
  - Download button

### 3. Download Clips
- Click "Download" button on any clip
- Or access clips directly in the `video_clips/` folder

## Output Files

### Clip Naming Convention
```
clip_[number]_[person_id]_frames_[start]-[end].mp4
```
Example: `clip_0001_person_0001_frames_1-100.mp4`

### Directory Structure
```
project_root/
├── video_clips/          # Generated video clips
│   ├── clip_0001_person_0001_frames_1-100.mp4
│   ├── clip_0002_person_0002_frames_101-120.mp4
│   └── ...
├── faces_db/             # Extracted face images
│   ├── person_0001/
│   └── person_0002/
└── main.py
```

## Technical Details

### Audio Processing
- Audio is extracted from the original video using ffmpeg
- Each clip includes the corresponding audio segment
- Audio codec: AAC
- If video has no audio, clips are created without audio

### Video Encoding
- Video codec: H.264 (libx264)
- Original resolution preserved
- Frame rate preserved from original video

## Requirements

### System Dependencies
- **FFmpeg** must be installed and available in PATH
  - Windows: Download from https://ffmpeg.org/download.html
  - Linux: `sudo apt install ffmpeg`
  - Mac: `brew install ffmpeg`

### Python Packages
- All packages from `pyproject.toml`
- FastAPI, OpenCV, InsightFace, etc.

## Troubleshooting

### Issue: FFmpeg not found
**Error:** `Failed to extract audio` or `Failed to create clip`
**Solution:** Install FFmpeg and add to PATH

### Issue: Clips have wrong person
**Solution:** Adjust `sim_threshold`:
- Increase if same person split into multiple IDs
- Decrease if different persons get same ID

### Issue: Too many short clips
**Solution:** Increase `min_clip_frames` value

### Issue: Missing important clips
**Solution:** Decrease `min_clip_frames` value

## Performance Tips

1. **Large Videos:** Processing time depends on:
   - Number of frames
   - Video resolution
   - CPU/GPU speed

2. **Disk Space:** Ensure sufficient space for:
   - Face images (faces_db/)
   - Video clips (video_clips/)

3. **Memory:** High-resolution videos may require more RAM

## Future Enhancements

Possible improvements:
- Batch processing multiple videos
- Custom clip naming templates
- Export metadata (CSV/JSON)
- Face quality filtering UI
- Clip merging options
