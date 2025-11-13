# Audio & Speaker Recognition Setup

## Install Additional Dependencies

You need to install the audio processing libraries:

```bash
# Install audio dependencies
uv add librosa soundfile pyannote-audio torch

# Also need ffmpeg for audio extraction
# On Ubuntu/Debian:
sudo apt-get install ffmpeg

# On Windows (using chocolatey):
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

## Pyannote Audio Setup (Speaker Diarization)

Pyannote requires HuggingFace authentication for some models:

1. **Create HuggingFace account:** https://huggingface.co/join
2. **Get access token:** https://huggingface.co/settings/tokens
3. **Accept model license:** https://huggingface.co/pyannote/speaker-diarization-3.1

### Option 1: Use Token in Code

Update `main.py` line with your token:
```python
speaker_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HUGGINGFACE_TOKEN_HERE"
)
```

### Option 2: Login via CLI

```bash
huggingface-cli login
```

## Output Format

After processing a video, you'll get:

### 1. `analysis_results.json`
Contains all data:
```json
{
  "video_info": {
    "total_frames": 5000,
    "fps": 30.0,
    "duration": 166.67
  },
  "persons": [
    {
      "person_id": "person_0001",
      "total_appearances": 5,
      "total_screen_time": 45.3,
      "periods": [
        {
          "segment_id": 1,
          "start_time": "00:00:05.000",
          "end_time": "00:00:15.500",
          "duration": 10.5,
          "start_frame": 150,
          "end_frame": 465,
          "audio_file": "person_0001_segment_001.wav",
          "speakers": [
            {
              "speaker": "SPEAKER_00",
              "start": 0.5,
              "end": 8.2,
              "duration": 7.7
            }
          ]
        }
      ]
    }
  ]
}
```

### 2. `audio_segments/` folder
Contains extracted audio clips:
- `person_0001_segment_001.wav`
- `person_0001_segment_002.wav`
- `person_0002_segment_001.wav`
- etc.

## Features

✅ **Timestamp tracking** - Records when each person appears  
✅ **Audio extraction** - Saves audio for each person's screen time  
✅ **Speaker diarization** - Identifies unique speakers in audio  
✅ **JSON export** - All data saved in structured format  
✅ **Screen time stats** - Total appearances and duration per person

## Processing Flow

1. Upload video → Face detection (only single-face frames)
2. Track continuous periods when each person is alone on screen
3. Extract audio segments for each person's periods
4. Run speaker diarization on each audio segment
5. Save results to JSON with timestamps and speaker IDs
