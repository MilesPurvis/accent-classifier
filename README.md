# Accent Classifier

Classify English accents from video or audio files using a fine-tuned Wav2Vec2 model.

## Supported Accents (6 Classes)

- **us** - United States English
- **england** - British English
- **indian** - Indian English
- **australia** - Australian English
- **canada** - Canadian English
- **latin** - Latin American Spanish-influenced English

## Features

- 🎬 **Video Support** - Automatic audio extraction from video files
- ✂️ **Audio Segmentation** - Split long recordings into segments for better accuracy
- 🤖 **Fine-tuned Model** - Wav2Vec2 model trained on Common Voice dataset
- 📊 **JSON Output** - Structured results with per-segment and aggregated predictions
- 🎯 **High Accuracy** - Ensemble predictions across segments

## Installation

### 1. Requirements

```bash
# Python 3.8+
# ffmpeg for audio/video processing
brew install ffmpeg  # macOS
```

### 2. Set Up Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Process video file
python classify_video.py video.mp4

# Process audio file
python classify_video.py audio.mp3

# Save results to JSON file
python classify_video.py video.mp4 --output results.json

# Verbose mode (show progress)
python classify_video.py video.mp4 --verbose --pretty
```

### Advanced Options

```bash
# Custom segment duration (default: 10 seconds)
python classify_video.py video.mp4 --segment-duration 5

# Pretty-print JSON output
python classify_video.py video.mp4 --pretty

# All options together
python classify_video.py video.mp4 \
  --segment-duration 5 \
  --output results.json \
  --verbose \
  --pretty
```

### Supported Formats

**Video**: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.flv`, `.wmv`, `.m4v`

**Audio**: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`

## Output Format

The script outputs JSON with the following structure:

```json
{
  "metadata": {
    "input_file": "video.mp4",
    "file_type": "video",
    "total_duration_seconds": 99.2,
    "segment_duration_seconds": 10,
    "num_segments": 10,
    "timestamp": "2025-11-27T14:53:39.440943Z",
    "model_path": "./accent_classifier_finetuned/final_model"
  },
  "overall_prediction": {
    "predicted_accent": "latin",
    "confidence": 0.619,
    "all_probabilities": {
      "australia": 0.048,
      "canada": 0.041,
      "england": 0.075,
      "indian": 0.173,
      "latin": 0.619,
      "us": 0.043
    }
  },
  "segment_predictions": [
    {
      "segment_number": 1,
      "predicted_accent": "latin",
      "confidence": 0.689,
      "all_probabilities": {
        "australia": 0.052,
        "canada": 0.048,
        "england": 0.077,
        "indian": 0.084,
        "latin": 0.689,
        "us": 0.049
      }
    }
    // ... more segments
  ]
}
```

## How It Works

1. **Video Input** → Extract audio using ffmpeg (16kHz, mono)
2. **Audio Segmentation** → Split into fixed-duration chunks (default 10s)
3. **Classification** → Process each segment with fine-tuned Wav2Vec2 model
4. **Aggregation** → Average predictions across all segments
5. **JSON Output** → Return structured results with metadata

## Model Details

- **Architecture**: Wav2Vec2ForSequenceClassification
- **Base Model**: `dima806/english_accents_classification` (fine-tuned)
- **Training Data**: Mozilla Common Voice 17.0 (English)
- **Training Samples**: 100 per accent class
- **Model Location**: `accent_classifier_finetuned/final_model/`

## Project Structure

```
accent-classifier/
├── classify_video.py               # Main script (USE THIS)
├── accent_classifier_finetuned/    # Fine-tuned model directory
│   ├── final_model/                # Saved model (378 MB)
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── preprocessor_config.json
│   └── label_encoder.pkl
├── requirements.txt
├── archive/                        # Legacy scripts (for reference)
└── README.md
```

## Examples

### Example 1: Quick Classification

```bash
python classify_video.py sample.mp3
```

Output (stdout):
```json
{
  "overall_prediction": {
    "predicted_accent": "us",
    "confidence": 0.82
  }
}
```

### Example 2: Detailed Analysis

```bash
python classify_video.py interview.mp4 \
  --segment-duration 5 \
  --output analysis.json \
  --verbose \
  --pretty
```

Console output:
```
✂️  Segmenting audio (120.5s) into 5s chunks
   Created 25 segments
🤖 Loading accent classifier model
🔍 Classifying 25 segments
   Segment 1/25: us (78.2%)
   Segment 2/25: us (81.5%)
   ...
📊 Aggregating results
✅ Results saved to: analysis.json
```

## Performance Notes

### Expected Accuracy

- High confidence (>70%): Strong accent indicators present
- Medium confidence (50-70%): Mixed or unclear accent features
- Low confidence (<50%): Multiple competing accents or unclear audio

### Best Practices

1. **Audio Quality** - Use clear audio without background noise
2. **Segment Duration** - 5-10 seconds works best for most cases
3. **Longer Recordings** - More segments = more reliable aggregated results
4. **Multiple Speakers** - Process each speaker separately for best results

### Limitations

1. **Audio Quality** - Background noise affects performance
2. **Short Audio** - Very short clips (<3 seconds) may be unreliable
3. **Non-native Speakers** - Trained primarily on native speakers
4. **Mixed Accents** - May show lower confidence with mixed accent patterns

## Training the Model (Optional)

The repository includes a pre-trained model. If you want to retrain:

```bash
# See archive/train_accent_model.py
python archive/train_accent_model.py
```

**Requirements for training:**
- HuggingFace account with Common Voice access
- ~15 GB disk space
- GPU recommended (2-4 hours training time)

## License

This project uses:
- **Wav2Vec2**: Facebook's pretrained model
- **Base Model**: dima806/english_accents_classification
- **Common Voice**: Mozilla's open dataset (CC0 license)
- **Libraries**: HuggingFace Transformers, librosa, scikit-learn

Check individual licenses for production use.

## Credits

- **Base Model**: [dima806/english_accents_classification](https://huggingface.co/dima806/english_accents_classification)
- **Wav2Vec2**: [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
- **Dataset**: [Mozilla Common Voice 17.0](https://commonvoice.mozilla.org/)
- **Libraries**: HuggingFace Transformers, librosa, scikit-learn

## Troubleshooting

### ffmpeg not found

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from: https://ffmpeg.org/download.html
```

### Model not found

Make sure the model files are in `accent_classifier_finetuned/final_model/`. If missing, you need to train the model first (see archive/train_accent_model.py).

### Out of memory

Reduce segment duration to process smaller chunks:

```bash
python classify_video.py large_file.mp4 --segment-duration 5
```
