# Docker Usage Guide

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t accent-classifier .
```

### 2. Run with Docker

**Process a single audio/video file:**

```bash
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/app/output \
  accent-classifier \
  /input/audio.mp3 --output /app/output/results.json --verbose --pretty
```

**Using docker-compose:**

```bash
# Create input/output directories
mkdir -p input output

# Copy your audio/video file to input/
cp ~/Downloads/audio.mp3 input/

# Run classification
docker-compose run --rm accent-classifier \
  /input/audio.mp3 --output /app/output/results.json --verbose --pretty

# Check results
cat output/results.json
```

## Configuration

### Environment Variables

- `MODEL_ID`: HuggingFace model ID (default: `MilesPurvis/english-accent-classifier`)
- `PYTHONUNBUFFERED`: Set to 1 for real-time logging

### Custom Model

To use a different model:

```bash
docker run --rm \
  -e MODEL_ID="your-username/your-model" \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/app/output \
  accent-classifier \
  /input/audio.mp3 --output /app/output/results.json
```

## Usage Examples

### Basic Classification

```bash
docker run --rm \
  -v $(pwd)/input:/input \
  accent-classifier /input/audio.wav
```

### With Verbose Output

```bash
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/app/output \
  accent-classifier \
  /input/video.mp4 --output /app/output/results.json --verbose --pretty
```

### Custom Segment Duration

```bash
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/app/output \
  accent-classifier \
  /input/audio.mp3 --segment-duration 5 --output /app/output/results.json
```

### Process Video File

```bash
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/app/output \
  accent-classifier \
  /input/video.mp4 --output /app/output/results.json --verbose
```

## Supported Formats

**Video:** .mp4, .mov, .avi, .mkv, .webm, .flv, .wmv, .m4v
**Audio:** .wav, .mp3, .flac, .ogg, .m4a, .aac

## Output

Results are saved as JSON with the following structure:

```json
{
  "metadata": {
    "input_file": "/input/audio.mp3",
    "file_type": "audio",
    "total_duration_seconds": 99.24,
    "segment_duration_seconds": 10,
    "num_segments": 10,
    "execution_time_seconds": 4.94,
    "timestamp": "2025-11-27T16:47:15.395297Z"
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
  "segment_predictions": [...]
}
```

## Troubleshooting

### Permission Issues

If you encounter permission errors with volumes:

```bash
# Create directories with proper permissions
mkdir -p input output
chmod 777 input output
```

### Out of Memory

For large video files, increase Docker memory limit:

```bash
docker run --rm --memory=4g \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/app/output \
  accent-classifier /input/large_video.mp4
```

### Model Download Issues

The model (378MB) will be downloaded on first run. Ensure:
- Internet connection is available
- At least 500MB free disk space
- HuggingFace Hub is accessible

## Development

### Build for Different Architectures

```bash
# For ARM64 (Apple Silicon, AWS Graviton)
docker build --platform linux/arm64 -t accent-classifier:arm64 .

# For AMD64 (x86_64)
docker build --platform linux/amd64 -t accent-classifier:amd64 .
```

### Interactive Shell

```bash
docker run --rm -it \
  -v $(pwd)/input:/input \
  --entrypoint /bin/bash \
  accent-classifier
```

## Deployment

### Push to Registry

```bash
# Tag for Docker Hub
docker tag accent-classifier your-username/accent-classifier:latest

# Push to Docker Hub
docker push your-username/accent-classifier:latest
```

### Run from Registry

```bash
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/app/output \
  your-username/accent-classifier:latest \
  /input/audio.mp3 --output /app/output/results.json
```
