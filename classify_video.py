#!/usr/bin/env python3
"""
VIDEO/AUDIO ACCENT CLASSIFIER
Main script for accent classification from video or audio files

Process:
1. Extract audio from video (if video file)
2. Segment audio into chunks using ffmpeg
3. Classify each segment using fine-tuned Wav2Vec2 model
4. Aggregate results across segments
5. Output results as JSON

Usage:
    python classify_video.py input.mp4 --output results.json
    python classify_video.py input.mp3 --segment-duration 10
"""

import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import sys
import os
import json
import pickle
import warnings
import subprocess
import tempfile
import shutil
import argparse
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', category=UserWarning)

# Model configuration
MODEL_PATH = "./accent_classifier_finetuned/final_model"
LABEL_ENCODER_PATH = "./accent_classifier_finetuned/label_encoder.pkl"

# HuggingFace Hub model (optional)
# Set via environment variable MODEL_ID or use command line --model flag
# Example: MODEL_ID="username/accent-classifier-wav2vec2-6class"
HF_MODEL_ID = os.getenv("MODEL_ID", None)

# Supported video extensions
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}


def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_audio_from_video(video_path, output_path=None):
    """
    Extract audio from video file using ffmpeg

    Args:
        video_path: Path to video file
        output_path: Path for output audio file (default: temp file)

    Returns:
        Path to extracted audio file
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg not found. Please install: brew install ffmpeg")

    if output_path is None:
        # Create temp file
        output_path = tempfile.mktemp(suffix='.wav', prefix='extracted_audio_')

    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # PCM 16-bit
        '-ar', '16000',  # 16kHz sample rate
        '-ac', '1',  # Mono
        '-y',  # Overwrite
        '-loglevel', 'error',
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")


def get_audio_duration(audio_path):
    """Get audio duration using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(audio_path)
    ]

    try:
        duration_str = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return float(duration_str)
    except subprocess.CalledProcessError:
        return None


def segment_audio(audio_path, segment_duration=10, output_dir=None):
    """
    Segment audio file using ffmpeg

    Args:
        audio_path: Path to input audio file
        segment_duration: Duration of each segment in seconds
        output_dir: Directory to save segments

    Returns:
        List of segment file paths, output directory
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='accent_segments_')
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Get audio duration
    total_duration = get_audio_duration(audio_path)
    if total_duration is None:
        raise RuntimeError("Failed to get audio duration")

    num_segments = int(np.ceil(total_duration / segment_duration))
    segment_files = []

    # Split audio into segments
    for i in range(num_segments):
        start_time = i * segment_duration
        output_file = os.path.join(output_dir, f"segment_{i:03d}.wav")

        cmd = [
            'ffmpeg',
            '-i', str(audio_path),
            '-ss', str(start_time),
            '-t', str(segment_duration),
            '-ar', '16000',
            '-ac', '1',
            '-y',
            '-loglevel', 'error',
            output_file
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            segment_files.append(output_file)
        except subprocess.CalledProcessError:
            pass  # Skip failed segments

    return segment_files, output_dir, total_duration


def load_model(model_id=None):
    """
    Load the fine-tuned Wav2Vec2 model

    Args:
        model_id: Optional HuggingFace model ID (e.g., "username/model-name")
                  If None, loads from local MODEL_PATH

    Returns:
        feature_extractor, model, accents, label_encoder
    """
    # Determine model source
    if model_id is None:
        model_id = HF_MODEL_ID

    if model_id:
        # Load from HuggingFace Hub
        print(f"Loading model from HuggingFace Hub: {model_id}")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)

        # Get accents from model config
        id2label = model.config.id2label
        accents = [id2label[i] for i in range(len(id2label))]
        label_encoder = None
    else:
        # Load from local path
        if not Path(MODEL_PATH).exists():
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. "
                "Please train the model first or set HF_MODEL_ID."
            )

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)

        # Load accent labels
        if Path(LABEL_ENCODER_PATH).exists():
            with open(LABEL_ENCODER_PATH, "rb") as f:
                label_encoder = pickle.load(f)
            accents = label_encoder.classes_.tolist()
        else:
            id2label = model.config.id2label
            accents = [id2label[i] for i in range(len(id2label))]
            label_encoder = None

    model.eval()
    return feature_extractor, model, accents, label_encoder


def classify_audio(audio_path, feature_extractor, model, accents, label_encoder):
    """
    Classify accent from audio file

    Returns:
        Predicted accent, probability dictionary
    """
    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=16000)

    # Process audio
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()

    # Get accent name and probabilities
    if label_encoder is not None:
        predicted_accent = label_encoder.inverse_transform([pred_id])[0]
        prob_dict = {
            label_encoder.inverse_transform([i])[0]: float(probs[i].item())
            for i in range(len(accents))
        }
    else:
        predicted_accent = model.config.id2label[pred_id]
        prob_dict = {
            model.config.id2label[i]: float(probs[i].item())
            for i in range(len(accents))
        }

    return predicted_accent, prob_dict


def aggregate_predictions(segment_predictions):
    """
    Aggregate predictions from multiple segments

    Args:
        segment_predictions: List of (accent, prob_dict) tuples

    Returns:
        Overall predicted accent, aggregated probability dictionary
    """
    aggregated_scores = defaultdict(list)

    for accent, prob_dict in segment_predictions:
        for acc, prob in prob_dict.items():
            aggregated_scores[acc].append(prob)

    # Calculate mean probability across segments
    mean_scores = {
        acc: float(np.mean(probs)) for acc, probs in aggregated_scores.items()
    }

    # Get predicted accent
    predicted_accent = max(mean_scores.items(), key=lambda x: x[1])[0]

    return predicted_accent, mean_scores


def process_video_or_audio(input_path, segment_duration=10, verbose=False):
    """
    Main processing function

    Args:
        input_path: Path to video or audio file
        segment_duration: Duration of each segment in seconds
        verbose: Print progress messages

    Returns:
        Dictionary with results
    """
    # Start timing
    start_time = time.time()

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Check file type
    is_video = input_path.suffix.lower() in VIDEO_EXTENSIONS
    is_audio = input_path.suffix.lower() in AUDIO_EXTENSIONS

    if not (is_video or is_audio):
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    # Track temp files for cleanup
    temp_files = []
    temp_dirs = []

    try:
        # Step 1: Extract audio from video if needed
        if is_video:
            if verbose:
                print(f"📹 Extracting audio from video: {input_path.name}")
            audio_path = extract_audio_from_video(input_path)
            temp_files.append(audio_path)
        else:
            audio_path = input_path

        # Get audio duration
        total_duration = get_audio_duration(audio_path)

        # Step 2: Segment audio
        if verbose:
            print(f"✂️  Segmenting audio ({total_duration:.1f}s) into {segment_duration}s chunks")

        segment_files, segment_dir, duration = segment_audio(
            audio_path,
            segment_duration=segment_duration
        )
        temp_dirs.append(segment_dir)

        if verbose:
            print(f"   Created {len(segment_files)} segments")

        # Step 3: Load model
        if verbose:
            print(f"🤖 Loading accent classifier model")

        feature_extractor, model, accents, label_encoder = load_model()

        # Step 4: Classify each segment
        if verbose:
            print(f"🔍 Classifying {len(segment_files)} segments")

        segment_predictions = []
        segment_details = []

        for i, segment_file in enumerate(segment_files):
            accent, prob_dict = classify_audio(
                segment_file,
                feature_extractor,
                model,
                accents,
                label_encoder
            )

            segment_predictions.append((accent, prob_dict))

            segment_details.append({
                "segment_number": i + 1,
                "predicted_accent": accent,
                "confidence": prob_dict[accent],
                "all_probabilities": prob_dict
            })

            if verbose:
                print(f"   Segment {i+1}/{len(segment_files)}: {accent} ({prob_dict[accent]:.1%})")

        # Step 5: Aggregate results
        if verbose:
            print(f"📊 Aggregating results")

        overall_accent, overall_scores = aggregate_predictions(segment_predictions)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Build result dictionary
        result = {
            "metadata": {
                "input_file": str(input_path),
                "file_type": "video" if is_video else "audio",
                "total_duration_seconds": float(total_duration),
                "segment_duration_seconds": segment_duration,
                "num_segments": len(segment_files),
                "execution_time_seconds": round(execution_time, 2),
                "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            },
            "overall_prediction": {
                "predicted_accent": overall_accent,
                "confidence": overall_scores[overall_accent],
                "all_probabilities": overall_scores
            },
            "segment_predictions": segment_details
        }

        return result

    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(
        description='Classify accents from video or audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video file
  python classify_video.py video.mp4 --output results.json

  # Process audio file with 5-second segments
  python classify_video.py audio.mp3 --segment-duration 5 --output results.json

  # Verbose mode
  python classify_video.py video.mp4 --verbose

Supported video formats: .mp4, .mov, .avi, .mkv, .webm, .flv, .wmv, .m4v
Supported audio formats: .wav, .mp3, .flac, .ogg, .m4a, .aac
        """
    )

    parser.add_argument('input', help='Path to video or audio file')
    parser.add_argument('--output', '-o', help='Output JSON file path (default: stdout)')
    parser.add_argument('--segment-duration', type=int, default=10,
                       help='Duration of each segment in seconds (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print progress messages')
    parser.add_argument('--pretty', action='store_true',
                       help='Pretty-print JSON output')
    parser.add_argument('--model', '-m', dest='model_id',
                       help='HuggingFace model ID (e.g., username/model-name). If not specified, uses local model.')

    args = parser.parse_args()

    try:
        # Process file
        result = process_video_or_audio(
            args.input,
            segment_duration=args.segment_duration,
            verbose=args.verbose
        )

        # Output JSON
        json_kwargs = {'indent': 2} if args.pretty else {}
        json_output = json.dumps(result, **json_kwargs)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_output)
            if args.verbose:
                print(f"\n✅ Results saved to: {args.output}")
        else:
            print(json_output)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
