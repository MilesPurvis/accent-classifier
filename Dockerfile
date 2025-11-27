# Dockerfile for Accent Classifier Service
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY classify_video.py .

# Create directory for results
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_ID="MilesPurvis/english-accent-classifier"

# Default command (can be overridden)
ENTRYPOINT ["python", "classify_video.py"]
CMD ["--help"]
