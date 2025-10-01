# Lightweight Dockerfile for the Camera Tracking app (app_backup)
# Notes:
# - This uses python:3.11-slim. If you need GPU support (CUDA) or a specific PyTorch wheel,
#   replace the base image with a suitable CUDA/PyTorch image and pin torch versions.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps required by OpenCV and other libs
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       ffmpeg \
       libgl1 \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY app_backup/requirements.txt ./requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app_backup/ .

# Expose the port used by uvicorn (matches main.py config)
EXPOSE 8000

# Default command - use uvicorn to serve the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
