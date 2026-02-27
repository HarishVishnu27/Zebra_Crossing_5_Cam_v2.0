# Zebra Crossing Vehicle Analytics v2
# Optimized for NVIDIA Jetson Orin (L4T-based)
# For non-Jetson: use nvidia/cuda base image instead

# --- Jetson Orin Build ---
# Use: nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3 for Jetson
# For standard GPU: nvidia/cuda:12.2.0-runtime-ubuntu22.04

ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.json .
COPY config.py .
COPY app.py .
COPY database.py .
COPY routes.py .
COPY vision_processing.py .

# Copy templates and static files
COPY templates/ templates/
COPY static/ static/

# Create runtime directories
RUN mkdir -p static/processed static/temp

# Expose port
EXPOSE 3000

# Environment variables
ENV ZC_CONFIG_FILE=config.json
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:3000/login || exit 1

# Run the application
CMD ["python", "app.py"]
