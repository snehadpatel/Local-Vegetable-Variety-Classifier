# Use a lightweight official Python runtime as base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV, Pillow, and rembg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to take advantage of Docker caching
COPY requirements.txt .

# Install dependencies, using CPU-only version of PyTorch to reduce image size
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-download the rembg U-2-Net model so server boots up instantly on deployments
RUN python -c "from rembg import remove; import numpy as np; remove(np.zeros((10, 10, 3), dtype=np.uint8))"

# Copy the rest of the application files (including weight .pth files and static/templates)
COPY . .

# Expose port 8000
EXPOSE 8000

# Start FastAPI server via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
