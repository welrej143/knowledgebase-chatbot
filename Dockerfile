# Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHROMA_TELEMETRY_DISABLED=1 \
    TOKENIZERS_PARALLELISM=false

# System packages: libreoffice (doc→pdf), tesseract (OCR), fonts, and deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    tesseract-ocr \
    fonts-dejavu-core \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python deps early to leverage docker cache
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy code
COPY . /app

# Create directories Render will mount disks to
RUN mkdir -p /app/data /app/storage/chroma

# Render provides the port in $PORT
ENV HOST=0.0.0.0 \
    PORT=8000 \
    CHROMA_DIR=/app/storage/chroma \
    CORS_ORIGINS=*

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
