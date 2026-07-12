# docint API image — Python 3.11 (the canonical runtime), CPU inference.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV/Paddle runtime libs + Tesseract with English/Marathi/Hindi models.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-mar \
        tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first for build caching.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src
COPY api ./api
RUN pip install --no-deps -e .

# Run unprivileged; PaddleOCR caches model weights under ~/.paddleocr.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser
ENV HOME=/home/appuser

EXPOSE 8000

# Mount a volume at /home/appuser/.paddleocr to persist model downloads
# across container restarts.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
