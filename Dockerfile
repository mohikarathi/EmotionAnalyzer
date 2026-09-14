# Production-grade container for EmotionAnalyzer Speech Emotion Recognition
FROM python:3.11-slim

# Set environment variables (forcing CPU-only mode to prevent 20s CUDA probing timeouts)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    CUDA_VISIBLE_DEVICES="-1" \
    TF_CPP_MIN_LOG_LEVEL="2" \
    TF_NUM_INTRAOP_THREADS="2" \
    TF_NUM_INTEROP_THREADS="1" \
    OMP_NUM_THREADS="2"

# Install system dependencies (libsndfile for soundfile, ffmpeg for decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies first to optimize caching layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, templates, static, and trained models
COPY . .

# Create uploads directory and non-root service user for container security
RUN mkdir -p uploads && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 10000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Start via production Gunicorn WSGI server with preloaded in-memory models
CMD ["sh", "-c", "gunicorn --preload --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 120 --access-logfile - --error-logfile - app:app"]

