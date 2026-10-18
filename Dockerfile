# Heart disease risk API: a small image with only what serving needs.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install libraries first, so Docker can reuse this layer when only code changes.
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY models/ models/

# Run as a normal user, not root (safer if the container is ever compromised).
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Docker marks the container "unhealthy" if /health stops answering.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
