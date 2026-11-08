FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY demo/frontend/package.json demo/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY demo/frontend/ ./
RUN npm run typecheck && npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    OCR_WORKERS=2 APP_DATA_DIR=/state \
    APP_FRONTEND_DIST=/opt/stratum/demo/frontend/dist HOME=/tmp
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libgl1 libglib2.0-0 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/stratum/demo/backend
COPY demo/backend/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
    && python -m pip freeze --all > /opt/requirements.resolved.lock
COPY demo/backend/ ./
COPY --from=frontend /build/dist /opt/stratum/demo/frontend/dist
RUN mkdir -p /state && chown 1001:1001 /state
USER 1001:1001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4)"
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","1","--proxy-headers"]
