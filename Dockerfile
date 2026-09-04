FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ORT_INTRA_OP_THREADS=2 \
    ORT_INTER_OP_THREADS=1

WORKDIR $HOME/app


COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user ml_layer/api.py ml_layer/__init__.py ml_layer/inference_engine.py \
     ml_layer/georeference.py ml_layer/export_payload.py ml_layer/
COPY --chown=user ml_layer/weights/best.onnx ml_layer/weights/best.onnx

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3).status==200 else 1)"


CMD ["uvicorn", "ml_layer.api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]