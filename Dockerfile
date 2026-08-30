FROM python:3.13.9

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn ml_layer.api:app --host 0.0.0.0 --port ${PORT:-10000}"]