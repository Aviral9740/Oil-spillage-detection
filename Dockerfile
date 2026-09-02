FROM python:3.13.9

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

CMD ["sh", "-c", "uvicorn ml_layer.api:app --host 0.0.0.0 --port 7860"]