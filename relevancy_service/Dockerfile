FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3 python3-venv python3-pip git \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts

EXPOSE 8080

ENV DEVICE=cuda
ENV MODEL_NAME=google/siglip-so400m-patch14-384

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
