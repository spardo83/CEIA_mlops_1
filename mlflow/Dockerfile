FROM python:3.11-slim

COPY requirements.txt /requirements.txt

# Minimal system deps and cleanup to keep the image slim
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir mlflow[extras]==2.10.2 && \
    pip install --no-cache-dir -r /requirements.txt

EXPOSE 5000
