FROM python:3.10-slim

# Install system dependencies for librosa
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api.py .
COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Expose port 8080
EXPOSE 8080

CMD uvicorn api:app --host 0.0.0.0 --port 8080
