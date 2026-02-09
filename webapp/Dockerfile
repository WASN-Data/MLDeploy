FROM python:3.10-slim

# Install system dependencies for librosa and matplotlib
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app.py .
COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Expose port 8081
EXPOSE 8081

CMD streamlit run app.py --server.port 8081 --server.address 0.0.0.0
