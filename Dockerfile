# Use an official PyTorch image
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Clone your GitHub repository (replace with your repo URL)
RUN apt-get update && apt-get install -y git && \
    git clone https://github.com/Deathrenderr/deepfake_detector.git . 

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download model weights
RUN mkdir -p checkpoints_v3_fixed && \
    pip install gdown && \
    gdown --id 13UCdd2OAek_pct3gNtdnG7r0ZerEkD1m -O checkpoints_v3_fixed/latest_checkpoint_v3.pth

# Expose port for Flask
EXPOSE 8080

# Set environment variable for Flask
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run Flask app
CMD ["python", "app.py"]
