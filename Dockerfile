FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for OpenCV, PIL, etc. (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire repository into the container
COPY . .

# Hugging Face Spaces routes traffic to port 7860
EXPOSE 7860

# Run the FastAPI app via Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
