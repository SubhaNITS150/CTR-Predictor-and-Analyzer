# -----------------------------
# Base image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Install Python dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy application code + model
# -----------------------------
COPY . .

# -----------------------------
# Expose port (Render uses 10000)
# -----------------------------
EXPOSE 10000

# -----------------------------
# Start FastAPI
# -----------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
