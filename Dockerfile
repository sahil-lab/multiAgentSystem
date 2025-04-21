FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model directory
RUN mkdir -p models
RUN mkdir -p chroma_db

# Environment variables
ENV MODEL_PATH="/app/models/model.gguf"
ENV NUM_AGENTS=3
ENV MAX_ITERATIONS=10
ENV CONVERGENCE_THRESHOLD=0.98
ENV TEMPERATURE_STRATEGY="decreasing"

# Volume for models and vector DB
VOLUME ["/app/models", "/app/chroma_db"]

# Expose port for Gradio
EXPOSE 7860

# Run the application
CMD ["python", "app.py"] 