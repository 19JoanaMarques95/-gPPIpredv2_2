FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 1000 user
WORKDIR /app

# Switch to root temporarily to ensure global install (more stable for Spaces)
# This avoids the "user path" vs "system path" confusion
RUN pip install --no-cache-dir --upgrade pip

# 1. Install Torch CPU
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# 2. Install GNN binaries
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    -f https://data.pyg.org/whl/torch-2.5.1+cpu.html

# 3. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# ... after pip install -r requirements.txt
RUN python -c "import pytz; import gradio; print('Dependencies verified!')"
# 4. Copy application code
COPY . .

# Ensure the user owns the /app directory
RUN chown -R user:user /app
USER user

# Set port for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py"]
