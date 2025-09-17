FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1 \
    NVCC_FLAGS="-Xcompiler -Wno-register"

# System dependencies (build + audio + espeak-ng)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    pkg-config \
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-fmt-all \
    espeak-ng \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project
COPY . ./

# Torch triplet pinned for CUDA 12.4
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torchvision==0.21.0+cu124

# Install project (from pyproject.toml)
RUN python3 -m pip install --no-cache-dir -e .

# Compile extras from source for CUDA 12.4 / Torch 2.6.0
RUN python3 -m pip install --no-cache-dir --no-build-isolation \
      --no-binary flash-attn --no-binary mamba-ssm --no-binary causal-conv1d \
      'flash-attn==2.7.4.post1' 'mamba-ssm==2.2.4' 'causal-conv1d==1.5.0.post8'

# Expose Gradio default port
EXPOSE 7860

# Default entrypoint
CMD ["python3", "gradio_interface.py"]
