FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
RUN pip install uv

# Install system dependencies for eSpeak and building flash-attn/mamba-ssm/causal-conv1d
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
      espeak-ng \
      build-essential \
      git \
      wget \
      curl \
      ca-certificates \
      python3-dev \
      pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Ensure CUDA toolchain env vars are visible to build steps
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1 \
    CUDA_HOME=/usr/local/cuda \
    NVCC_FLAGS="-Xcompiler -Wno-register"

WORKDIR /app
COPY . ./

# Install base deps and then compile extras from source for CUDA 12.4 / torch 2.6
RUN uv pip install --system -e . && \
    uv pip install --system --no-build-isolation \
      --no-binary flash-attn --no-binary mamba-ssm --no-binary causal-conv1d \
      -e .[compile]

CMD ["python", "gradio_interface.py"]
