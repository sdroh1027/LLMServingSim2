#!/bin/bash

# launch pytorch docker
# NGC 25.01-py3 is the first PyTorch container with Blackwell (sm_120) enabled,
# so it works with RTX Pro 6000. Transformers is upgraded to a version that
# natively recognizes model_type="qwen3_5_moe".
docker run --name llm_profile \
  --gpus all \
  -it \
  -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
  -v $(pwd):/workspace \
  --volume ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size=16g \
  nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -lc 'set -euo pipefail; \
    apt-get update && apt-get install -y git ninja-build cmake && \
    pip install -U pip setuptools wheel packaging && \
    pip install -U "transformers==5.8.1" && \
    exec bash'