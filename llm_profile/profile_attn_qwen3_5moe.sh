#!/bin/bash

#######################  Profile attention layers  #######################
# Single RTX Pro 6000 (Blackwell, 96 GB) → tp_size=1 only.

CUDA_VISIBLE_DEVICES=0 \
python -m profiler.attention.main \
  --model "Qwen/Qwen3.5-397B-A17B" \
  --hardware RTXPro6000 \
  --max-len 65536 \
  --tp-size "1" \
  --warmup 5 \
  --repeat 10 \
  --device cuda
