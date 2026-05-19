#!/bin/bash

#####################  Profile non-attention layers  #####################
# Single RTX Pro 6000 (Blackwell, 96 GB) → tp_size=1 only.

CUDA_VISIBLE_DEVICES=0 \
python3 -m profiler.layers.main \
  --hardware RTXPro6000 \
  --model "Qwen/Qwen3.5-397B-A17B" \
  --num-layers 1 \
  --tp-size "1" \
  --warmup 5 \
  --repeat 10 \
  --max-len 2048 \
  --device cuda
