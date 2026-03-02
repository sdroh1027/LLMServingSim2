#!/bin/bash

#####################  Profile non-attention layers  #####################

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m profiler.layers.main \
  --hardware H100 \
  --model "Qwen/Qwen3-32B" \
  --num-layers 1 \
  --tp-size "1, 2, 4" \
  --warmup 5 \
  --repeat 10 \
  --max-len 2048 \
  --device cuda
