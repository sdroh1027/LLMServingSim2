#!/bin/bash

#######################  Profile attention layers  #######################

CUDA_VISIBLE_DEVICES=0 \
python -m profiler.attention.main \
  --model "Qwen/Qwen3-32B" \
  --hardware H100 \
  --max-len 65536 \
  --tp-size "1, 2, 4" \
  --warmup 5 \
  --repeat 10 \
  --device cuda