#!/bin/bash
# Windows용 LLMServingSim2 실행 스크립트
# 사용법: bash run_windows.sh [추가 인자]
# 예시: bash run_windows.sh --num-req 100

export PYTHONIOENCODING=utf-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python main.py \
  --cluster-config 'cluster_config/single_node_single_instance.json' \
  --fp 16 --block-size 16 \
  --dataset 'dataset/example_trace.jsonl' \
  --output 'output/example_single_run.csv' \
  --num-req 100 --log-interval 1.0 \
  "$@"
