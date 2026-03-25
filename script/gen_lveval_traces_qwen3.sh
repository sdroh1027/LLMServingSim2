#!/bin/bash
# LVEval -> LLMServingSim2 trace 배치 생성 스크립트 (Qwen3-32B)
#
# 사용법:
#   cd LLMServingSim2
#   bash script/gen_lveval_traces_qwen3.sh
#
# 생성 파일 예시:
#   dataset/lveval_hotpotwikiqa_mixup_16k_qwen3-32b_rate1.0.jsonl

set -e

MODEL="Qwen/Qwen3-32B"
MODEL_TAG="qwen3-32b"
NUM_REQ=9999   # 각 데이터셋 전체 항목 사용 (실제 항목 수가 상한)
RATE=1.0
SEED=42

DATASETS=(hotpotwikiqa_mixup multifieldqa_en_mixup loogle_SD_mixup factrecall_en)
LENGTHS=(16k 32k 64k 128k 256k)

TOTAL=$(( ${#DATASETS[@]} * ${#LENGTHS[@]} ))
DONE=0

for DATASET in "${DATASETS[@]}"; do
  for LEN in "${LENGTHS[@]}"; do
    DONE=$(( DONE + 1 ))
    INPUT="../LVEval/data/${DATASET}/${DATASET}_${LEN}.jsonl"
    OUTPUT="dataset/lveval_${DATASET}_${LEN}_${MODEL_TAG}_rate${RATE}.jsonl"

    echo "=========================================="
    echo "[${DONE}/${TOTAL}] ${DATASET} / ${LEN}"
    echo "  input : ${INPUT}"
    echo "  output: ${OUTPUT}"
    echo "=========================================="

    python script/lveval_to_trace.py \
      --input        "$INPUT" \
      --output       "$OUTPUT" \
      --model        "$MODEL" \
      --num-req      "$NUM_REQ" \
      --arrival-rate "$RATE" \
      --seed         "$SEED"
  done
done

echo ""
echo "All done. Generated ${TOTAL} trace files in dataset/"
