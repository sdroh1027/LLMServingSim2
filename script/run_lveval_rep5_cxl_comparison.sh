#!/bin/bash
# LVEval 5회반복 트레이스 CXL 유/무 비교 실험
# - 모델: Qwen3-32B on H100 (96GB)
# - 트레이스: hotpotwikiqa_mixup 32k, rate20, rep5 (620 requests)
# - 구성: CXL 2560GB 있음 vs CXL 없음 (CPU 1TB only)

set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

OUTDIR="output/lveval_rep5_cxl_comparison"
mkdir -p "$OUTDIR"

CONFIG_CXL="cluster_config/single_node_h100_qwen3-32b_96g_cxl2560.json"
CONFIG_NOCXL="cluster_config/single_node_h100_qwen3-32b.json"
TRACE="dataset/lveval_hotpotwikiqa_mixup_32k_rep5_qwen3-32b_rate20.jsonl"

COMMON_ARGS="--fp 16 --block-size 16 --bypass-astrasim --num-req 620 --max-batch 32 --max-num-batched-tokens 65536 --log-interval 1 --log-level WARNING"

echo "============================================"
echo " LVEval rep5 CXL Comparison (32k, rate20)"
echo "============================================"
echo ""

# ---- Experiment 1: CXL + prefix caching ----
echo "[1/2] 32k trace - WITH CXL prefix caching"
python main.py --cluster-config "$CONFIG_CXL" \
    $COMMON_ARGS \
    --enable-prefix-caching --prefix-storage "CXL" \
    --dataset "$TRACE" \
    --output "$OUTDIR/32k_rate20_cxl.csv"
echo "  -> Done: $OUTDIR/32k_rate20_cxl.csv"
echo ""

# ---- Experiment 2: No CXL (CPU prefix caching) ----
echo "[2/2] 32k trace - NO CXL (CPU prefix caching)"
python main.py --cluster-config "$CONFIG_NOCXL" \
    $COMMON_ARGS \
    --enable-prefix-caching --prefix-storage "CPU" \
    --dataset "$TRACE" \
    --output "$OUTDIR/32k_rate20_nocxl_cpu.csv"
echo "  -> Done: $OUTDIR/32k_rate20_nocxl_cpu.csv"
echo ""

echo "============================================"
echo " All experiments complete!"
echo " Results in: $OUTDIR/"
echo "============================================"
echo ""

# ---- Summary ----
echo "=== Results Summary ==="
echo ""
for f in "$OUTDIR"/32k_rate20_*.csv; do
    name=$(basename "$f" .csv)
    echo "--- $name ---"
    total=$(tail -n +2 "$f" | wc -l)
    avg_latency=$(tail -n +2 "$f" | awk -F',' '{sum+=$8; n++} END {if(n>0) printf "%.2f", sum/n; else print "N/A"}')
    avg_ttft=$(tail -n +2 "$f" | awk -F',' '{sum+=$10; n++} END {if(n>0) printf "%.2f", sum/n; else print "N/A"}')
    avg_tpot=$(tail -n +2 "$f" | awk -F',' '{sum+=$11; n++} END {if(n>0) printf "%.6f", sum/n; else print "N/A"}')
    echo "  Requests: $total"
    echo "  Avg Latency (s): $avg_latency"
    echo "  Avg TTFT (s):    $avg_ttft"
    echo "  Avg TPOT (s):    $avg_tpot"
    echo ""
done
