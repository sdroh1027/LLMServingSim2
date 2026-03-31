"""
Multi-turn LVEval trace 생성 스크립트 (8k + 16k + 32k)

동일 문서(32k row)를 context 8k → 16k → 32k로 truncate해서 3개 turn을 생성한다.
- Turn 1 (8k):  ctx32[:8k]  + sep + question
- Turn 2 (16k): ctx32[:16k] + sep + question
- Turn 3 (32k): ctx32       + sep + question (전체)

Turn 1의 input_ids는 Turn 2 input_ids의 prefix, Turn 2는 Turn 3의 prefix.
→ Radix tree prefix caching에서 turn2는 turn1 캐시에서 ~8k hit, turn3는 turn2 캐시에서 ~16k hit.

사용법:
  python -X utf8 script/gen_multiturn_trace.py \
    --input-32k ../LVEval/data/hotpotwikiqa_mixup/hotpotwikiqa_mixup_32k.jsonl \
    --output dataset/lveval_hotpotwikiqa_mixup_8k+16k+32k_llama3-8b_rate20.jsonl \
    --model meta-llama/Llama-3.1-8B \
    --num-conv 100 \
    --rate 20
"""

import argparse
import json
import random
import sys
from pathlib import Path


def load_tokenizer(model_name: str):
    """AutoTokenizer 로드. 실패 시 GPT-2로 fallback."""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        loaded_name = model_name
        print(f"[tokenizer] Loaded: {loaded_name}")
    except Exception as e:
        loaded_name = "gpt2"
        print(f"[tokenizer] Failed to load {model_name} ({e}), falling back to {loaded_name}")
        tok = AutoTokenizer.from_pretrained(loaded_name)
    return loaded_name, tok


def generate_arrival_times(n: int, rate: float, seed: int = 42) -> list:
    """포아송 프로세스로 n개의 arrival_time_ns 생성 (rate: req/sec)"""
    rng = random.Random(seed)
    times = []
    t = 0
    for _ in range(n):
        interval_ns = int(rng.expovariate(rate) * 1e9)
        t += interval_ns
        times.append(t)
    return times


def load_rows(path: str, num: int) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= num:
                break
    return rows


def main():
    parser = argparse.ArgumentParser(description="Multi-turn LVEval trace 생성 (8k+16k+32k)")
    parser.add_argument("--input-32k", required=True,
                        help="LVEval 32k JSONL 파일 (동일 row에서 8k/16k/32k truncation)")
    parser.add_argument("--output", required=True,
                        help="출력 JSONL 파일 경로")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B",
                        help="HuggingFace 토크나이저 모델명")
    parser.add_argument("--num-conv", type=int, default=100,
                        help="conversation 수 (각 3 turn → 총 num_conv×3 requests)")
    parser.add_argument("--rate", type=float, default=20.0,
                        help="전체 요청 arrival rate req/sec (conv rate = rate/3)")
    parser.add_argument("--intra-gap", type=float, default=1.0,
                        help="같은 conversation 내 turn 간 간격 (초, default 1.0)")
    parser.add_argument("--target-8k", type=int, default=8192,
                        help="Turn 1 context 토큰 수 (default 8192)")
    parser.add_argument("--target-16k", type=int, default=16384,
                        help="Turn 2 context 토큰 수 (default 16384)")
    parser.add_argument("--max-ctx", type=int, default=32768,
                        help="Turn 3 context 최대 토큰 수 (default 32768, GPT2 fallback 시 cap 용)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer_name, tokenizer = load_tokenizer(args.model)

    print(f"[load] Reading {args.num_conv} rows from {args.input_32k} ...")
    rows = load_rows(args.input_32k, args.num_conv)

    if len(rows) < args.num_conv:
        print(f"[warn] 32k file has only {len(rows)} rows (requested {args.num_conv})")

    num_conv = min(len(rows), args.num_conv)
    print(f"[conv]  Generating {num_conv} conversations ({num_conv * 3} total requests)")

    # Conversation 시작 도착 시간: Poisson(conv_rate)
    conv_rate = args.rate / 3.0
    conv_arrival_times = generate_arrival_times(num_conv, conv_rate, args.seed)
    intra_gap_ns = int(args.intra_gap * 1e9)

    sep_ids = tokenizer.encode("\n\n", add_special_tokens=False)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    records = []
    first_ctx_ids = None  # 검증용 첫 번째 conv context

    for i in range(num_conv):
        row = rows[i]

        ctx_ids = tokenizer.encode(row.get("context", ""), add_special_tokens=False)
        q_ids   = tokenizer.encode(row.get("input", ""),   add_special_tokens=False)

        if i == 0:
            first_ctx_ids = ctx_ids

        # Turn 1: 앞 target_8k 토큰
        t1_ctx = ctx_ids[: args.target_8k]
        turn1_ids = t1_ctx + sep_ids + q_ids

        # Turn 2: 앞 target_16k 토큰
        t2_ctx = ctx_ids[: args.target_16k]
        turn2_ids = t2_ctx + sep_ids + q_ids

        # Turn 3: max_ctx 토큰까지 사용
        t3_ctx = ctx_ids[: args.max_ctx]
        turn3_ids = t3_ctx + sep_ids + q_ids

        answers = row.get("answers", [""])
        out_ids = tokenizer.encode(answers[0] if answers else " ", add_special_tokens=False) or [0]

        t0 = conv_arrival_times[i]
        for turn_ids, t in [
            (turn1_ids, t0),
            (turn2_ids, t0 + intra_gap_ns),
            (turn3_ids, t0 + 2 * intra_gap_ns),
        ]:
            if not turn_ids:
                turn_ids = [0]
            records.append({
                "input_toks":      len(turn_ids),
                "output_toks":     len(out_ids),
                "arrival_time_ns": t,
                "input_tok_ids":   turn_ids,
                "output_tok_ids":  out_ids,
            })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{num_conv} conversations done...", flush=True)

    # prefix 공유 검증: 정렬 전 conv0의 3 turn = records[0..2]
    t1 = records[0]["input_tok_ids"]
    t2 = records[1]["input_tok_ids"]
    t3 = records[2]["input_tok_ids"]
    shared_12 = t1[:args.target_8k] == t2[:args.target_8k]
    shared_23 = t2[:args.target_16k] == t3[:args.target_16k]

    max_tok = max(r["input_toks"] for r in records)
    print(f"[check] Max input_toks = {max_tok}")

    # arrival_time_ns 오름차순 정렬 (시뮬레이터가 순서대로 읽음)
    records.sort(key=lambda r: r["arrival_time_ns"])

    with open(args.output, "w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done]  {len(records)} records -> {args.output}")

    avg_in  = sum(r["input_toks"]  for r in records) / len(records)
    avg_out = sum(r["output_toks"] for r in records) / len(records)
    print(f"[stats] tokenizer_model  = {tokenizer_name}")
    print(f"[stats] avg input_toks   = {avg_in:.1f}")
    print(f"[stats] avg output_toks  = {avg_out:.1f}")
    print(f"[stats] num_conv         = {num_conv}")
    print(f"[stats] total requests   = {len(records)}")
    print(f"[stats] arrival_rate     = {args.rate} req/s")
    print(f"[verify] conv[0] turn1/turn2 share 8k prefix:  {shared_12}")
    print(f"[verify] conv[0] turn2/turn3 share 16k prefix: {shared_23}")

    meta_path = str(args.output).replace(".jsonl", "") + ".meta.json"
    meta = {
        "tokenizer_model":    tokenizer_name,
        "source_32k":         str(args.input_32k),
        "num_conv":           num_conv,
        "total_req":          len(records),
        "arrival_rate_req_per_sec": args.rate,
        "intra_gap_sec":      args.intra_gap,
        "target_8k_tokens":   args.target_8k,
        "target_16k_tokens":  args.target_16k,
        "seed":               args.seed,
    }
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)
    print(f"[meta]  Saved -> {meta_path}")


if __name__ == "__main__":
    main()
