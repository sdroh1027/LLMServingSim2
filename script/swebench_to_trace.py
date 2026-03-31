"""
SWE-bench → LLMServingSim2 trace 변환 스크립트

SWE-bench Lite (300 tasks) 또는 Full 데이터셋을 LLMServingSim2 trace JSONL로 변환.

Input 구성:
  problem_statement + hints_text (+ 옵션: repo context padding)
Output 구성:
  patch (diff)

사용법:
  python -X utf8 script/swebench_to_trace.py \
    --output dataset/swebench_lite_qwen3-32b_rate5.jsonl \
    --model Qwen/Qwen3-32B \
    --variant lite \
    --rate 5
"""

import argparse
import json
import random
from pathlib import Path


def load_tokenizer(model_name: str):
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
    rng = random.Random(seed)
    times = []
    t = 0
    for _ in range(n):
        interval_ns = int(rng.expovariate(rate) * 1e9)
        t += interval_ns
        times.append(t)
    return times


def main():
    parser = argparse.ArgumentParser(description="SWE-bench → LLMServingSim2 trace converter")
    parser.add_argument("--output", required=True, help="출력 JSONL 파일 경로")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="HuggingFace 토크나이저 모델명")
    parser.add_argument("--variant", choices=["lite", "full"], default="lite",
                        help="SWE-bench variant (lite: 300, full: ~2294)")
    parser.add_argument("--rate", type=float, default=5.0, help="Poisson arrival rate req/sec")
    parser.add_argument("--num-req", type=int, default=0,
                        help="변환할 최대 요청 수 (0 = 전부)")
    parser.add_argument("--context-padding", type=int, default=0,
                        help="각 요청에 추가할 dummy context 토큰 수 (repo code 시뮬레이션용)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="데이터셋 반복 횟수 (KV cache hit 측정용)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer_name, tokenizer = load_tokenizer(args.model)

    # Load SWE-bench
    from datasets import load_dataset
    dataset_name = "princeton-nlp/SWE-bench_Lite" if args.variant == "lite" else "princeton-nlp/SWE-bench"
    print(f"[load] Loading {dataset_name} ...")
    ds = load_dataset(dataset_name, split="test")
    print(f"[load] {len(ds)} tasks loaded")

    rows = list(ds)
    if args.num_req > 0:
        rows = rows[:args.num_req]

    # Repeat
    rows = rows * args.repeat
    num_req = len(rows)
    print(f"[conv] {len(ds)} tasks × {args.repeat} repeat = {num_req} total, tokenizing...")

    arrival_times = generate_arrival_times(num_req, args.rate, args.seed)

    # Context padding: dummy tokens to simulate repo code context
    padding_ids = []
    if args.context_padding > 0:
        # Use repeating pattern as padding (simulates code context)
        base_text = "# Repository source code context\n" * 100
        base_ids = tokenizer.encode(base_text, add_special_tokens=False)
        while len(padding_ids) < args.context_padding:
            padding_ids.extend(base_ids)
        padding_ids = padding_ids[:args.context_padding]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    records = []
    for i, (row, arrival_ns) in enumerate(zip(rows, arrival_times)):
        # Input: problem_statement + hints
        input_text = row["problem_statement"]
        hints = row.get("hints_text", "")
        if hints:
            input_text += "\n\n" + hints

        # Output: patch
        output_text = row["patch"]

        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        output_ids = tokenizer.encode(output_text, add_special_tokens=False)

        # Add context padding
        if padding_ids:
            input_ids = padding_ids + input_ids

        if not input_ids:
            input_ids = [0]
        if not output_ids:
            output_ids = [0]

        records.append({
            "input_toks":      len(input_ids),
            "output_toks":     len(output_ids),
            "arrival_time_ns": arrival_ns,
            "input_tok_ids":   input_ids,
            "output_tok_ids":  output_ids,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{num_req} done...", flush=True)

    with open(args.output, "w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] {len(records)} records -> {args.output}")

    avg_in  = sum(r["input_toks"]  for r in records) / len(records)
    avg_out = sum(r["output_toks"] for r in records) / len(records)
    print(f"[stats] tokenizer_model   = {tokenizer_name}")
    print(f"[stats] avg input_toks    = {avg_in:.1f}")
    print(f"[stats] avg output_toks   = {avg_out:.1f}")
    print(f"[stats] num_req           = {num_req}")
    print(f"[stats] context_padding   = {args.context_padding}")
    print(f"[stats] arrival_rate      = {args.rate} req/s")

    meta_path = str(args.output).replace(".jsonl", "") + ".meta.json"
    meta = {
        "tokenizer_model": tokenizer_name,
        "source_dataset":  dataset_name,
        "variant":         args.variant,
        "num_req":         num_req,
        "repeat":          args.repeat,
        "context_padding": args.context_padding,
        "arrival_rate_req_per_sec": args.rate,
        "seed":            args.seed,
    }
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)
    print(f"[meta] Saved -> {meta_path}")


if __name__ == "__main__":
    main()
