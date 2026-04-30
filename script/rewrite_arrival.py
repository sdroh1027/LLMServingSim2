"""
기존 LLMServingSim2 JSONL 트레이스의 arrival_time_ns만 새 도착률(rate)로 재생성.

input_toks/output_toks/input_tok_ids/output_tok_ids는 그대로 유지하고,
arrival_time_ns만 포아송 프로세스로 다시 생성한다.

함께 존재하는 .meta.json 파일이 있으면 arrival_rate_req_per_sec/seed를 갱신한다.

사용법:
  python script/rewrite_arrival.py \
    --src dataset/lveval_hotpotwikiqa_mixup_16k_qwen3-32b_rate10.0_rep5.jsonl \
    --dst dataset/lveval_hotpotwikiqa_mixup_16k_qwen3-32b_rate20.0_rep5.jsonl \
    --arrival-rate 20.0 \
    --seed 42
"""

import argparse
import json
import random
from pathlib import Path


def generate_arrival_times(n: int, rate: float, seed: int = 42) -> list:
    """포아송 프로세스로 arrival_time_ns 생성 (rate: req/sec)."""
    rng = random.Random(seed)
    times = []
    t = 0
    for _ in range(n):
        interval_ns = int(rng.expovariate(rate) * 1e9)
        t += interval_ns
        times.append(t)
    return times


def main():
    parser = argparse.ArgumentParser(description="Rewrite arrival_time_ns of an existing trace JSONL")
    parser.add_argument("--src",          required=True, help="Source JSONL trace file")
    parser.add_argument("--dst",          required=True, help="Destination JSONL trace file")
    parser.add_argument("--arrival-rate", type=float, required=True, help="New Poisson arrival rate (req/sec)")
    parser.add_argument("--seed",         type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if src.resolve() == dst.resolve():
        raise SystemExit(f"--src and --dst must differ (got same path: {src})")

    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise SystemExit(f"No records in {src}")

    arrival_times = generate_arrival_times(len(rows), args.arrival_rate, args.seed)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fout:
        for row, arrival_ns in zip(rows, arrival_times):
            row["arrival_time_ns"] = arrival_ns
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[rewrite] {len(rows)} records, rate={args.arrival_rate} req/s, seed={args.seed}")
    print(f"[rewrite] {src} -> {dst}")
    print(f"[rewrite] last arrival = {arrival_times[-1] / 1e9:.3f} s")

    src_meta = Path(str(src).replace(".jsonl", "") + ".meta.json")
    if src_meta.exists():
        with open(src_meta, encoding="utf-8") as mf:
            meta = json.load(mf)
        meta["arrival_rate_req_per_sec"] = args.arrival_rate
        meta["seed"] = args.seed
        dst_meta = Path(str(dst).replace(".jsonl", "") + ".meta.json")
        with open(dst_meta, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2, ensure_ascii=False)
        print(f"[meta]    {src_meta} -> {dst_meta}")
    else:
        print(f"[meta]    no source meta file ({src_meta}), skipping")


if __name__ == "__main__":
    main()
