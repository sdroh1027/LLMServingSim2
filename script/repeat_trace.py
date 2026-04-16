#!/usr/bin/env python3
"""
Repeat a LLMServingSim2 JSONL trace N times.

Rep 0: original arrival times
Rep 1: last_arrival_of_rep0 + 1s gap + original arrival time
Rep N: N * (last_arrival + gap) + original arrival time

Usage:
  python script/repeat_trace.py \
    --src dataset/qwen_traceA_blksz_16.jsonl \
    --dst dataset/qwen_traceA_blksz_16_rep2.jsonl \
    --repeat 2 \
    --num-req 43058
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Repeat a LLMServingSim2 JSONL trace")
    parser.add_argument("--src", required=True, help="Source JSONL file")
    parser.add_argument("--dst", required=True, help="Destination JSONL file")
    parser.add_argument("--repeat", type=int, default=2, help="Number of repetitions (default: 2)")
    parser.add_argument("--num-req", type=int, default=None, help="Max requests to read from source")
    parser.add_argument("--gap", type=float, default=1.0, help="Gap between repeats in seconds (default: 1.0)")
    args = parser.parse_args()

    # Read source
    rows = []
    with open(args.src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.num_req is not None and i >= args.num_req:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    base_count = len(rows)
    if base_count == 0:
        print("No rows found.")
        return

    last_arrival_ns = max(r["arrival_time_ns"] for r in rows)
    gap_ns = int(args.gap * 1_000_000_000)

    with open(args.dst, "w", encoding="utf-8") as fout:
        for rep in range(args.repeat):
            time_offset_ns = rep * (last_arrival_ns + gap_ns)

            for row in rows:
                record = {
                    "input_toks": row["input_toks"],
                    "output_toks": row["output_toks"],
                    "arrival_time_ns": row["arrival_time_ns"] + time_offset_ns,
                    "input_tok_ids": row["input_tok_ids"],
                    "output_tok_ids": row["output_tok_ids"],
                }
                fout.write(json.dumps(record, separators=(",", ":")) + "\n")

    total = base_count * args.repeat
    print(f"{base_count} requests x {args.repeat} repeat = {total} total -> {args.dst}")
    print(f"Rep 0 arrival: [0, {last_arrival_ns/1e9:.3f}s]")
    for rep in range(1, args.repeat):
        offset_s = rep * (last_arrival_ns + gap_ns) / 1e9
        print(f"Rep {rep} arrival: [{offset_s:.3f}s, {offset_s + last_arrival_ns/1e9:.3f}s]")


if __name__ == "__main__":
    main()
