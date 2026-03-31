#!/usr/bin/env python3
"""
Convert qwen-bailian-usagetraces-anon trace to LLMServingSim2 format.

Source format (per line):
  {"chat_id", "parent_chat_id", "timestamp" (sec), "input_length", "output_length",
   "type", "turn", "hash_ids": [...]}

Target format (per line):
  {"input_toks", "output_toks", "arrival_time_ns",
   "input_tok_ids": [...], "output_tok_ids": [...]}

hash_ids are block-level (16 tokens per block). We expand each hash_id to
BLOCK_SIZE entries so the simulator's per-token radix tree works correctly.
output_tok_ids use globally-unique negative IDs (no prefix reuse on outputs).
"""

import argparse
import json
import sys
from pathlib import Path

BLOCK_SIZE = 16


def convert(src: Path, dst: Path, num_req: int | None = None):
    out_id_counter = -1  # unique negative IDs for output tokens

    with open(src) as fin, open(dst, "w") as fout:
        for i, line in enumerate(fin):
            if num_req is not None and i >= num_req:
                break

            row = json.loads(line)
            input_toks = row["input_length"]
            output_toks = row["output_length"]
            arrival_ns = int(row["timestamp"] * 1_000_000_000)

            # Expand block-level hash_ids to token-level
            input_tok_ids = []
            for hid in row["hash_ids"]:
                input_tok_ids.extend([hid] * BLOCK_SIZE)
            # Trim or pad to exact input_toks length
            if len(input_tok_ids) >= input_toks:
                input_tok_ids = input_tok_ids[:input_toks]
            else:
                # Pad with continuation of last hash (shouldn't normally happen)
                last = input_tok_ids[-1] if input_tok_ids else 0
                input_tok_ids.extend([last] * (input_toks - len(input_tok_ids)))

            # Unique output IDs (no prefix sharing on outputs)
            output_tok_ids = list(range(out_id_counter, out_id_counter - output_toks, -1))
            out_id_counter -= output_toks

            record = {
                "input_toks": input_toks,
                "output_toks": output_toks,
                "arrival_time_ns": arrival_ns,
                "input_tok_ids": input_tok_ids,
                "output_tok_ids": output_tok_ids,
            }
            fout.write(json.dumps(record, separators=(",", ":")) + "\n")

        count = i + 1 if num_req is None else min(num_req, i + 1)
    print(f"Converted {count} requests -> {dst}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert qwen trace to LLMServingSim2 format")
    parser.add_argument("--src", required=True, help="Source JSONL file")
    parser.add_argument("--dst", required=True, help="Destination JSONL file")
    parser.add_argument("--num-req", type=int, default=None, help="Max requests to convert")
    args = parser.parse_args()

    convert(Path(args.src), Path(args.dst), args.num_req)


if __name__ == "__main__":
    main()
