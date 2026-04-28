"""
Expand rep5 trace into rep5 with 6-variant prefix per unique request.

Original trace structure:
  rep1 (124 unique A..Z), rep2 (random sample 124), ..., rep5 (random sample 124)
  -> 620 total records

We first recover the 124 unique requests (A..Z). For each unique request X we
build 6 variants X1..X6 by prepending i-1 "variant" tokens to the input (so X1
is identical to original, X2 gets 1 prepended, etc.). The prepended token at
position i is the SAME token across all requests (variant_token[i]), so
  - X1 and X2 differ at position 0 -> no KV hit between them
  - X2 and Y2 share prepended token 0 -> prefix hit possible across different
    originals at the same variant-index.

Output order (rep1..rep5), each rep a full pass of:
    X1 X2 X3 X4 X5 X6 Y1 Y2 ... Z1 Z2 Z3 Z4 Z5 Z6
For rep2..rep5 the uniques are randomly sampled (with replacement) like before.

Total = 124 * 6 * 5 = 3720 requests.
Poisson arrival times regenerated at the same rate.
"""

import argparse, json, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="dataset/lveval_hotpotwikiqa_mixup_16k_qwen3-32b_rate10.0_rep5.jsonl")
    ap.add_argument("--output", default="dataset/lveval_hotpotwikiqa_mixup_16k_qwen3-32b_rate10.0_rep5_6x.jsonl")
    ap.add_argument("--unique", type=int, default=124)
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--repeat",  type=int, default=5)
    ap.add_argument("--arrival-rate", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    # Reserve a distinct token id range for variants. Qwen3 vocab=151936; we
    # use the top end so prepended ids never collide with real tokens used in
    # the inputs.
    ap.add_argument("--variant-base-id", type=int, default=151000)
    args = ap.parse_args()

    # 1) Read original 620 records and recover 124 unique rep1 rows.
    all_rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_rows.append(json.loads(line))

    uniques = all_rows[: args.unique]
    print(f"[recover] {len(uniques)} unique rows from {len(all_rows)} total")

    # 2) Define variant tokens. variant_tokens[0] will never be prepended (X1
    #    is identical to original); variant_tokens[i] for i>=1 is prepended to
    #    Xi (so X2 gets [v1], X3 gets [v1, v2]? -- NO: spec says A1/A2 differ
    #    by 1 token at front, so we prepend a single DIFFERENT token per
    #    variant index. X_i has exactly (i-1) prepended tokens, where the
    #    (i-1)-th prepended token is variant_tokens[i-2]. But to let X2/Y2
    #    share at variant-index 2, the *sequence* of prepended tokens must be
    #    identical across originals at the same variant index. The simplest
    #    scheme: Xi gets a prefix of length (i-1) taken from a shared list
    #    [v1, v2, v3, v4, v5]. Then X2 prefix = [v1], Y2 prefix = [v1] -> they
    #    share token v1. X1 prefix = [] so A1->A2 diverges immediately at
    #    position 0.)
    variant_prefix_tokens = [args.variant_base_id + i for i in range(args.variants - 1)]
    print(f"[variants] prepended token ids: {variant_prefix_tokens}")

    def expand(row, variant_idx):
        """variant_idx: 0..variants-1; 0 -> identical, i -> prepend first i shared tokens."""
        prefix = variant_prefix_tokens[:variant_idx]
        new_ids = prefix + list(row["input_tok_ids"])
        return {
            "input_toks":     len(new_ids),
            "output_toks":    row["output_toks"],
            # arrival filled in later
            "arrival_time_ns": 0,
            "input_tok_ids":  new_ids,
            "output_tok_ids": list(row["output_tok_ids"]),
        }

    # 3) Build rep list: rep1 uniques in order, rep2..rep5 random sample.
    rng = random.Random(args.seed)
    rep_sources = [list(uniques)]
    for _ in range(args.repeat - 1):
        rep_sources.append([rng.choice(uniques) for _ in range(len(uniques))])

    # 4) Flatten: for each rep, for each unique X in that rep's order, emit X1..X6.
    out_rows = []
    for rep_idx, src in enumerate(rep_sources):
        for row in src:
            for v in range(args.variants):
                out_rows.append(expand(row, v))
    print(f"[expand] {len(uniques)} * {args.variants} * {args.repeat} = {len(out_rows)} rows")

    # 5) Regenerate Poisson arrivals.
    t = 0
    rng_arr = random.Random(args.seed)
    for r in out_rows:
        interval_ns = int(rng_arr.expovariate(args.arrival_rate) * 1e9)
        t += interval_ns
        r["arrival_time_ns"] = t

    # 6) Write out.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 7) Meta.
    meta_path = args.output.replace(".jsonl", "") + ".meta.json"
    meta = {
        "source_trace": args.input,
        "unique": args.unique,
        "variants": args.variants,
        "repeat": args.repeat,
        "total_req": len(out_rows),
        "arrival_rate_req_per_sec": args.arrival_rate,
        "seed": args.seed,
        "variant_prefix_tokens": variant_prefix_tokens,
        "note": "Xi gets (i-1) shared prepended tokens; Xi and Yi share the prepended prefix, enabling cross-original KV hits at the same variant index. X1 and X2 differ at position 0 so A1->A2 has no KV hit.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 8) Stats.
    avg_in  = sum(r["input_toks"]  for r in out_rows) / len(out_rows)
    avg_out = sum(r["output_toks"] for r in out_rows) / len(out_rows)
    print(f"[stats] total={len(out_rows)}, avg input_toks={avg_in:.1f}, avg output_toks={avg_out:.2f}")
    print(f"[stats] max input_toks = {max(r['input_toks'] for r in out_rows)}")
    print(f"[out]   {args.output}")
    print(f"[meta]  {meta_path}")

if __name__ == "__main__":
    main()
