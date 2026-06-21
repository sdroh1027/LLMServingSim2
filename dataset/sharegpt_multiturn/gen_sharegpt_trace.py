"""
Aeala/ShareGPT_Vicuna_unfiltered → LLMServingSim2 multi-turn JSONL.

For each ShareGPT conversation (after merging same-prefix splits), this script
emits one JSON object per (human, gpt) turn-pair with:

  {
    "input_toks":           int,          # cumulative input length at this turn
    "output_toks":          int,          # gpt response length
    "arrival_time_ns":      int,
    "input_tok_ids":        [int, ...],   # = prev_input_tok_ids + prev_output_tok_ids + new_human_tok_ids
    "output_tok_ids":       [int, ...],
    "session_id":           str,
    "turn_idx":             int,          # 0-based per session, strictly contiguous
    "intra_session_gap_ns": int           # 0 for turn 0; user think-time for N>0
  }

Prefix-extension invariant (load-bearing for KV-cache reuse measurement):
    input_tok_ids[N][:len(input_tok_ids[N-1]) + len(output_tok_ids[N-1])]
        == input_tok_ids[N-1] + output_tok_ids[N-1]

The script builds this incrementally in TOKEN space, never re-tokenizing
concatenated strings, so BPE boundary drift cannot break the invariant.

Default tokenizer matches the existing LLMServingSim2 sharegpt traces
(meta-llama/Llama-3.1-8B). See dataset_analysis.md for distribution context
that motivated the default caps.
"""

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


SUFFIX_RE = re.compile(r"^(.*?)_(\d+)$")


def parse_args():
    here = Path(__file__).resolve().parent
    default_input = (here / "../../../ShareGPT_Traces/raw/ShareGPT_V4.3_unfiltered_cleaned_split.json").resolve()
    default_output = here / "sharegpt_multiturn.jsonl"

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default=str(default_input),
                    help=f"Raw ShareGPT JSON (default: {default_input})")
    ap.add_argument("--output", default=str(default_output),
                    help="Output JSONL path")
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B")

    # Caps
    ap.add_argument("--max-sessions", type=int, default=0,
                    help="cap on emitted sessions (0 = no cap)")
    ap.add_argument("--max-requests", type=int, default=0,
                    help="cap on total turns emitted (0 = no cap). Applied AFTER per-session capping.")
    ap.add_argument("--max-turns", type=int, default=20,
                    help="per-session turn cap (truncate later turns)")
    ap.add_argument("--min-turns", type=int, default=1,
                    help="drop sessions with fewer than this many usable turns")
    ap.add_argument("--max-input-length", type=int, default=16384,
                    help="cumulative input cap per turn; session truncated at first turn that exceeds")
    ap.add_argument("--max-output-length", type=int, default=1024,
                    help="output cap per turn; longer responses are truncated (session continues)")
    ap.add_argument("--max-kv-length", type=int, default=0,
                    help="cumulative input+output cap per turn (0 = no cap)")

    # Timing
    ap.add_argument("--session-rate", type=float, default=2.0,
                    help="Poisson session start rate (sessions/sec). First session at t=0. 0 = burst.")
    ap.add_argument("--gap-dist", choices=["lognormal", "fixed", "zero"], default="lognormal",
                    help="intra_session_gap_ns distribution for turn N>0")
    ap.add_argument("--gap-mean-s", type=float, default=5.0,
                    help="mean of gap distribution in seconds (lognormal/fixed)")
    ap.add_argument("--gap-sigma", type=float, default=1.0,
                    help="sigma of underlying normal for lognormal (ignored otherwise)")
    ap.add_argument("--gap-max-s", type=float, default=60.0,
                    help="cap on sampled gap in seconds")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-self-check", action="store_true",
                    help="skip prefix-invariant verification (faster, not recommended)")
    return ap.parse_args()


def group_by_prefix(data):
    """Group raw splits by id-prefix (suffix '_N' indicates split index)."""
    groups = defaultdict(list)  # prefix -> list of (split_idx, conversations)
    for row in data:
        rid = row.get("id", "")
        m = SUFFIX_RE.match(rid)
        if m:
            prefix, sidx = m.group(1), int(m.group(2))
        else:
            prefix, sidx = rid, 0
        groups[prefix].append((sidx, row.get("conversations") or []))
    return groups


def extract_pairs(merged_msgs):
    """Skip leading non-human; pair (human, gpt) sequentially; skip stray messages."""
    i = 0
    while i < len(merged_msgs) and merged_msgs[i].get("from") != "human":
        i += 1
    pairs = []
    j = i
    while j + 1 < len(merged_msgs):
        a, b = merged_msgs[j], merged_msgs[j + 1]
        if a.get("from") == "human" and b.get("from") == "gpt":
            pairs.append((a.get("value") or "", b.get("value") or ""))
            j += 2
        else:
            j += 1  # try to resync
    return pairs


def session_start_times(num_sessions, rate, seed):
    if num_sessions <= 0:
        return []
    out = [0]
    if rate <= 0:
        return out + [0] * (num_sessions - 1)
    rng = random.Random(seed)
    t = 0
    for _ in range(num_sessions - 1):
        t += int(rng.expovariate(rate) * 1e9)
        out.append(t)
    return out


def make_gap_sampler(args):
    if args.gap_dist == "zero":
        return lambda rng: 0
    if args.gap_dist == "fixed":
        ns = int(args.gap_mean_s * 1e9)
        return lambda rng: ns
    # lognormal: mean of the log = log(mean) - sigma^2/2 so that E[X] = mean_s
    mean_s = max(args.gap_mean_s, 1e-6)
    sigma = max(args.gap_sigma, 1e-6)
    mu = math.log(mean_s) - 0.5 * sigma * sigma
    max_ns = int(args.gap_max_s * 1e9)

    def sample(rng):
        x = rng.lognormvariate(mu, sigma)
        ns = int(x * 1e9)
        return max(0, min(ns, max_ns))

    return sample


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {in_path}", flush=True)
    data = json.load(open(in_path, encoding="utf-8"))
    print(f"[load] raw splits: {len(data):,}", flush=True)

    # Tokenizer
    print(f"[tokenizer] loading {args.tokenizer}", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    # Group + flatten
    groups = group_by_prefix(data)
    print(f"[group] logical sessions: {len(groups):,}", flush=True)

    # Decide session emission order: stable by prefix for reproducibility
    prefixes = sorted(groups.keys())
    rng.shuffle(prefixes)  # avoid alphabetical bias when truncating

    gap_sample = make_gap_sampler(args)

    records = []                     # final emitted rows
    emitted_session_starts = []      # for assigning arrival_time_ns
    emitted_sids = []                # in emission order

    n_dropped_empty = 0
    n_dropped_no_pair = 0
    n_dropped_short = 0
    n_truncated_by_len = 0
    n_truncated_by_turns = 0

    for prefix in tqdm(prefixes, desc="sessions"):
        if args.max_sessions and len(emitted_sids) >= args.max_sessions:
            break
        if args.max_requests and len(records) >= args.max_requests:
            break

        splits = sorted(groups[prefix], key=lambda x: x[0])
        merged = []
        for _, convs in splits:
            merged.extend(convs)
        if not merged:
            n_dropped_empty += 1
            continue
        pairs = extract_pairs(merged)
        if not pairs:
            n_dropped_no_pair += 1
            continue

        # Build turns in token space
        ctx_ids = []
        session_turns = []  # list of (in_ids, out_ids)
        truncated_by_len = False
        for turn_idx, (u_text, a_text) in enumerate(pairs):
            if args.max_turns and turn_idx >= args.max_turns:
                n_truncated_by_turns += 1
                break
            u_ids = tok(u_text, add_special_tokens=False).input_ids
            a_ids = tok(a_text, add_special_tokens=False).input_ids
            if not u_ids:
                u_ids = [0]
            if not a_ids:
                a_ids = [0]
            if args.max_output_length and len(a_ids) > args.max_output_length:
                a_ids = a_ids[: args.max_output_length]
            in_ids = ctx_ids + u_ids
            if args.max_input_length and len(in_ids) > args.max_input_length:
                truncated_by_len = True
                break
            if args.max_kv_length and (len(in_ids) + len(a_ids)) > args.max_kv_length:
                truncated_by_len = True
                break
            session_turns.append((in_ids, a_ids))
            ctx_ids = in_ids + a_ids

        if truncated_by_len:
            n_truncated_by_len += 1

        if len(session_turns) < args.min_turns:
            n_dropped_short += 1
            continue

        # Compose arrival timing
        sid = prefix  # use shared prefix as session id
        emitted_sids.append(sid)
        # Will assign session start time after we know total #sessions; for now stash turn timing offsets.
        # We materialize records here with a placeholder arrival, then patch up after start times known.
        per_turn_gaps = [0]
        for _ in range(1, len(session_turns)):
            per_turn_gaps.append(gap_sample(rng))

        cum_arrival = 0
        for turn_idx, ((in_ids, a_ids), gap) in enumerate(zip(session_turns, per_turn_gaps)):
            cum_arrival += gap
            rec = {
                "session_id": sid,
                "turn_idx": turn_idx,
                "input_toks": len(in_ids),
                "output_toks": len(a_ids),
                "arrival_time_ns": int(cum_arrival),  # offset within session for now
                "intra_session_gap_ns": int(gap),
                "input_tok_ids": list(in_ids),
                "output_tok_ids": list(a_ids),
            }
            records.append(rec)
            if args.max_requests and len(records) >= args.max_requests:
                break
        emitted_session_starts.append(None)  # filled below

    # Assign Poisson session start times and patch arrival_time_ns
    starts = session_start_times(len(emitted_sids), args.session_rate, args.seed)
    sid_to_start = {sid: starts[i] for i, sid in enumerate(emitted_sids)}
    for r in records:
        r["arrival_time_ns"] = int(r["arrival_time_ns"] + sid_to_start[r["session_id"]])

    # Final sort: contiguous per session, by turn_idx
    records.sort(key=lambda r: (sid_to_start[r["session_id"]], r["session_id"], r["turn_idx"]))

    # Self-check
    if not args.no_self_check:
        by_sid = defaultdict(list)
        for r in records:
            by_sid[r["session_id"]].append(r)
        fail = []
        for sid, lst in by_sid.items():
            lst.sort(key=lambda r: r["turn_idx"])
            for i, r in enumerate(lst):
                if r["turn_idx"] != i:
                    fail.append(f"{sid}: turn_idx not contiguous at pos {i} (got {r['turn_idx']})")
                    break
            for i in range(1, len(lst)):
                prev = lst[i - 1]
                cur = lst[i]
                expected_prefix = prev["input_tok_ids"] + prev["output_tok_ids"]
                if cur["input_tok_ids"][: len(expected_prefix)] != expected_prefix:
                    fail.append(f"{sid}: turn {i} input is NOT prefix-extension of turn {i-1} (in+out)")
                    break
        if fail:
            print("[FAIL] invariants violated:", file=sys.stderr)
            for f in fail[:10]:
                print(f"  {f}", file=sys.stderr)
            sys.exit(2)

    # Write JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Meta + debug
    input_lens = [r["input_toks"] for r in records]
    output_lens = [r["output_toks"] for r in records]
    by_sid = defaultdict(list)
    for r in records:
        by_sid[r["session_id"]].append(r)
    turns_per_session = [len(v) for v in by_sid.values()]
    # Two per-turn-transition (N>=1) reuse measures, both turn-sampled:
    #   input_only : input_toks[N-1] / input_toks[N]
    #                -> fraction of turn N's prompt that overlaps the PREVIOUS PROMPT only.
    #                   Conservative lower bound (matches swe_multiturn meta.json def).
    #   incl_output: (input_toks[N-1] + output_toks[N-1]) / input_toks[N]
    #                -> fraction already in KV cache after turn N-1 (prev prompt + decoded
    #                   output). This is the real reusable-prefix fraction the radix tree sees,
    #                   since output_tok_ids[N-1] is carried verbatim into input_tok_ids[N].
    prefix_share_input_only = []
    prefix_share_incl_output = []
    for sid, lst in by_sid.items():
        lst.sort(key=lambda r: r["turn_idx"])
        for i in range(1, len(lst)):
            cur_in = max(lst[i]["input_toks"], 1)
            prev_in = lst[i - 1]["input_toks"]
            prev_out = lst[i - 1]["output_toks"]
            prefix_share_input_only.append(prev_in / cur_in)
            prefix_share_incl_output.append((prev_in + prev_out) / cur_in)

    def pct(xs, p):
        return float(np.percentile(np.asarray(xs), p)) if xs else 0.0

    meta = {
        "tokenizer": args.tokenizer,
        "input_file": str(in_path),
        "config": {
            "max_sessions": args.max_sessions,
            "max_requests": args.max_requests,
            "max_turns": args.max_turns,
            "min_turns": args.min_turns,
            "max_input_length": args.max_input_length,
            "max_output_length": args.max_output_length,
            "max_kv_length": args.max_kv_length,
            "session_rate": args.session_rate,
            "gap_dist": args.gap_dist,
            "gap_mean_s": args.gap_mean_s,
            "gap_sigma": args.gap_sigma,
            "gap_max_s": args.gap_max_s,
            "seed": args.seed,
        },
        "counts": {
            "num_sessions": len(by_sid),
            "num_requests": len(records),
            "dropped_empty": n_dropped_empty,
            "dropped_no_pair": n_dropped_no_pair,
            "dropped_short": n_dropped_short,
            "sessions_truncated_by_length": n_truncated_by_len,
            "sessions_truncated_by_turn_cap": n_truncated_by_turns,
        },
        "turns_per_session": {
            "mean": float(np.mean(turns_per_session)) if turns_per_session else 0,
            "p50": pct(turns_per_session, 50),
            "p95": pct(turns_per_session, 95),
            "max": int(max(turns_per_session)) if turns_per_session else 0,
        },
        "input_toks": {
            "mean": float(np.mean(input_lens)) if input_lens else 0,
            "p50": pct(input_lens, 50),
            "p95": pct(input_lens, 95),
            "p99": pct(input_lens, 99),
            "max": int(max(input_lens)) if input_lens else 0,
        },
        "output_toks": {
            "mean": float(np.mean(output_lens)) if output_lens else 0,
            "p50": pct(output_lens, 50),
            "p95": pct(output_lens, 95),
            "max": int(max(output_lens)) if output_lens else 0,
        },
        # Reuse ratios (turn-sampled over all N>=1 transitions). See computation comment above.
        "prefix_share_ratio_mean": float(np.mean(prefix_share_input_only)) if prefix_share_input_only else 0.0,
        "kv_reuse_ratio_input_only_mean": float(np.mean(prefix_share_input_only)) if prefix_share_input_only else 0.0,
        "kv_reuse_ratio_incl_output_mean": float(np.mean(prefix_share_incl_output)) if prefix_share_incl_output else 0.0,
        "kv_reuse_ratio_note": "input_only = input[N-1]/input[N] (lower bound, prev prompt only); "
                               "incl_output = (input[N-1]+output[N-1])/input[N] (real reusable prefix, "
                               "since prev output is carried verbatim into next input).",
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    debug_path = out_path.with_suffix(".debug.jsonl")
    with open(debug_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "session_id": r["session_id"],
                "turn_idx": r["turn_idx"],
                "input_toks": r["input_toks"],
                "output_toks": r["output_toks"],
                "arrival_time_ns": r["arrival_time_ns"],
                "intra_session_gap_ns": r["intra_session_gap_ns"],
            }) + "\n")

    print(f"[done] {len(records)} requests / {len(by_sid)} sessions -> {out_path}")
    print(f"[done] meta  -> {meta_path}")
    print(f"[done] debug -> {debug_path}")
    print(f"[stats] turns/session  mean={meta['turns_per_session']['mean']:.2f} "
          f"p95={meta['turns_per_session']['p95']:.0f} max={meta['turns_per_session']['max']}")
    print(f"[stats] input_toks     mean={meta['input_toks']['mean']:.1f} "
          f"p95={meta['input_toks']['p95']:.0f} max={meta['input_toks']['max']}")
    print(f"[stats] output_toks    mean={meta['output_toks']['mean']:.1f} "
          f"p95={meta['output_toks']['p95']:.0f} max={meta['output_toks']['max']}")
    print(f"[stats] kv_reuse       input_only={meta['kv_reuse_ratio_input_only_mean']:.3f} "
          f"incl_output={meta['kv_reuse_ratio_incl_output_mean']:.3f}")
    print(f"[stats] dropped empty/no-pair/short = "
          f"{n_dropped_empty}/{n_dropped_no_pair}/{n_dropped_short}; "
          f"truncated len/turns = {n_truncated_by_len}/{n_truncated_by_turns}")


if __name__ == "__main__":
    main()
