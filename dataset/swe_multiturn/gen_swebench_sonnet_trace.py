"""
swebench + claude-sonnet-4-6 multi-turn trace → LLMServingSim2 JSONL.

For each session, every turn's `input` (a list of OpenAI-style messages) is a
strict prefix-extension of the prior turn (verified in
data/swebench_sonnet_session_prefix_check.csv: 110/110 sessions). This script
preserves that property at the token level by tokenizing each appended message
incrementally and concatenating, then writes a JSONL with session_id/turn_idx/
intra_session_gap_ns fields so LLMServingSim2's modified scheduler can enforce
strict dependency: turn N+1 admission is gated on turn N completion.

Output JSONL row (one per turn):
  {
    "input_toks":           int,
    "output_toks":          int,
    "arrival_time_ns":      int,
    "input_tok_ids":        [int, ...],
    "output_tok_ids":       [int, ...],
    "session_id":           str,
    "turn_idx":             int,                # 0-based per session
    "intra_session_gap_ns": int                 # pre_gap (s) -> ns, 0 for turn 0
  }

Default tokenizer: Qwen/Qwen3.5-122B-A10B. GPT-2 fallback if unavailable.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        return model_name, tok
    except Exception as e:
        print(f"[tokenizer] Failed to load {model_name} ({e}), falling back to gpt2")
        return "gpt2", AutoTokenizer.from_pretrained("gpt2")


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()


def _norm_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    # tool_calls / structured content -> stable JSON
    try:
        return json.dumps(x, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        return str(x)


def serialize_msg(msg: dict) -> str:
    """Stable per-message text for tokenization. Same dict -> same string."""
    parts = [f"<|{msg.get('role', 'unknown')}|>"]
    c = msg.get("content", "")
    if c is not None and (not hasattr(c, "__len__") or len(c) > 0):
        parts.append(_norm_str(c))
    tc = msg.get("tool_calls")
    if tc is not None and hasattr(tc, "__len__") and len(tc) > 0:
        parts.append(_norm_str(list(tc) if isinstance(tc, np.ndarray) else tc))
    tcid = msg.get("tool_call_id")
    if tcid:
        parts.append(f"[tcid:{tcid}]")
    name = msg.get("name")
    if name:
        parts.append(f"[name:{name}]")
    return "\n".join(parts)


def msg_cache_key(msg: dict) -> tuple:
    return (
        msg.get("role", ""),
        _hash(_norm_str(msg.get("content", ""))),
        _hash(_norm_str(msg.get("tool_calls"))) if msg.get("tool_calls") is not None and hasattr(msg.get("tool_calls"), "__len__") and len(msg.get("tool_calls")) > 0 else "",
        msg.get("tool_call_id") or "",
        msg.get("name") or "",
    )


def session_start_times(num_sessions: int, rate: float, seed: int) -> list:
    """Poisson-spaced session start times.

    First session always starts at t=0. Subsequent sessions are spaced by
    expovariate(rate) seconds. rate=0 → all sessions burst at t=0.
    """
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


def synth_output_ids(session_id: str, length: int, vocab_size: int) -> list:
    """Deterministic synthetic IDs for the last turn's output (no real text)."""
    out = []
    base = int(hashlib.sha1(f"{session_id}|final".encode()).hexdigest(), 16)
    # Stay in a high band to avoid colliding with realistic content tokens.
    band_lo = max(500, vocab_size - 5000)
    band_hi = vocab_size - 1
    span = band_hi - band_lo
    if span <= 0:
        span = max(1, vocab_size - 1)
        band_lo = 0
    for i in range(length):
        out.append(band_lo + ((base + i * 1315423911) % span))
    return out


def main():
    here = Path(__file__).resolve().parent
    default_parquet = (here / "../../../LMCache_Agentic_Traces/data/swebench_sonnet.parquet").resolve()
    default_output = here / "swebench_sonnet_qwen3.5-122b_rate2.jsonl"

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-parquet", default=str(default_parquet),
                    help=f"Filtered swebench+sonnet parquet (default: {default_parquet})")
    ap.add_argument("--output", default=str(default_output),
                    help=f"Output JSONL path (default: {default_output})")
    ap.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B",
                    help="HF tokenizer model name (default Qwen/Qwen3.5-122B-A10B; gpt2 fallback)")
    ap.add_argument("--rate", type=float, default=2.0,
                    help="Session-start Poisson rate (sessions/sec). "
                         "First session always at t=0; subsequent sessions Poisson-spaced. "
                         "0 = all sessions burst at t=0.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[load] reading {args.input_parquet} ...", flush=True)
    df = pd.read_parquet(args.input_parquet)
    print(f"[load] {len(df)} rows, {df['session_id'].nunique()} sessions", flush=True)

    tok_name, tok = load_tokenizer(args.model)
    print(f"[tokenizer] using {tok_name}", flush=True)
    vocab_size = tok.vocab_size if hasattr(tok, "vocab_size") and tok.vocab_size else 32000

    # Group preserving file order (already verified as turn order).
    sessions = []  # list of (session_id, df_subset)
    for sid, g in df.groupby("session_id", sort=False):
        sessions.append((sid, g.reset_index(drop=True)))

    starts = session_start_times(len(sessions), args.rate, args.seed)

    records = []
    debug_rows = []
    msg_id_cache = {}  # (cache_key) -> [token_ids]
    cache_hits = 0
    cache_misses = 0

    total_claude_olen = 0
    total_qwen_olen = 0
    per_turn_olen_diff = []

    for s_idx, (sid, g) in enumerate(sessions):
        prev_messages_len = 0
        prev_tok_ids = []
        prev_input_msgs = None  # cached list view of previous turn's input

        # Pre-extract appended assistant message text from turn N+1 for turn N output.
        turn_messages = [list(row["input"]) for _, row in g.iterrows()]
        n_turns = len(turn_messages)

        # Build per-turn output_tok_ids (for the previous turn, from this turn's new assistant msg).
        # For the final turn, synthesize.
        out_ids_per_turn = [None] * n_turns
        for t in range(n_turns - 1):
            prev_msgs = turn_messages[t]
            curr_msgs = turn_messages[t + 1]
            new_msgs = curr_msgs[len(prev_msgs):]
            asst = next((m for m in new_msgs if m.get("role") == "assistant"), None)
            if asst is not None:
                ids = tok.encode(serialize_msg(asst), add_special_tokens=False)
                if not ids:
                    ids = [0]
                out_ids_per_turn[t] = ids
            else:
                # No assistant turn appended — synthesize from output_length.
                olen = int(g.iloc[t]["output_length"]) or 1
                out_ids_per_turn[t] = synth_output_ids(f"{sid}#{t}", olen, vocab_size)

        # Final turn: synthesize from output_length.
        final_olen = int(g.iloc[-1]["output_length"]) or 1
        out_ids_per_turn[n_turns - 1] = synth_output_ids(f"{sid}#final", final_olen, vocab_size)

        arrival_ns = starts[s_idx]
        for t in range(n_turns):
            row = g.iloc[t]
            curr_msgs = turn_messages[t]
            # Incremental tokenize: only messages appended after prev_messages_len.
            new_msgs = curr_msgs[prev_messages_len:]
            new_ids = []
            for m in new_msgs:
                key = msg_cache_key(m)
                ids = msg_id_cache.get(key)
                if ids is None:
                    ids = tok.encode(serialize_msg(m), add_special_tokens=False)
                    if not ids:
                        ids = [0]
                    msg_id_cache[key] = ids
                    cache_misses += 1
                else:
                    cache_hits += 1
                new_ids.extend(ids)

            input_tok_ids = prev_tok_ids + new_ids
            input_toks = len(input_tok_ids)
            if input_toks == 0:
                input_tok_ids = [0]
                input_toks = 1

            # output_toks: prefer dataset's measured output_length (Claude tokenizer count).
            claude_olen = int(row["output_length"])
            qwen_olen = len(out_ids_per_turn[t])
            output_toks = claude_olen
            output_tok_ids = out_ids_per_turn[t]
            # Reconcile id list length with output_toks: pad/truncate.
            if len(output_tok_ids) < output_toks:
                pad = synth_output_ids(f"{sid}#pad{t}", output_toks - len(output_tok_ids), vocab_size)
                output_tok_ids = output_tok_ids + pad
            elif len(output_tok_ids) > output_toks:
                output_tok_ids = output_tok_ids[:output_toks]

            total_claude_olen += claude_olen
            total_qwen_olen += qwen_olen
            per_turn_olen_diff.append(qwen_olen - claude_olen)

            # arrival/intra_session_gap
            pre_gap_s = float(row["pre_gap"])
            intra_gap_ns = int(pre_gap_s * 1e9) if t > 0 else 0
            if t == 0:
                arrival_ns = starts[s_idx]
            else:
                # JSONL arrival is just a lower-bound hint; the simulator's
                # dependency gate enforces max(arrival, prev_completion + intra_gap)
                # at runtime, so no analytic exec-time estimate is needed here.
                arrival_ns = arrival_ns + intra_gap_ns

            rec = {
                "session_id": str(sid),
                "turn_idx": t,
                "input_toks": input_toks,
                "output_toks": output_toks,
                "arrival_time_ns": int(arrival_ns),
                "intra_session_gap_ns": intra_gap_ns,
                "input_tok_ids": list(input_tok_ids),
                "output_tok_ids": list(output_tok_ids),
            }
            records.append(rec)
            debug_rows.append({
                "session_id": str(sid),
                "turn_idx": t,
                "total_turns": n_turns,
                "input_toks": input_toks,
                "expected_prefix_hit": len(prev_tok_ids),
                "output_toks": output_toks,
                "qwen_output_toks": qwen_olen,
                "arrival_time_ns": int(arrival_ns),
                "intra_session_gap_ns": intra_gap_ns,
            })

            # Roll forward.
            prev_messages_len = len(curr_msgs)
            prev_tok_ids = input_tok_ids

        if (s_idx + 1) % 10 == 0 or s_idx + 1 == len(sessions):
            print(f"  [session] {s_idx + 1}/{len(sessions)} done", flush=True)

    # ===== Self-checks =====
    # Group records by session_id to verify prefix-extension and turn_idx contiguity.
    by_sid = defaultdict(list)
    for r in records:
        by_sid[r["session_id"]].append(r)

    fail = []
    for sid, lst in by_sid.items():
        lst.sort(key=lambda r: r["turn_idx"])
        for i, r in enumerate(lst):
            if r["turn_idx"] != i:
                fail.append(f"{sid}: turn_idx not contiguous ({r['turn_idx']} at pos {i})")
        for i in range(1, len(lst)):
            a, b = lst[i - 1]["input_tok_ids"], lst[i]["input_tok_ids"]
            if len(a) > len(b) or b[: len(a)] != a:
                fail.append(f"{sid}: turn {i} input is NOT prefix-extension of turn {i-1}")
                break

    if fail:
        print("[FAIL] Prefix/turn invariants violated:", file=sys.stderr)
        for f in fail[:10]:
            print(f"  {f}", file=sys.stderr)
        sys.exit(2)

    # Group by session_id for readability (all turns of a session contiguous,
    # ordered by turn_idx). Router.generate() re-sorts by arrival_time_ns when
    # adding to scheduler.request, so on-wire JSONL order does not affect
    # simulation correctness.
    records.sort(key=lambda r: (r["session_id"], r["turn_idx"]))

    # Per-session turn_idx contiguity check (must be 0..K-1 in order).
    last_turn_per_sid = {}
    for r in records:
        sid = r["session_id"]
        expected = last_turn_per_sid.get(sid, -1) + 1
        if r["turn_idx"] != expected:
            print(f"[FAIL] {sid}: turn order broken (got {r['turn_idx']}, expected {expected})", file=sys.stderr)
            sys.exit(2)
        last_turn_per_sid[sid] = r["turn_idx"]

    # Output paths.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    debug_path = out_path.with_suffix(".debug.jsonl")
    with open(debug_path, "w", encoding="utf-8") as f:
        for r in debug_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats.
    input_lens = [r["input_toks"] for r in records]
    output_lens = [r["output_toks"] for r in records]
    prefix_share = []
    for sid, lst in by_sid.items():
        lst.sort(key=lambda r: r["turn_idx"])
        for i in range(1, len(lst)):
            prefix_share.append(lst[i - 1]["input_toks"] / lst[i]["input_toks"])

    meta = {
        "tokenizer_model": tok_name,
        "vocab_size": int(vocab_size),
        "input_parquet": str(args.input_parquet),
        "num_sessions": len(by_sid),
        "num_requests": len(records),
        "session_start_rate_per_sec": args.rate,
        "seed": args.seed,
        "input_toks": {
            "min": int(min(input_lens)), "max": int(max(input_lens)),
            "mean": float(sum(input_lens) / len(input_lens)),
        },
        "output_toks": {
            "min": int(min(output_lens)), "max": int(max(output_lens)),
            "mean": float(sum(output_lens) / len(output_lens)),
        },
        "claude_vs_qwen_output_length": {
            "total_claude": int(total_claude_olen),
            "total_qwen": int(total_qwen_olen),
            "ratio_qwen_over_claude": float(total_qwen_olen / max(total_claude_olen, 1)),
        },
        "prefix_share_ratio_mean": float(sum(prefix_share) / len(prefix_share)) if prefix_share else 0.0,
        "msg_token_cache_hits": cache_hits,
        "msg_token_cache_misses": cache_misses,
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[done] {len(records)} requests -> {out_path}")
    print(f"[done] debug -> {debug_path}")
    print(f"[done] meta  -> {meta_path}")
    print(f"[stats] avg input_toks  = {meta['input_toks']['mean']:.1f}")
    print(f"[stats] avg output_toks = {meta['output_toks']['mean']:.1f}")
    print(f"[stats] prefix_share    = {meta['prefix_share_ratio_mean']:.3f}")
    print(f"[stats] msg cache hits/misses = {cache_hits}/{cache_misses}")
    print(f"[stats] Qwen/Claude output_length ratio = {meta['claude_vs_qwen_output_length']['ratio_qwen_over_claude']:.3f}")


if __name__ == "__main__":
    main()
