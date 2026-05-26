"""Synthesize a tiny multi-turn trace for quick dependency-gating verification.

Produces 3 sessions x 3 turns = 9 small requests. Each session's turn N input
is a strict prefix-extension of turn N-1. arrival_time_ns intentionally set
so that turns 1+ of a session would arrive BEFORE the predecessor finishes if
the simulator did not gate — the simulator's session_state hook must enforce
the dependency.
"""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "smoke_multiturn.jsonl"

records = []
NUM_SESSIONS = 3
NUM_TURNS = 3
BASE_INPUT_LEN = 32   # tokens per turn-0
PER_TURN_GROWTH = 16  # added each turn
OUTPUT_LEN = 8

# Stagger session starts very close together so all turn-0s land near t=0.
for s in range(NUM_SESSIONS):
    sid = f"smoke-sess-{s}"
    prev_ids = []
    for t in range(NUM_TURNS):
        # Build cumulative input IDs (turn N strictly prefix-extends turn N-1).
        if t == 0:
            ids = [1000 + s * 100 + i for i in range(BASE_INPUT_LEN)]
        else:
            new_ids = [2000 + s * 100 + t * 50 + i for i in range(PER_TURN_GROWTH)]
            ids = prev_ids + new_ids
        prev_ids = ids

        # Output IDs distinct from input space.
        out_ids = [5000 + s * 100 + t * 10 + i for i in range(OUTPUT_LEN)]

        # arrival_time_ns: deliberately pack all 9 requests near t=0
        # (impossible if gating works). turn 0 gets staggered slightly per session.
        arrival_ns = s * 1_000_000  # 1ms per session start
        intra_gap_ns = 5_000_000 if t > 0 else 0  # 5ms intra-session gap

        records.append({
            "input_toks":           len(ids),
            "output_toks":          len(out_ids),
            "arrival_time_ns":      arrival_ns,
            "input_tok_ids":        ids,
            "output_tok_ids":       out_ids,
            "session_id":           sid,
            "turn_idx":             t,
            "intra_session_gap_ns": intra_gap_ns,
        })

# Global sort by arrival (ties broken so all turn-0s come first).
records.sort(key=lambda r: (r["arrival_time_ns"], r["turn_idx"], r["session_id"]))

out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"wrote {len(records)} records to {out}")
for r in records:
    print(f"  arr={r['arrival_time_ns']:>10} sid={r['session_id']} turn={r['turn_idx']} in={r['input_toks']} out={r['output_toks']} gap={r['intra_session_gap_ns']}")
