# SWE-bench Multi-turn Trace for LLMServingSim2

Multi-turn agentic SWE-bench trace converted from
[LMCache/Agentic_Traces](https://github.com/LMCache/Agentic_Traces) so it can
drive `LLMServingSim2` with realistic per-turn prefix-cache reuse and strict
turn-by-turn dependency enforcement.

The Claude-Sonnet-4.6 subset is included (110 sessions, 2,559 turns). Other
models can be added by re-filtering the upstream parquet shards.

## Get the trace

The 271 MB main JSONL is not tracked in git (GitHub 100 MB limit). Fetch it
from the Hugging Face Hub dataset repo before running the simulator:

```bash
pip install huggingface_hub
huggingface-cli download noddu/swe_multiturn-claude-trace-tokenized-qwen3.5 \
  swebench_sonnet_qwen3.5-122b_rate2.jsonl \
  --repo-type dataset \
  --local-dir dataset/swe_multiturn
```

Or regenerate it from the upstream parquet via `gen_swebench_sonnet_trace.py`
(see "Build pipeline" below).

## Files

| File | Purpose |
|---|---|
| `gen_swebench_sonnet_trace.py` | Main generator (parquet → JSONL + meta + debug). |
| `make_smoke_trace.py` | 9-request synthetic trace to verify dependency gating in seconds. |
| `swebench_sonnet_qwen3.5-122b_rate2.jsonl` | Simulator input. One JSON object per turn. |
| `swebench_sonnet_qwen3.5-122b_rate2.meta.json` | Summary stats (token-length distributions, tokenizer, prefix-share ratio). |
| `swebench_sonnet_qwen3.5-122b_rate2.debug.jsonl` | Per-turn metadata sidecar without the heavy `*_tok_ids` lists — for quick lookup / cross-referencing simulator output. |
| `smoke_multiturn.jsonl` | Generated synthetic trace (output of `make_smoke_trace.py`). |

## JSONL schema (simulator input)

One object per turn. Sessions are grouped (all turns of a session contiguous,
ordered by `turn_idx`) for human readability. The simulator's `Router.generate()`
re-sorts by `arrival_time_ns` internally — on-wire order does **not** affect
simulation correctness.

```json
{
  "session_id": "swebench__django__django-13568__claude",
  "turn_idx": 0,
  "input_toks": 3182,
  "output_toks": 196,
  "arrival_time_ns": 0,
  "intra_session_gap_ns": 0,
  "input_tok_ids": [..., ..., ...],
  "output_tok_ids": [..., ..., ...]
}
```

| Field | Type | Meaning |
|---|---|---|
| `session_id` | str | Original upstream session id. Same id = multi-turn conversation chain. |
| `turn_idx` | int (0-based) | Position within the session. Strictly contiguous 0..K-1 per session. |
| `input_toks` | int | Cumulative input length at this turn (system + all prior assistant/tool messages + current user). |
| `output_toks` | int | Model output length. **Claude's measured token count is preferred** (`pre_gap` data column from upstream); `output_tok_ids` is padded/truncated to match. |
| `arrival_time_ns` | int | Lower-bound admission hint. For turn 0 = session start; for turn N>0 = previous-turn `arrival_time_ns + intra_session_gap_ns`. Actual admission = `max(this, predecessor.end_time + intra_session_gap_ns)` enforced by the simulator at runtime. |
| `intra_session_gap_ns` | int | Real wall-clock gap between previous turn's completion and this turn's request (thinking + tool exec time, from upstream `pre_gap` × 1e9). 0 for turn 0. |
| `input_tok_ids` | list[int] | Tokenized via Qwen3.5 (per-message, concatenated → preserves token-level prefix-extension across turns). Used by simulator's radix tree. |
| `output_tok_ids` | list[int] | Tokenized assistant response. For turns 1..K-1 extracted from the next turn's appended `assistant` message; for the final turn synthesized deterministically (no real text exists in dataset). |

### Prefix-extension invariant (load-bearing)

For every session and every turn N>0:

```
input_tok_ids[N][:len(input_tok_ids[N-1])] == input_tok_ids[N-1]
```

Verified by `gen_swebench_sonnet_trace.py` self-check before write; abort on
violation. Enables LLMServingSim2's RadixCache to compute exact per-turn hit
length.

## Build pipeline

### Step 1 — Filter upstream parquet (one-time)

The upstream dataset bundles many models; we only keep Sonnet on swebench
sessions. Produces `swebench_sonnet.parquet` (2,559 rows, 110 sessions).
This file already exists under `../../../LMCache_Agentic_Traces/data/`.

```python
import pandas as pd, glob
dfs = []
for f in sorted(glob.glob("data/train-*.parquet")):
    df = pd.read_parquet(f)
    sub = df[df["session_id"].str.startswith("swebench__") & (df["model"] == "claude-sonnet-4-6")]
    if len(sub):
        dfs.append(sub)
out = pd.concat(dfs, ignore_index=True)
out.to_parquet("data/swebench_sonnet.parquet", index=False)
```

### Step 2 — Generate the JSONL trace

```bash
cd LLMServingSim2/dataset/swe_multiturn
python -X utf8 gen_swebench_sonnet_trace.py \
  --rate 2.0 \
  --seed 42
```

Defaults pick up the parquet from `../../../LMCache_Agentic_Traces/data/swebench_sonnet.parquet`
and write the trace + meta + debug into the current directory. Override with
`--input-parquet` / `--output` / `--model` / `--rate` / `--seed`.

Internally the generator:

1. **Groups by `session_id`** preserving file row order (verified upstream as
   turn order — all 110 sessions have strict monotonic input growth).
2. **Per-message tokenize + cache** — each message dict is serialized as
   `<|role|>\n{content}\n{tool_calls_json}\n[tcid:...]` and tokenized once
   with `tokenizer.encode(..., add_special_tokens=False)`. A per-process
   `(role, sha1(content), sha1(tool_calls), tool_call_id, name) → token_ids`
   cache amortizes repeated system prompts (~12KB) and tool prefaces.
3. **Incremental prefix extension** — turn N's `input_tok_ids` = turn N-1's
   `input_tok_ids` + tokens of the newly appended messages. This *constructs*
   the prefix-extension invariant rather than hoping BPE preserves it.
4. **Extracts output_tok_ids** — for turns 0..K-2, finds the appended
   assistant message in turn N+1's input and tokenizes it. For the final
   turn (no successor exists in dataset), synthesizes deterministic IDs of
   length `output_length` from the upstream column.
5. **Computes arrival/intra_session_gap** — turn 0 gets the Poisson session
   start time (`expovariate(rate)`, first session always t=0); turn N>0
   uses `arrival = prev_arrival + intra_session_gap_ns` where
   `intra_session_gap_ns = pre_gap_seconds × 1e9`. No analytic exec-time
   estimate is added; the simulator enforces the real `max(arrival,
   predecessor_end + intra_gap)` dynamically.
6. **Self-checks** — prefix-extension invariant per session, turn_idx
   contiguity 0..K-1, monotonic arrival within session.

### Step 3 — Run in LLMServingSim2

The required simulator modifications are already merged on this branch. From
the project root:

```bash
cd LLMServingSim2
python -X utf8 -u main.py \
  --cluster-config cluster_config/single_node_h100_qwen3-32b.json \
  --fp 16 --block-size 16 \
  --enable-prefix-caching \
  --dataset dataset/swe_multiturn/swebench_sonnet_qwen3.5-122b_rate2.jsonl \
  --output output/swebench_sonnet_run.csv \
  --num-req 2559 \
  --max-num-batched-tokens 70000 \
  --bypass-astrasim
```

Quick verification end-to-end (1.2s wall-clock):

```bash
python -X utf8 dataset/swe_multiturn/make_smoke_trace.py
python -X utf8 -u main.py \
  --cluster-config cluster_config/single_node_h100_qwen3-32b.json \
  --fp 16 --block-size 16 \
  --enable-prefix-caching \
  --dataset dataset/swe_multiturn/smoke_multiturn.jsonl \
  --output output/smoke_multiturn.csv \
  --num-req 9 --max-num-batched-tokens 2048 --bypass-astrasim
```

Expected smoke result: 9/9 done, prefix_cache_hit per turn = (0, 32, 48).

## Simulator-side modifications (already merged)

This trace cannot run on stock LLMServingSim2 — it requires the following
modifications. They are gated on `session_id` being present, so legacy
single-turn traces continue to work unchanged.

| File | Change |
|---|---|
| `inference_serving/request.py` | `Request.__init__` gets `session_id`, `turn_idx`, `intra_session_gap_ns` (defaults: None, 0, 0). |
| `inference_serving/router.py` | `Router.__init__` creates `self.session_state = {}` and injects the same dict into every scheduler so cross-instance turn completions are visible. `generate()` parses the new JSONL fields and sorts by `arrival_time_ns` before adding requests. |
| `inference_serving/scheduler.py` | `_session_admissible()` admission gate (predecessor must be in `done` + `effective = max(arrival, last_completion + intra_gap) ≤ current`). `add_done()` writes to `shared_session_state` when a request fully completes. `add_request`/`add_decode` use `bisect.insort` so `self.request` stays arrival-sorted regardless of insertion order. `get_next_admissible_time()` returns the earliest future admissible moment considering both arrival and dependency gate effective time — used by the bypass-mode wakeup timer to prevent deadlock when all queued reqs are dependency-gated. |
| `inference_serving/controller.py` | `BypassController.submit_event(..., is_timer=False)` — timer-only filler events use `is_timer=True` and a sentinel iteration `-1` so they do not advance the iteration counter (otherwise the next real batch's completion event desyncs from its `batch_id` in `add_done`). `parse_output` regex accepts negative iteration ids. |
| `main.py` | The `new_req == None` branch in the event loop submits filler timer events with `is_timer=True` and uses `get_next_admissible_time` (not `get_next_arrival_time`). |

## Statistics (default rate=2, seed=42)

From `swebench_sonnet_qwen3.5-122b_rate2.meta.json`:

| | |
|---|---|
| Sessions | 110 |
| Total turns | 2,559 |
| Mean turns per session | 23.3 |
| Mean `input_toks` per turn | 20,247.9 |
| Max `input_toks` | 65,960 |
| Mean `output_toks` per turn | 288.7 |
| Mean `intra_session_gap` | 0.94 s (max 41 s, source `pre_gap`) |
| Per-session prefix share ratio | 0.939 (avg `prev_turn_input / current_input`) |
| Qwen vs. Claude token-length ratio | 0.969 (Qwen counts within 3% of Claude's, in aggregate) |

`prefix_share_ratio ≈ 0.94` means turn N's prompt is on average 94% the same
tokens as turn N-1's prompt — large KV reuse opportunity. Real benefit
depends on RadixCache page alignment (block_size=16 by default).
