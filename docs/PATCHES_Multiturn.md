# LLMServingSim2 Patches for Multi-turn Trace Support

Minimal diff applied to upstream to run `swe_multiturn` traces correctly.

## 1. `inference_serving/request.py`

Add per-turn session fields to `Request.__init__`:

```
session_id, turn_idx, intra_session_gap_ns, original_arrival
```

`arrival` is mutated at runtime to the effective admission time; the JSONL
value is preserved in `original_arrival`.

## 2. `inference_serving/router.py`

- `Router.__init__`: create `self.session_turns = {}` and inject reference
  into every scheduler (`s.router_session_turns = self.session_turns`).
- `Router.generate`:
  - Sort the JSONL by `arrival_time_ns` before iterating.
  - Parse `session_id`, `turn_idx`, `intra_session_gap_ns`.
  - For turn N>0: pass arrival sentinel `1<<62` so the req is treated as
    "not yet arrived" until `add_done(predecessor)` releases it.
  - Pass session metadata to `add_request`; register the returned req in
    `session_turns[sid][turn_idx] = (req, sched)`.
  - Save JSONL arrival on `req.original_arrival` for later release.

## 3. `inference_serving/scheduler.py`

- `Scheduler.__init__`: add `self.router_session_turns = {}` slot
  (Router overwrites it with the shared dict).
- `add_request`: signature gains `session_id`, `turn_idx`,
  `intra_session_gap_ns`. Uses `bisect.insort(... key=lambda r: r.arrival)`
  to keep `self.request` arrival-sorted. Returns the new `Request`.
- `add_decode`: also uses `bisect.insort` for the same invariant.
- `add_done`: after a req moves to `self.done`, look up
  `router_session_turns[sid][turn_idx+1]`, set
  `child.arrival = max(child.original_arrival, finish + child.intra_session_gap_ns)`,
  `remove` from owning scheduler's request list, and `bisect.insort` back.

The admission filter and `self.request[0].arrival > current` short-circuit
are unchanged — `req.arrival` is now the single source of truth.

## 4. `inference_serving/memory_model.py`

After computing `mem_for_kv`, raise if it cannot hold one max-sized batch
of KV: `self.get_kv(max_num_batched_tokens) > mem_for_kv`. Error message
suggests reducing `--max-num-batched-tokens`, increasing NPU memory, or
raising TP.

## 5. `inference_serving/controller.py`

- `BypassController.submit_event` gains `is_timer=False`. When `True`, the
  event uses sentinel iteration `-1` and does NOT bump the per-NPU
  iteration counter. Without this, filler wakeup timers desync the
  `iteration - 1 == batch_id` mapping in `Scheduler.add_done`.
- `parse_output` regex now accepts negative iteration ids
  (`iteration (-?\d+)`).

## 6. `main.py`

- Bypass-mode batch submission: emit one completion event per NPU in the
  instance (`for offset in range(instance["npu_num"]): submit_event(..., sys+offset)`),
  not just `sys`. Required because `Scheduler.add_done` waits for both
  `start_npu` and `start_npu + npu_num - 1` to appear in `batch.end`
  before processing.
- `elif new_req == None` branch:
  - Skip the wakeup-timer dance entirely for non-start NPUs (they are
    driven by batch-completion events from `start_npu`).
  - Filler timer uses `is_timer=True`.
- (The earlier-added `get_next_admissible_time` call has been reverted
  to `get_next_arrival_time` after the arrival-mutation refactor.)
