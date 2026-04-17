# Prefix Caching Scheduling Deadlock Fix Notes

Written: 2026-04-13
Target simulation: `--enable-prefix-caching`, any workload with a large number of init (prefill) requests queued simultaneously (e.g., high arrival rate, batch-mode submission, long-context datasets such as LVEval).

---

## Summary

Fix for a bug where, when many init requests are queued and the KV cache is nearly full, the scheduler fails to form a new batch — throughput drops to 0 and the simulator enters a permanent deadlock. The bug is **not dataset-specific**; it manifests in any scenario that piles up init requests faster than they can be scheduled.

---

## Symptom

- `prompt throughput: 0.0`, `generation throughput: 0.0` for tens to hundreds of seconds
- `inflight 0 reqs`, `waiting` keeps growing
- KV Cache at 99.9%+ usage

## Diagnostic log

```
[DIAG] temp_len=0 (consecutive=20) batch=924(init=923,dec=1)
       useable=1003.2MB(avail=3.2+evict=1000.0)
       prot=98288_tok evict=4000_tok kv_for_1=1820.0MB
```

---

## Root cause analysis

### Stage 1: evicted decode's full-reload cost > available space

In the eviction loop, when a decode request is evicted:
- `unlock_prefix` → `erase_prefix_info` → `npu_cache_hit = 0`
- Next scheduling makes `get_block_kv` compute a full reload (hit=0)
- Example: input=7000 → **kv_for_1 = 1820MB**

But `useable` is only 1003MB → cannot schedule even 1 request.

### Stage 2: why useable is small — excessive `lock_prefix` on init

```
On schedule_with_prefix entry:
  batch_req = 900+ waiting requests (900 init + 1 decode)

  for req in batch_req:        ← iterates over all 900
      if req.is_init:
          prefix_match(req)
          lock_prefix(req)     ← locks the matched path in the radix tree

Result: 900 init requests lock most of the radix tree
  → prot = 98288 tokens (of 102288 total)
  → evictable = 4000 tokens (1000MB) left
  → useable = avail(3.2MB) + evictable(1000MB) = 1003.2MB
```

### Stage 3: deadlock mechanism

```
tick N:
  1. 900 init lock_prefix → evictable exhausted
  2. useable(1003MB) < kv_for_1(1820MB) → temp_len=0
  3. eviction loop: evict 1 decode (unlock) → small evictable bump
  4. gen_req=[] → rollback (init unlock) → return None

tick N+1:
  1. 900+ init lock_prefix → evictable exhausted again
  2. same failure → return None
  ... loops forever
```

The rollback unlocks init, but the next tick locks them again — so the recovery has no lasting effect.

---

## Fix

### File: `inference_serving/scheduler.py` — `schedule_with_prefix()`

**Cap `batch_len` based on KV capacity** to prevent excessive `lock_prefix`.

```python
# Before
batch_req = batch_req[:batch_len]
# → batch_len = 900+ (every waiting request)
# → 900 init all call lock_prefix → evictable exhausted

# After
batch_req = batch_req[:batch_len]

# Cap batch_len so that total estimated KV does not exceed KV capacity
_kv_budget = self.memory.mem_for_kv
_kv_accum = 0
for i in range(batch_len):
    _kv_accum += self.memory.get_kv(batch_req[i].input)
    if _kv_accum > _kv_budget:
        batch_len = max(i, 1)
        batch_req = batch_req[:batch_len]
        break
```

**Estimation method:**
- Use `get_kv(req.input)` (full KV size) for every request
- When cumulative KV > total KV capacity (`mem_for_kv`), cap `batch_len`
- Guarantee at least 1 (`max(i, 1)`)

**Effect:**
- Only include requests that fit within KV capacity
- Fewer `lock_prefix` calls → `evictable` preserved
- Evicted decode regains schedulable space

### Refactoring: `inference_serving/memory_model.py`

**Activation memory reservation & `npu_used` simplification:**

If the KV cache claims the entire NPU memory, there is no room for activation (intermediate tensors), which can stall scheduling. Reserve activation memory upfront based on `max_num_batched_tokens` (vLLM/SGLang style).

```python
# npu_used tracks weight + activation + KV total; limit is npu_mem
self.npu_used = self.weight + self.activation_reserve
self.mem_for_kv = self.npu_mem - self.weight - self.activation_reserve
# Check: npu_used + size > npu_mem
```

- Removed `npu_kv_limit`; use `npu_mem` directly
- Renamed `free_weight()` → `free_weight_activation()`

---

## Residual issues

### init's `lock_prefix` is the root cause

The `batch_len` cap is an **indirect** fix. The structural problem is that `lock_prefix` runs before the feasibility check, exhausting `evictable`.

Proper fix: run `prefix_match` first (for hit computation only) and defer `lock_prefix` until after the batch is finalized. The current fix only reduces lock count via `batch_len` to work around the issue.

### batch_len = len(gen_req) = 0 issue

After the eviction loop evicts every decode, `batch_len = len(gen_req) = 0`, so init requests are never examined. The `batch_len` cap reduces frequency but does not fully eliminate it.

---

## Reproduction command (one example)

Any workload that queues many init requests simultaneously under memory pressure will reproduce the deadlock. The following LVEval command is one example:

```bash
python main.py \
  --dataset dataset/lveval_hotpotwikiqa_mixup_16k_llama3-8b_rate100_rep5.jsonl \
  --max-num-batched-tokens 50000 \
  --num-req 100 \
  --output output_lveval_prefix.csv \
  --enable-prefix-caching \
  --log-level WARNING
```
