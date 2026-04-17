# Bug Fix History

## Fixed

### 1. `avail_size()` double unit conversion (bytes × bytes/token)

> Commit: `2ea4743` (2026-03-30) — Details: [BUGFIX_prefix_caching.md §Bug 1](./BUGFIX_prefix_caching.md#bug-1-avail_size-double-unit-conversion-bytes--bytestoken)

### 2. `apply_kv_cache_events` — free/alloc order

> Commit: `2ea4743` (2026-03-30) — Details: [BUGFIX_prefix_caching.md §Bug 2](./BUGFIX_prefix_caching.md#bug-2-apply_kv_cache_events--alloc-before-free)

### 3. Radix tree chained hashing (fixes `_npu_cache_hashtolen` duplicate count)

> Commit: `2ea4743` (2026-03-30) — Details: [BUGFIX_prefix_caching.md §Bug 3](./BUGFIX_prefix_caching.md#bug-3-_npu_cache_hashtolen--double-count-on-duplicate-page-hashes)

### 4. Second-tier prefix cache (CPU/CXL) block size hardcoded

> Commit: `2ea4743` (2026-03-30)

**File**: `inference_serving/memory_model.py` — `__init__()` RadixCache construction

**Symptom**: With CPU/CXL storage, block-level management ran at token granularity (page_size=1), mismatching the NPU block granularity.

**Cause**: NPU prefix cache uses `page_size=self.block_size` (default 16), but second-tier (CPU/CXL) storage was hardcoded to `page_size=1`.

**Fix**: `page_size=1` → `page_size=self.block_size`

### 5. Empty batch guard

> Commit: `c1b61e4` (2026-03-30)

**File**: `inference_serving/trace_generator.py`

**Symptom**: Empty batches (`total_len=0`, `len(requests)=0`) reaching trace generation raised errors.

**Fix**: Added a guard to log and skip empty batches.

### 6. Input size validation

> Commit: `0a76bdf` (2026-03-30)

**File**: `inference_serving/scheduler.py`

**Symptom**: Requests with `req.input > max_num_batched_tokens` caused an infinite scheduling loop.

**Fix**: Validate at `add_request()` entry and raise `ValueError` with a clear message.

### 7. Cross-tree reference crash during prefix cache eviction

> Fix date: 2026-04-01

**File**: `inference_serving/memory_model.py`

**Symptom**: `AttributeError: 'NoneType' object has no attribute 'lock_ref'` when `--enable-prefix-caching` + `--prefix-storage CXL` (or `CPU`) + KV cache eviction of a decoding request under GPU memory pressure.

**Cause**: `storage_cache_evicted_req()` passed a node from the second_tier (CXL/CPU) radix tree into the NPU radix tree's `inc_lock_ref()`. The two trees are separate instances, so the parent chain never reached NPU root and crashed at the storage tree root's `parent = None`.

**Fix**: Locate the NPU node directly via `match_prefix()` on the NPU radix tree and lock it:

```python
token_ids = (req.input_hash_ids + req.output_hash_ids)[:req.input]
npu_result = self.npu_prefix_cache.match_prefix(token_ids)
if npu_result.last_device_node is not None:
    self.npu_prefix_cache.inc_lock_ref(npu_result.last_device_node)
```


### 8. CXL prefix cache eviction unsupported + hardcoded memory usage in logs

> Fix date: 2026-04-07

**Files**: `inference_serving/memory_model.py`, `main.py`

**Symptom 1**: With `--prefix-storage CXL`, when CXL ran out of space the simulator raised an error instead of dropping overflow KV:
`RuntimeError: Trying to evict prefix cache to unsupported device Device.CXL`

**Cause**: `evict_prefix_cache()` was missing the `Device.CXL` branch; only `Device.CPU` was handled.

**Fix**: `elif device == Device.CPU:` → `elif device == Device.CPU or device == Device.CXL:`

**Symptom 2**: CXL/CPU memory usage in logs diverged from reality (0.3% displayed, actual 62%).

**Cause**: `total_size() * 131072` was hardcoded. `total_size()` returns token count, and a fixed constant was used instead of the model-specific kv_size. The `cxl_util` calculation was also missing `* 100`.

**Fix**: Use `total_memory_usage()` + add `* 100` to `cxl_util` (3 sites).

### 9. `storage_cache_evicted_req` NPU lock leak

> Fix date: 2026-04-07

**File**: `inference_serving/memory_model.py`

**Symptom**: With prefix_storage enabled, evicted requests' NPU radix tree nodes retained a permanent lock, excluding the prefix from eviction candidates.

**Cause**: After storing the evicted req's KV cache to CPU/CXL, `storage_cache_evicted_req()` called `inc_lock_ref()` on the NPU radix tree but did not save the node in `req.npu_last_node`. Since `dec_lock_ref()` was never called afterwards, the lock leaked.

**Fix**: Use CPU/CXL (`second_tier_prefix_cache`) lock instead of NPU lock; it now pairs correctly with `unlock_prefix(req, Device.CPU)` at re-scheduling.

```python
# Before (NPU lock — leaked)
self.npu_prefix_cache.inc_lock_ref(new_last_node)

# After (CPU/CXL lock — paired with unlock at re-scheduling)
self.second_tier_prefix_cache.inc_lock_ref(new_last_node)
```

### 10. Prefix caching scheduling deadlock — batch_len KV capacity cap

> Fix date: 2026-04-13 — Details: [BUGFIX_scheduling_deadlock.md](./BUGFIX_scheduling_deadlock.md)

### 11. Activation memory reservation & npu_used simplification

> Fix date: 2026-04-13 — Details: [BUGFIX_scheduling_deadlock.md §Refactoring](./BUGFIX_scheduling_deadlock.md#refactoring-inference_servingmemory_modelpy)

### 12. Prefill attention kv_cache_size reflects prefix cache hit

> Fix date: 2026-04-17

**File**: `inference_serving/scheduler.py` lines 193, 474

**Symptom**: Even on a prefix cache hit, `prefill_k_list` was always set to 0, so `_make_attn_db_key` queried with `kv_cache_size=0` → attention latency underestimated.

**Cause**: KV for prefix cache hit tokens is already in HBM, so attention must compute Q(miss) × K(hit + miss). `prefill_k_list.append(0)` dropped the hit info.

**Fix**: `prefill_k_list.append(req.prefix_cache_hit)`

**Example** (input=1024, hit=512):

| | Before | After |
|---|---|---|
| prefill_k_list | `[0]` | `[512]` |
| DB lookup key | `(0, 512)` → 6,066ns | `(512, 512)` → 23,683ns |

### 13. Prefill attention kv_cache_size per-request lookup

> Fix date: 2026-04-17

**File**: `inference_serving/trace_generator.py` `_make_attn_db_key()` lines 2320–2331

**Symptom**: Multiple prefill requests in a batch had their `kv_cache_size` summed (`sum(prefill_k_list)`) into a single lookup key. Since per-request attention is independent, collapsing them into one key produced inaccurate latency.

**Fix**: Build per-request `(kv_i, q_i)` keys and look them up individually, then sum.

```python
# After
prefill_keys = []
for i in range(batch.num_prefill):
    kv = batch.prefill_k_list[i]
    kv = ((kv + _gran - 1) // _gran) * _gran
    q = batch.prefill_q_list[i]
    q = (ceil(q / _chunk_gran)) * _chunk_gran
    prefill_keys.append((kv, q))
```

**Example** (2 requests in a batch):

| Request | q_len | kv_cache |
|---|---|---|
| A | 512 | 512 |
| B | 256 | 1024 |

| | Before | After |
|---|---|---|
| Lookup | `(1536, chunk)` once | `(512, 512)` + `(1024, 256)` each, then sum |

## Unfixed (Known Issues)

### 1. `lock_prefix` timing for init (runs before feasibility check)

> Details: [BUGFIX_scheduling_deadlock.md §Residual issues](./BUGFIX_scheduling_deadlock.md#residual-issues)

**File**: `inference_serving/scheduler.py` — `schedule_with_prefix()` lines 280–297

`lock_prefix` runs before the feasibility check (`temp_len` computation), exhausting `evictable`.
Full fix: run `prefix_match` first and defer `lock_prefix` until the batch is finalized.
For now, §10 batch_len cap reduces lock count and works around it **indirectly**.

### 2. `batch_len = len(gen_req) = 0` deadlock

> Details: [BUGFIX_scheduling_deadlock.md §Residual issues](./BUGFIX_scheduling_deadlock.md#residual-issues)

When the eviction loop evicts every decode request, `batch_len = len(gen_req) = 0`, so init requests are never examined. §10 batch_len cap reduces frequency but does not fully eliminate it.
