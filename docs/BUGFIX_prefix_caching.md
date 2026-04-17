# Prefix Caching Bug Analysis and Fix Notes

Written: 2026-03-27
Last updated: 2026-04-16
Target simulation: `--enable-prefix-caching`, LLaMA-3.1-8B on A6000, LVEval dataset

---

## Summary

Three bugs that cause OOM (`tried to load XMB but only YMB available`) when running long traces with prefix caching enabled — all fixed.

1. `avail_size()` double-converts units (bytes vs bytes/token), so eviction never triggers.
2. `apply_kv_cache_events` runs alloc before free, causing transient capacity overflow.
3. `_npu_cache_hashtolen` used **position-agnostic content-only hashes** as keys, so pages with the same hash got created at multiple radix tree nodes, and on erase only one of them was removed.

---

## Bug 1: `avail_size()` double unit conversion (bytes × bytes/token)

> Fix date: 2026-03-30 (commit `2ea4743`) — Fixed

**File**: `inference_serving/memory_model.py` — `avail_size()` method

### Symptom
OOM immediately after simulation start (~4s). Eviction never triggers.

### Cause
```python
# Before (buggy)
def avail_size(self, device):
    if device == Device.NPU:
        return self.npu_prefix_cache.avail_size() * self._bytes_per_token
        #      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^
        #      already returns bytes                 multiplies by bytes/token
        #      → bytes × (bytes/token) = meaningless huge value
```

`RadixCache.avail_size()` already returns `capacity - total_memory_usage()` in **bytes**.
Multiplying by `_bytes_per_token` again yields `bytes × (bytes/token)` — an absurdly large value.

Scheduler computes `evict_size = kv_need - avail_size(NPU)`. Because `avail_size` appears as trillions of bytes, `evict_size ≤ 0` → nothing evicted → OOM.

### Fix
```python
# After
def avail_size(self, device):
    if device == Device.NPU:
        return self.npu_prefix_cache.avail_size()  # already in bytes
```

---

## Bug 2: `apply_kv_cache_events` — alloc before free

> Fix date: 2026-03-30 (commit `2ea4743`) — Fixed

**File**: `inference_serving/memory_model.py` — `apply_kv_cache_events()` method

### Symptom
OOM even in batches with zero net memory change — i.e., when memory is nearly full and we evict and re-allocate the same amount in a single step.

### Cause
```python
# Before (wrong order)
if npu_byte_alloc > 0:
    self.allocate(npu_byte_alloc, Device.NPU)  # Alloc first → capacity overflow
if npu_byte_free > 0:
    self.free(npu_byte_free, Device.NPU)       # Free afterwards
```

When evict (free) and alloc coexist in the same batch, attempting alloc before free causes transient capacity overflow and raises `RuntimeError`.

### Fix
```python
# After (free first)
if npu_byte_free > 0:
    self.free(npu_byte_free, Device.NPU)
if npu_byte_alloc > 0:
    self.allocate(npu_byte_alloc, Device.NPU)
```

---

## Bug 3: `_npu_cache_hashtolen` — double count on duplicate page hashes

> Fix date: 2026-03-30 (commit `2ea4743`) — Fixed (chained hashing introduced)

**Files**: `inference_serving/radix_tree.py`, `inference_serving/memory_model.py`

### Symptom
Memory drift grows gradually during simulation, eventually leading to OOM.
Logs repeat two warnings:
```
WARNING  BlockStored duplicate hash -4638695184844254712 (prev_tlen=16, new_tlen=16)
WARNING  BlockRemoved unknown hash  -4638695184844254712 dict_size=15043
```

### Cause: content-only hashes collide across radix tree branches

The pre-fix code computed block hash purely from **page token content**:
```python
# Before — content-only hash (radix_tree.py, old version)
block_hash = hash(tuple(page_tokens))
```

LVEval reuses the same long document (e.g., Wikipedia articles) across many questions. If a 16-token pattern repeats within the document:

```
  system_prompt[0:16]   == system_prompt[256:272]  (identical token sequence)
  → hash(tokens[0:16])  == hash(tokens[256:272])

Two radix tree branches at different positions independently hold
tree nodes with the same hash:

  root
   └─ shared_prefix_node (tokens[0:256])
       ├─ [page H at pos 256] → branch_A ...
       └─ (a copy of page H from pos 0:16 reappears on another branch)
```

**Failure sequence**:

| Step | Event | hashtolen state | npu_used |
|------|-------|-----------------|----------|
| 1 | BlockStored(H): Node A stored | `[H → 16]` | +1KB |
| 2 | BlockStored(H): Node B stored (duplicate!) | `[H → 16]` (overwrite) | **+1KB again** (double count!) |
| 3 | BlockRemoved(H): Node A evicted | `{}` (pop ok) | -1KB |
| 4 | BlockRemoved(H): Node B evicted | `{}` (pop fails → 0) | **no change** (free missed!) |
| Result | | | npu_used +1KB residual (drift) |

Both A and B are evicted from the radix tree → `avail_size()` grows.
But npu_used is stuck at +1KB → `npu_mem - npu_used` shrinks.
→ `radix_avail > npu_avail` gap widens → next batch cannot evict enough → OOM.

### Fix: introduce chained hashing (supersedes initial refcount proposal)

Adopted the content-addressed chained hashing used by vLLM/SGLang.

```python
# After (radix_tree.py:575-601)
# Chained hashing: block_hash = hash(page_tokens + (parent_hash,))
# Identical token content at different prefix positions → different hashes.
parent_block_hash = node.parent.get_last_hash_value()
node.hash_value = []
for start in range(0, len(node.key), self.page_size):
    page_tokens = node.key[start : start + self.page_size]
    block_hash = hash(tuple(page_tokens) + (parent_block_hash,))
    node.hash_value.append(block_hash)
    self.kv_event_queue.append(
        BlockStored(
            block_hashes=[block_hash],
            parent_block_hash=parent_block_hash,
            token_ids=page_tokens,
            block_size=len(page_tokens),
            lora_id=None,
        )
    )
    parent_block_hash = block_hash
```

Each block hash depends on the parent chain, so identical token content at different positions in the tree produces **distinct hashes**. No key collision in `_npu_cache_hashtolen`, so double-count on store and missed free on remove are eliminated at the source.

**Side fixes**:
- `_split_node` now splits `hash_value` by page count into parent/child correctly
- `_record_store_event` root check fixed (`node != root_node` → `node == root_node`)
- Parent's last hash used as seed via new `get_last_hash_value()` (`radix_tree.py:99-103`)

### Status
- Chained hashing applied (`radix_tree.py:575-601`)
- A defensive refcount (`_npu_cache_hashref`) remains in `memory_model.py:575-591`, but since chained hashing prevents duplicates it is effectively a no-op
- Verified: duplicate-hash warning and unknown-hash warning do not recur

---

## Applied fixes

| # | File | Change | Status |
|---|------|--------|--------|
| 1 | `inference_serving/memory_model.py` | `avail_size()` — remove `× _bytes_per_token` | Done |
| 2 | `inference_serving/memory_model.py` | `apply_kv_cache_events` — free-before-alloc order | Done |
| 3 | `inference_serving/radix_tree.py` | chained hashing (`hash(tokens + (parent_hash,))`) + `_split_node` `hash_value` split | Done |

---

## Reproduction command

```bash
python main.py \
  --dataset dataset/lveval_hotpotwikiqa_mixup_16k_llama3-8b_rate100_rep5.jsonl \
  --max-num-batched-tokens 50000 \
  --num-req 100 \
  --output output_lveval_prefix.csv \
  --enable-prefix-caching \
  --log-level WARNING
```
