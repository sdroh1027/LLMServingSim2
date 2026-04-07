# 버그 수정 이력

## 수정 완료

### 1. Radix tree node split 시 chained hashing 오류

> 커밋: `2ea4743` (2026-03-30)

**파일**: `inference_serving/radix_tree.py`, `inference_serving/memory_model.py`

**증상**: prefix caching 활성화 시 메모리 drift → 점진적 OOM

**원인**: 동일 토큰이 다른 prefix 위치에 있을 때 같은 hash가 생성됨.
BlockStored에서 메모리를 이중 할당, BlockRemoved에서 한 번만 해제.

**수정**:
- Chained hashing 도입: `block_hash = hash(page_tokens + (parent_block_hash,))`
  → 위치별로 고유 hash 생성, 중복 원천 차단
- Node split 시 `hash_value` 리스트를 page count 기준으로 올바르게 분할
- Root 조건 체크 오류 수정 (`node != root_node` → `node == root_node`)
- Parent의 마지막 hash를 seed로 사용하도록 변경

### 2. `avail_size()` 이중 단위 변환 (bytes × bytes/token)

> 커밋: `2ea4743` (2026-03-30)

**파일**: `inference_serving/memory_model.py`

**증상**: 시뮬레이션 시작 직후 OOM. eviction이 전혀 일어나지 않음.

**원인**: `RadixCache.avail_size()`가 이미 bytes 단위를 반환하는데
`_bytes_per_token`을 추가로 곱하여 사용 가능 공간이 수천 배 과대 보고
→ `evict_size ≤ 0` → 아무것도 evict 안 함 → OOM.

**수정**: `× _bytes_per_token` 제거

### 3. `apply_kv_cache_events` — free/alloc 순서

> 커밋: `2ea4743` (2026-03-30)

**파일**: `inference_serving/memory_model.py`

**증상**: net 메모리 변화가 0인 배치에서도 OOM.

**원인**: 같은 배치에서 evict(free)와 alloc이 동시에 발생할 때,
alloc을 먼저 실행하면 일시적 용량 초과로 `RuntimeError`.

**수정**: free → alloc 순서로 변경

### 4. Empty batch guard

> 커밋: `c1b61e4` (2026-03-30)

**파일**: `inference_serving/trace_generator.py`

**증상**: `total_len=0`, `len(requests)=0`인 빈 배치가 trace 생성에 도달하면 에러.

**수정**: 빈 배치 감지 시 로그 출력 후 스킵하도록 guard 추가.

### 5. Input size 초과 검증

> 커밋: `0a76bdf` (2026-03-30)

**파일**: `inference_serving/scheduler.py`

**증상**: `req.input > max_num_batched_tokens`인 request가 스케줄링되면 무한 루프.

**수정**: `add_request()` 시점에 검증하여 명확한 에러 메시지와 함께 `ValueError` 발생.

### 6. Prefix cache eviction 시 cross-tree 참조 크래시

> 수정일: 2026-04-01

**파일**: `inference_serving/memory_model.py`

**증상**: `--enable-prefix-caching` + `--prefix-storage CXL`(또는 `CPU`) +
GPU 메모리 부족으로 decode 중인 request의 KV cache가 evict될 때
`AttributeError: 'NoneType' object has no attribute 'lock_ref'` 발생.

**원인**: `storage_cache_evicted_req()`에서 second_tier(CXL/CPU) radix tree의 노드를
NPU radix tree의 `inc_lock_ref()`에 전달. 두 트리는 별개 인스턴스이므로
parent 체인이 NPU root에 도달하지 못하고 storage tree root의 `parent = None`에서 크래시.

**수정**: NPU radix tree에서 해당 request의 prefix를 직접 `match_prefix()`로 찾아서 lock:

```python
token_ids = (req.input_hash_ids + req.output_hash_ids)[:req.input]
npu_result = self.npu_prefix_cache.match_prefix(token_ids)
if npu_result.last_device_node is not None:
    self.npu_prefix_cache.inc_lock_ref(npu_result.last_device_node)
```


### 7. CXL prefix cache eviction 미지원 + 로그 메모리 사용량 하드코딩

> 수정일: 2026-04-07

**파일**: `inference_serving/memory_model.py`, `main.py`

**증상 1**: `--prefix-storage CXL` 사용 시 CXL 공간 부족하면 `RuntimeError: Trying to evict prefix cache to unsupported device Device.CXL`

**원인**: `evict_prefix_cache()`에 `Device.CXL` 분기 누락. `Device.CPU`만 처리.

**수정**: `elif device == Device.CPU:` → `elif device == Device.CPU or device == Device.CXL:`

**증상 2**: CXL/CPU 로그의 메모리 사용량이 실제와 다름 (0.3% 표시 → 실제 62%)

**원인**: `total_size() * 131072` 하드코딩. `total_size()`는 토큰 수인데 모델별 kv_size 대신 고정 상수 사용. 또한 `cxl_util` 계산 시 `* 100` 누락.

**수정**: `total_memory_usage()` 사용 + `cxl_util`에 `* 100` 추가 (3곳)

### 8. `storage_cache_evicted_req` NPU lock 누수

> 수정일: 2026-04-07

**파일**: `inference_serving/memory_model.py`

**증상**: prefix_storage 사용 시 evicted request의 NPU radix tree 노드에 lock이 영구적으로 남아 해당 prefix가 eviction 대상에서 제외됨.

**원인**: `storage_cache_evicted_req()`에서 evicted req의 KV cache를 CPU/CXL에 저장한 뒤, NPU radix tree에 `inc_lock_ref()`를 호출하지만 해당 노드를 `req.npu_last_node`에 저장하지 않음. 이후 어디서도 `dec_lock_ref()`가 호출되지 않아 lock 누수 발생.

**수정**: NPU lock 대신 CPU/CXL(`second_tier_prefix_cache`) lock으로 변경. 재스케줄 시 `unlock_prefix(req, Device.CPU)`와 올바르게 쌍을 이룸.

```python
# Before (NPU lock — leaked)
self.npu_prefix_cache.inc_lock_ref(new_last_node)

# After (CPU/CXL lock — paired with unlock at re-scheduling)
self.second_tier_prefix_cache.inc_lock_ref(new_last_node)
```

## 미수정 (알려진 문제)

### 1. Prefill attention의 kv_cache_size 미반영

**파일**: `scheduler.py` line 437 / line 193

```python
prefill_k_list.append(0)   # ← 항상 0, prefix cache hit 토큰 수를 반영하지 않음
```

prefix cache hit이 있으면 해당 토큰의 KV가 이미 HBM에 존재하므로,
attention은 Q(miss 토큰) × K(hit + miss 전체)를 계산해야 한다.
하지만 `prefill_k_list`가 항상 0이므로 `_make_attn_db_key`에서 `kv_cache_size=0`으로 조회.

**영향**: prefix cache hit 시 attention latency 과소 추정

**예시** (input=1024, hit=512):

| | 현재 | 올바른 값 |
|---|---|---|
| prefill_k_list | `[0]` | `[512]` |
| DB 조회 key | `(0, 512)` → 6,066ns | `(512, 512)` → 23,683ns |

**수정 방향**: `prefill_k_list.append(req.prefix_cache_hit)`

### 2. Prefill attention의 kv_cache_size 합산 방식

**파일**: `trace_generator.py` `_make_attn_db_key()` line 2331

```python
prefill_agg_kv_cache_size = sum(batch.prefill_k_list)
```

배치 내 여러 prefill request의 `kv_cache_size`를 **합산**하여 하나의 key로 조회.
실제 attention은 request별로 독립적이므로, per-request로 조회하여 합산해야 한다.

**예시** (배치 내 2개 request):

| Request | q_len | kv_cache |
|---|---|---|
| A | 512 | 512 |
| B | 256 | 1024 |

| | 현재 | 올바른 방식 |
|---|---|---|
| 조회 | `(1536, chunk)` 1회 | `(512, 512)` + `(1024, 256)` 각각 조회 후 합산 |

**영향**: 배치 내 여러 prefill request가 있을 때 attention latency 부정확

