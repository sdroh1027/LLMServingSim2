# Prefix Caching 버그 분석 및 수정 패치노트

작성일: 2026-03-27
대상 시뮬레이션: `--enable-prefix-caching`, LLaMA-3.1-8B on A6000 (48GB), LVEval 데이터셋

---

## 요약

Prefix caching 활성화 시 OOM(`tried to load XMB but only YMB available`)이 발생하는 버그 3종 발견 및 수정.
근본 원인은 `_npu_cache_hashtolen`의 메모리 추적 로직과 `avail_size()` 계산 오류이며,
그 중 하나는 같은 페이지 토큰을 공유하는 radix tree 브랜치의 **중복 hash** 문제다.

---

## Bug 1: A6000 메모리 크기 오설정

**파일**: `cluster_config/single_node_single_instance.json`

### 증상
모델 weight + KV cache 공간 부족으로 초반부터 OOM.

### 원인
```json
"npu_mem": { "mem_size": 40, ... }
```
A6000 실제 VRAM은 48GB인데 40GB로 잘못 설정.

### 수정
```json
"npu_mem": { "mem_size": 48, ... }
```

---

## Bug 2: `avail_size()` 이중 단위 변환 (bytes × bytes/token)

**파일**: `inference_serving/memory_model.py` — `avail_size()` 메서드

### 증상
시뮬레이션 시작 직후(~4초) OOM. eviction이 전혀 일어나지 않음.

### 원인
```python
# 수정 전 (잘못된 코드)
def avail_size(self, device):
    if device == Device.NPU:
        return self.npu_prefix_cache.avail_size() * self._bytes_per_token
        #      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^
        #      이미 bytes 단위로 반환               bytes/token 을 또 곱함
        #      → bytes × (bytes/token) = 잘못된 단위, 실제보다 수천 배 큰 값 반환
```

`RadixCache.avail_size()`는 이미 `capacity - total_memory_usage()` 형태로 **bytes** 단위를 반환한다.
그런데 `_bytes_per_token`을 한 번 더 곱하면 `bytes × (bytes/token)` = 의미 없이 큰 값이 된다.

Scheduler 에서 `evict_size = kv_need - avail_size(NPU)` 를 계산할 때,
`avail_size`가 수조 bytes로 보이므로 `evict_size ≤ 0` → 아무것도 evict 안 함 → OOM.

### 수정
```python
# 수정 후
def avail_size(self, device):
    if device == Device.NPU:
        return self.npu_prefix_cache.avail_size()  # 이미 bytes 단위
```

---

## Bug 3: `apply_kv_cache_events` — free보다 alloc을 먼저 실행

**파일**: `inference_serving/memory_model.py` — `apply_kv_cache_events()` 메서드

### 증상
net 메모리 변화가 0인 배치에서도 OOM. 즉 evict하고 같은 양을 alloc할 때 실패.

### 원인
```python
# 수정 전 (잘못된 순서)
if npu_byte_alloc > 0:
    self.allocate(npu_byte_alloc, Device.NPU)  # 먼저 alloc → 공간 초과 오류
if npu_byte_free > 0:
    self.free(npu_byte_free, Device.NPU)       # 그 다음 free
```

같은 배치에서 evict(free)와 alloc이 동시에 발생할 때, free 전에 alloc을 시도하면
일시적으로 용량 초과가 발생해 `RuntimeError`가 터짐.

### 수정
```python
# 수정 후 (free 먼저)
if npu_byte_free > 0:
    self.free(npu_byte_free, Device.NPU)
if npu_byte_alloc > 0:
    self.allocate(npu_byte_alloc, Device.NPU)
```

---

## Bug 4: `_npu_cache_hashtolen` — 중복 page hash에 대한 이중 카운트 (미수정, 수정 필요)

**파일**: `inference_serving/memory_model.py` — `apply_kv_cache_events()` BlockStored 처리

### 증상
시뮬레이션 진행 중 메모리 drift가 점진적으로 커지다 OOM.
로그에 다음 두 경고가 반복:
```
WARNING  BlockStored duplicate hash -4638695184844254712 (prev_tlen=16, new_tlen=16)
WARNING  BlockRemoved unknown hash  -4638695184844254712 dict_size=15043
```

### 원인: 같은 16토큰 페이지가 서로 다른 radix tree 브랜치에 중복으로 존재

**예시**:

```
LVEval 데이터셋은 동일한 긴 본문(Wikipedia article 등)을 여러 질문에 반복 사용.
본문 내부에 반복되는 16토큰 패턴이 있으면:

  system_prompt[0:16]   == system_prompt[256:272]  (동일한 토큰 시퀀스)
  → hash(tokens[0:16])  == hash(tokens[256:272])

그 결과 radix tree의 두 브랜치가 서로 다른 위치에서 같은 hash를 가진
tree node를 독립적으로 보유:

  root
   └─ shared_prefix_node (tokens[0:256])
       ├─ [page H at pos 256] → branch_A ...
       └─ (이미 pos 0:16에 있던 page H의 사본이 다른 브랜치에서 재등장)
```

**결함 시퀀스**:

| 단계 | 이벤트 | hashtolen 상태 | npu_used |
|------|--------|----------------|----------|
| 1 | BlockStored(H): Node A 저장 | `[H → 16]` | +1KB |
| 2 | BlockStored(H): Node B 저장 (중복!) | `[H → 16]` (overwrite) | **+1KB 또 추가** (이중 카운트!) |
| 3 | BlockRemoved(H): Node A evict | `{}` (pop 성공) | -1KB |
| 4 | BlockRemoved(H): Node B evict | `{}` (pop 실패 → 0) | **변화 없음** (free 누락!) |
| 결과 | | | npu_used +1KB 잔존 (drift) |

radix tree는 A, B 모두 evict 완료 → `avail_size()` 증가
npu_used는 여전히 +1KB → `npu_mem - npu_used` 감소
→ `radix_avail > npu_avail` drift → 다음 배치에서 eviction 부족 → OOM

실제로 vLLM/SGLang에서는 동일 hash 페이지는 **물리 블록 공유** (content-addressed storage),
ref count로 관리해 한 번만 메모리를 할당한다.

### 수정 방향: reference count 도입

```python
# 초기화
self._npu_cache_hashtolen = {}  # hash → tlen
self._npu_cache_hashref   = {}  # hash → ref_count

# BlockStored 처리
if h in self._npu_cache_hashref:
    self._npu_cache_hashref[h] += 1
    # npu_byte_alloc에 추가 안 함 (물리 블록 공유, 이미 카운트됨)
else:
    self._npu_cache_hashref[h] = 1
    self._npu_cache_hashtolen[h] = tlen
    npu_byte_alloc += self.get_kv(tlen)  # 첫 등장 시만 카운트

# BlockRemoved 처리
ref = self._npu_cache_hashref.get(h, 0)
if ref == 0:
    self.logger.warning("BlockRemoved unknown hash %s", h)
elif ref == 1:
    tlen = self._npu_cache_hashtolen.pop(h)
    del self._npu_cache_hashref[h]
    npu_byte_free += self.get_kv(tlen)  # 마지막 참조 해제 시만 free
else:
    self._npu_cache_hashref[h] -= 1
    # 아직 다른 노드가 이 페이지를 참조 중 → free 안 함
```

### 현재 상태
- 버그 확인 완료 (중복 hash 로그, unknown hash 로그 모두 확인)
- 수정 코드 설계 완료, 적용 예정

---

## 적용된 수정 목록

| # | 파일 | 수정 내용 | 상태 |
|---|------|-----------|------|
| 1 | `cluster_config/single_node_single_instance.json` | `mem_size: 40 → 48` (A6000 실제 용량) | 완료 |
| 2 | `inference_serving/memory_model.py` | `avail_size()` — `× _bytes_per_token` 제거 | 완료 |
| 3 | `inference_serving/memory_model.py` | `apply_kv_cache_events` — free-before-alloc 순서 변경 | 완료 |
| 4 | `inference_serving/memory_model.py` | `_npu_cache_hashtolen` reference count 도입 | **미완료** |

---

## 재현 명령어

```bash
python main.py \
  --dataset dataset/lveval_hotpotwikiqa_mixup_16k_llama3-8b_rate100_rep5.jsonl \
  --max-num-batched-tokens 50000 \
  --num-req 100 \
  --output output_lveval_prefix.csv \
  --enable-prefix-caching \
  --log-level WARNING
```
