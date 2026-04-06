# `--bypass-astrasim` 옵션

> 추가일: 2026-03-31
> 최적화: 2026-04-01 (in-memory bypass, 메모리 전송 비용, 주기별 타이밍 출력)
> 업데이트: 2026-04-02 (CPU/CXL bw/latency 소스 수정, 함수 정리)

## 개요

Single-node 시뮬레이션에서 AstraSim 프로세스와 Chakra 그래프 변환을 건너뛰고,
trace 파일의 compute time을 직접 합산하여 시뮬레이션 속도를 **최대 ~59배 향상**시키는 옵션.

## 사용법

```bash
# 기본 (in-memory bypass, 가장 빠름)
python main.py \
  --cluster-config cluster_config/... \
  --dataset dataset/... \
  --bypass-astrasim \
  ...

# 파일 기반 bypass (디버깅/호환용)
python main.py \
  --cluster-config cluster_config/... \
  --dataset dataset/... \
  --bypass-astrasim --bypass-use-file \
  ...
```

기존 명령어에 `--bypass-astrasim`만 추가하면 된다. 나머지 옵션은 동일.
`--bypass-use-file`을 추가하면 최적화 이전의 파일 기반 경로를 사용한다.

## 배경 및 동기

기존 시뮬레이션 루프의 병목:

```
매 배치마다:
  1. generate_trace()     → perf DB lookup → trace .txt 파일 생성
  2. generate_graph()     → subprocess.run(python -m chakra ...) ← Python 프로세스 매번 spawn
  3. AstraSim 바이너리    → trace 읽기 → 실행 → stdout으로 cycle 반환
  4. Python               → stdout 파싱 → 다음 스케줄링
```

- **2번**에서 매 배치마다 새 Python 프로세스를 spawn하여 Chakra ET로 변환
- **3번**에서 AstraSim 바이너리와 stdin/stdout IPC

Single-node, npu_num=1 환경에서는 네트워크 통신이 없으므로 AstraSim이 하는 일은
**trace의 compute time 합산 + 메모리 전송 시간 계산**뿐이다.

## 구현 내용

### 수정 파일

| 파일 | 변경 내용 |
|------|-----------|
| `main.py` | `--bypass-astrasim`, `--bypass-use-file` CLI 인자, in-memory/file 분기 로직, 주기별 prefill/decode 타이밍 출력 |
| `inference_serving/controller.py` | `BypassController` 클래스, `compute_trace_cycles()`, `compute_trace_cycles_mem_detail()`, `_calc_transfer_ns()` |
| `inference_serving/trace_generator.py` | `generate_trace_bypass()`, `_open_or_buf()` 헬퍼, `_synthesize_trace`/`_synthesize_interleaved_trace`에 `_file_obj` 파라미터 |
| `inference_serving/scheduler.py` | `get_next_arrival_time()` 메서드 추가 |
| `inference_serving/utils.py` | `get_config()` 캐싱 (`_config_cache`) |
| `inference_serving/memory_model.py` | `storage_cache_evicted_req()` cross-tree 참조 버그 수정 |

### BypassController

AstraSim 프로세스 대신 Python 내부 이벤트 큐(min-heap)로 시뮬레이션 시간을 관리.

```python
class BypassController:
    def submit_event(finish_cycle_ns, npu_id)   # 배치 완료 이벤트 등록
    def read_wait(p=None)                        # 다음 이벤트 pop (AstraSim stdout 대체)
    def write_flush(p, input_str)                # No-op
    def parse_output(output)                     # 동일한 정규식 파싱
    def check_end(p=None)                        # 종료 메시지 출력
```

### cycle 계산 함수

| 함수 | 입력 | 반환 | 용도 |
|------|------|------|------|
| `compute_trace_cycles(path, ...)` | 파일 경로 | `int` (total_ns) | `--bypass-use-file` (디버깅용) |
| `compute_trace_cycles_mem_detail(layers, ...)` | in-memory 리스트 | `dict` (breakdown) | 기본 bypass 경로 |

`compute_trace_cycles_mem_detail` 반환값:

```python
{
    'total_ns':         # 총 배치 시간 (compute + exposed comm)
    'compute_ns':       # 순수 레이어 연산 시간
    'cpu_transfer_ns':  # kv_load/kv_evict의 CPU 메모리 전송 시간
    'cxl_transfer_ns':  # kv_load/kv_evict의 CXL 전송 시간
    'comm_ns':          # 기타 (non-kv) remote/CXL 전송 시간
    'exposed_comm_ns':  # compute와 overlap되지 않는 통신 시간
}
```

### 메모리 전송 비용 계산

각 location 태그별로 해당 메모리의 bw/latency를 사용:

| location 태그 | 메모리 | bw/latency 소스 (cluster config) |
|---|---|---|
| `LOCAL` | GPU HBM | 전송 비용 0 (compute에 포함) |
| `REMOTE:N` | CPU DRAM | `nodes[N].cpu_mem.mem_bw` / `mem_latency` |
| `CXL:N` | CXL 디바이스 | `cxl_mem.mem_bw` / `cxl_mem.mem_latency` |

> **주의**: `link_bw`/`link_latency`는 multi-node 간 TP/PP 통신용이며,
> KV cache 전송과는 무관하다.

`_calc_transfer_ns(loc, data_size, cpu_bw, cpu_latency, cxl_bw, cxl_latency)`:
- `REMOTE` → `cpu_latency + data_size / cpu_bw`
- `CXL` → `cxl_latency + data_size / cxl_bw`
- `LOCAL` → 0

### generate_trace_bypass()

`generate_trace()`의 in-memory 변형:

- `_synthesize_trace()`에 `io.StringIO` 버퍼를 전달 (`_file_obj` 파라미터)하여 디스크 I/O 제거
- trace 데이터를 `list[list[str]]`로 직접 반환
- 최종 파일 재쓰기(write #2) 생략

### 시뮬레이션 루프 변경

```
[기존]                              [File bypass]                    [Memory bypass (기본)]
generate_trace()                    generate_trace()                 generate_trace_bypass()
  └ _synthesize_trace → 파일         └ _synthesize_trace → 파일       └ _synthesize_trace → StringIO
generate_graph() ← Chakra 변환      (스킵)                            (스킵)
write_flush() ← AstraSim IPC        compute_trace_cycles() ← 파일    compute_trace_cycles_mem_detail() ← 메모리
read_wait() ← AstraSim stdout       BypassController.read_wait()     BypassController.read_wait()
```

### 배치당 I/O 비교

| | 기존 (AstraSim) | File bypass | Memory bypass |
|---|---|---|---|
| _synthesize_trace → 파일 쓰기 | O | O | **StringIO** |
| 파일 재읽기 | O | O | **메모리 파싱** |
| 최종 파일 재쓰기 | O | O | **생략** |
| compute 파일 읽기 | - | O | **생략** |
| Chakra 프로세스 spawn | O | 생략 | 생략 |
| AstraSim IPC | O | 생략 | 생략 |

### 주기별 타이밍 출력

Memory bypass 모드에서 `--log-interval`마다 prefill/decode 배치의 평균 시간을 출력:

```
[1.0s] Avg prompt throughput: 115.0 tokens/s, Avg generation throughput: 158.0 tokens/s
        [Interval] Prefill(139 tok): (cpu/cxl/comp) = (0.00/0.00/26.81) ms  (7 batches, 139 tokens)
        [Interval] Decode(133 tok):  (cpu/cxl/comp) = (0.00/0.00/24.60) ms  (33 batches, 133 tokens)
        [Total]    Prefill(139 tok): (cpu/cxl/comp) = (0.00/0.00/26.81) ms  (7 batches, 139 tokens)
        [Total]    Decode(133 tok):  (cpu/cxl/comp) = (0.00/0.00/24.60) ms  (33 batches, 133 tokens)
```

- **Interval**: 직전 로그 이후 구간의 배치 평균 (매 구간 리셋)
- **Total**: 시뮬레이션 시작부터 누적 배치 평균
- **(cpu/cxl/comp)**: CPU 메모리 전송 / CXL 전송 / 순수 계산 (ms, 배치 평균)
- **Prefill**: `num_prefill > 0`인 배치 (prefill+decode 혼합 배치 포함)
- **Decode**: 순수 decode-only 배치

### get_config() 캐싱

`inference_serving/utils.py`의 `get_config(model_name)`이 매 호출마다 JSON 파일을 열어 파싱하던 것을
모듈 레벨 `_config_cache` dict로 캐싱. 모델당 1회만 파일 읽기, 이후 dict lookup.

- `calculate_sizes()` 등에서 배치당 ~15회 호출 → 300 req 기준 ~18,000회 파일 I/O 제거

## 정합성 검증

### File bypass vs Memory bypass

두 bypass 경로는 **비트 단위 동일 결과**를 생성한다.

| Trace | Requests | 비교 | 결과 |
|-------|----------|------|------|
| `example_trace.jsonl` | 10 | File vs Mem | **EXACT MATCH** |
| `sharegpt_req300_rate10_llama.jsonl` | 300 | File vs Mem | **EXACT MATCH** |

### AstraSim (ground truth) 대비 오차

**example_trace.jsonl (10 req, single_node_single_instance)**

| 항목 | 최대 상대 오차 |
|------|---------------|
| end_time | 0.0009% |
| latency | 0.0037% |
| TTFT | 0.0202% |
| TPOT | 0.0014% |

**sharegpt_req300_rate10_llama.jsonl (300 req, single_node_single_instance)**

| 항목 | 최대 상대 오차 |
|------|---------------|
| end_time | 1.42% |
| latency | 1.68% |
| TTFT | ~1.4% |
| TPOT | ~1.7% |

요청 수가 많아지면 배치 스케줄링 차이가 누적되어 오차가 커지지만,
이는 bypass 방식 자체의 특성이며 in-memory 최적화와 무관하다.

오차 원인: AstraSim은 per-layer 수준으로 compute/communication을 overlap하지만,
bypass는 batch-level로 단순화.

### 속도 비교

**example_trace.jsonl (10 req)**

| 모드 | 실행 시간 | AstraSim 대비 |
|------|----------|---------------|
| AstraSim | 17.9s | 1.0x |
| File bypass | 4.6s | 3.9x |
| **Memory bypass** | **1.86s** | **9.6x** |

**sharegpt_req300_rate10_llama.jsonl (300 req)**

| 모드 | 실행 시간 | AstraSim 대비 |
|------|----------|---------------|
| AstraSim | 156.8s | 1.0x |
| File bypass | 29.1s | 5.4x |
| **Memory bypass** | **2.65s** | **59.2x** |

요청 수가 많을수록 파일 I/O 제거 + get_config 캐싱 효과가 커져 speedup이 증가한다.

## Bugfix: prefix cache eviction 시 cross-tree 참조 크래시

> 수정일: 2026-04-01

### 발생 조건

- `--enable-prefix-caching` + `--prefix-storage CXL` (또는 `CPU`)
- GPU 메모리 부족으로 decode 중인 request의 KV cache가 evict될 때

### 원인

`memory_model.py:storage_cache_evicted_req()`에서:

```python
new_last_node = self.second_tier_prefix_cache.cache_unfinished_req(req)  # storage 트리 노드
self.npu_prefix_cache.inc_lock_ref(new_last_node)  # ← 다른 트리의 노드를 전달
```

`new_last_node`는 **second_tier(CXL/CPU) radix tree의 노드**인데,
`npu_prefix_cache.inc_lock_ref()`는 **NPU radix tree**에서 parent 체인을 순회.
두 트리는 별개 인스턴스이므로 parent 체인이 NPU root에 도달하지 못하고
storage tree root의 `parent = None`에서 `AttributeError: 'NoneType' object has no attribute 'lock_ref'` 발생.

### 수정

NPU radix tree에서 해당 request의 prefix를 직접 `match_prefix()`로 찾아서 lock:

```python
token_ids = (req.input_hash_ids + req.output_hash_ids)[:req.input]
npu_result = self.npu_prefix_cache.match_prefix(token_ids)
if npu_result.last_device_node is not None:
    self.npu_prefix_cache.inc_lock_ref(npu_result.last_device_node)
```

수정 파일: `inference_serving/memory_model.py`

## 버그 수정 및 알려진 문제

→ [docs/bugfixes.md](bugfixes.md) 참조

## 제한사항

- **Single-node, npu_num=1 전용**: multi-node나 npu_num>1에서는 네트워크 통신 시뮬레이션이
  필요하므로 AstraSim을 사용해야 한다.
- **MoE 모델**: EXPERT 마커 행은 스킵하고 실제 expert 연산 행의 comp_time만 합산.
  multi-GPU expert parallelism이 있는 경우 정확도가 떨어질 수 있다.
- **PIM offloading**: PIM 마커 행은 스킵. PIM 채널 병렬성은 고려하지 않는다.

## Trace 파일 포맷 참고

```
COLOCATED       model_parallel_NPU_group: 1     ← Line 1: 병렬화 타입
387                                              ← Line 2: 레이어 수
Layername  comp_time  input_loc  input_size  weight_loc  weight_size  output_loc  output_size  comm_type  comm_size  misc
embedding_0    413878  REMOTE:0   32824       LOCAL       1050673152   LOCAL       67223552     NONE       0          NONE
...
```

### location 태그

- `LOCAL`: GPU HBM — 전송 비용 없음 (attention comp_time에 HBM 접근 포함)
- `REMOTE:N`: CPU DRAM — `cpu_mem.mem_bw`/`cpu_mem.mem_latency` 기반 전송 시간
- `CXL:N`: CXL 디바이스 — `cxl_mem.mem_bw`/`cxl_mem.mem_latency` 기반 전송 시간

### location 필드별 매핑

| 필드 | 결정 방식 | 가능한 값 |
|------|-----------|-----------|
| `input_loc` | 대부분 `LOCAL` 하드코딩. embedding/lm_head만 `REMOTE:N` | `LOCAL`, `REMOTE:N` |
| `weight_loc` | `get_device(placement, ...)` → config에 따라 결정 | `LOCAL`, `REMOTE:N`, `CXL:N` |
| `output_loc` | 대부분 `LOCAL` 하드코딩. lm_head만 `REMOTE:N` | `LOCAL`, `REMOTE:N` |

kv_load/kv_evict의 경우 `weight_loc` = `get_device(placement, None, None, 'kv_evict_loc')`:
- `kv_evict_loc: "npu"` → `LOCAL`
- `kv_evict_loc: "cpu"` → `REMOTE:N`
- `kv_evict_loc: "cxl"` → `CXL`



## bypass-astrasim 관련 bugfix

### 1. `compute_trace_cycles`에서 CXL 전송 비용 누락

> 수정일: 2026-04-01

**파일**: `inference_serving/controller.py`, `main.py`

**증상**: `kv_evict_loc=cxl` 설정에서 kv_load/kv_evict의 CXL 전송 비용이 0으로 처리.

**원인**: `REMOTE`로 시작하는 location만 체크하여 `CXL:0` 등을 무시.

**수정**: `_calc_transfer_ns()` 헬퍼 도입, `CXL` location 지원.

### 2. `compute_trace_cycles`에서 CPU 메모리 전송에 `link_bw` 사용

> 수정일: 2026-04-02

**파일**: `inference_serving/controller.py`, `main.py`

**증상**: `REMOTE:N`(CPU 메모리) 접근 시 `link_bw`/`link_latency`(노드간 TP/PP 통신)를
전송 비용에 사용. 실제로는 `cpu_mem.mem_bw`/`cpu_mem.mem_latency`를 사용해야 함.

**수정**: `link_bw`/`link_latency` → `cpu_mem.mem_bw`/`cpu_mem.mem_latency`로 변경.
함수 파라미터명도 `link_bw` → `cpu_bw`로 전면 교체.


## bypass-astrasim 관련 미수정 (알려진 문제)

### 1. Exposed communication의 batch-level 계산 (bypass 전용)

**파일**: `controller.py` `compute_trace_cycles_mem_detail()`

```python
exposed_comm = max(0, total_comm - total_compute)  # 배치 전체 합산 후 1회 계산
```

현재는 모든 레이어의 compute/comm을 합산한 뒤 한 번에 exposed comm을 계산.
실제로는 레이어가 순차 실행되므로, per-layer로 계산해야 정확하다.

**예시**:

| 레이어 | compute | comm |
|---|---|---|
| kv_load | 0 ns | 50,000 ns |
| embedding | 7,000 ns | 0 ns |

| | 현재 (batch-level) | per-layer |
|---|---|---|
| exposed_comm | `max(0, 50000 - 7000) = 43,000` | kv_load: 50,000 + embedding: 7,000 = **57,000** |

batch-level은 다른 레이어의 compute로 comm을 가릴 수 있다고 가정하여 과소 추정.

**수정 방향**: per-layer exposed comm 계산

```python
total = 0
for layer in layers:
    layer_exposed = max(0, layer_comm - layer_compute)
    total += layer_compute + layer_exposed
```
