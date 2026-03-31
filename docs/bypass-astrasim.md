# `--bypass-astrasim` 옵션

> 추가일: 2026-03-31
> 최적화: 2026-04-01 (in-memory bypass)

## 개요

Single-node 시뮬레이션에서 AstraSim 프로세스와 Chakra 그래프 변환을 건너뛰고,
trace 파일의 compute time을 직접 합산하여 시뮬레이션 속도를 **최대 ~38배 향상**시키는 옵션.

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
| `main.py` | `--bypass-astrasim`, `--bypass-use-file` CLI 인자, in-memory/file 분기 로직 |
| `inference_serving/controller.py` | `BypassController` 클래스, `compute_trace_cycles()`, `compute_trace_cycles_mem()` |
| `inference_serving/trace_generator.py` | `generate_trace_bypass()`, `_open_or_buf()` 헬퍼, `_synthesize_trace`/`_synthesize_interleaved_trace`에 `_file_obj` 파라미터 |
| `inference_serving/scheduler.py` | `get_next_arrival_time()` 메서드 추가 |

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

### compute_trace_cycles() / compute_trace_cycles_mem()

trace 데이터에서 총 실행 시간(ns)을 계산:

1. 각 레이어의 `comp_time` (column 1) 합산
2. `REMOTE` 메모리 접근의 데이터 전송 시간 계산 (`data_size / link_bw + link_latency`)
3. Exposed communication = `max(0, total_comm - total_compute)` (batch-level overlap)

- `compute_trace_cycles(path)`: trace .txt 파일에서 읽기 (file-based bypass용)
- `compute_trace_cycles_mem(layers)`: in-memory 리스트에서 직접 계산 (기본 bypass용)

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
write_flush() ← AstraSim IPC        compute_trace_cycles() ← 파일    compute_trace_cycles_mem() ← 메모리
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

### 속도 비교 (3-way)

**example_trace.jsonl (10 req)**

| 모드 | 실행 시간 | AstraSim 대비 |
|------|----------|---------------|
| AstraSim | 17.9s | 1.0x |
| File bypass | 4.6s | 3.9x |
| **Memory bypass** | **2.1s** | **8.7x** |

**sharegpt_req300_rate10_llama.jsonl (300 req)**

| 모드 | 실행 시간 | AstraSim 대비 |
|------|----------|---------------|
| AstraSim | 156.8s | 1.0x |
| File bypass | 29.1s | 5.4x |
| **Memory bypass** | **4.1s** | **37.8x** |

요청 수가 많을수록 파일 I/O 제거 효과가 커져 speedup이 증가한다.

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

- `comp_time` (column 1): 레이어 실행 시간 (nanoseconds)
- `REMOTE:N`: N번 노드의 원격 메모리 접근 → 전송 시간 발생
- `LOCAL`: 로컬 메모리 접근 → 전송 시간 없음
