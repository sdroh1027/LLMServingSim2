# TODO: Chunked Prefill 구현

## 배경

현재 시뮬레이터는 prefill 요청의 전체 input 토큰을 한 iteration에 처리한다.
실제 서빙 시스템(vLLM 등)에서는 긴 prefill을 chunk 단위로 나눠 decode 요청과 interleave하여,
prefill이 decode를 블로킹하는 지연을 줄인다. 이를 시뮬레이터에 반영해야 한다.

## 현재 상태

- `scheduler.py:192` — `# For now, we don't assume chunked prefill` 주석 존재
- 긴 input이 `max_num_batched_tokens`를 초과하면 에러로 reject (`scheduler.py:598-607`)
- 성능 DB(`trace_generator.py`)에는 `prefill_chunk_size` 기반 latency 룩업이 이미 존재
- profiler(`batch_sampling.py`)도 chunk size별 프로파일링을 지원

## TODO

### 1. CLI 옵션 추가 (`main.py`)
- [ ] `--enable-chunked-prefill` (bool): chunked prefill 활성화
- [ ] `--prefill-chunk-size` (int): chunk 크기 (토큰 단위, e.g., 512, 1024)

### 2. Request 클래스 확장 (`request.py`)
- [ ] `remaining_prefill_tokens`: 아직 처리하지 않은 prefill 토큰 수
- [ ] `prefill_position`: 현재까지 처리한 prefill 위치
- [ ] `is_chunked_prefill`: chunked prefill 진행 중 여부
- [ ] prefill 완료 판정 로직: `remaining_prefill_tokens == 0`일 때 decode 전환

### 3. Scheduler 수정 (`scheduler.py`)
- [ ] `schedule_base()` / `schedule_with_prefix()`에서 chunked prefill 처리
  - prefill 요청의 q_list 값을 `min(chunk_size, remaining_tokens)`로 설정
  - chunk 처리 후 `remaining_prefill_tokens` 갱신
  - 완료되지 않은 prefill 요청은 request queue에 유지 (삭제하지 않음)
- [ ] prefill chunk + decode 요청 co-scheduling 지원
  - 한 batch에 prefill chunk와 decode 요청을 함께 배치
  - `max_num_batched_tokens` 내에서 prefill chunk + decode 토큰 합산 관리
- [ ] `max_num_batched_tokens` 초과 요청을 reject 대신 chunk로 분할 처리

### 4. Memory 모델 수정 (`memory_model.py`)
- [ ] chunk 단위 점진적 KV cache 할당 방식 검토
  - 옵션 A: 첫 chunk에서 전체 KV cache 미리 할당 (현재 방식 유지, 간단)
  - 옵션 B: chunk마다 해당 chunk 분량만 추가 할당 (더 정확하나 복잡)
- [ ] prefix cache hit 시 chunk 시작 위치 조정

### 5. Trace 생성 수정 (`trace_generator.py`)
- [ ] `_make_attn_db_key()`에서 실제 chunk size를 반영
  - 현재: RMS 기반 집계 → chunk 미반영
  - 변경: chunked prefill 시 실제 chunk size를 DB 키로 사용
- [ ] `prefill_k_list`에 이전 chunk의 KV cache 크기 반영 (현재 항상 0)

### 6. Batch 클래스 확장 (`request.py` — Batch)
- [ ] chunked prefill 진행 중인 요청 추적
- [ ] iteration 간 partial prefill 상태 전달

### 7. 통계/로깅
- [ ] chunk 단위 prefill latency 기록
- [ ] chunked prefill로 인한 decode latency 개선 효과 측정
- [ ] 요청별 총 prefill chunk 수 / iteration 수 로깅

## 관련 파일

| 파일 | 수정 내용 |
|------|----------|
| `main.py` | CLI 옵션 추가 |
| `inference_serving/request.py` | Request/Batch 필드 추가 |
| `inference_serving/scheduler.py` | chunk 분할 스케줄링 로직 |
| `inference_serving/memory_model.py` | 점진적 KV 할당 검토 |
| `inference_serving/trace_generator.py` | chunk size 기반 DB 키 생성 |

## 참고

- 성능 DB에 `(kv_cache_size, prefill_chunk_size)` 키가 이미 존재하므로, latency 룩업 인프라는 준비됨
- profiler의 `PREFILL_CHUNK_SIZE_SPACE`가 32~64K 범위를 커버
- vLLM의 chunked prefill 구현을 참고할 것
