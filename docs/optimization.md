# 최적화 이력

### 1. `get_config()` 모델 설정 파일 캐싱

> 수정일: 2026-04-07

**파일**: `inference_serving/utils.py`

**문제**: `get_config(model_name)` 호출 시마다 JSON 파일을 매번 디스크에서 읽어옴. 배치마다 반복 호출되어 불필요한 I/O 발생.

**수정**: `_config_cache` 딕셔너리로 모델 이름 기준 캐싱. 동일 모델에 대해 최초 1회만 파일 읽기 수행.

### 2. Radix tree eviction 후 node compaction

> 수정일: 2026-04-07

**파일**: `inference_serving/radix_tree.py`

**문제**: `evict()` 메서드에서 leaf 노드 삭제 후, 부모 노드에 자식이 하나만 남는 경우에도 병합(compact)하지 않음. radix tree의 핵심 불변조건(single-child 노드 제거)이 깨져 tree가 선형 체인으로 퇴화. 장시간 실행 시 `match_prefix`, `insert` 등의 탐색 성능 저하.

**수정**: `_try_compact()` 메서드 추가. eviction에서 leaf 삭제 후 부모에 대해 호출하여 single-child 노드를 자식과 병합.


## 중간 점검

**프로파일 결과** (compaction 적용 후, 100 req):

조건: `qwen_traceA_blksz_16.jsonl`, Qwen3-32B, H100 96GB + CXL 2560GB, block_size=16
```
python -m cProfile -s cumtime main.py \
  --cluster-config cluster_config/single_node_h100_qwen3-32b_96g_cxl2560.json \
  --fp 16 --block-size 16 --bypass-astrasim --num-req 100 --max-batch 32 \
  --max-num-batched-tokens 65536 --enable-prefix-caching --prefix-storage CXL \
  --dataset dataset/qwen_traceA_blksz_16.jsonl
```

| 함수 | 호출 수 | 총 시간 | 비중 |
|---|---|---|---|
| `_total_size_helper` | 12,586 | 83.8s | 1위 |
| `_insert_helper` | 68,296 | 29.1s | 2위 |
| `_match_prefix_helper` | 68,296 | 26.1s | 3위 |
| `_key_match_page_size1` | 16,684,531 | 9.5s | 4위 |
| `_key_match_paged` | 1,182,919 | 2.5s | - |

### 3. `total_size()` O(1) 캐싱

> 수정일: 2026-04-08

**파일**: `inference_serving/radix_tree.py`, `main.py`

**문제**: `total_size()`가 매 호출마다 BFS로 전체 트리 순회 (`_total_size_helper`). 12,586번 호출에 83.8초 소요 (전체 1위 병목).

**수정**: `evictable_size_ + protected_size_` 반환으로 O(1) 변경. 두 변수는 insert/delete/lock/unlock에서 이미 정확히 관리됨. 12,586번 호출에서 BFS 결과와 불일치 0건 검증 완료. 시뮬레이션 100초마다 `verify_total_size()`로 BFS와 정합성 주기 검증.


## 수정이후 평가

● Before vs After 비교

  ┌───────────────────────────┬────────────────────┬───────────────────┬─────────────┐
  │           함수            │ Before (호출/시간) │ After (호출/시간) │    변화     │
  ├───────────────────────────┼────────────────────┼───────────────────┼─────────────┤
  │ _total_size_helper        │ 12,586 / 83.8s     │ 0 / 0s            │ 제거됨      │
  ├───────────────────────────┼────────────────────┼───────────────────┼─────────────┤
  │ avail_size (memory_model) │ 9,422 / 78.9s      │ 9,386 / 0.02s     │ 3,900x 빠름 │
  ├───────────────────────────┼────────────────────┼───────────────────┼─────────────┤
  │ _insert_helper            │ 68,296 / 29.1s     │ 68,296 / 28.7s    │ 동일        │
  ├───────────────────────────┼────────────────────┼───────────────────┼─────────────┤
  │ _match_prefix_helper      │ 68,296 / 26.1s     │ 68,296 / 25.5s    │ 동일        │
  ├───────────────────────────┼────────────────────┼───────────────────┼─────────────┤
  │ _key_match_page_size1     │ 16,684,531 / 9.5s  │ 16,684,533 / 9.5s │ 동일        │
  ├───────────────────────────┼────────────────────┼───────────────────┼─────────────┤
  │ schedule_with_prefix      │ 3,123 / 141s       │ 3,113 / 57s       │ 2.5x 빠름   │
  └───────────────────────────┴────────────────────┴───────────────────┴─────────────┘

  total_size() O(1) 캐싱으로 83.8초가 완전히 제거되었고, 이에 의존하던 avail_size도 78.9s → 0.02s로 급감했습니다. 전체 스케줄링 시간이 141초 → 57초로 2.5배 빨라졌습니다.

  현재 병목은 _insert_helper(28.7s)와 _match_prefix_helper(25.5s)이고, 이들의 핵심은 _key_match_page_size1(9.5s, 1670만회 호출)입니다. 이 부분은 Phase 2 (C++ extension)의
  대상입니다.


