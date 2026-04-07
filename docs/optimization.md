# 최적화 이력

### 1. Radix tree eviction 후 node compaction

> 수정일: 2026-04-07

**파일**: `inference_serving/radix_tree.py`

**문제**: `evict()` 메서드에서 leaf 노드 삭제 후, 부모 노드에 자식이 하나만 남는 경우에도 병합(compact)하지 않음. radix tree의 핵심 불변조건(single-child 노드 제거)이 깨져 tree가 선형 체인으로 퇴화. 장시간 실행 시 `match_prefix`, `insert` 등의 탐색 성능 저하.

**수정**: `_try_compact()` 메서드 추가. eviction에서 leaf 삭제 후 부모에 대해 호출하여 single-child 노드를 자식과 병합.

### 2. `get_config()` 모델 설정 파일 캐싱

> 수정일: 2026-04-07

**파일**: `inference_serving/utils.py`

**문제**: `get_config(model_name)` 호출 시마다 JSON 파일을 매번 디스크에서 읽어옴. 배치마다 반복 호출되어 불필요한 I/O 발생.

**수정**: `_config_cache` 딕셔너리로 모델 이름 기준 캐싱. 동일 모델에 대해 최초 1회만 파일 읽기 수행.
