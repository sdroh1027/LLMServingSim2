# LVEval Trace 생성 가이드

LVEval 데이터셋을 LLMServingSim2 시뮬레이터용 trace JSONL로 변환하는 방법을 설명합니다.

---

## 파일 구조

```
LLMServingSim2/
├── script/
│   ├── lveval_to_trace.py          # 단일 변환 스크립트
│   └── gen_lveval_traces_qwen3.sh  # 전체 배치 생성 스크립트 (Qwen3-32B)
├── dataset/
│   └── lveval_<dataset>_<len>_<model>_rate<rate>.jsonl   # 생성된 trace
│   └── lveval_<dataset>_<len>_<model>_rate<rate>.meta.json  # 메타데이터
└── docs/
    └── lveval_traces.md            # 이 문서
```

---

## Trace 포맷

각 줄이 하나의 요청:
```json
{
  "input_toks":      30020,
  "output_toks":     4,
  "arrival_time_ns": 1020060287,
  "input_tok_ids":   [14374, 98475, ...],
  "output_tok_ids":  [32, 1440, ...]
}
```

| 필드 | 설명 |
|---|---|
| `input_toks` | 입력 토큰 수 (context + question) |
| `output_toks` | 출력 토큰 수 (정답) |
| `arrival_time_ns` | 요청 도착 시각 (나노초, 누적) |
| `input_tok_ids` | 입력 토큰 ID 배열 |
| `output_tok_ids` | 출력 토큰 ID 배열 |

---

## Poisson 도착 모델

`arrival_time_ns`는 **포아송 프로세스**로 생성됩니다:

```
inter-arrival time ~ Exponential(arrival_rate)
평균 간격 = 1 / arrival_rate  (초)
```

| `--arrival-rate` | 평균 간격 | 용도 |
|---|---|---|
| 0.5 | 2초 | 여유 트래픽 |
| 1.0 | 1초 | 기본값 |
| 5.0 | 0.2초 | 중간 부하 |
| 10.0 | 0.1초 | 고부하 / burst |

시뮬레이터에서 arrival rate가 높을수록 동시 처리 요청이 많아져 KV cache 활용률과 prefill 지연이 달라집니다.

---

## 단일 변환

```bash
cd LLMServingSim2
python script/lveval_to_trace.py \
  --input        ../LVEval/data/hotpotwikiqa_mixup/hotpotwikiqa_mixup_16k.jsonl \
  --output       dataset/lveval_hotpotwikiqa_mixup_16k_qwen3-32b_rate1.0.jsonl \
  --model        Qwen/Qwen3-32B \
  --num-req      9999 \
  --arrival-rate 1.0 \
  --seed         42
```

| 인자 | 설명 |
|---|---|
| `--input` | LVEval JSONL 파일 경로 |
| `--output` | 출력 JSONL 파일 경로 |
| `--model` | HuggingFace 토크나이저 모델명 (없으면 gpt2 fallback) |
| `--num-req` | 최대 변환 요청 수 (9999 = 전체) |
| `--arrival-rate` | 포아송 도착률 req/s |
| `--seed` | 랜덤 시드 |

---

## 전체 배치 생성 (Qwen3-32B)

```bash
cd LLMServingSim2
bash script/gen_lveval_traces_qwen3.sh
```

4개 데이터셋 × 5개 컨텍스트 길이 = **20개 파일** 생성.

### 생성된 파일 목록 및 통계 (Qwen/Qwen3-32B, rate=1.0)

| 파일 | 레코드 수 | avg input_toks | avg output_toks |
|---|---|---|---|
| lveval_hotpotwikiqa_mixup_16k_qwen3-32b_rate1.0.jsonl | 124 | 30,020 | 3.7 |
| lveval_hotpotwikiqa_mixup_32k_qwen3-32b_rate1.0.jsonl | 124 | 53,695 | 3.7 |
| lveval_hotpotwikiqa_mixup_64k_qwen3-32b_rate1.0.jsonl | 124 | 100,741 | 3.7 |
| lveval_hotpotwikiqa_mixup_128k_qwen3-32b_rate1.0.jsonl | 124 | 194,951 | 3.7 |
| lveval_hotpotwikiqa_mixup_256k_qwen3-32b_rate1.0.jsonl | 124 | 385,330 | 3.7 |
| lveval_multifieldqa_en_mixup_16k_qwen3-32b_rate1.0.jsonl | 101 | 29,155 | 14.9 |
| lveval_multifieldqa_en_mixup_32k_qwen3-32b_rate1.0.jsonl | 101 | 54,478 | 14.9 |
| lveval_multifieldqa_en_mixup_64k_qwen3-32b_rate1.0.jsonl | 101 | 104,504 | 14.9 |
| lveval_multifieldqa_en_mixup_128k_qwen3-32b_rate1.0.jsonl | 101 | 203,904 | 14.9 |
| lveval_multifieldqa_en_mixup_256k_qwen3-32b_rate1.0.jsonl | 101 | 402,491 | 14.9 |
| lveval_loogle_SD_mixup_16k_qwen3-32b_rate1.0.jsonl | 160 | 23,464 | 13.9 |
| lveval_loogle_SD_mixup_32k_qwen3-32b_rate1.0.jsonl | 160 | 47,244 | 13.9 |
| lveval_loogle_SD_mixup_64k_qwen3-32b_rate1.0.jsonl | 160 | 87,622 | 13.9 |
| lveval_loogle_SD_mixup_128k_qwen3-32b_rate1.0.jsonl | 160 | 165,263 | 13.9 |
| lveval_loogle_SD_mixup_256k_qwen3-32b_rate1.0.jsonl | 160 | 316,637 | 13.9 |
| lveval_factrecall_en_16k_qwen3-32b_rate1.0.jsonl | 200 | 15,171 | 5.0 |
| lveval_factrecall_en_32k_qwen3-32b_rate1.0.jsonl | 200 | 30,177 | 5.0 |
| lveval_factrecall_en_64k_qwen3-32b_rate1.0.jsonl | 200 | 60,096 | 5.0 |
| lveval_factrecall_en_128k_qwen3-32b_rate1.0.jsonl | 200 | 118,998 | 5.0 |
| lveval_factrecall_en_256k_qwen3-32b_rate1.0.jsonl | 200 | 233,909 | 5.0 |

> **참고:** LVEval의 컨텍스트 길이 레이블(16k 등)은 **문자(character) 기준**이며,
> 실제 Qwen3 토크나이저 토큰 수는 약 2배 수준입니다 (영어 기준 1토큰 ≈ 4자).

---

## 주의사항

- **128k / 256k 파일**: Qwen3-32B의 기본 컨텍스트 한계(131,072 토큰)를 초과하는 샘플 존재.
  토크나이징은 정상 완료되나, 시뮬레이터에서 해당 요청을 어떻게 처리할지 별도 고려 필요.
- **파일명 규칙**: `lveval_{dataset}_{length}_{model_tag}_rate{rate}.jsonl`
  → 모델과 arrival rate가 다른 trace를 추가 생성해도 파일명으로 구분 가능.
- **소스 데이터**: `KVCache_Simulator/LVEval/data/{dataset}/{dataset}_{length}.jsonl`
