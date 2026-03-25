"""
LVEval -> LLMServingSim2 trace 변환 스크립트

LVEval 형식:
  { "input": "질문", "context": "긴 문서", "answers": ["정답"], ... }

LLMServingSim2 형식 (JSONL):
  { "input_toks": int, "output_toks": int, "arrival_time_ns": int,
    "input_tok_ids": [...], "output_tok_ids": [...] }

사용법:
  python script/lveval_to_trace.py \
    --input  ../LVEval/data/hotpotwikiqa_mixup_16k.jsonl \
    --output dataset/lveval_hotpot_16k.jsonl \
    --model  meta-llama/Llama-3.1-8B \
    --num-req 200 \
    --arrival-rate 1.0
"""

import argparse
import json
import random
import sys
from pathlib import Path


def load_tokenizer(model_name: str):
    """AutoTokenizer 로드. 실패 시 GPT-2로 fallback. (model_name, tokenizer) 반환."""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        loaded_name = model_name
        print(f"[tokenizer] Loaded: {loaded_name}")
    except Exception as e:
        loaded_name = "gpt2"
        print(f"[tokenizer] Failed to load {model_name} ({e}), falling back to {loaded_name}")
        tok = AutoTokenizer.from_pretrained(loaded_name)
    return loaded_name, tok


def generate_arrival_times(n: int, rate: float, seed: int = 42) -> list:
    """포아송 프로세스로 arrival_time_ns 생성 (rate: req/sec)"""
    rng = random.Random(seed)
    times = []
    t = 0
    for _ in range(n):
        interval_ns = int(rng.expovariate(rate) * 1e9)
        t += interval_ns
        times.append(t)
    return times


def convert(input_path: str, output_path: str, tokenizer_name: str, tokenizer,
            num_req: int, arrival_rate: float, seed: int = 42, repeat: int = 1):

    raw_rows = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_rows.append(json.loads(line))
            if len(raw_rows) >= num_req:
                break

    if not raw_rows:
        print(f"[error] No data found in {input_path}", file=sys.stderr)
        sys.exit(1)

    # repeat: 동일한 행을 repeat회 반복 (KV cache hit 측정용)
    rows = raw_rows * repeat
    print(f"[convert] {len(raw_rows)} entries × {repeat} repeat = {len(rows)} total, tokenizing...")

    arrival_times = generate_arrival_times(len(rows), arrival_rate, seed)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0

    # 메타데이터를 별도 파일로 저장 (JSONL 파일과 같은 이름, .meta.json 확장자)
    meta_path = str(output_path).replace(".jsonl", "") + ".meta.json"
    meta = {
        "tokenizer_model": tokenizer_name,
        "source_dataset": str(input_path),
        "num_req": num_req,
        "repeat": repeat,
        "total_req": len(rows),
        "arrival_rate_req_per_sec": arrival_rate,
        "seed": seed,
    }
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)
    print(f"[meta]    Saved metadata -> {meta_path}")

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, (row, arrival_ns) in enumerate(zip(rows, arrival_times)):

            # ---- input: context + "\n\n" + question ----
            context = row.get("context", "")
            question = row.get("input", "")
            input_text = context + "\n\n" + question

            # ---- output: 첫 번째 정답 (없으면 빈 문자열) ----
            answers = row.get("answers", [""])
            output_text = answers[0] if answers else ""
            if not output_text:
                output_text = " "  # 최소 1토큰 보장

            input_ids  = tokenizer.encode(input_text,  add_special_tokens=False)
            output_ids = tokenizer.encode(output_text, add_special_tokens=False)

            if not input_ids:
                input_ids = [0]
            if not output_ids:
                output_ids = [0]

            record = {
                "input_toks":      len(input_ids),
                "output_toks":     len(output_ids),
                "arrival_time_ns": arrival_ns,
                "input_tok_ids":   input_ids,
                "output_tok_ids":  output_ids,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(rows)} done...", flush=True)

    print(f"[convert] Done. {written} records -> {output_path}")

    records = [json.loads(l) for l in open(output_path, encoding="utf-8")]
    avg_in  = sum(r["input_toks"]  for r in records) / len(records)
    avg_out = sum(r["output_toks"] for r in records) / len(records)
    print(f"[stats]   tokenizer_model = {tokenizer_name}")
    print(f"[stats]   avg input_toks  = {avg_in:.1f}")
    print(f"[stats]   avg output_toks = {avg_out:.1f}")
    print(f"[stats]   arrival_rate    = {arrival_rate} req/s")


def main():
    parser = argparse.ArgumentParser(description="LVEval -> LLMServingSim2 trace converter")
    parser.add_argument("--input",        required=True,  help="LVEval JSONL 파일 경로")
    parser.add_argument("--output",       required=True,  help="출력 JSONL 파일 경로")
    parser.add_argument("--model",        default="gpt2", help="HuggingFace 토크나이저 모델명 (default: gpt2)")
    parser.add_argument("--num-req",      type=int, default=200, help="변환할 최대 요청 수 (default: 200)")
    parser.add_argument("--arrival-rate", type=float, default=1.0,
                        help="포아송 도착률 req/sec (default: 1.0)")
    parser.add_argument("--seed",         type=int, default=42, help="랜덤 시드")
    parser.add_argument("--repeat",       type=int, default=1,  help="데이터셋을 반복할 횟수 (default: 1, KV cache hit 측정용)")
    args = parser.parse_args()

    tokenizer_name, tokenizer = load_tokenizer(args.model)
    convert(args.input, args.output, tokenizer_name, tokenizer, args.num_req, args.arrival_rate, args.seed, args.repeat)


if __name__ == "__main__":
    main()
