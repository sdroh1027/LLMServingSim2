"""
LVEval 16k 소스 데이터셋의 input/output 토큰 통계 계산.

lveval_to_trace.py 와 동일한 토크나이즈 방식:
  - input_text  = context + "\n\n" + question(input)
  - output_text = answers[0]   (없으면 " ")
  - tokenizer.encode(..., add_special_tokens=False)

대상: 4개 데이터셋의 16k 길이 파일.
"""

import json
import math
import os
import sys

MODEL = "Qwen/Qwen3-32B"
LVEVAL_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "LVEval", "data")

DATASETS = ["hotpotwikiqa_mixup", "multifieldqa_en_mixup", "loogle_SD_mixup", "factrecall_en"]
LENGTH = "16k"


def stats(vals):
    n = len(vals)
    if n == 0:
        return (0, 0, 0, 0.0)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return (mean, min(vals), max(vals), math.sqrt(var))


def main():
    from transformers import AutoTokenizer
    print(f"[tokenizer] loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)

    rows_out = []
    all_in, all_out = [], []

    for ds in DATASETS:
        path = os.path.join(LVEVAL_ROOT, ds, f"{ds}_{LENGTH}.jsonl")
        if not os.path.exists(path):
            print(f"[skip] missing {path}", file=sys.stderr)
            continue
        in_toks, out_toks = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                context = row.get("context", "")
                question = row.get("input", "")
                input_text = context + "\n\n" + question
                answers = row.get("answers", [""])
                output_text = answers[0] if answers else ""
                if not output_text:
                    output_text = " "
                iids = tok.encode(input_text, add_special_tokens=False) or [0]
                oids = tok.encode(output_text, add_special_tokens=False) or [0]
                in_toks.append(len(iids))
                out_toks.append(len(oids))
        ai, mni, mxi, sdi = stats(in_toks)
        ao, mno, mxo, sdo = stats(out_toks)
        all_in.extend(in_toks)
        all_out.extend(out_toks)
        rows_out.append((f"{ds}_{LENGTH}", len(in_toks), ai, mni, mxi, sdi, ao, mno, mxo, sdo))
        print(f"[done] {ds}_{LENGTH}: n={len(in_toks)} avg_in={ai:.1f} avg_out={ao:.1f}", flush=True)

    print("\n================ PER-FILE (16k) ================")
    hdr = (f"{'file':36s} {'n':>4s} {'avg_in':>9s} {'min_in':>7s} {'max_in':>8s} "
           f"{'std_in':>8s} {'avg_out':>8s} {'min_out':>7s} {'max_out':>7s} {'std_out':>7s}")
    print(hdr)
    for r in rows_out:
        print(f"{r[0]:36s} {r[1]:>4d} {r[2]:>9.1f} {r[3]:>7d} {r[4]:>8d} {r[5]:>8.1f} "
              f"{r[6]:>8.1f} {r[7]:>7d} {r[8]:>7d} {r[9]:>7.1f}")

    ai, mni, mxi, sdi = stats(all_in)
    ao, mno, mxo, sdo = stats(all_out)
    print("\n================ OVERALL (4 x 16k files) ================")
    print(f"records      : {len(all_in)}")
    print(f"input_toks   : avg={ai:.1f}  min={mni}  max={mxi}  std={sdi:.1f}")
    print(f"output_toks  : avg={ao:.1f}  min={mno}  max={mxo}  std={sdo:.1f}")


if __name__ == "__main__":
    main()
