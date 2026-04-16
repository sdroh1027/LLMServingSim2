"""
두 실험 CSV 결과로 비교 플롯 생성
- Prompt/Gen Throughput per interval (CSV end_time 기반)
- Cumulative Prompt Tokens (CSV end_time 기반, 정확)
"""
import csv
import ast
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "output/lveval_hotpot_16k_qwen3-32b_rate10.0_rep5_comparison"
os.makedirs(OUTDIR, exist_ok=True)

CXL_CSV = "output/lveval_hotpot_16k_qwen3-32b_rate10.0_rep5_cxl"
GPU_CSV = "output/lveval_hotpot_16k_qwen3-32b_rate10.0_rep5_onlygpu"

INTERVAL_S = 2.0

def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def build_timeseries(rows, interval_s):
    """Build throughput timeseries from CSV: bin requests by end_time."""
    # CSV "output" = total seq length (input + generated), so decode tokens = output - input
    data = [(int(r["arrival"]), int(r["end_time"]), int(r["input"]), int(r["output"]) - int(r["input"])) for r in rows]
    min_t = min(d[0] for d in data)
    max_t = max(d[1] for d in data)
    interval_ns = int(interval_s * 1e9)

    times, prompt_thr, gen_thr = [], [], []
    cum_prompt, cum_gen = [], []
    total_p, total_g = 0, 0

    t = min_t
    while t < max_t + interval_ns:
        t_end = t + interval_ns
        p, g = 0, 0
        for arr, et, inp, out in data:
            if t <= et < t_end:
                p += inp
                g += out
        total_p += p
        total_g += g
        times.append((t - min_t) / 1e9)
        prompt_thr.append(p / interval_s)
        gen_thr.append(g / interval_s)
        cum_prompt.append(total_p)
        cum_gen.append(total_g)
        t = t_end

    return (np.array(times), np.array(prompt_thr), np.array(gen_thr),
            np.array(cum_prompt), np.array(cum_gen))

print("Loading CSVs...")
rows_cxl = load_csv(CXL_CSV)
rows_gpu = load_csv(GPU_CSV)

print("Building timeseries...")
t_cxl, p_cxl, g_cxl, cp_cxl, cg_cxl = build_timeseries(rows_cxl, INTERVAL_S)
t_gpu, p_gpu, g_gpu, cp_gpu, cg_gpu = build_timeseries(rows_gpu, INTERVAL_S)

# ---- Plot 1: Prompt Throughput ----
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(t_cxl, p_cxl / 1000, label="CXL Prefix Caching", alpha=0.85, linewidth=0.9, color="#1f77b4")
ax.plot(t_gpu, p_gpu / 1000, label="GPU Only (No Prefix)", alpha=0.85, linewidth=0.9, color="#ff7f0e")
ax.set_xlabel("Simulation Time (s)", fontsize=12)
ax.set_ylabel("Prefill Throughput (k tok/s)", fontsize=12)
ax.set_title(f"Prefill (Prompt) Throughput per {INTERVAL_S}s Interval  (Qwen3-32B, H100, 16k context, rate=10, rep=5)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "prompt_throughput.png"), dpi=150)
plt.close()
print("  prompt_throughput.png")

# ---- Plot 2: Generation Throughput ----
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(t_cxl, g_cxl, label="CXL Prefix Caching", alpha=0.85, linewidth=0.9, color="#1f77b4")
ax.plot(t_gpu, g_gpu, label="GPU Only (No Prefix)", alpha=0.85, linewidth=0.9, color="#ff7f0e")
ax.set_xlabel("Simulation Time (s)", fontsize=12)
ax.set_ylabel("Decode Throughput (tok/s)", fontsize=12)
ax.set_title(f"Decode (Generation) Throughput per {INTERVAL_S}s Interval  (Qwen3-32B, H100, 16k context, rate=10, rep=5)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "gen_throughput.png"), dpi=150)
plt.close()
print("  gen_throughput.png")

# ---- Plot 3: Cumulative Prompt Tokens ----
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(t_cxl, cp_cxl / 1e6, label="CXL Prefix Caching", linewidth=2, color="#1f77b4")
ax.plot(t_gpu, cp_gpu / 1e6, label="GPU Only (No Prefix)", linewidth=2, color="#ff7f0e")
ax.set_xlabel("Simulation Time (s)", fontsize=12)
ax.set_ylabel("Cumulative Prompt Tokens (M)", fontsize=12)
ax.set_title("Cumulative Prompt Token Progress", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "cumulative_tokens.png"), dpi=150)
plt.close()
print("  cumulative_tokens.png")

# Verify totals
print(f"\nCXL  cumulative total: {cp_cxl[-1]:,} prompt tokens")
print(f"GPU  cumulative total: {cp_gpu[-1]:,} prompt tokens")
print(f"Done. Plots in {OUTDIR}/")
