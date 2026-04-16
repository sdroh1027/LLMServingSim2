"""
4-way experiment comparison: onlygpu / cxl2560 / RDMA352 / RDMA2560
Generates plots + markdown report from output CSVs.
"""
import csv, ast, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "output/lveval_hotpot_16k_qwen3-32b_rate10.0_rep5_comparison"
os.makedirs(OUTDIR, exist_ok=True)

EXPERIMENTS = {
    "GPU Only":   "output/lveval_hotpot_16k_qwen3-32b_rate10.0_rep5_onlygpu",
    "CXL 2560GB": "output/lveval_hotpot_16k_qwen3-32b_rate10.0_rep5_cxl2560",
    "RDMA 352GB": "output/lveval_hotpot_16k_qwen3-32b_rate10.0_rep5_RDMA352",
    "RDMA 2560GB":"output/lveval_hotpot_16k_qwen3-32b_rate10.0_rep5_RDMA2560",
}

COLORS = {
    "GPU Only":    "#ff7f0e",
    "CXL 2560GB":  "#1f77b4",
    "RDMA 352GB":  "#2ca02c",
    "RDMA 2560GB": "#d62728",
}

INTERVAL_S = 2.0

def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def ns_to_ms(v): return float(v) / 1e6

def compute_stats(rows):
    n = len(rows)
    latencies = [ns_to_ms(r["latency"]) for r in rows]
    ttfts     = [ns_to_ms(r["TTFT"]) for r in rows]
    tpots     = [ns_to_ms(r["TPOT"]) for r in rows]
    all_itls = []
    for r in rows:
        try:
            itl_list = ast.literal_eval(r["ITL"])
            all_itls.extend([v / 1e6 for v in itl_list])
        except: pass

    total_input  = sum(int(r["input"]) for r in rows)
    total_output = sum(int(r["output"]) - int(r["input"]) for r in rows)
    arrivals  = [int(r["arrival"])  for r in rows]
    end_times = [int(r["end_time"]) for r in rows]
    total_time_s = (max(end_times) - min(arrivals)) / 1e9

    npu_hits     = sum(int(r["npu_cache_hit"])     for r in rows)
    storage_hits = sum(int(r["storage_cache_hit"]) for r in rows)
    prefix_hits  = sum(int(r["prefix_cache_hit"])  for r in rows)

    return {
        "num_req": n,
        "total_input_tokens":  total_input,
        "total_output_tokens": total_output,
        "total_time_s":        total_time_s,
        "req_throughput":      n / total_time_s,
        "prefill_throughput":  total_input / total_time_s,
        "decode_throughput":   total_output / total_time_s,
        "mean_latency_ms":     np.mean(latencies),
        "mean_ttft_ms":        np.mean(ttfts),
        "median_ttft_ms":      np.median(ttfts),
        "p99_ttft_ms":         np.percentile(ttfts, 99),
        "mean_tpot_ms":        np.mean(tpots),
        "median_tpot_ms":      np.median(tpots),
        "p99_tpot_ms":         np.percentile(tpots, 99),
        "mean_itl_ms":         np.mean(all_itls) if all_itls else 0,
        "median_itl_ms":       np.median(all_itls) if all_itls else 0,
        "p99_itl_ms":          np.percentile(all_itls, 99) if all_itls else 0,
        "npu_cache_hit":       npu_hits,
        "storage_cache_hit":   storage_hits,
        "prefix_cache_hit":    prefix_hits,
    }

def build_timeseries(rows, interval_s):
    data = [(int(r["arrival"]), int(r["end_time"]), int(r["input"]), int(r["output"]) - int(r["input"])) for r in rows]
    min_t = min(d[0] for d in data)
    max_t = max(d[1] for d in data)
    interval_ns = int(interval_s * 1e9)
    times, prefill_thr, decode_thr, cum_prefill = [], [], [], []
    total_p = 0
    t = min_t
    while t < max_t + interval_ns:
        t_end = t + interval_ns
        p, g = 0, 0
        for arr, et, inp, out in data:
            if t <= et < t_end:
                p += inp
                g += out
        total_p += p
        times.append((t - min_t) / 1e9)
        prefill_thr.append(p / interval_s)
        decode_thr.append(g / interval_s)
        cum_prefill.append(total_p)
        t = t_end
    return np.array(times), np.array(prefill_thr), np.array(decode_thr), np.array(cum_prefill)

# Load all
print("Loading CSVs...")
all_rows = {}
all_stats = {}
all_ts = {}
for name, path in EXPERIMENTS.items():
    rows = load_csv(path)
    all_rows[name] = rows
    all_stats[name] = compute_stats(rows)
    all_ts[name] = build_timeseries(rows, INTERVAL_S)
    print(f"  {name}: {len(rows)} rows, {all_stats[name]['total_time_s']:.1f}s")

# ---- Plot 1: Prefill Throughput ----
fig, ax = plt.subplots(figsize=(15, 5))
for name in EXPERIMENTS:
    t, p, _, _ = all_ts[name]
    ax.plot(t, p / 1000, label=name, alpha=0.8, linewidth=0.9, color=COLORS[name])
ax.set_xlabel("Simulation Time (s)", fontsize=12)
ax.set_ylabel("Prefill Throughput (k tok/s)", fontsize=12)
ax.set_title(f"Prefill (Prompt) Throughput per {INTERVAL_S}s  (Qwen3-32B, H100, 16k, rate=10, rep=5)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "prefill_throughput_4way.png"), dpi=150)
plt.close()
print("  prefill_throughput_4way.png")

# ---- Plot 2: Decode Throughput ----
fig, ax = plt.subplots(figsize=(15, 5))
for name in EXPERIMENTS:
    t, _, g, _ = all_ts[name]
    ax.plot(t, g, label=name, alpha=0.8, linewidth=0.9, color=COLORS[name])
ax.set_xlabel("Simulation Time (s)", fontsize=12)
ax.set_ylabel("Decode Throughput (tok/s)", fontsize=12)
ax.set_title(f"Decode (Generation) Throughput per {INTERVAL_S}s  (Qwen3-32B, H100, 16k, rate=10, rep=5)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "decode_throughput_4way.png"), dpi=150)
plt.close()
print("  decode_throughput_4way.png")

# ---- Plot 3: Cumulative Prefill Tokens ----
fig, ax = plt.subplots(figsize=(15, 5))
for name in EXPERIMENTS:
    t, _, _, cp = all_ts[name]
    ax.plot(t, cp / 1e6, label=name, linewidth=2, color=COLORS[name])
ax.set_xlabel("Simulation Time (s)", fontsize=12)
ax.set_ylabel("Cumulative Prefill Tokens (M)", fontsize=12)
ax.set_title("Cumulative Prefill Token Progress", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "cumulative_tokens_4way.png"), dpi=150)
plt.close()
print("  cumulative_tokens_4way.png")

# ---- Plot 4: TTFT CDF ----
fig, ax = plt.subplots(figsize=(10, 5))
for name in EXPERIMENTS:
    ttfts = sorted([ns_to_ms(r["TTFT"]) for r in all_rows[name]])
    ax.plot(ttfts, np.linspace(0, 1, len(ttfts)), label=name, linewidth=1.5, color=COLORS[name])
ax.set_xlabel("TTFT (ms)", fontsize=12)
ax.set_ylabel("CDF", fontsize=12)
ax.set_title("TTFT CDF Comparison", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "ttft_cdf_4way.png"), dpi=150)
plt.close()
print("  ttft_cdf_4way.png")

# ---- Plot 5: Bar chart summary ----
names = list(EXPERIMENTS.keys())
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
metrics = [
    ("total_time_s",       "Total Sim Time (s)"),
    ("prefill_throughput",  "Prefill Throughput (tok/s)"),
    ("mean_ttft_ms",       "Mean TTFT (ms)"),
    ("median_tpot_ms",     "Median TPOT (ms)"),
]
for ax, (key, label) in zip(axes, metrics):
    vals = [all_stats[n][key] for n in names]
    bars = ax.bar(range(len(names)), vals, color=[COLORS[n] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax.set_title(label, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{v:,.0f}", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "bar_summary_4way.png"), dpi=150)
plt.close()
print("  bar_summary_4way.png")

print(f"\nAll plots saved to {OUTDIR}/")

# ---- Generate markdown table data ----
def fmt(v, prec=1):
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.{prec}f}"
        return f"{v:.{prec}f}"
    return f"{v:,}"

gpu_stats = all_stats["GPU Only"]
md_lines = []

md_lines.append("# LVEval 16k Qwen3-32B Rate=10 Rep=5: 4-Way Comparison")
md_lines.append("")
md_lines.append("## Experiment Configuration")
md_lines.append("")
md_lines.append("| Config | Storage | Size | BW (GB/s) | Latency (ns) | Devices |")
md_lines.append("|---|---|---|---|---|---|")
md_lines.append("| GPU Only | None | - | - | - | - |")
md_lines.append("| CXL 2560GB | CXL | 2,560 GB | 19.5 | 500 | 10 |")
md_lines.append("| RDMA 352GB | RDMA | 352 GB | 16 | 5,000 | 1 |")
md_lines.append("| RDMA 2560GB | RDMA | 2,560 GB | 16 | 5,000 | 1 |")
md_lines.append("")
md_lines.append("Common: Qwen3-32B, H100 96GB, 620 req (124 unique x 5 rep, random sampling), rate=10 req/s, FP16, block=16")
md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## Summary Comparison")
md_lines.append("")

header = "| Metric |"
sep = "|---|"
for n in names:
    header += f" {n} |"
    sep += "---|"
md_lines.append(header)
md_lines.append(sep)

rows_md = [
    ("Total Sim Time (s)",           "total_time_s",       1),
    ("Request Throughput (req/s)",    "req_throughput",     2),
    ("Prefill Throughput (tok/s)",    "prefill_throughput", 0),
    ("Decode Throughput (tok/s)",     "decode_throughput",  2),
    ("Mean TTFT (ms)",               "mean_ttft_ms",       0),
    ("Median TTFT (ms)",             "median_ttft_ms",     0),
    ("P99 TTFT (ms)",                "p99_ttft_ms",        0),
    ("Mean TPOT (ms)",               "mean_tpot_ms",       0),
    ("Median TPOT (ms)",             "median_tpot_ms",     0),
    ("P99 TPOT (ms)",                "p99_tpot_ms",        0),
    ("Mean ITL (ms)",                "mean_itl_ms",        0),
    ("Median ITL (ms)",              "median_itl_ms",      0),
    ("P99 ITL (ms)",                 "p99_itl_ms",         0),
    ("GPU Prefix Hit",               "npu_cache_hit",      0),
    ("Storage Prefix Hit",           "storage_cache_hit",  0),
    ("Total Prefix Hit",             "prefix_cache_hit",   0),
]
for label, key, prec in rows_md:
    line = f"| **{label}** |"
    for n in names:
        v = all_stats[n][key]
        line += f" {fmt(v, prec)} |"
    md_lines.append(line)

md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## Plots")
md_lines.append("")
md_lines.append("### Prefill (Prompt) Throughput")
md_lines.append("![Prefill Throughput](prefill_throughput_4way.png)")
md_lines.append("")
md_lines.append("### Decode (Generation) Throughput")
md_lines.append("![Decode Throughput](decode_throughput_4way.png)")
md_lines.append("")
md_lines.append("### Cumulative Prefill Token Progress")
md_lines.append("![Cumulative](cumulative_tokens_4way.png)")
md_lines.append("")
md_lines.append("### TTFT CDF")
md_lines.append("![TTFT CDF](ttft_cdf_4way.png)")
md_lines.append("")
md_lines.append("### Summary Bar Chart")
md_lines.append("![Bar Summary](bar_summary_4way.png)")

md_path = os.path.join(OUTDIR, "experiment_report_4way.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"Report saved to {md_path}")
