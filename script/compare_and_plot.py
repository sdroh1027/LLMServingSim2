"""
두 실험 결과 비교표 + Throughput per {args.interval} 시각화
Usage:
  python script/compare_and_plot.py \
    --cxl   output/lveval_hotpot_16k_qwen3-32b_rate10.0_rep5_cxl \
    --gpu   output/lveval_hotpot_16k_qwen3-32b_rate10.0_rep5_onlygpu \
    --out   output/lveval_hotpot_16k_qwen3-32b_rate10.0_rep5_comparison
"""

import argparse
import csv
import os
import ast
import numpy as np

def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def ns_to_ms(v): return float(v) / 1e6
def ns_to_s(v):  return float(v) / 1e9

def compute_stats(rows):
    n = len(rows)
    latencies = [ns_to_ms(r["latency"]) for r in rows]
    ttfts     = [ns_to_ms(r["TTFT"]) for r in rows]
    tpots     = [ns_to_ms(r["TPOT"]) for r in rows]

    # ITL: stored as stringified list — parse each
    all_itls = []
    for r in rows:
        itl_raw = r["ITL"]
        try:
            itl_list = ast.literal_eval(itl_raw)
            all_itls.extend([v / 1e6 for v in itl_list])  # ns -> ms
        except:
            pass

    total_input  = sum(int(r["input"])  for r in rows)
    total_output = sum(int(r["output"]) for r in rows)

    # total sim time: max end_time - min arrival
    arrivals  = [int(r["arrival"])  for r in rows]
    end_times = [int(r["end_time"]) for r in rows]
    total_time_s = (max(end_times) - min(arrivals)) / 1e9

    # prefix hit
    npu_hits     = sum(int(r["npu_cache_hit"])     for r in rows)
    storage_hits = sum(int(r["storage_cache_hit"]) for r in rows)
    prefix_hits  = sum(int(r["prefix_cache_hit"])  for r in rows)

    return {
        "num_req": n,
        "total_input_tokens":  total_input,
        "total_output_tokens": total_output,
        "total_time_s":        total_time_s,
        "req_throughput":      n / total_time_s,
        "prompt_throughput":   total_input / total_time_s,
        "gen_throughput":      total_output / total_time_s,
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

def compute_throughput_timeseries(rows, interval_s=0.5):
    """Reconstruct throughput per interval from output CSV.
    For each interval, count tokens that completed (end_time falls in interval)."""
    end_times = [(int(r["end_time"]), int(r["input"]), int(r["output"])) for r in rows]
    if not end_times:
        return [], [], []
    min_t = min(int(r["arrival"]) for r in rows)
    max_t = max(et for et, _, _ in end_times)
    interval_ns = int(interval_s * 1e9)

    time_points = []
    prompt_thr = []
    gen_thr = []

    t = min_t
    while t < max_t:
        t_end = t + interval_ns
        p_toks = 0
        g_toks = 0
        for et, inp, out in end_times:
            if t <= et < t_end:
                p_toks += inp
                g_toks += out
        time_points.append((t - min_t) / 1e9)  # seconds from start
        prompt_thr.append(p_toks / interval_s)
        gen_thr.append(g_toks / interval_s)
        t = t_end

    return time_points, prompt_thr, gen_thr

def print_comparison(stats_cxl, stats_gpu):
    fmt = "{:<45s} {:>20s} {:>20s}"
    sep = "-" * 87

    print(sep)
    print(fmt.format("Metric", "CXL Prefix", "GPU Only"))
    print(sep)

    def row(label, key, unit="", prec=2):
        vc = stats_cxl[key]
        vg = stats_gpu[key]
        if isinstance(vc, float):
            sc = f"{vc:.{prec}f}{unit}"
            sg = f"{vg:.{prec}f}{unit}"
        else:
            sc = f"{vc}{unit}"
            sg = f"{vg}{unit}"
        print(fmt.format(label, sc, sg))

    row("Requests",             "num_req")
    row("Total input tokens",   "total_input_tokens")
    row("Total output tokens",  "total_output_tokens")
    row("Total sim time",       "total_time_s",       " s")
    print(sep)
    row("Request throughput",   "req_throughput",     " req/s")
    row("Prompt throughput",    "prompt_throughput",  " tok/s", 1)
    row("Generation throughput","gen_throughput",     " tok/s", 1)
    print(sep)
    row("Mean Latency",         "mean_latency_ms",    " ms", 1)
    row("Mean TTFT",            "mean_ttft_ms",       " ms", 1)
    row("Median TTFT",          "median_ttft_ms",     " ms", 1)
    row("P99 TTFT",             "p99_ttft_ms",        " ms", 1)
    row("Mean TPOT",            "mean_tpot_ms",       " ms", 1)
    row("Median TPOT",          "median_tpot_ms",     " ms", 1)
    row("P99 TPOT",             "p99_tpot_ms",        " ms", 1)
    row("Mean ITL",             "mean_itl_ms",        " ms", 1)
    row("Median ITL",           "median_itl_ms",      " ms", 1)
    row("P99 ITL",              "p99_itl_ms",         " ms", 1)
    print(sep)
    row("GPU prefix hit tokens",    "npu_cache_hit")
    row("Storage prefix hit tokens","storage_cache_hit")
    row("Total prefix hit tokens",  "prefix_cache_hit")
    print(sep)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxl", required=True, help="CXL experiment output CSV")
    parser.add_argument("--gpu", required=True, help="GPU-only experiment output CSV")
    parser.add_argument("--out", required=True, help="Output directory for plots and summary")
    parser.add_argument("--interval", type=float, default=0.5, help="Throughput interval (sec)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    rows_cxl = load_csv(args.cxl)
    rows_gpu = load_csv(args.gpu)

    stats_cxl = compute_stats(rows_cxl)
    stats_gpu = compute_stats(rows_gpu)

    # Print comparison table
    print_comparison(stats_cxl, stats_gpu)

    # Save comparison table to file
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_comparison(stats_cxl, stats_gpu)
    summary_path = os.path.join(args.out, "comparison_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"\nSummary saved to {summary_path}")

    # Throughput timeseries
    t_cxl, p_cxl, g_cxl = compute_throughput_timeseries(rows_cxl, args.interval)
    t_gpu, p_gpu, g_gpu = compute_throughput_timeseries(rows_gpu, args.interval)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Prompt throughput
    ax = axes[0]
    ax.plot(t_cxl, [v/1000 for v in p_cxl], label="CXL Prefix Caching", alpha=0.8, linewidth=1)
    ax.plot(t_gpu, [v/1000 for v in p_gpu], label="GPU Only", alpha=0.8, linewidth=1)
    ax.set_ylabel("Prompt Throughput (k tok/s)")
    ax.set_title(f"Prompt Throughput per {args.interval}s interval")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Generation throughput
    ax = axes[1]
    ax.plot(t_cxl, g_cxl, label="CXL Prefix Caching", alpha=0.8, linewidth=1)
    ax.plot(t_gpu, g_gpu, label="GPU Only", alpha=0.8, linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Generation Throughput (tok/s)")
    ax.set_title(f"Generation Throughput per {args.interval}s interval")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.out, "throughput_comparison.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()

    # TTFT CDF
    fig, ax = plt.subplots(figsize=(10, 5))
    ttft_cxl = sorted([ns_to_ms(r["TTFT"]) for r in rows_cxl])
    ttft_gpu = sorted([ns_to_ms(r["TTFT"]) for r in rows_gpu])
    ax.plot(ttft_cxl, np.linspace(0, 1, len(ttft_cxl)), label="CXL Prefix Caching", linewidth=1.5)
    ax.plot(ttft_gpu, np.linspace(0, 1, len(ttft_gpu)), label="GPU Only", linewidth=1.5)
    ax.set_xlabel("TTFT (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("TTFT CDF Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    cdf_path = os.path.join(args.out, "ttft_cdf_comparison.png")
    plt.savefig(cdf_path, dpi=150)
    print(f"TTFT CDF saved to {cdf_path}")
    plt.close()

if __name__ == "__main__":
    main()
