"""QPS timeseries for 3 experiments (CPU no-share / RDMA share / CXL share)."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "output/lveval_hotpot_16k_qwen3-32b_rate10.0_rep5_comparison"
os.makedirs(OUTDIR, exist_ok=True)

EXPERIMENTS = {
    "CPU (no share)":  "output/lveval_hotpot_16k_qwen3-32b_rate10.0_rep5_CPU_no_share.csv",
    "RDMA (share)":    "output/lveval_hotpot_16k_qwen3-32b_rate10.0_rep5_RDMA.csv",
    "CXL (share)":     "output/lveval_hotpot_16k_qwen3-32b_rate10.0_rep5_CXL.csv",
}
COLORS = {
    "CPU (no share)": "#ff7f0e",
    "RDMA (share)":   "#2ca02c",
    "CXL (share)":    "#1f77b4",
}
INTERVAL_S = 2.0

def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def qps_series(rows, interval_s):
    arrivals = [int(r["arrival"]) for r in rows]
    ends = [int(r["end_time"]) for r in rows]
    min_t = min(arrivals)
    max_t = max(ends)
    interval_ns = int(interval_s * 1e9)
    times, qps, cum = [], [], []
    done = 0
    t = min_t
    while t < max_t + interval_ns:
        t_end = t + interval_ns
        n = sum(1 for e in ends if t <= e < t_end)
        done += n
        times.append((t - min_t) / 1e9)
        qps.append(n / interval_s)
        cum.append(done)
        t = t_end
    return np.array(times), np.array(qps), np.array(cum)

data = {n: qps_series(load(p), INTERVAL_S) for n, p in EXPERIMENTS.items()}

# Plot 1: instantaneous QPS
fig, ax = plt.subplots(figsize=(14, 5))
for n in EXPERIMENTS:
    t, q, _ = data[n]
    ax.plot(t, q, label=n, linewidth=1.2, color=COLORS[n], alpha=0.85)
ax.set_xlabel("Simulation Time (s)")
ax.set_ylabel("QPS (req/s)")
ax.set_title(f"Instantaneous QPS per {INTERVAL_S}s interval  (Qwen3-32B, 5x H100, 16k, rate=10, rep=5)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
p1 = os.path.join(OUTDIR, "qps_3way.png")
plt.savefig(p1, dpi=150); plt.close()
print(p1)

# Plot 2: cumulative completed requests
fig, ax = plt.subplots(figsize=(14, 5))
for n in EXPERIMENTS:
    t, _, c = data[n]
    ax.plot(t, c, label=n, linewidth=2, color=COLORS[n])
ax.set_xlabel("Simulation Time (s)")
ax.set_ylabel("Completed Requests")
ax.set_title("Cumulative Completed Requests")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
p2 = os.path.join(OUTDIR, "qps_cumulative_3way.png")
plt.savefig(p2, dpi=150); plt.close()
print(p2)

# Summary
print("\nAvg QPS (total reqs / total sim time):")
for n, p in EXPERIMENTS.items():
    rows = load(p)
    arrivals = [int(r["arrival"]) for r in rows]
    ends = [int(r["end_time"]) for r in rows]
    total_s = (max(ends) - min(arrivals)) / 1e9
    print(f"  {n:20s}: {len(rows)/total_s:.2f} req/s  (total {total_s:.1f}s)")
