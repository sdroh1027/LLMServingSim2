"""Benchmark: original bypass (file-based) vs optimized bypass (in-memory).

Runs the same simulation twice with the same JSONL trace and compares:
  1. Per-request cycle correctness (every request must match)
  2. Wall-clock time of each run
"""
import subprocess, sys, time, csv, os

DATASET   = "dataset/sharegpt_req300_rate10_llama.jsonl"
CLUSTER   = "cluster_config/single_node_single_instance.json"
NUM_REQ   = 300
FP        = 16
BLOCK_SZ  = 16
LOG_INT   = 1.0

BASE_CMD  = [
    sys.executable, "main.py",
    "--cluster-config", CLUSTER,
    "--fp", str(FP),
    "--block-size", str(BLOCK_SZ),
    "--dataset", DATASET,
    "--num-req", str(NUM_REQ),
    "--log-interval", str(LOG_INT),
    "--bypass-astrasim",
]

def run(label, output_csv, extra_args=None):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = BASE_CMD + ["--output", output_csv]
    if extra_args:
        cmd += extra_args
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    lines = r.stdout.strip().split('\n')
    for l in lines[-8:]:
        print(f"  {l}")
    if r.returncode != 0:
        print(f"  STDERR: {r.stderr[-500:]}")
    return elapsed

def read_csv(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

# ── Run both ──────────────────────────────────────────────
out_file  = "output/_bench_file.csv"
out_mem   = "output/_bench_mem.csv"

t_file = run("ORIGINAL (file-based bypass)", out_file, extra_args=["--bypass-use-file"])
t_mem  = run("OPTIMIZED (in-memory bypass)", out_mem)

# ── Compare ───────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RESULTS")
print(f"{'='*60}")
print(f"  Original (file):   {t_file:.2f} s")
print(f"  Optimized (mem):   {t_mem:.2f} s")
print(f"  Speedup:           {t_file/t_mem:.2f}x")

rows_f = read_csv(out_file)
rows_m = read_csv(out_mem)

print(f"\n  Requests: file={len(rows_f)}, mem={len(rows_m)}")
mismatches = 0
for rf, rm in zip(rows_f, rows_m):
    if rf['request id'] != rm['request id']:
        print(f"  !! Request ID mismatch: {rf['request id']} vs {rm['request id']}")
        mismatches += 1
        continue
    if rf['end_time'] != rm['end_time']:
        diff_ns = abs(int(rf['end_time']) - int(rm['end_time']))
        print(f"  !! req {rf['request id']}: end_time diff = {diff_ns} ns  (file={rf['end_time']}, mem={rm['end_time']})")
        mismatches += 1
    if rf['latency'] != rm['latency']:
        diff_ns = abs(int(rf['latency']) - int(rm['latency']))
        print(f"  !! req {rf['request id']}: latency diff = {diff_ns} ns")
        mismatches += 1

if mismatches == 0:
    print(f"  All {len(rows_f)} requests MATCH perfectly.")
else:
    print(f"  {mismatches} mismatches found!")
