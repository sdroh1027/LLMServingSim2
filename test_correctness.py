"""Correctness test: AstraSim (ground truth) vs file-bypass vs mem-bypass.

Runs the same simulation three ways and compares every request's
end_time, latency, TTFT, and TPOT across all pairs.
"""
import subprocess, sys, time, csv, os

TESTS = [
    {
        "name": "example_trace (10 req)",
        "dataset": "dataset/example_trace.jsonl",
        "cluster": "cluster_config/single_node_single_instance.json",
        "num_req": 100,
    },
    {
        "name": "sharegpt_llama (300 req)",
        "dataset": "dataset/sharegpt_req300_rate10_llama.jsonl",
        "cluster": "cluster_config/single_node_single_instance.json",
        "num_req": 300,
    },
]

FIELDS_TO_COMPARE = ["end_time", "latency", "TTFT", "TPOT"]

def run_sim(label, dataset, cluster, num_req, output_csv, extra_args=None):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        sys.executable, "main.py",
        "--cluster-config", cluster,
        "--fp", "16",
        "--block-size", "16",
        "--dataset", dataset,
        "--num-req", str(num_req),
        "--log-interval", "1.0",
        "--output", output_csv,
    ]
    if extra_args:
        cmd += extra_args
    print(f"  Running {label}...", end=" ", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"FAILED ({elapsed:.1f}s)")
        print(f"    stderr: {r.stderr[-300:]}")
        return None, elapsed
    print(f"done ({elapsed:.1f}s)")
    return read_csv(output_csv), elapsed


def read_csv(path):
    with open(path, 'r') as f:
        return {row['request id']: row for row in csv.DictReader(f)}


def compare(name_a, rows_a, name_b, rows_b):
    """Compare two result sets. Returns (num_checked, mismatches_list)."""
    if rows_a is None or rows_b is None:
        return 0, [f"  !! Skipped — one of the runs failed"]

    ids_a = set(rows_a.keys())
    ids_b = set(rows_b.keys())
    mismatches = []

    if ids_a != ids_b:
        mismatches.append(f"  !! Request ID sets differ: {name_a} has {len(ids_a)}, {name_b} has {len(ids_b)}")
        if ids_a - ids_b:
            mismatches.append(f"     Only in {name_a}: {sorted(ids_a - ids_b)[:10]}")
        if ids_b - ids_a:
            mismatches.append(f"     Only in {name_b}: {sorted(ids_b - ids_a)[:10]}")

    common = sorted(ids_a & ids_b, key=int)
    max_rel_err = {}
    for field in FIELDS_TO_COMPARE:
        max_rel_err[field] = 0.0

    for rid in common:
        ra, rb = rows_a[rid], rows_b[rid]
        for field in FIELDS_TO_COMPARE:
            va, vb = int(ra[field]), int(rb[field])
            if va != vb:
                rel = abs(va - vb) / max(abs(va), 1) * 100
                if rel > max_rel_err[field]:
                    max_rel_err[field] = rel
                if rel > 1.0:  # only report >1% difference
                    mismatches.append(
                        f"  !! req {rid} {field}: {name_a}={va}  {name_b}={vb}  diff={abs(va-vb)}  ({rel:.3f}%)"
                    )

    return len(common), mismatches, max_rel_err


def print_header(text):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {text}")
    print(bar)


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
os.makedirs("output", exist_ok=True)
all_pass = True

for test in TESTS:
    print_header(f"TEST: {test['name']}")

    # 1) AstraSim (ground truth)
    rows_astrasim, t_astrasim = run_sim(
        "AstraSim (ground truth)",
        test["dataset"], test["cluster"], test["num_req"],
        "output/_test_astrasim.csv",
    )

    # 2) File-based bypass
    rows_file, t_file = run_sim(
        "File-based bypass",
        test["dataset"], test["cluster"], test["num_req"],
        "output/_test_file.csv",
        extra_args=["--bypass-astrasim", "--bypass-use-file"],
    )

    # 3) In-memory bypass
    rows_mem, t_mem = run_sim(
        "In-memory bypass",
        test["dataset"], test["cluster"], test["num_req"],
        "output/_test_mem.csv",
        extra_args=["--bypass-astrasim"],
    )

    # ── Timing ────────────────────────────────────────────
    print(f"\n  Timing:")
    print(f"    AstraSim:       {t_astrasim:7.2f}s")
    print(f"    File bypass:    {t_file:7.2f}s  ({t_astrasim/t_file:.1f}x vs AstraSim)")
    print(f"    Memory bypass:  {t_mem:7.2f}s  ({t_astrasim/t_mem:.1f}x vs AstraSim)")

    # ── Correctness ───────────────────────────────────────
    pairs = [
        ("AstraSim", rows_astrasim, "File-bypass", rows_file),
        ("AstraSim", rows_astrasim, "Mem-bypass",  rows_mem),
        ("File-bypass", rows_file,  "Mem-bypass",  rows_mem),
    ]

    print(f"\n  Correctness:")
    for name_a, ra, name_b, rb in pairs:
        n, mismatches, max_err = compare(name_a, ra, name_b, rb)
        label = f"{name_a} vs {name_b}"

        if not mismatches:
            # Show max relative error even if under threshold
            err_summary = ", ".join(f"{k}={v:.4f}%" for k, v in max_err.items() if v > 0)
            if err_summary:
                print(f"    {label}: {n} requests OK (max rel err: {err_summary})")
            else:
                print(f"    {label}: {n} requests EXACT MATCH")
        else:
            all_pass = False
            print(f"    {label}: {len(mismatches)} issues")
            for m in mismatches[:10]:
                print(f"      {m}")
            if len(mismatches) > 10:
                print(f"      ... and {len(mismatches)-10} more")

# ── Cleanup ───────────────────────────────────────────────
for f in ["output/_test_astrasim.csv", "output/_test_file.csv", "output/_test_mem.csv"]:
    if os.path.exists(f):
        os.remove(f)

print_header("FINAL VERDICT")
if all_pass:
    print("  ALL TESTS PASSED")
else:
    print("  SOME TESTS FAILED — see above")
