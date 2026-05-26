import re
import heapq
from .logger import get_logger

class Controller():
    def __init__(self, total_num):
        self.end_dict = {}
        self.total_num = total_num
        self.logger = get_logger(self.__class__)
        for i in range(total_num):
            self.end_dict[i] = -1


    def read_wait(self, p):
        out = [""]
        while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
            line = p.stdout.readline()
            # For debugging
            # print(line, end='')
            out.append(line)
            p.stdout.flush()
        return out

    def check_end(self, p):
        out = ["",""]
        while out[-2] != "All Request Has Been Exited\n" and out[-2] != "ERROR: Some Requests Remain\n":
            out.append(p.stdout.readline())
            p.stdout.flush()
        print(out[-4], end='')
        print(out[-2], end='')
        return out

    def write_flush(self, p, input):
        # For debugging
        # print(input)
        p.stdin.write(input+'\n')
        p.stdin.flush()
        return

    def parse_output(self, output):
        pattern = r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, exposed communication (\d+) cycles."
        match = re.search(pattern, output)
        if match:
            sys = int(match.group(1))
            id = int(match.group(2))
            cycle = int(match.group(3))
            com_cycle = int(match.group(4))

            if self.end_dict[sys] != id:
                self.logger.info(
                    "NPU[%d] iteration %d finished, %d cycles, exposed communication %d cycles.",
                    sys,
                    id,
                    cycle,
                    com_cycle,
                )
                self.end_dict[sys] = id
            return {'sys': sys, 'id': id, 'cycle': cycle}
        return


class BypassController():
    """AstraSim bypass: compute total cycles directly from trace files."""

    def __init__(self, total_num):
        self.total_num = total_num
        self.logger = get_logger(self.__class__)
        # Per-NPU iteration counter (matches AstraSim's iteration numbering)
        self.iteration = {i: 0 for i in range(total_num)}
        # Min-heap of (finish_cycle_ns, npu_id, iteration_id)
        self.events = []

    def submit_event(self, finish_cycle_ns, npu_id, is_timer=False):
        """Add a batch/event completion to the event queue.

        is_timer=True is used for wakeup/wait events that do NOT correspond to
        a real batch completion (initial startup tick, next-arrival wakeup).
        These must NOT advance the iteration counter, because Scheduler.add_done
        looks up batches by `iteration - 1 == batch_id`; an extra iteration
        bump from a filler timer event would permanently desync the mapping
        and leave the batch stuck inflight.
        """
        if is_timer:
            # Sentinel iteration id < 0 → add_done's batch lookup yields no match (no-op).
            heapq.heappush(self.events, (finish_cycle_ns, npu_id, -1))
        else:
            it = self.iteration[npu_id]
            self.iteration[npu_id] += 1
            heapq.heappush(self.events, (finish_cycle_ns, npu_id, it))

    def has_event_for_npu(self, npu_id):
        """Check if any pending event exists for the given NPU."""
        return any(e[1] == npu_id for e in self.events)

    def read_wait(self, p=None):
        """Pop next event from the queue (replaces AstraSim stdout read)."""
        if not self.events:
            return ["", "Checking Non-Exited Systems ...\n"]
        cycle, npu_id, it = heapq.heappop(self.events)
        line = f"sys[{npu_id}] iteration {it} finished, {cycle} cycles, exposed communication 0 cycles.\n"
        return ["", line, "Waiting\n"]

    def write_flush(self, p, input_str):
        """No-op — events are managed via submit_event."""
        pass

    def check_end(self, p=None):
        print("\nChecking Non-Exited Systems ...")
        print("All Request Has Been Exited")
        return ["", "", "Checking Non-Exited Systems ...\n", "All Request Has Been Exited\n"]

    def parse_output(self, output):
        # iteration may be a sentinel -1 for timer-only events (see submit_event)
        pattern = r"sys\[(\d+)\] iteration (-?\d+) finished, (\d+) cycles, exposed communication (\d+) cycles."
        match = re.search(pattern, output)
        if match:
            return {
                'sys': int(match.group(1)),
                'id': int(match.group(2)),
                'cycle': int(match.group(3)),
            }
        return None


# ─────────────────────────────────────────────────────────────────────
# Trace column layout:
#   idx 0: layer_name
#   idx 1: comp_time (ns)
#   idx 2-3: input_loc, input_size      ┐
#   idx 4-5: weight_loc, weight_size    ├─ memory transfer (LOCAL / REMOTE:N / CXL:N)
#   idx 6-7: output_loc, output_size    ┘
#   idx 8-9: comm_type, comm_size       ← TP collective (ALLREDUCE / ALLTOALL / NONE)
#   idx 10:  misc
#
# Memory transfer: per-device bw/latency (cpu_mem, cxl_mem)
# TP collective:   link_bw/link_latency (inter-node or NVLink)
#   → only relevant when npu_num > 1 (bypass mode is single-node only,
#     but the calculation is included for future multi-GPU support)
# ─────────────────────────────────────────────────────────────────────

def _calc_transfer_ns(loc, data_size, cpu_bw_bpns, cpu_latency, cxl_bw_bpns, cxl_latency):
    """Return transfer time (ns) for a single memory access given its location tag.

    Supports REMOTE (CPU memory) and CXL device locations.
    Returns 0 for LOCAL or when bandwidth is not configured.
    """
    if data_size <= 0:
        return 0
    if loc.startswith('REMOTE') and cpu_bw_bpns > 0:
        return cpu_latency + int(data_size / cpu_bw_bpns)
    if loc.startswith('CXL') and cxl_bw_bpns > 0:
        return cxl_latency + int(data_size / cxl_bw_bpns)
    return 0


def _calc_collective_ns(comm_type, comm_size, npu_num, link_bw_bpns, link_latency):
    """Return TP collective communication time (ns).

    Simplified ring-based model:
      ALLREDUCE:  2 * (npu_num-1)/npu_num * data_size / link_bw + link_latency
      ALLTOALL:   (npu_num-1)/npu_num * data_size / link_bw + link_latency
      ALLGATHER:  (npu_num-1)/npu_num * data_size / link_bw + link_latency

    Returns 0 when npu_num <= 1 or comm_type is NONE.
    """
    if npu_num <= 1 or comm_type == 'NONE' or link_bw_bpns <= 0 or comm_size <= 0:
        return 0
    ratio = (npu_num - 1) / npu_num
    if comm_type == 'ALLREDUCE':
        return link_latency + int(2 * ratio * comm_size / link_bw_bpns)
    elif comm_type in ('ALLTOALL', 'ALLGATHER', 'REDUCESCATTER'):
        return link_latency + int(ratio * comm_size / link_bw_bpns)
    return 0


def compute_trace_cycles(trace_path, cpu_bw=0, cpu_latency=0, cxl_bw=0, cxl_latency=0):
    """Read a trace .txt file and return total time in nanoseconds.

    Sums comp_time for all layers and adds exposed communication time
    for REMOTE/CXL memory accesses (input/weight/output), matching
    AstraSim's analytical backend behaviour.

    Args:
        trace_path: path to the trace .txt file
        cpu_bw: CPU memory bandwidth in GB/s (0 = ignore REMOTE)
        cpu_latency: CPU memory access latency in ns
        cxl_bw: CXL device bandwidth in GB/s (0 = ignore CXL)
        cxl_latency: per-access CXL latency in ns
    """
    with open(trace_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 3:
        return 0

    num_layers = int(lines[1].strip())
    total_compute_ns = 0
    cpu_transfer_ns = 0
    cxl_transfer_ns = 0
    other_comm_ns = 0
    # GB/s  →  bytes per ns  (1 GB/s == 1 byte/ns)
    cpu_bw_bpns = cpu_bw if cpu_bw > 0 else 0
    cxl_bw_bpns = cxl_bw if cxl_bw > 0 else 0
    has_remote = cpu_bw_bpns > 0 or cxl_bw_bpns > 0

    for i in range(3, min(3 + num_layers, len(lines))):
        cols = lines[i].split()
        if not cols or len(cols) < 2:
            continue

        name = cols[0]
        # Skip marker rows (MoE expert boundaries, PIM boundaries)
        if name in ('EXPERT', 'PIM') or name.startswith('EXPERT ') or name.startswith('PIM '):
            continue

        try:
            comp_time = int(cols[1])
            total_compute_ns += comp_time
        except (ValueError, IndexError):
            continue

        is_kv = name.startswith('kv_load') or name.startswith('kv_evict')

        # Trace columns: name comp input_loc input_size weight_loc weight_size
        #                output_loc output_size comm_type comm_size misc
        if has_remote and len(cols) >= 8:
            for loc_idx, size_idx in [(2, 3), (4, 5), (6, 7)]:
                try:
                    loc = cols[loc_idx]
                    data_size = int(cols[size_idx])
                    if data_size <= 0:
                        continue
                    if loc.startswith('REMOTE') and cpu_bw_bpns > 0:
                        t = cpu_latency + int(data_size / cpu_bw_bpns)
                        if is_kv:
                            cpu_transfer_ns += t
                        else:
                            other_comm_ns += t
                    elif loc.startswith('CXL') and cxl_bw_bpns > 0:
                        t = cxl_latency + int(data_size / cxl_bw_bpns)
                        if is_kv:
                            cxl_transfer_ns += t
                        else:
                            other_comm_ns += t
                except (ValueError, IndexError):
                    pass

    # AstraSim model: kv_load/kv_evict/input_load run in parallel (MEM nodes),
    # all must complete before first compute. Compute layers run sequentially.
    pre_comp_comm = max(cpu_transfer_ns, cxl_transfer_ns, other_comm_ns)
    return pre_comp_comm + total_compute_ns



def compute_trace_cycles_mem_detail(layers, cpu_bw=0, cpu_latency=0, cxl_bw=0, cxl_latency=0):
    """In-memory version of compute_trace_cycles with breakdown dict.

    Returns:
        dict with keys:
            total_ns:         total batch time (pre_comp_comm + compute)
            compute_ns:       pure layer computation time (sequential)
            cpu_transfer_ns:  kv_load/kv_evict via REMOTE (CPU→HBM)
            cxl_transfer_ns:  kv_load/kv_evict via CXL (CXL→HBM)
            comm_ns:          other (non-kv) remote/CXL transfer time
            pre_comp_comm_ns: max(cpu, cxl, other) — blocking time before compute
    """
    if not layers:
        return {'total_ns': 0, 'compute_ns': 0, 'cpu_transfer_ns': 0,
                'cxl_transfer_ns': 0, 'comm_ns': 0, 'exposed_comm_ns': 0}

    total_compute_ns = 0
    cpu_transfer_ns = 0
    cxl_transfer_ns = 0
    other_comm_ns = 0
    cpu_bw_bpns = cpu_bw if cpu_bw > 0 else 0
    cxl_bw_bpns = cxl_bw if cxl_bw > 0 else 0
    has_remote = cpu_bw_bpns > 0 or cxl_bw_bpns > 0

    for cols in layers:
        if not cols or len(cols) < 2:
            continue

        name = cols[0]
        # Skip marker rows (MoE expert boundaries, PIM boundaries)
        if name in ('EXPERT', 'PIM'):
            continue

        # Skip non-layer rows (comp_time must be a valid integer)
        try:
            comp_time = int(cols[1])
        except (ValueError, IndexError):
            continue

        total_compute_ns += comp_time
        is_kv = name.startswith('kv_load') or name.startswith('kv_evict')

        if has_remote and len(cols) >= 8:
            for loc_idx, size_idx in ((2, 3), (4, 5), (6, 7)):
                try:
                    loc = cols[loc_idx]
                    data_size = int(cols[size_idx])
                    if data_size <= 0:
                        continue
                    if loc.startswith('REMOTE') and cpu_bw_bpns > 0:
                        t = cpu_latency + int(data_size / cpu_bw_bpns)
                        if is_kv:
                            cpu_transfer_ns += t
                        else:
                            other_comm_ns += t
                    elif loc.startswith('CXL') and cxl_bw_bpns > 0:
                        t = cxl_latency + int(data_size / cxl_bw_bpns)
                        if is_kv:
                            cxl_transfer_ns += t
                        else:
                            other_comm_ns += t
                except (ValueError, IndexError):
                    pass

    # AstraSim execution model (from llm_converter.py):
    #   kv_evict ──┐
    #   kv_load  ──┤  (parallel, independent MEM nodes)
    #   input_load ┤
    #              ↓
    #        [embedding COMP] → [layernorm COMP] → [qkv COMP] → ...  (sequential)
    #
    # kv_load/kv_evict/input_load must ALL complete before first compute starts.
    # They run in parallel with each other, so the blocking time is max() of them.
    # Compute layers run sequentially after that.
    pre_comp_comm = max(cpu_transfer_ns, cxl_transfer_ns, other_comm_ns)
    total_ns = pre_comp_comm + total_compute_ns

    return {
        'total_ns': total_ns,
        'compute_ns': total_compute_ns,
        'cpu_transfer_ns': cpu_transfer_ns,
        'cxl_transfer_ns': cxl_transfer_ns,
        'comm_ns': other_comm_ns,
        'pre_comp_comm_ns': pre_comp_comm,
    }