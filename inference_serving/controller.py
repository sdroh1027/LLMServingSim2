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

    def submit_event(self, finish_cycle_ns, npu_id):
        """Add a batch/event completion to the event queue."""
        it = self.iteration[npu_id]
        self.iteration[npu_id] += 1
        heapq.heappush(self.events, (finish_cycle_ns, npu_id, it))

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
        pattern = r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, exposed communication (\d+) cycles."
        match = re.search(pattern, output)
        if match:
            return {
                'sys': int(match.group(1)),
                'id': int(match.group(2)),
                'cycle': int(match.group(3)),
            }
        return None


def _calc_transfer_ns(loc, data_size, link_bw_bpns, link_latency, cxl_bw_bpns, cxl_latency):
    """Return transfer time (ns) for a single data access given its location tag.

    Supports REMOTE (CPU memory) and CXL device locations.
    Returns 0 for LOCAL or when bandwidth is not configured.
    """
    if data_size <= 0:
        return 0
    if loc.startswith('REMOTE') and link_bw_bpns > 0:
        return link_latency + int(data_size / link_bw_bpns)
    if loc.startswith('CXL') and cxl_bw_bpns > 0:
        return cxl_latency + int(data_size / cxl_bw_bpns)
    return 0


def compute_trace_cycles(trace_path, link_bw=0, link_latency=0, cxl_bw=0, cxl_latency=0):
    """Read a trace .txt file and return total time in nanoseconds.

    Sums comp_time for all layers and adds exposed communication time
    for REMOTE/CXL memory accesses (input/weight/output), matching
    AstraSim's analytical backend behaviour.

    Args:
        trace_path: path to the trace .txt file
        link_bw: inter-node link bandwidth in GB/s (0 = ignore REMOTE)
        link_latency: per-access link latency in ns
        cxl_bw: CXL device bandwidth in GB/s (0 = ignore CXL)
        cxl_latency: per-access CXL latency in ns
    """
    with open(trace_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 3:
        return 0

    num_layers = int(lines[1].strip())
    total_compute_ns = 0
    total_comm_ns = 0
    # GB/s  →  bytes per ns  (1 GB/s == 1 byte/ns)
    link_bw_bpns = link_bw if link_bw > 0 else 0
    cxl_bw_bpns = cxl_bw if cxl_bw > 0 else 0
    has_remote = link_bw_bpns > 0 or cxl_bw_bpns > 0

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

        # Accumulate REMOTE/CXL data transfer time for the whole batch
        # Trace columns: name comp input_loc input_size weight_loc weight_size
        #                output_loc output_size comm_type comm_size misc
        if has_remote and len(cols) >= 8:
            for loc_idx, size_idx in [(2, 3), (4, 5), (6, 7)]:
                try:
                    total_comm_ns += _calc_transfer_ns(
                        cols[loc_idx], int(cols[size_idx]),
                        link_bw_bpns, link_latency, cxl_bw_bpns, cxl_latency)
                except (ValueError, IndexError):
                    pass

    # Exposed comm = comm time not overlapped with compute (batch-level)
    exposed_comm = max(0, total_comm_ns - total_compute_ns)
    return total_compute_ns + exposed_comm


def compute_trace_cycles_mem(layers, link_bw=0, link_latency=0, cxl_bw=0, cxl_latency=0):
    """In-memory version of compute_trace_cycles — no file I/O.

    Takes the layer list returned by generate_trace_bypass() directly
    instead of reading from a trace .txt file.

    Args:
        layers: list[list[str]] — tokenised layer rows from generate_trace_bypass().
        link_bw: inter-node link bandwidth in GB/s (0 = ignore REMOTE)
        link_latency: per-access link latency in ns
        cxl_bw: CXL device bandwidth in GB/s (0 = ignore CXL)
        cxl_latency: per-access CXL latency in ns
    """
    if not layers:
        return 0

    total_compute_ns = 0
    total_comm_ns = 0
    link_bw_bpns = link_bw if link_bw > 0 else 0
    cxl_bw_bpns = cxl_bw if cxl_bw > 0 else 0
    has_remote = link_bw_bpns > 0 or cxl_bw_bpns > 0

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

        if has_remote and len(cols) >= 8:
            for loc_idx, size_idx in ((2, 3), (4, 5), (6, 7)):
                try:
                    total_comm_ns += _calc_transfer_ns(
                        cols[loc_idx], int(cols[size_idx]),
                        link_bw_bpns, link_latency, cxl_bw_bpns, cxl_latency)
                except (ValueError, IndexError):
                    pass

    exposed_comm = max(0, total_comm_ns - total_compute_ns)
    return total_compute_ns + exposed_comm