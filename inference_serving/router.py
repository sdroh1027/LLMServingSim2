import pandas as pd
import random
from time import time
from .logger import get_logger

class Router:
    def __init__(
            self, 
            num_instances, 
            schedulers, req_num, 
            routing_policy="RR", 
            seed=42
    ):
        self.schedulers = schedulers
        self.num_instances = num_instances
        self.prefill_schedulers = [s for s in schedulers if s.pd_type != "decode"]
        self.prefill_instances = len(self.prefill_schedulers)
        self.decode_schedulers = [s for s in schedulers if s.pd_type == "decode"]
        self.decode_instances = len(self.decode_schedulers)
        self.req_num = req_num
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        self._rnd = random.Random(seed) if seed is not None else random
        self.instance_status = [0 for _ in range(num_instances)]
        self.prefill_rr_counter = 0
        self.decode_rr_counter = 0
        if self.routing_policy == "RR":
            self.routing_fn = self._rr_routing
        elif self.routing_policy == "RAND":
            self.routing_fn = self._rand_routing
        elif self.routing_policy == "CUSTOM":
            self.routing_fn = self._custom_routing_policy
        else:
            raise ValueError(f"Unknown routing_policy '{routing_policy}'. "
                             "Supported: RR, RAND, CUSTOM")
        self.logger = get_logger(self.__class__)

        # Cross-instance index of all queued turns per session, used by
        # Scheduler.add_done to find a completing req's successor turn (which
        # may live on a different scheduler under round-robin routing) and
        # advance its admission time directly. Each entry is
        # session_turns[session_id][turn_idx] = (req, owning_scheduler).
        # Replaces the older shared_session_state dict + per-req effective-time
        # gate: arrival is now the single source of truth for admission.
        self.session_turns = {}
        for s in self.schedulers:
            s.router_session_turns = self.session_turns

    def _rr_routing(self, request_ctr, num_instances):
        return request_ctr % num_instances

    def _rand_routing(self, request_ctr, num_instances):
        return self._rnd.randrange(num_instances)
    
    def _custom_routing_policy(self, request_ctr, num_instances):
        raise NotImplementedError("Implement custom routing policy.")

    def transfer_prefill_request(self, requests):
        for req in requests:
            instance_id = self.routing_fn(self.decode_rr_counter, self.decode_instances)
            self.decode_schedulers[instance_id].add_decode(req)
            self.decode_rr_counter += 1

    # generate request to each instance with routing policy
    def generate(self, path, enable_prefix_caching=False, is_init=True):
        path = f'../{path}'
        data = pd.read_json(path, lines=True)
        # JSONL may be session-grouped for human readability. Sort by
        # arrival_time_ns here so scheduler.request stays arrival-ordered,
        # which the scheduling fast-paths (short-circuit on request[0].arrival,
        # get_next_arrival_time) rely on.
        if 'arrival_time_ns' in data.columns:
            data = data.sort_values('arrival_time_ns', kind='stable').reset_index(drop=True)

        for index, row in data.iterrows():
            if index >= self.req_num:
                break
            input_length = int(row['input_toks'])
            output_length = int(row['input_toks']+row['output_toks'])
            arrival_time_ns = int(row['arrival_time_ns'])
            if enable_prefix_caching:
                # using token ids as hash ids for simplicity
                # change this to add your own hash function
                input_hash_ids = row['input_tok_ids']
                output_hash_ids = row['output_tok_ids']

            # Multi-turn session metadata (optional; pre-multi-turn traces lack these)
            session_id = row['session_id'] if 'session_id' in row and pd.notna(row['session_id']) else None
            turn_idx = int(row['turn_idx']) if 'turn_idx' in row and pd.notna(row['turn_idx']) else 0
            intra_session_gap_ns = int(row['intra_session_gap_ns']) if 'intra_session_gap_ns' in row and pd.notna(row['intra_session_gap_ns']) else 0

            # Multi-turn dependency: turn N>0 must not be admissible until the
            # predecessor turn completes, regardless of its JSONL arrival.
            # We park its `arrival` at a sentinel so the simple arrival filter
            # treats it as not-yet-arrived; add_done(predecessor) then sets
            # arrival = max(original_arrival, finish + intra_gap) and re-sorts.
            # The original JSONL value is preserved in req.original_arrival.
            effective_arrival = arrival_time_ns if (session_id is None or turn_idx == 0) else (1 << 62)

            if index == 0:
                # set first arrival time
                for scheduler in self.schedulers:
                    scheduler.first_arrival_time = arrival_time_ns

            instance_id = self.routing_fn(self.prefill_rr_counter, self.prefill_instances)
            # add only if instance id matches & add to only prefill schedulers
            if instance_id < 0 or instance_id >= self.prefill_instances:
                raise ValueError(f"Invalid instance_id {instance_id}")

            sched = self.prefill_schedulers[instance_id]
            if enable_prefix_caching:
                new_req = sched.add_request(
                    [index, sched.model, input_length, output_length, effective_arrival, instance_id, input_hash_ids, output_hash_ids],
                    is_init=is_init,
                    session_id=session_id, turn_idx=turn_idx, intra_session_gap_ns=intra_session_gap_ns,
                )
            else:
                new_req = sched.add_request(
                    [index, sched.model, input_length, output_length, effective_arrival, instance_id],
                    is_init=is_init,
                    session_id=session_id, turn_idx=turn_idx, intra_session_gap_ns=intra_session_gap_ns,
                )
            # Preserve the true JSONL arrival regardless of the sentinel above,
            # so add_done can compute max(original_arrival, finish + intra_gap)
            # when releasing this turn for admission.
            if session_id is not None and turn_idx > 0:
                new_req.original_arrival = arrival_time_ns
            # Register in the global session->turns index so the predecessor's
            # add_done can find this turn (possibly on a different scheduler)
            # and advance its admission time when it completes.
            if session_id is not None:
                self.session_turns.setdefault(session_id, {})[turn_idx] = (new_req, sched)
            self.prefill_rr_counter += 1
        
        for scheduler in self.schedulers:
            self.logger.info(
                "Added %d requests to scheduler[%d] (%s type) ",
                len(scheduler.request),
                scheduler.instance_id,
                scheduler.pd_type
            )
        return