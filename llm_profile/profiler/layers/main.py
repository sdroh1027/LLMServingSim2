"""
LLM non-attention layer profiler.

Goal
----
Measure each major sub-layer's GPU cost in isolation so a downstream simulator
can re-assemble them and predict full-model latency under different shapes,
TP degrees, batch configurations, etc.

Ground truth
------------
Production runs the model under torch.compile / CUDAGraph, where kernels are
launched back-to-back with negligible Python dispatch overhead and minimal
inter-kernel idle. So a faithful "what does this op cost in production" is
approximated by *kernel execution time*, NOT by raw eager wall-clock (which
includes Python dispatch, launch overhead, and idle gaps that compiled
production removes).

Timer strategy
--------------
1. **Per-component measurement uses `record_function` cuda_time** (kernel-only).
   - record_function spans + chrome-trace parsing sum the correlated kernel
     durations inside each span. This is the time the GPU actually spent
     executing that op's kernels — ignoring Python overhead between launches.
   - For fused single-kernel ops (lm_head, grouped_mm MoE block) this equals
     the kernel's own runtime, which is also what production sees.
   - For sequential micro-ops (e.g. eager MoE Python loop) this correctly
     EXCLUDES the per-iter Python/dispatch overhead — production wouldn't
     pay that anyway because the fused kernel doesn't iterate in Python.
   - Note: the RecordFunctionTracer must look at BOTH `cuda_runtime` and
     `cuda_driver` categories (PyTorch 25.01 routes single-kernel launches
     through cuda_driver), otherwise some ops silently report 0 (lm_head
     was the canonical example before that patch).

2. **CUDA Event measurements are NOT a substitute** for per-op profiling.
   - Wrapping an op with start/end Events and elapsed_time includes any GPU
     idle gap in the stream — including dispatch overhead that production
     does not have. Such values over-state the op's production cost.
   - The earlier CUDA-event `experts_call` measurement (95 ms in eager mode)
     was misleading for this reason; the production kernel cost is the
     ~4 ms record_function sum (eager) or ~11 ms (grouped_mm fused).

3. **Sanity / consistency check uses a CUDA Graph capture** of the full
   forward, replayed and timed end-to-end. CUDA Graph replay launches the
   captured kernel sequence with near-zero CPU overhead, mimicking the
   production execution model. Sum of per-op record_function cuda_times
   should match this CUDA-Graph wall-clock within a few percent; if it
   diverges, either an op is missing instrumentation or our trace parser
   is dropping a launch.
   - An eager-mode full_forward CUDA Event wall-clock is NOT a valid baseline:
     it includes per-op Python dispatch that production strips out, so the
     "unaccounted" gap there is profiler-environment noise, not real cost.

4. **Which experts implementation to profile**: production uses the default
   (`grouped_mm`) fused kernel. Profile that. Don't force `_experts_implementation
   = "eager"` — the per-expert breakdown isn't needed by the simulator and the
   Python iteration noise is irrelevant to production timing.

TODO: full_forward baseline improvement
---------------------------------------
Current sanity baseline tries CUDA Graph capture but falls back to eager
wall-clock for MoE models because the routing path uses dynamic-shape ops
(e.g. `.nonzero()` on the expert hit mask) that CUDAGraph cannot capture.
The eager fallback over-states the gap by the cumulative Python dispatch
overhead between sub-blocks (~10 ms for a 1-layer 35B-A3B prefill at len=256).
Follow-ups to make the baseline truly production-equivalent:
  - try `torch.compile(model, mode="reduce-overhead", dynamic=True)`
  - or replace MoE routing with a CUDAGraph-safe static path (Megatron-style
    fixed-capacity routing)
  - or wrap only the static prefix/suffix of the forward in CUDAGraph and
    leave the MoE block as eager (hybrid baseline)
Until then, treat `timer_sum` as the authoritative production-realistic
per-layer-cost sum; the `full_forward` value is an upper-bound diagnostic.
"""

import torch

from collections import defaultdict
from tqdm import tqdm
import csv
import os
import gc
import argparse

from transformers import AutoConfig
from transformers.utils import logging
from transformers.cache_utils import DynamicCache

from profiler.common.timer_stats_store import TimerStatsStore
from profiler.utils import *
from profiler.utils.record_function_tracer import RecordFunctionTracer
from profiler.utils.logger import *


logging.set_verbosity_error()   # error only to avoid warnings from transformers

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Non-Attention Layer profiler")

    # Model parameters
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name (e.g., meta-llama/Llama-3.1-8B).")
    parser.add_argument("--hardware", type=str,  required=True,
                        help="Hardware name for metadata logging.")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="Number of transformer layers to profile.")
    # Tensor parallelism configuration
    parser.add_argument("--tp-size", type=str, default="1",
                        help="Comma-separated list of tensor parallel degrees (e.g., '1,2,4').")
    # Batch conditions
    parser.add_argument("--max-len", type=int, default=2048,
                        help="Maximum request length.")
    # Profiling parameters
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup iterations.")
    parser.add_argument("--repeat", type=int, default=30,
                        help="Number of repeated profiling steps.")
    # Profiling method
    parser.add_argument("--profile-method", default="record_function", choices=[e.value for e in ProfileMethod],
        help="Method to use for measuring time taken by operations (default: %(default)s)")
    # Device selection
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'. FlashAttention requires GPU.")
    
    parser.add_argument("--append", action="store_true", help="Append to CSV instead of overwrite")

    parser.add_argument("--verbose", action="store_true")
    # Which stages to run
    parser.add_argument("--legacy", action="store_true", help="Use legacy attention with no prediction (not recommended)")

    return parser.parse_args()


def _create_past_key_values(config, kv_len, device):
    """
    Create a DynamicCache with preallocated KV tensors for a single TP rank.

    - tp_size is the logical tensor-parallel degree.
    - We allocate KV tensors with num_key_value_heads / tp_size heads, i.e.
      the per-rank KV-head count in a tensor-parallel setup.

    This matches the per-rank KV cache shape used in vLLM / Megatron-style TP:
        K, V: [batch, num_kv_heads_local, kv_len, head_dim]
    """
    num_layers = config.num_hidden_layers

    # Total KV heads in the (global) config
    num_kv_heads_total = config.num_key_value_heads // config.tp_size

    # Head dim is always based on total attention heads (not KV heads)
    # if config has attribute "head_dim":
    if hasattr(config, "head_dim"):
        head_dim = config.head_dim 
    else:
        head_dim = config.hidden_size // (config.num_attention_heads) # config has been already divided by tp_size

    # Choose dtype from config
    if getattr(config, "dtype", None) == torch.float16:
        dtype = torch.float16
    else:
        # Fallback; you can extend this if you want bfloat16, etc.
        dtype = torch.float32

    # Preallocate per-rank KV tensors:
    key_states = torch.zeros(
        (1, num_kv_heads_total, kv_len, head_dim),
        device=device,
        dtype=dtype,
    )
    value_states = torch.zeros(
        (1, num_kv_heads_total, kv_len, head_dim),
        device=device,
        dtype=dtype,
    )

    # Dummy input to satisfy rotary embedding forward
    dummy_x = torch.zeros((1, kv_len, head_dim), device=device, dtype=dtype)
    position_ids = torch.arange(kv_len, device=device).unsqueeze(0)  # shape: (1, kv_len)

    if "llama" in config.model_type:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        rope = LlamaRotaryEmbedding(config, device=device)
        cos, sin = rope(dummy_x, position_ids)
    elif "mixtral" in config.model_type:
        from transformers.models.mixtral.modeling_mixtral import MixtralRotaryEmbedding
        rope = MixtralRotaryEmbedding(config, device=device)
        cos, sin = rope(dummy_x, position_ids)
    elif "phimoe" in config.model_type:
        from transformers.models.phimoe.modeling_phimoe import PhimoeRotaryEmbedding
        rope = PhimoeRotaryEmbedding(config)
        cos, sin = rope(dummy_x, kv_len)
    elif "qwen3_5_moe" in config.model_type:
        # NB: must precede the 'qwen3' branch — substring would otherwise catch us
        from models.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
        rope = Qwen3_5MoeTextRotaryEmbedding(config)
        bsz = dummy_x.shape[0]
        position_ids = torch.arange(kv_len, device=dummy_x.device).unsqueeze(0).expand(bsz, -1)
        cos, sin = rope(dummy_x, position_ids)
    elif "qwen3" in config.model_type:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        rope = Qwen3RotaryEmbedding(config)
        bsz = dummy_x.shape[0]
        position_ids = torch.arange(kv_len, device=dummy_x.device).unsqueeze(0).expand(bsz, -1)
        cos, sin = rope(dummy_x, position_ids)
    else:
        raise NotImplementedError("Only LLaMA, Mixtral, Phi-MoE, Qwen3, Qwen3.5-MoE models are supported in profiling. We will add more models soon.")

    # Hybrid models (e.g. Qwen3.5-MoE) interleave linear-attention and full-attention
    # layers. Build the cache from config so each slot has the correct mixin type;
    # only write K/V into full-attention slots.
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        cache = DynamicCache(config=config)
    else:
        cache = DynamicCache()
    for layer_idx in range(num_layers):
        if layer_types is not None and layer_types[layer_idx] != "full_attention":
            continue
        cache.update(
            key_states,
            value_states,
            layer_idx,
            {
                "cos": cos,
                "sin": sin,
                "cache_position": position_ids,
            },
        )
    return cache

def run_profile(
    hardware="A6000",
    model_name="meta-llama/Llama-3.1-8B",
    num_layers=None,
    input_lengths=[128, 256, 512, 1024, 2048],
    is_prefill=True,
    tp_size=1,
    device="cuda",
    warmup=10,
    repeat=100,
    profile_method="record_function",
    csv_append=True,
    verbose=False,
):

    config = AutoConfig.from_pretrained(model_name)
    # Multimodal Qwen3.5-MoE: drop into the text sub-config for LLM profiling
    if 'qwen3_5_moe' in config.model_type and hasattr(config, 'text_config'):
        config = config.text_config
    original_num_layers = config.num_hidden_layers
    original_layer_types = list(getattr(config, 'layer_types', []) or [])
    config.num_hidden_layers = num_layers
    config.dtype = torch.float16
    config.pad_token_id = 1
    config.tp_size = tp_size
    # Qwen3.5-MoE is a hybrid linear+full attention model. This layers profiler
    # measures only the GDN (linear_attention) token mixer and the shared per-block
    # components; full-attention cost is profiled separately by profile_attn_*.sh
    # and combined downstream via the layer_type fractions stored above.
    # The MoE block uses the default (grouped_mm) fused kernel, timed end-to-end
    # by the CUDA events embedded in Qwen3_5MoeSparseMoeBlock (key "experts_call").
    if 'qwen3_5_moe' in config.model_type:
        config.layer_types = ["linear_attention"] * num_layers
    # Call singletone instance TimerStatsStore to set profile method
    timer_stats_store = TimerStatsStore(profile_method=profile_method)

    if 'llama' in config.model_type:
        from models.llama import LlamaForCausalLM
        model = LlamaForCausalLM(config)
    elif 'mixtral' in config.model_type:
        from models.mixtral import MixtralForCausalLM
        # If you want to collect router stats during profiling, turn this on
        config.collect_router_stats = True
        model = MixtralForCausalLM(config)
    elif 'phimoe' in config.model_type:
        from models.phimoe import PhimoeForCausalLM
        # If you want to collect router stats during profiling, turn this on
        config.collect_router_stats = False
        model = PhimoeForCausalLM(config)
    elif 'qwen3_5_moe' in config.model_type:
        # NB: must precede the 'qwen3' branch — substring would otherwise catch us
        from models.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
        model = Qwen3_5MoeForCausalLM(config)
        # When the cache has no full-attention layer (we profile linear-only),
        # DynamicCache.get_seq_length / get_mask_sizes raise. For a fresh cache
        # (kv_len=0) the equivalents are 0 / (q_len, 0); patch fallbacks in.
        from transformers.cache_utils import DynamicCache as _DC
        if not getattr(_DC, "_profile_no_attn_patched", False):
            _orig_gsl = _DC.get_seq_length
            def _safe_gsl(self, layer_idx=0):
                try:
                    return _orig_gsl(self, layer_idx)
                except (ValueError, StopIteration):
                    return 0
            _DC.get_seq_length = _safe_gsl

            _orig_gms = _DC.get_mask_sizes
            def _safe_gms(self, query_length, layer_idx=0):
                try:
                    return _orig_gms(self, query_length, layer_idx)
                except (ValueError, StopIteration):
                    return query_length, 0
            _DC.get_mask_sizes = _safe_gms

            _DC._profile_no_attn_patched = True
    elif 'qwen3' in config.model_type:
        from models.qwen3 import Qwen3ForCausalLM
        # If you want to collect router stats during profiling, turn this on
        config.collect_router_stats = False
        model = Qwen3ForCausalLM(config)
    else:
        raise NotImplementedError("Only LLaMA, Mixtral, Phi-MoE, Qwen3, Qwen3.5-MoE models are supported in profiling. We will add more models soon.")
    
    model.eval()
    model.to(config.dtype)
    model.to(device)

    if is_prefill:
        kv_lengths = [0]
    else:
        kv_lengths = input_lengths # should run all possible input/kv combinations for decode
        raise log_warning(f"This deprecated decode profiling will profile {len(kv_lengths) * len(input_lengths)} configurations.")
    results = defaultdict(float)
   
    total_tasks = len(input_lengths) * len(kv_lengths)
    csv_rows = []
    log_info(f"Starting profiling for hardware={hardware}, model={model_name}, tp_size={tp_size}")
    for (input_len, kv_len) in tqdm([(l, k) for l in input_lengths for k in kv_lengths], total=total_tasks, desc="Profiling configs"):
        if input_len + kv_len > config.max_position_embeddings:
            continue  # Skip if input length exceeds max position embeddings
        if verbose:
            log_info(f"Running input={input_len}, kv={kv_len}, tp={tp_size}")

        input_ids = torch.randint(low=0, high=config.vocab_size // tp_size, size=(1, input_len), device=device)

        num_layers = config.num_hidden_layers
        # For kv_len=0 (prefill), let the model construct its own cache. Manually
        # pre-filled caches need to contain the right layer-type mixins (hybrid
        # models like Qwen3.5-MoE need *both* attention and linear-attention
        # layers present), and passing None avoids that fragility.
        def _make_pkv():
            if kv_len == 0:
                return None
            return _create_past_key_values(config, kv_len, device)

        # Warm-up phase
        for _ in range(warmup):
            past_key_values = _make_pkv()
            with torch.no_grad():
                _ = model(input_ids, past_key_values=past_key_values, use_cache=True)

        torch.cuda.synchronize()
        timer_stats_store.clear_stats()

        # ---- Profile pass: per-op record_function cuda_time ----
        if profile_method == ProfileMethod.RECORD_FUNCTION.value:

            trace_output_dir = f"perf_models/{hardware}/{model_name}/tp{tp_size}"
            record_function_tracer = RecordFunctionTracer(trace_output_dir)

            with record_function_tracer:
                for _ in range(repeat):
                    with torch.no_grad():
                        _ = model(input_ids, past_key_values=_make_pkv(), use_cache=True)

            torch.cuda.synchronize()
            time_stats = record_function_tracer.get_operation_time_stats()
            record_function_tracer.clean_up()

        else:
            for _ in range(repeat):
                with torch.no_grad():
                    _ = model(input_ids, past_key_values=_make_pkv(), use_cache=True)

            torch.cuda.synchronize()
            time_stats = timer_stats_store.get_stats()

        # ---- Sanity pass: production-like wall-clock baseline ----
        # See module docstring for rationale. CUDA Graph replay strips Python
        # dispatch overhead; falls back to eager wall-clock if capture fails
        # (MoE dynamic-shape routing isn't graph-safe — see TODO in docstring).
        # Only sampled at input_len multiples of SANITY_STRIDE because the
        # sanity baseline is much more expensive than the profile pass and
        # one diagnostic per ~256-token bucket is enough to flag regressions.
        SANITY_STRIDE = 256
        if input_len % SANITY_STRIDE == 0:
            import numpy as _np
            full_forward_times_ms = []
            graph_used = False
            try:
                static_pkv = _make_pkv()
                for _ in range(3):
                    with torch.no_grad():
                        _ = model(input_ids, past_key_values=static_pkv, use_cache=True)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.no_grad(), torch.cuda.graph(graph):
                    _ = model(input_ids, past_key_values=static_pkv, use_cache=True)

                for _ in range(repeat):
                    s_evt = torch.cuda.Event(enable_timing=True)
                    e_evt = torch.cuda.Event(enable_timing=True)
                    s_evt.record()
                    graph.replay()
                    e_evt.record()
                    torch.cuda.synchronize()
                    full_forward_times_ms.append(s_evt.elapsed_time(e_evt))
                graph_used = True
            except Exception as ex:
                log_warning(
                    f"CUDA Graph capture failed ({type(ex).__name__}: {ex}); "
                    "falling back to eager wall-clock baseline. The sanity gap "
                    "will include Python dispatch overhead that production strips."
                )
                for _ in range(repeat):
                    s_evt = torch.cuda.Event(enable_timing=True)
                    e_evt = torch.cuda.Event(enable_timing=True)
                    with torch.no_grad():
                        s_evt.record()
                        _ = model(input_ids, past_key_values=_make_pkv(), use_cache=True)
                        e_evt.record()
                    torch.cuda.synchronize()
                    full_forward_times_ms.append(s_evt.elapsed_time(e_evt))

            full_forward_median_ms = float(_np.median(full_forward_times_ms))
            timer_sum_ms = sum(v.get("median", 0.0) for v in time_stats.values())
            diff_pct = (full_forward_median_ms - timer_sum_ms) / full_forward_median_ms * 100.0 if full_forward_median_ms > 0 else 0.0
            baseline_tag = "cuda_graph" if graph_used else "eager"
            log_info(
                f"[sanity] input={input_len} kv={kv_len} "
                f"full_forward_median={full_forward_median_ms:.3f} ms ({baseline_tag}), "
                f"timer_sum={timer_sum_ms:.3f} ms, "
                f"unaccounted={diff_pct:+.1f}%"
            )


        profile_keys = ["embedding", "input_layernorm", "q_proj", "k_proj", "v_proj", "rope", "attn", "o_proj", "post_layernorm", "gate_proj", "up_proj", "act_fn", "down_proj", "final_layernorm", "lm_head"]
        if 'mixtral' in config.model_type or 'phimoe' in config.model_type:
            profile_keys += ["gate", "expert.w1", "expert.w2", "expert.w3"]
        elif 'qwen3_5_moe' in config.model_type:
            # Linear-attention (GDN) path + MoE routing/shared expert + the
            # fused MoE block timed end-to-end via CUDA events ("experts_call").
            profile_keys += [
                "gdn_in_proj", "gdn_kernel", "gdn_post",
                "router", "experts_call",
                "shared_expert.mlp", "shared_expert_gate",
            ]

        for key, value in time_stats.items():
            if key in profile_keys:
                if verbose:
                    log_info(f"input={input_len}, kv={kv_len}, tp={tp_size}, layer={key}, time={value['median']*1000:.2f} us")
                results_key = (input_len, kv_len, key)
                results[results_key] = value['median']*1000
                csv_rows.append({
                    "layer_name": key,
                    "input": input_len,
                    "kv_cache": kv_len,
                    "tp_size": tp_size,
                    "latency(ns)": int(value['median']* 1000_000)  # convert ms to ns
                })
            else:
                log_warning(f"Skipping layer={key} not in profile keys.")

        embedding = results.get((input_len, kv_len, "embedding"), 0.0)
        final_norm = results.get((input_len, kv_len, "final_layernorm"), 0.0)
        lm_head = results.get((input_len, kv_len, "lm_head"), 0.0)
        if 'llama' in config.model_type:
            block_components = ["input_layernorm", "q_proj", "k_proj", "v_proj", "rope", "attn", "o_proj", "post_layernorm", "gate_proj", "up_proj", "act_fn", "down_proj"]
        elif 'mixtral' in config.model_type or 'phimoe' in config.model_type:
            block_components = ["input_layernorm", "q_proj", "k_proj", "v_proj", "rope", "attn", "o_proj", "post_layernorm", "gate"]
        elif 'qwen3_5_moe' in config.model_type:
            # Shared per-layer ops only (norms + MoE block). Token-mixer cost is added
            # below with layer_type weighting since layers alternate linear/full attention.
            block_components = ["input_layernorm", "post_layernorm",
                                "router", "shared_expert.mlp", "shared_expert_gate"]
        elif 'qwen3' in config.model_type:
            block_components = ["input_layernorm", "q_proj", "k_proj", "v_proj", "rope", "attn", "o_proj", "post_layernorm", "gate_proj", "up_proj", "act_fn", "down_proj"]
        else:
            raise NotImplementedError("Only LLaMA, Mixtral, Phi-MoE, Qwen3, Qwen3.5-MoE models are supported in profiling. We will add more models soon.")

        per_block_time = sum(results.get((input_len, kv_len, comp), 0.0) for comp in block_components)

        # Runs experts sequentially in huggungface implementation
        if 'mixtral' in config.model_type or 'phimoe' in config.model_type:
            moe_components = ["expert.w1", "expert.w2", "expert.w3", "act_fn"]
            n_tok = max(input_len // config.num_local_experts // tp_size, 1)
            for moe_comp in moe_components:
                per_block_time += results.get((n_tok, kv_len, moe_comp), 0.0) * (config.num_local_experts // tp_size)
        elif 'qwen3_5_moe' in config.model_type:
            # We profile a single linear-attention layer (full-attention is profiled
            # separately). Per-block estimate = shared + GDN (raw, unweighted) + MoE.
            # No layer-type fraction weighting here; downstream combines layer types.
            per_block_time += sum(
                results.get((input_len, kv_len, c), 0.0)
                for c in ["gdn_in_proj", "gdn_kernel", "gdn_post"]
            )

            # MoE block: CUDA-event measurement of the whole self.experts(...)
            # call. Covers the fused grouped_mm kernel as a single op (no need
            # to multiply by num_experts).
            per_block_time += results.get((input_len, kv_len, "experts_call"), 0.0)

            # Sparse experts: each token routes to top_k experts → avg per-expert load
            # = input_len * top_k / num_experts. Sum over all experts.
            num_experts = getattr(config, 'num_experts', 256)
            top_k = getattr(config, 'num_experts_per_tok', 8)
            n_tok = max(input_len * top_k // num_experts // tp_size, 1)
            per_block_time += results.get((n_tok, kv_len, "expert.mlp"), 0.0) * num_experts

        full_latency_estimate = embedding + final_norm + lm_head + per_block_time * original_num_layers

        if verbose:
            log_info(f"Estimated latency: {(full_latency_estimate / 1000):.2f} ms")

    output_path = f"perf_models/{hardware}/{model_name}/tp{tp_size}/layers.csv"
    if csv_append:
        mode = "a"
    else:
        mode = "w"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["layer_name", "input", "kv_cache", "tp_size", "latency(ns)"])
        if not csv_append:
            writer.writeheader()
        writer.writerows(csv_rows)
    log_success(f"[{hardware}/{model_name} TP={tp_size}] Writing profiled results to {output_path}")

def main():

    args = parse_args()

    # Convert tp_size string into list
    tp_sizes = [int(x.strip()) for x in args.tp_size.split(",")]

    # Load model config once per script run
    model_config = AutoConfig.from_pretrained(args.model) # token="hf_xxx"
    # Multimodal Qwen3.5-MoE: validate TP against the text sub-config
    if 'qwen3_5_moe' in model_config.model_type and hasattr(model_config, 'text_config'):
        model_config = model_config.text_config

    for tp_size in tp_sizes:
        if validate_tp_size(tp_size, model_config.num_attention_heads):
            log_warning(f"Skipping invalid TP degree {tp_size}.")
            continue

        # ---------- Prefill sweep ----------
        run_profile(
            hardware=args.hardware,
            model_name=args.model,
            input_lengths=range(1, args.max_len + 1),
            is_prefill=True,
            num_layers=args.num_layers,
            tp_size=tp_size,
            device=args.device,
            warmup=args.warmup,
            repeat=args.repeat,
            profile_method=args.profile_method,
            csv_append=False, # prefill first
            verbose=args.verbose,
        )
        torch.cuda.empty_cache()
        gc.collect()

        # ---------- Decode sweep (legacy) ----------
        if args.legacy:
            log_warning(
                "Deprecated: running legacy profiler. "
                "We recommend using attention prediction instead."
            )
            run_profile(
                hardware=args.hardware,
                model_name=args.model,
                input_lengths=range(1, args.max_len + 1),
                is_prefill=False,
                num_layers=args.num_layers,
                tp_size=tp_size,
                device=args.device,
                warmup=args.warmup,
                repeat=args.repeat,
                profile_method=args.profile_method,
                csv_append=True, # append decode results
                verbose=args.verbose,
            )
            torch.cuda.empty_cache()
            gc.collect()

    
if __name__ == "__main__":
    main()