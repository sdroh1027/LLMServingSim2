# llm_profile

A PyTorch-based profiling tool for measuring LLM layer latencies, attention latencies, and
GPU/system-level power consumption. The outputs are used by LLMServingSim as performance and
power models.

To profile a new model or hardware target for use with LLMServingSim, follow the steps below.
See also the [Adding a New Model & Hardware](../README.md#adding-a-new-model--hardware) section
in the top-level README.

## Overview

`llm_profile` loads models from Hugging Face and inserts PyTorch profiler hooks into key
layers to measure execution time on GPU. It supports dense and MoE architectures and
produces per-layer latency CSVs and a scikit-learn-based attention latency predictor.
GPU and system-level power consumption are measured via `nvidia-smi` and `ipmitool`,
and the results feed into LLMServingSim's power model.

## Usage

### 1. Environment

Run inside the provided Docker container or a native PyTorch + CUDA environment:

```bash
./docker.sh
```

For models that require access approval (e.g., LLaMA), provide your Hugging Face token
as described in `docker.sh`.

### 2. Profile layers and attention

```bash
./profile_layers.sh    # Measures compute latency for non-attention layers
./profile_attn.sh      # Measures attention latency across batch sizes and sequence lengths
```

To reduce profiling time and memory usage, decrease the number of layers via `--num-layer`
in the respective profiling scripts.

### 3. Profile power (optional)

For power measurement, we provide example scripts under `profiler/power/` that use
`nvidia-smi` to measure GPU power consumption and `ipmitool` to measure system-level power:

```bash
./profiler/power/profile_gpu_power.sh      # GPU power via nvidia-smi
./profiler/power/profile_server_power.sh   # System-level power via ipmitool
```

Power profiling results are used by LLMServingSim's power model when a cluster config with
power settings is provided (e.g., `cluster_config/single_node_power_instance.json`).

### 4. Build the attention predictor

```bash
./build_predictor.sh
```

This trains a scikit-learn model on the profiled attention data to support real-time latency
prediction during simulation (`--enable-attn-prediction`). The inference space covered by
the predictor can be controlled via `--max-batch` and `--max-len`.

## Output structure

Results are written to:

```
perf_models/{hardware}/{model}/tp{tp_size}/
  layers.csv                              # Per-layer compute latency
  attention.csv                           # Attention latency by (batch_size, seq_len)
  predictions/
    attn_decode_predictions.csv           # Predictor output for decode attention
    attn_prefill_predictions.csv          # Predictor output for prefill attention
```

These files are loaded automatically by LLMServingSim at runtime.

## Supported models

Model-specific profiling code is located in `models/`:

- `llama.py` — Llama architecture (Llama-3.1-8B, Llama-3.1-70B)
- `mixtral.py` — Mixtral-8x7B (MoE)
- `phimoe.py` — Phi-mini-MoE-instruct (MoE)
- `qwen3.py` — Qwen3 (Qwen3-8B, Qwen3-30B-A3B, etc.)

## Adding a new model or hardware

1. Add a model profiling script in `models/` following the existing examples.
2. Set the target hardware name and model identifier in the profiling shell scripts.
3. Run the profiling and predictor build steps above.
4. Create a `cluster_config` entry referencing the new hardware name.

## Internal design

### Profiler role separation

`attn` (attention kernel time) is **not** measured by the layer profiler. It is measured
independently by the attention profiler and stored in `attention.csv`. The two profilers
produce complementary outputs that LLMServingSim merges at runtime.

| Profiler | Entry point | Timer keys | Output |
|---|---|---|---|
| Layer profiler | `profiler/layers/main.py` | embedding, input_layernorm, q_proj, k_proj, v_proj, rope, o_proj, post_layernorm, gate_proj, up_proj, act_fn, down_proj, final_layernorm, lm_head | `layers.csv` |
| Attention profiler | `profiler/attention/main.py` | attn_prefill, attn_decode | `attention.csv` |

### Layer profiler — how it works

Measurement is done by `Timer` context managers inserted directly into the model forward
methods (not post-hoc hooks). Each `Timer(name=...)` wraps exactly one operation and
registers its median latency with `TimerStatsStore`.

Latency estimation formula after each `(input_len, kv_len)` config:

```
full_latency = embedding + final_layernorm + lm_head
             + per_block_time × original_num_layers
```

where `per_block_time` is the sum of all block-level timer keys. For MoE models
(Mixtral, PhiMoE) the expert contribution is estimated as:

```
expert_time += latency(n_tok) × (num_local_experts / tp_size)
n_tok = max(input_len // num_local_experts // tp_size, 1)
```

This corrects for HuggingFace's sequential expert execution to approximate real
parallel-dispatch latency.

**TP simulation:** actual distributed execution is not performed. Instead, projection
matrices and embedding tables are sliced by `tp_size` at model construction time so that
a single-GPU run approximates per-rank cost.

**Weights:** `AutoConfig` is used to construct model structure only — pretrained weights
are never loaded. Latency is measured on random-weight tensors.

### Attention profiler — how it works

Calls `flash_attn_varlen_func` directly (bypassing the model) to isolate FA2 kernel cost.

Input space covered:

| Stage | Variables swept |
|---|---|
| Prefill (chunked) | `prefill_chunk_size` × `kv_cache_size` (multiples of chunk), batch=1 |
| Prefill (full) | `seq_len` in stepped range up to `max_len`, kv=0, batch=1 |
| Decode | `kv_cache_size` × `batch_size` combinations |

Configurations exceeding the estimated KV cache memory budget
(`is_under_memory_limit`) are filtered before profiling begins.

### Model-specific notes

**Qwen3**
- `q_norm` and `k_norm` (per-head RMSNorm, Qwen3-specific) are folded inside the
  `q_proj` and `k_proj` timers respectively. Their cost is not measured separately.
- Uses sliding window attention on alternating layers (`layer_types` in config).
  The layer profiler profiles with `num_layers=1` so only one layer type is captured
  at a time; ensure the target layer type is represented in `--num-layers`.

**MoE models (Mixtral, PhiMoE)**
- `config.collect_router_stats` can be toggled in `layers/main.py` to capture routing
  statistics alongside latency.
- Expert timers (`expert.w1`, `expert.w2`, `expert.w3`) measure a single expert's
  projection at `n_tok = input_len // num_experts // tp_size` tokens.
