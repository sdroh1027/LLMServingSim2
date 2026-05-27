# Aeala/ShareGPT_Vicuna_unfiltered — Dataset Analysis

Analysis of the upstream ShareGPT corpus that feeds the planned `sharegpt_multiturn`
traces. Goal: pick reasonable filter caps (max turns, max KV length) and confirm
whether per-turn cumulative-input growth matches the assumed KV-cache reuse pattern.

## Source

- **Repo:** `Aeala/ShareGPT_Vicuna_unfiltered` (HF Hub, dataset)
- **File:** `ShareGPT_V4.3_unfiltered_cleaned_split.json` — 464 MB
- **Tokenizer:** `meta-llama/Llama-3.1-8B` (same default as `dataset/sharegpt_parser.py`)
- **Analyzer:** `ShareGPT_Traces/analyze.py` → `ShareGPT_Traces/stats.json`

## Sessions & turns

The raw file is a flat list of "splits" — long conversations are pre-chopped and stored
under IDs like `hRPPgZT_0`, `hRPPgZT_11`. Splits with the same id-prefix must be
re-merged (suffix-ordered) to recover a *logical session*. About 36% of raw splits
start with `gpt` and only make sense after this merge.

| | Value |
|---|---|
| Raw splits | **68,623** |
| `human`-start / `gpt`-start splits | 44,012 / 24,611 |
| Logical sessions (merged by id prefix) | **40,698** |
| Sessions with ≥1 (human, gpt) pair | 40,697 |
| Dropped (empty / unpairable) | 1 |

### Turns per session (n = 40,697)

| mean | p50 | p90 | p95 | p99 | p99.9 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 5.53 | 2 | 12 | 19 | 47 | 141 | 318 |

Heavily long-tailed. Median session is just 2 turns, but the top 1% exceed 47 turns.

## Token statistics (overall, n = 224,995 turn-pairs)

Two distinct "input" measurements per turn:

- **`in_cum` (cumulative input)** = the prompt the model actually sees at turn *k*:
  `(prior turns' prompts + responses) + the new user utterance`. This is the
  prefill length and the relevant figure for KV-cache reuse analysis.
- **`in_new` (new prompt only)** = just this turn's user message.

At turn 0 the two are identical; from turn 1 onward `in_cum − in_new` is the
prior-context length carried into this turn.

| Metric | mean | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| per-turn **INPUT (cumulative, `in_cum`)** | 4,594 | 1,407 | 11,475 | 20,228 | 50,797 | 126,774 |
| per-turn input (new prompt only, `in_new`) | 77 | 17 | 127 | 314 | 1,265 | 20,438 |
| per-turn **OUTPUT** | 289 | 262 | 576 | 707 | 769 | 31,695 |
| session total input (Σ `in_cum` over turns) | 25,397 | 758 | 23,773 | 61,883 | 379,565 | 12.4 M |
| session final-turn KV footprint (in_cum + out) | 2,025 | 847 | 4,472 | 7,198 | 18,097 | 127,543 |

## Per-turn-index breakdown

For each turn slot *k* (0-indexed), how many sessions reach that turn and what
their lengths look like at that point.

### Coverage decay (sessions that reach turn *k*)

| turn k | sessions | % of total |
|---:|---:|---:|
| 0 | 40,697 | 100.0% |
| 1 | 26,694 | 65.6% |
| 2 | 20,106 | 49.4% |
| 3 | 15,839 | 38.9% |
| 4 | 12,847 | 31.6% |
| 5 | 10,592 | 26.0% |
| 7 | 7,454 | 18.3% |
| 10 | 4,892 | 12.0% |
| 15 | 2,822 | 6.9% |
| 20 | 1,780 | 4.4% |
| 30 | 918 | 2.3% |
| 50 | 362 | 0.9% |
| 100 | 86 | 0.2% |
| 200 | 11 | 0.03% |

### Length per turn slot (first 20 turns)

Column meaning (all token counts, Llama-3.1 tokenizer):

- **`in_cum_*`** — **Cumulative input** the model sees at this turn:
  `(prior turns' prompts + responses) + this turn's new user message`.
  This is the prefill length, i.e. the figure relevant to KV-cache reuse.
- **`in_new_*`** — **New prompt only**: just this turn's user utterance,
  with no carried-over context.
- **`out_*`** — This turn's gpt response length.

Per turn, `in_cum - in_new` ≈ the prior-context length (= candidate for KV
prefix hit). At turn 0 the two are identical because there is no prior context.

|  k  |  sess  | in_cum_mean | in_cum_p50 | in_cum_p95 | in_new_mean | in_new_p50 | in_new_p95 | out_mean | out_p50 | out_p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  0 | 40,697 |   128 |   24 |   592 |   128 |   24 |   592 |  280 |  251 |  694 |
|  1 | 26,694 |   472 |  343 | 1,240 |    67 |   15 |   264 |  293 |  267 |  692 |
|  2 | 20,106 |   826 |  670 | 1,851 |    64 |   15 |   221 |  303 |  277 |  705 |
|  3 | 15,839 | 1,193 | 1,012 | 2,522 |    67 |   15 |   225 |  307 |  282 |  709 |
|  4 | 12,847 | 1,557 | 1,360 | 3,148 |    64 |   16 |   220 |  309 |  280 |  727 |
|  5 | 10,592 | 1,928 | 1,705 | 3,752 |    61 |   16 |   205 |  304 |  278 |  723 |
|  6 |  8,806 | 2,296 | 2,055 | 4,440 |    61 |   16 |   211 |  301 |  278 |  699 |
|  7 |  7,454 | 2,646 | 2,400 | 5,081 |    64 |   16 |   221 |  303 |  276 |  726 |
|  8 |  6,371 | 3,011 | 2,753 | 5,686 |    59 |   16 |   217 |  295 |  272 |  714 |
|  9 |  5,564 | 3,357 | 3,095 | 6,321 |    64 |   16 |   217 |  288 |  265 |  699 |
| 10 |  4,892 | 3,704 | 3,458 | 6,884 |    61 |   16 |   238 |  292 |  270 |  704 |
| 11 |  4,351 | 4,030 | 3,786 | 7,416 |    58 |   16 |   208 |  293 |  265 |  712 |
| 12 |  3,845 | 4,398 | 4,152 | 8,020 |    59 |   17 |   205 |  292 |  268 |  717 |
| 13 |  3,421 | 4,732 | 4,485 | 8,619 |    61 |   16 |   199 |  288 |  261 |  731 |
| 14 |  3,109 | 5,101 | 4,801 | 9,185 |    65 |   16 |   257 |  282 |  252 |  700 |
| 15 |  2,822 | 5,441 | 5,110 | 9,743 |    67 |   16 |   257 |  281 |  252 |  703 |
| 16 |  2,563 | 5,752 | 5,435 | 10,290 |   62 |   15 |   236 |  284 |  257 |  714 |
| 17 |  2,329 | 6,099 | 5,783 | 10,873 |   70 |   16 |   300 |  275 |  248 |  676 |
| 18 |  2,098 | 6,434 | 6,093 | 11,255 |   72 |   17 |   247 |  276 |  252 |  678 |
| 19 |  1,941 | 6,743 | 6,411 | 11,718 |   50 |   17 |   217 |  282 |  261 |  690 |
| 20 |  1,780 | 7,070 | 6,724 | 12,293 |   58 |   16 |   182 |  278 |  251 |  711 |

### Observations

- **`in_cum` grows ~340 tok/turn** (≈ 60 new prompt + 280 prior output). Roughly linear.
- **`out` is flat at ~290 tok across all turn slots.** The model isn't becoming
  more verbose as the session progresses.
- **Turn 0 `in_new` is ~2× longer than later turns** (mean 128 vs ~60). Users
  open with the long question, then mostly follow up with short clarifications.
- **`in_cum − in_new` (= prior context, the KV-prefix candidate) reaches ~12k tok
  at turn 20 (p95).** This is the gap that prefix-cache reuse is supposed to
  eliminate from prefill cost.
- **By turn 20 `in_cum` p95 ≈ 12k tok** — beyond a 4k context window. Sims with
  ≤4k context need to either drop late turns or truncate context.

## Implications for the simulator trace

| Decision | Recommendation | Reason |
|---|---|---|
| Group by id prefix before pairing | **required** | Otherwise 36% of splits start mid-conversation. |
| Build cumulative input in **token-id space**, not by re-tokenizing concatenated strings | **required** | Preserves bit-exact prefix → KV-cache hit measurement is meaningful. |
| `max_turns_per_session` | 20 (or 30) | Keeps p95 sessions; drops the 4.4% long tail that would dominate batch dwell time. |
| `max_kv_length` | match target model | 4k cuts in around p65–p70; 8k around p85; 32k passes nearly everyone. |
| `max_output_length` | 1,024 is plenty | p99 output = 769; p99.9 = 804. |
| Session arrival distribution | Poisson on **turn 0** only | Subsequent turns are gated by `intra_session_gap_ns`. |
| `intra_session_gap_ns` distribution | lognormal(μ=log 5 s, σ=1) — or fixed for ablation | The raw dataset has no real timestamps, so this is a modeling choice. |

## Reproduce

```powershell
# from C:\Users\cruha\Documents\workspace\KVCache_Simulator\ShareGPT_Traces
python analyze.py                                  # full run, ~3 min
python analyze.py --limit 2000                     # smoke test
python analyze.py --no-cumulative-input            # report prompt-only input
```

Outputs `stats.json` next to the script.
