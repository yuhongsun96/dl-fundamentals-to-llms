# The KV-Cache Problem

This is the load-bearing file of Part 7. Every attention variant in 7.1, the whole of 7.2, and half the motivation for 7.3 exist because of the arithmetic below. Read it with a calculator and the rest of the part becomes obvious.

**Convention:** `L` layers, `H` query heads, `n_kv` key/value heads, `D_h` per-head dim, `S` sequence length, bf16 = 2 bytes/value. 8B/70B/405B configs from [ARCHITECTURE.md](../../ARCHITECTURE.md).

## What gets cached, and why

Generation is autoregressive: to produce token `t+1` you run one token through the stack. But attention at position `t+1` needs the **keys and values of every previous position** — and those don't change when new tokens arrive, because the causal mask means position `i`'s representation never depends on anything after `i` ([5.1/05](../../part5_transformer_rebuilt/5.1_self_attention/05_causal_and_bidirectional_masking.md)). So you compute each position's `K` and `V` once and keep them:

```
prefill:  run the whole prompt in parallel → fill the cache with K,V for all prompt positions
decode:   per new token, compute its q,k,v; append k,v to the cache;
          attend the single new q against the ENTIRE cache
```

Without the cache, generating token `t` would re-run the whole prefix — `O(S²)` work per token, `O(S³)` per sequence. With it, decode is `O(S)` per token. The cache is non-negotiable; the problem is that it's **enormous**.

Note what is *not* cached: queries (used once, at their own position and never again) and activations (no backward pass at inference). Just `K` and `V`.

### Prefill: the phase with no cache yet

Prefill deserves naming precisely, because it's the one phase where **nothing is cached and everything must be computed from scratch.** When a request arrives you hold only token IDs; there is no `K`, no `V`, no state. So the entire prompt is pushed through the full stack, all `S` positions at once, and for every layer that means:

| Work | Cost for `S` prompt tokens, per layer |
|---|---|
| `W_Q, W_K, W_V, W_O` projections | `≈ 2 · 4 d_model² · S` |
| Attention scores `QKᵀ` and the weighted sum `AV` | `≈ 4 S² d_model` — the **quadratic** term |
| FFN (SwiGLU: three matrices) | `≈ 2 · 3 d_model · d_ff · S` |

Summed over layers, the dense-matmul part is just `2N·S` — **every weight touched once per prompt token** — plus the attention term that grows with `S²`. Verified for a 70B model:

| Prompt `S` | Prefill FLOPs | attention's share | Time to first token | Then, per decoded token |
|---|---|---|---|---|
| 2,048 | 3.0e14 | 3.7% | **301 ms** | 41.8 ms |
| 8,192 | 1.3e15 | 13.3% | **1.34 s** | 41.8 ms |
| 131,072 | 6.3e16 | 71.1% | **64 s** | 41.8 ms |

Three things to take from this:

- **Prefill is compute-bound, and that's why it's fast per token.** All `S` tokens share one read of the weights, so intensity `≈ S` — at `S = 8192` that's 8192 FLOP/byte against a machine balance of ~296, deep in compute-bound territory. Prefill processes thousands of tokens in the time decode produces one.
- **The cache is prefill's by-product, not extra work.** Those `K`/`V` values had to be computed anyway to attend within the prompt; storing them is what lets decode skip recomputing them. At `S = 8192` you write 2.5 GiB and thereby save re-doing 1.1e15 FLOPs on *every* subsequent token.
- **Prefill sets time-to-first-token; decode sets tokens-per-second.** They're separate user-visible metrics with separate bottlenecks — compute for one, memory bandwidth for the other ([7.2/01](../7.2_efficient_attention/01_the_memory_bandwidth_wall.md)) — which is why the two phases are optimized, and often scheduled, independently (Part 9.2).

## The arithmetic

Per token, the cache stores `K` and `V` for every layer and every KV head:

```
bytes/token = 2 (K and V) × L × n_kv × D_h × bytes_per_value
```

Multiply by sequence length for one sequence, and by batch size for a serving replica. The numbers, all verified:

| Model | `L` | `n_kv` | KV bytes/token | at `S = 8192` | at `S = 128K` |
|---|---|---|---|---|---|
| Llama-3 8B | 32 | 8 | **128 KiB** | 1.0 GiB | 16 GiB |
| Llama-3 70B | 80 | 8 | **320 KiB** | 2.5 GiB | **40 GiB** |
| Llama-3 405B | 126 | 8 | **504 KiB** | 3.9 GiB | 63 GiB |

Now notice the `n_kv` column: **8** for all three, even though these models have `H = 32`, `64`, and `128` query heads respectively. That's not a typo — it's **grouped-query attention** (GQA, [file 02](02_mqa_and_gqa.md)) already applied. Every modern model ships with this mitigation, so the table above shows the cache *after* the fix. Set `n_kv = H` to see what plain multi-head attention would have cost the same configs:

| Model | MHA bytes/token | at `S = 128K` |
|---|---|---|
| Llama-3 8B (`H=32`) | 512 KiB | 64 GiB |
| Llama-3 70B (`H=64`) | 2,560 KiB | **320 GiB** |
| Llama-3 405B (`H=128`) | 8,064 KiB | **1,008 GiB** |

A single 128K-token conversation with an MHA-70B would need 320 GiB of KV cache — more than four H100s' worth of HBM, for **one user**. That is the number that killed MHA.

## Three consequences that shape everything downstream

**1. The cache overtakes the weights.** A 70B model in bf16 is ~140 GiB of weights — a fixed cost, amortized across all concurrent users. KV is **per sequence**. With GQA at 128K, `140 / 40 = 3.5` concurrent sequences saturate as much memory as the entire model; with MHA it's **0.44** — you couldn't fit one. Since throughput ≈ how many sequences you can batch, KV size *is* your serving cost.

*Why batch size is a memory question at all:* a decode step processes one token from each of `B` sequences, and each sequence attends over **its own** history — so each needs its own cache resident. You cannot batch a sequence whose cache isn't in HBM. Hence

```
max_B  ≈  (HBM − weights − overhead) / (context × KV bytes per token)
```

That divisor is the cache because **the cache is the only large per-sequence cost.** Everything else in HBM is either fixed or negligible (70B-class, `d_model = 8192`, `d_ff = 28672`):

| Occupant | Size | Scales with |
|---|---|---|
| Weights (bf16) | **130.4 GiB** | nothing — fixed, shared |
| **KV cache** | 0.62 GiB @ 2K · 2.5 GiB @ 8K · **40 GiB @ 128K** | **per sequence × context** |
| Activations, in-flight batch | **18 MiB** at batch 256 | batch (decode) / prompt length (prefill) |
| Logits (fp32, for sampling) | 125 MiB at batch 256 | batch × vocab |
| Framework overhead | ~1–2 GiB | CUDA context, cuBLAS workspaces, fragmentation, TP collective buffers |

Activations are tiny at decode for a specific reason: **there's no backward pass**, so nothing is retained for gradients (contrast [2.2/04](../../part2_neural_network_fundamentals/2.2_backpropagation/supplementary/04_gradient_checkpointing.ipynb), where retaining them *is* the problem) — and each sequence contributes only one token. At 128K with batch 256 the cache is ~**580,000×** the activations, which is why the formula can ignore every row but one. Two exceptions worth knowing: prefill activations scale with prompt length (576 MiB at `S = 8192`), which is why serving stacks do **chunked prefill**; and computing logits for *all* prefill positions would cost `S × V × 4` = **3.9 GiB** at `S = 8192`, which is why only the last position's logits are materialized.

In practice frameworks don't estimate this — they **measure** it: load weights, profile one forward pass for peak activations, then convert all remaining memory under a utilization cap (vLLM's default ~0.9) into KV blocks ([7.2/03](../7.2_efficient_attention/03_kv_cache_memory_management.md)). The resulting block count *is* the concurrency limit. On 8×H100 (640 GiB) with ~130 GiB of weights: **710** sequences at 2K, **177** at 8K, **11** at 128K — to be read against the ~300 that consequence 2 says you need.

**2. Decode is memory-bound, badly.** Generating one token with a 70B dense model costs `≈ 2N = 1.4e11` FLOPs and requires reading `≈ 2N = 1.4e11` bytes of weights (bf16) — an **arithmetic intensity of 1 FLOP/byte**. An H100 does ~990 TFLOP/s against ~3.35 TB/s of HBM, a machine balance of **~296 FLOP/byte**. Decode at batch 1 is ~300× off compute-bound: the GPU is idle, waiting on memory. Two implications, both central to Part 9:
   - **Batching is the only fix** — you need ~300 concurrent sequences to reach compute-bound, and (consequence 1) the KV cache is what stops you.
   - **Fewer bytes is faster**, even at identical FLOPs. This is why quantization and MoE speed up decode, and why shrinking the KV cache is a latency win and not just a capacity win.

**3. Prefill and decode are different machines.** Prefill processes `S` tokens at once — high arithmetic intensity, compute-bound, FLOPs matter. Decode processes one — memory-bound, bytes matter. The same model on the same GPU has two performance regimes with opposite optimizations, which is why Part 9 schedules them separately.

## The design space this opens

Every term in `2 × L × n_kv × D_h × bytes` is a target, and each has a real technique attached:

| Attack the term | Technique | Where |
|---|---|---|
| `n_kv` | share KV across query heads (MQA/GQA) | [file 02](02_mqa_and_gqa.md) |
| `D_h` (effectively) | compress KV to a low-rank latent (MLA) | [file 03](03_multi_head_latent_attention.md) |
| `S` | don't attend to everything (sliding window, sparsity) | [file 04](04_local_and_sparse_patterns.md) |
| `bytes` | quantize the cache to int8/fp8 | Part 9.2 |
| `L` | share KV across *layers* (cross-layer attention) | rarer; noted in [file 04](04_local_and_sparse_patterns.md) |
| the whole thing | replace the growing cache with a fixed-size state | [7.3](../7.3_alternatives_to_attention/01_linear_attention.md) |

Read that table as the syllabus for the rest of 7.1–7.3.

## Why it matters in modern LLM work

- **It's the number that explains model cards.** `n_kv = 8` on nearly every modern model is this arithmetic, not fashion.
- **It's your capacity planning.** Max concurrent sequences ≈ `(HBM − weights) / (KV bytes/token × context)`. That single formula prices a deployment.
- **It reframes "efficiency."** At inference the scarce resource is bytes moved, not operations performed — the opposite of the training-time intuition Part 6 built.

## Self-check

1. Why are `K` and `V` cached but not `Q`, and what property of the causal mask makes caching valid at all?
2. Compute KV bytes/token for a hypothetical `L=48`, `n_kv=4`, `D_h=128` model in bf16. At what context length does one sequence's cache reach 10 GiB?
3. A 70B model has 140 GiB of weights. Explain why "the weights dominate memory" is true at 2K context and false at 128K.
4. Decode has arithmetic intensity ~1 FLOP/byte on a machine balanced at ~296. Name the two distinct fixes and why the KV cache obstructs one of them.
5. Why does the same model need *opposite* optimizations for prefill and decode?

### Answers

1. `Q` is used exactly once — at its own position, against the existing cache — and never referenced again, so there's nothing to reuse. `K` and `V` are consulted by *every* future position, so they're reused `S − i` times. Caching is valid because the causal mask makes position `i`'s `K`/`V` independent of all later tokens: they're final the moment they're computed. (A bidirectional model has no such guarantee — appending a token changes every representation, which is why encoders can't cache incrementally, [6.1/03](../../part6_pretraining_paradigms/6.1_pretraining_objectives/03_why_decoder_only_won.md).)
2. `2 × 48 × 4 × 128 × 2 = 98,304` bytes = **96 KiB/token**. 10 GiB / 96 KiB ≈ **109,000 tokens**.
3. The weights are a *fixed* 140 GiB regardless of context or users; KV is `320 KiB × S` **per sequence**. At `S = 2048` one sequence is 0.64 GiB — you'd need ~220 concurrent sequences before KV rivals the weights, so weights dominate. At `S = 128K` one sequence is 40 GiB, so **3.5 sequences** match the entire model's footprint. Same model, same hardware; the crossover moved by two orders of magnitude because KV scales with context and weights don't.
4. (a) **Increase batch size** — more sequences reuse each weight byte read, raising intensity toward the machine balance; (b) **read fewer bytes** — quantize weights, or activate fewer of them (MoE, [7.4](../7.4_mixture_of_experts/01_the_sparse_ffn.md)). The KV cache obstructs (a): batching is limited by memory left over after weights, and each added sequence costs `KV bytes/token × S`, so at long context you run out of room long before ~300 concurrent sequences.
5. Prefill runs `S` tokens through each weight matrix in one pass, so each weight byte read is amortized over `S` tokens — high intensity, compute-bound, and FLOP reductions (sparsity, better kernels) pay. Decode runs *one* token per weight read — intensity ~1, memory-bound, and only byte reductions (quantization, fewer active params, smaller cache) pay. FLOP-saving tricks can even *hurt* decode if they add memory traffic.

## Exercise

Build the capacity calculator you'll actually reuse. Write a function taking `(L, H, n_kv, D_h, bytes_per_value, N_params, HBM_GiB)` and returning max concurrent sequences vs. context length. (a) Plot it for Llama-3-70B on 8×H100 (640 GiB) across `S ∈ [4K, 1M]`, for GQA (`n_kv=8`) and MHA (`n_kv=64`) — note where each hits 1 sequence. (b) Add an int8-KV line and quantify the win. (c) Compute the context length at which KV cache equals weight memory for a *single* sequence, for the 8B, 70B, and 405B configs — you should find the crossover scales with model size in a way that surprises you at first; explain it in one sentence using the bytes/token column above.
