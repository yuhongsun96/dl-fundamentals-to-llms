# Where Quadratic Actually Bites

"Attention is `O(S²)`" is the most-repeated fact about Transformers and one of the most misleading, because it invites the conclusion that attention FLOPs are the long-context bottleneck. They usually aren't. This file does the arithmetic to locate the real walls, which turn out to be three different things in three different regimes.

**Convention:** `d_model = 4096`, `d_ff = 3.5 d_model` (SwiGLU), `D_h = 128`, per [ARCHITECTURE.md](../../ARCHITECTURE.md). Per-token, per-layer FLOPs; multiply-add counted as 2.

## The crossover: attention FLOPs vs. everything else

A block's per-token cost has two parts that scale differently:

```
constant in S:   qkvo projections   ≈ 2 · 4 d_model²        = 1.34e8
                 SwiGLU FFN         ≈ 2 · 3 d_model · d_ff   = 3.52e8
                                                     total   ≈ 4.86e8
grows with S:    QKᵀ and AV         ≈ 4 · S · d_model
```

Setting them equal, attention's score-and-weighted-sum work overtakes *all* the matmuls at:

```
S = 4.86e8 / (4 · 4096) ≈ 29,700 tokens
```

Verified, with the share of block FLOPs attention consumes:

| `S` | attention share of block FLOPs |
|---|---|
| 2,048 | **6.5%** |
| 8,192 | 21.6% |
| 32,768 | 52.5% |
| 131,072 | **81.5%** |
| 250,000 | 89.4% |

Read this carefully, because it cuts both ways. **At the sequence lengths of 2019–2023 (512–8K), attention was a small minority of compute** — which is why the entire "efficient attention" literature of that era produced disappointing end-to-end speedups ([7.1/04](../7.1_attention_variants/04_local_and_sparse_patterns.md)): halving 6.5% of the work is invisible. But **at 128K it's 82%**, so the asymptotics genuinely arrive; the quadratic term was never wrong, just premature. Note also that `S ≈ 30K` crossover is a *ratio* of `d_ff` to `d_model` and shifts with architecture — a wider FFN pushes the crossover later.

## Wall 1 (training): activation memory — now solved

Before FlashAttention, the binding constraint at long context wasn't FLOPs at all but the `S × S` score matrix, which had to be *stored* for the backward pass: `O(S²)` memory per head per layer. At `S = 8192` that's ~256 MiB per head ([7.2/01](../7.2_efficient_attention/01_the_memory_bandwidth_wall.md)) — multiply by heads and layers and long-context training was simply impossible.

FlashAttention made this `O(S)` ([7.2/02](../7.2_efficient_attention/02_flashattention.md)), which is why 128K training exists. **This wall is gone**, and it's worth being explicit about that because much long-context discussion is still written as though it weren't.

## Wall 2 (training): the sequence doesn't fit on one device

What replaced it. Even with `O(S)` attention memory, a single 128K–1M token sequence's activations exceed one GPU: the residual stream alone is `S × d_model × 2 bytes × L` for anything you must keep, plus optimizer states and weights. You cannot shard this with data parallelism, because it is **one sample** — data parallelism splits the batch, and here the batch is 1.

So you must split the **sequence dimension** across devices, which requires attention to work across a distributed sequence — every query needs every key. That's [file 02](02_training_at_long_context.md) (ring attention and friends), and it's the actual reason long-context training is hard.

## Wall 3 (inference): the KV cache — the real one

At serving time, FLOPs are essentially never the constraint. From [7.1/01](../7.1_attention_variants/01_the_kv_cache_problem.md):

- 70B at 128K = **40 GiB of KV cache per sequence** (with GQA; 320 GiB with MHA).
- Weights are 140 GiB, so **3.5 concurrent sequences** consume as much memory as the entire model.
- Decode is memory-bound at intensity ≈ 1 ([7.2/01](../7.2_efficient_attention/01_the_memory_bandwidth_wall.md)), and the cache is what prevents the batching that would fix it.

The asymmetry worth internalizing:

| | Prefill (process the prompt) | Decode (generate) |
|---|---|---|
| Scales as | `O(S²)` compute — quadratic, real | `O(S)` per token — cache reads |
| Bound by | **compute** | **memory bandwidth** |
| Long-context pain | latency to first token | cache size limits batch ⇒ throughput |

So "quadratic attention" is a genuine prefill problem (time-to-first-token on a 200K-token document is dominated by it, and that's where sparse attention like NSA pays off — [7.1/04](../7.1_attention_variants/04_local_and_sparse_patterns.md)) and essentially a *non-issue* for decode, where the cost is linear-but-memory-bound.

## The summary table

Long context is not one problem. It's four, with four different owners:

| Wall | Regime | Status | Owner |
|---|---|---|---|
| `O(S²)` activation memory | training | **solved** | FlashAttention ([7.2/02](../7.2_efficient_attention/02_flashattention.md)) |
| Sequence exceeds one device | training | managed | context parallelism ([file 02](02_training_at_long_context.md)) |
| `O(S²)` prefill compute | inference | partially attacked | sparse attention ([7.1/04](../7.1_attention_variants/04_local_and_sparse_patterns.md)) |
| KV cache size | inference | **the live problem** | GQA/MLA ([7.1](../7.1_attention_variants/02_mqa_and_gqa.md)–[03](../7.1_attention_variants/03_multi_head_latent_attention.md)), paging ([7.2/03](../7.2_efficient_attention/03_kv_cache_memory_management.md)), SSMs ([7.3](../7.3_alternatives_to_attention/02_state_space_models.md)) |

And a fifth that isn't computational at all: **the model may not actually use the context it accepts** ([file 03](03_evaluating_long_context.md)). Advertised context length is a memory-allocation claim, not a capability claim.

## Why it matters in modern LLM work

- **It stops you optimizing the wrong thing.** "Reduce attention FLOPs" is the right instinct for 200K-token prefill and the wrong one for 4K-context decode; the crossover table tells you which regime you're in.
- **It explains the research timeline** — why efficient attention failed in 2020 and sparse attention returned in 2025 is one number (`S`), moving.
- **It's the map for the rest of 7.5** and a good final check on Part 7: every technique in this part attaches to exactly one row of that table.

## Self-check

1. Derive the `S ≈ 30K` crossover from the two FLOP expressions, and state what architectural change moves it later.
2. At `S = 2048`, attention is 6.5% of block FLOPs. What does that imply about a paper claiming a 2× attention speedup at that length?
3. Which long-context wall did FlashAttention remove, and which one did it *expose*?
4. Why can't data parallelism help fit a single 1M-token sequence?
5. Explain why "quadratic attention" is a prefill problem but not really a decode problem.
6. Rank the four walls by how much they constrain a production deployment serving 128K contexts, and justify the top one.

### Answers

1. Constant-in-`S` per-token work is `2·4d² + 2·3d·d_ff = 1.34e8 + 3.52e8 ≈ 4.86e8` FLOPs; attention's score+AV work is `4·S·d`. Equate: `S = 4.86e8/(4·4096) ≈ 29,700`. A **wider FFN** (larger `d_ff/d_model`) raises the constant term and pushes the crossover to larger `S`; so does anything that adds `S`-independent compute per token.
2. That the end-to-end gain is at most ~3.3% (halving 6.5%), so the paper's speedup is a *microbenchmark* result that will be invisible in training or serving throughput. This is precisely why the 2020 efficient-attention wave underdelivered — correct kernel-level claims about an operation that wasn't the bottleneck at the lengths in use.
3. It removed the **`O(S²)` activation-memory** wall (making attention memory linear in `S`), which is what made long-context training possible at all. It exposed the **KV cache** as the next binding constraint at inference — and, at training, the fact that a single long sequence's activations exceed one device. Solving a bottleneck promotes the next one.
4. Because data parallelism splits along the **batch** dimension, replicating the model and giving each device different samples. A single 1M-token sequence is *one sample* — there's nothing to split that way. You must partition the **sequence** dimension instead, which breaks attention's requirement that every query see every key, hence the need for ring/context parallelism ([file 02](02_training_at_long_context.md)).
5. Prefill processes all `S` tokens at once, computing the full `S × S` score structure — genuinely quadratic, compute-bound, and the dominant term past `S ≈ 30K`, so it sets time-to-first-token. Decode processes **one** query against `S` cached keys: that's `O(S)` per token, and it's memory-bandwidth-bound rather than compute-bound ([7.2/01](../7.2_efficient_attention/01_the_memory_bandwidth_wall.md)), so the cost is *reading the cache*, not the arithmetic. The quadratic term never appears in decode.
6. **KV cache size** first — it caps concurrent sequences at ~3.5 per model-memory-equivalent on a 70B at 128K, which directly caps batch size and therefore throughput in a memory-bound regime; everything else is secondary. Then **prefill compute** (dominates time-to-first-token at 82% of block FLOPs at 128K). Then **sequence-doesn't-fit**, which is a training-only concern. Last, **activation memory**, already solved. (Arguably tied with the fifth: a model that can't *use* 128K makes the whole ranking moot — [file 03](03_evaluating_long_context.md).)

## Exercise

Build the crossover calculator and use it to make a decision. (a) Write `block_flops(S, d_model, d_ff_mult, D_h)` returning projection, FFN, and attention FLOPs separately; reproduce the share table above and find the crossover for `d_ff_mult ∈ {2.67, 3.5, 8}` — note how much architecture moves it. (b) Add a memory model: activation memory with and without FlashAttention, plus KV cache from [7.1/01](../7.1_attention_variants/01_the_kv_cache_problem.md)'s formula. Plot all three against `S ∈ [2K, 1M]` on log axes and mark which is largest in each region — you should see three distinct regimes. (c) You're given a fixed engineering budget and a product serving 200K-token document Q&A with a p50 of 3 output tokens (heavy prefill, light decode) — and a second product doing 4K-context chat with long generations. For each, name from your plots which wall to attack and which Part 7 technique you'd deploy first. (d) One sentence on why the answers differ.
