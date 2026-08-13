# FlashAttention

The most consequential systems result in the Transformer era, and unusually satisfying because the whole thing rests on one algebraic trick you can implement in twenty lines. FlashAttention (Dao et al., 2022) makes attention several times faster and reduces its memory from `O(S²)` to `O(S)` **without changing the output by a single bit beyond float rounding.** This file derives the trick, then covers what v2 and v3 added.

## The problem restated

From [file 01](01_the_memory_bandwidth_wall.md): naive attention writes and re-reads an `S × S` score matrix through HBM — **256 MiB per head at `S = 8192`** versus 6 MiB to stream `Q`, `K`, `V` once, a ~43× traffic ratio. The fix is obvious and blocked: process the scores in tiles that fit in SRAM, never materializing the full matrix. What blocks it is **softmax**, which is a *global* operation over each row — it needs the row max (for stability, [1.2/03](../../part1_math_foundations/1.2_calculus_probability/03_softmax_logsumexp.md)) and the row sum, neither of which you know until you've seen the whole row.

So the question is: **can you compute a softmax-weighted sum incrementally, seeing the row only one tile at a time?** Yes.

## Online softmax — the trick

Keep three running quantities as you stream tiles: the max seen so far `m`, the running denominator `l`, and the running weighted-sum accumulator `acc`. On a new tile with scores `s`:

```
m_new = max(m, max(s))                    # new running max
corr  = exp(m − m_new)                    # correction factor for everything so far
acc   = acc · corr + exp(s − m_new) @ V_tile
l     = l   · corr + sum(exp(s − m_new))
m     = m_new
```
and at the end, `out = acc / l`.

The insight is the `corr` term. Every previously accumulated value was scaled by `exp(−m_old)`; when a larger max arrives, everything must be re-based to `exp(−m_new)`. Multiplying by `exp(m_old − m_new)` does exactly that — and because it's a *scalar* multiplying both `acc` and `l`, it costs almost nothing and needs no revisiting of old tiles. You are, in effect, carrying the numerator and denominator of the softmax separately and renormalizing lazily.

**Verified exact.** With `S = 500`, `D_h = 32` in float64, comparing against a standard full-matrix `softmax(Kq) @ V`:

| Tile size | max abs difference |
|---|---|
| 1 | 2.2e-15 |
| 7 | 1.6e-15 |
| 64 | 1.3e-15 |
| 500 (one tile) | 1.1e-15 |

Machine epsilon, for *any* tile size — including tiles of 1 and 7, which don't divide the sequence length. **FlashAttention is not an approximation**; that distinction is what separates it from the entire 2020 efficient-attention literature ([7.1/04](../7.1_attention_variants/04_local_and_sparse_patterns.md)) and is why it was adopted universally rather than as a quality tradeoff.

## The full algorithm

Wrap the online softmax in a two-level tiling:

```
for each block of Q rows (kept in SRAM):
    init  m = −∞, l = 0, acc = 0
    for each block of K,V columns (streamed into SRAM):
        s = Q_block @ K_blockᵀ · scale      # computed in SRAM, never written to HBM
        apply causal mask (skip the block entirely if fully masked)
        update m, l, acc with the online-softmax step
    write out = acc / l                     # the ONLY HBM write, size (block, D_h)
```

Three properties follow:

- **HBM traffic is `O(S·D_h)`, not `O(S²)`** — each of `Q`, `K`, `V` is read a small number of times and only the output is written.
- **Memory is `O(S)`**, so attention stops being the thing that limits sequence length.
- **Causal masking becomes a compute saving**, not just a correctness step: whole `(Q_block, K_block)` pairs above the diagonal are skipped, roughly halving work — a real advantage over the naive version, which computes all `S²` scores and then masks.

**Backward pass and recomputation.** Backprop needs the attention probabilities, which were never stored. FlashAttention **recomputes** them tile-by-tile in the backward pass from `Q`, `K`, `V` plus the saved per-row softmax statistics (`m`, `l` — `O(S)` values). This is gradient checkpointing ([2.2/04](../../part2_neural_network_fundamentals/2.2_backpropagation/supplementary/04_gradient_checkpointing.ipynb)) applied surgically to one operation, and the trade is the same: more FLOPs, far less memory traffic — which on memory-bound code is a strict win.

## v1 → v2 → v3

Each version is a systems refinement, not a new algorithm:

- **v1 (2022)** — the tiling + online softmax + recomputation result above. ~2–4× faster than naive, memory linear in `S`.
- **v2 (2023)** — better *parallelism and work partitioning*: parallelize over the sequence dimension (not just batch × heads, which starves the GPU at small batch — exactly the long-context, small-batch regime that matters), move the `1/l` rescaling out of the inner loop, and reduce non-matmul work. Roughly another ~2× and a much better occupancy story.
- **v3 (2024)** — Hopper-specific: exploit **FP8** tensor cores and asynchrony (warp specialization, overlapping the softmax with the matmuls via the TMA engine). Reported ~1.5–2× over v2 on H100 and closer to peak utilization.

The pattern across versions is worth noting: after v1 solved the *algorithmic* problem, all remaining gains came from mapping onto specific hardware — which is why FlashAttention is versioned against GPU generations and why the fastest kernel is always somewhat hardware-locked.

## The consequences that ripple outward

FlashAttention changed the field's incentives, not just its speeds:

- **It killed most approximate attention.** If exact attention is fast and linear-memory, a method that trades quality for speed must beat a *moving* baseline — and most 2020-era methods didn't. This is the direct cause of the "efficient Transformer" wave receding ([7.1/04](../7.1_attention_variants/04_local_and_sparse_patterns.md)).
- **It made long context feasible at all** — `O(S)` memory is a precondition for 128K training, which is why [7.5/02](../7.5_long_context/02_training_at_long_context.md) can even discuss the problem.
- **It shifted the bottleneck to the KV cache.** Once attention compute and activation memory were handled, what remained was the cache — which is why 7.1's variants (GQA, MLA) are a *post-*FlashAttention research program.
- **It created a kernel-compatibility tax.** Because everyone depends on these kernels, novel attention variants must ship a matching kernel to be usable — a real adoption barrier for MLA ([7.1/03](../7.1_attention_variants/03_multi_head_latent_attention.md)) and for sparse patterns.

## Why it matters in modern LLM work

- **You use it constantly**, whether or not you call it: `F.scaled_dot_product_attention` dispatches to it, and every serving stack depends on it.
- **The online-softmax idea generalizes** — the same "carry numerator, denominator, and a running max; renormalize lazily" pattern appears in distributed softmax (ring attention, [7.5/02](../7.5_long_context/02_training_at_long_context.md)), in streaming top-k, and in log-sum-exp reductions generally.
- **It's the canonical example of the roofline lesson:** the winning move was to spend FLOPs to save bytes.

## Self-check

1. Why does softmax obstruct tiling, and precisely which two row-statistics are the obstruction?
2. Write the correction factor and explain in one sentence why a *scalar* suffices to fix all previously accumulated work.
3. FlashAttention does more FLOPs than naive attention. Give both reasons, and explain why it's still faster.
4. Why is the exactness (to float rounding) strategically important, not just aesthetically nice?
5. What does v2 change, and why does that specifically matter for long-context training?
6. Why did FlashAttention *increase* the relative importance of GQA and MLA?

### Answers

1. Softmax normalizes across the whole row, so it needs the row **maximum** (for numerical stability — subtracting it prevents `exp` overflow) and the row **sum** of exponentials (the denominator). Neither is known until every column has been seen, so a naive tiled implementation can't emit correct partial results.
2. `corr = exp(m_old − m_new)`. Everything accumulated so far was computed as `exp(s − m_old)`; rebasing to the new max requires multiplying by `exp(m_old − m_new)`, and since that factor is identical for every term already in `acc` and `l`, one scalar multiply on each running quantity re-bases the entire history — no need to revisit old tiles.
3. (a) The **backward pass recomputes** the attention probabilities instead of reading stored ones; (b) modest redundant reads of `K`/`V` tiles across `Q` blocks. It's still faster because attention is memory-bound: the ~43× reduction in HBM traffic at `S = 8192` dwarfs a small arithmetic increase, and on a machine balanced at ~296 FLOP/byte, FLOPs are the cheap resource.
4. Because it removes the quality question entirely, so adoption requires no evaluation, no risk, and no per-model validation — a drop-in replacement. Approximate methods must be re-validated for every model and task and can fail unpredictably; exactness let FlashAttention become infrastructure. It also means it competes with approximations on speed *while dominating them on quality*, which is what collapsed that research direction.
5. v2 improves parallelism and work partitioning — notably parallelizing across the **sequence dimension** rather than only batch × heads, plus moving rescaling out of the inner loop. It matters for long context because that regime has *small batch and few heads per device but huge `S`*: parallelizing only over batch × heads leaves most of the GPU idle, so sequence-dimension parallelism is what keeps utilization high exactly where long-context training lives.
6. By solving the other two attention costs. Before, attention's `O(S²)` activation memory and traffic were the visible bottleneck; once those were gone, the remaining scaling obstacle at long context was the **KV cache** — which FlashAttention does nothing about. Fixing one bottleneck promotes the next, and GQA/MLA are the response to the newly-exposed one.

## Exercise

Implement online softmax attention yourself — this is the single highest-value exercise in Part 7. (a) Write `online_attention(q, K, V, tile)` maintaining `(m, l, acc)` exactly as above, in float64, and verify against `torch.softmax(K @ q, 0) @ V` for tile sizes `{1, 7, 64, S}` — you should get ~1e-15 everywhere, including tiles that don't divide `S`. (b) Deliberately break it two ways: drop the `corr` factor, and use a per-tile max without tracking the global max; report the resulting error and explain which numerical failure each produces. (c) Extend to a full `(S, D_h)` query block with a causal mask and count how many `(Q_block, K_block)` pairs you can skip entirely at `S = 1024`, block 128 — compare to `S²/2`. (d) Compute predicted HBM traffic for naive vs. tiled at `S ∈ {2048, 8192, 32768}` and check the ratio grows linearly in `S`.
