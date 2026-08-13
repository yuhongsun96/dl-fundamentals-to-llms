# Relative Positions

Absolute encodings ([file 02](02_sinusoidal_and_learned_absolute.md)) tag each token with its slot index. But language doesn't care whether a clause starts at token 4 or token 4004 — it cares that a verb is *three tokens after* its subject, that a closing bracket matches an opening one *some distance back*, that recent tokens matter more than far ones. What matters is the **offset `i − j`** between a query and a key, its magnitude and its sign. Relative encodings drop the absolute index entirely and encode that offset directly — and, crucially, they do it **inside attention**, biasing the score `q_i · k_j` by a function of `i − j`, so position never has to be carried up the [residual stream](../5.1_self_attention/01_qkv_projections.md).

**Convention:** row-vector (`Y = X W`). The attention score matrix is `(B, H, S, S)`; a relative scheme adds a term indexed by `(i, j)` to entry `[·, ·, i, j]`. Dimensions from [NOTATION.md](../../NOTATION.md); 8B anchor `H = 32`, `S` up to 8192 from [ARCHITECTURE.md](../../ARCHITECTURE.md).

The unifying idea for this whole file:

```
score(i, j) = (q_i · k_j) / √D_h  +  b(i − j)      # add a bias that depends on the offset
```

`b(·)` is a function of the relative distance. The two schemes below differ only in what `b` is: **learned** (T5) or a **fixed linear penalty** (ALiBi).

## T5 relative position bias (Raffel et al., 2019)

T5 adds a **learned scalar** to each attention logit, indexed by the (bucketed) relative distance and shared across all layers:

```
score(i, j) = (q_i · k_j) / √D_h  +  r[ bucket(i − j), head ]
```

- `r` is a small learned table: one scalar per `(relative-distance bucket, head)` pair. Per head, because different heads want different distance preferences (one tracks the previous token, another spans a clause).
- **No positional vector is added to the input at all** — the only positional signal is this additive bias on the scores. That's the key move away from file 02: position lives in attention, computed fresh at every layer, not carried in the stream. (In T5 the table is even *shared across layers*, so it's a tiny number of parameters — **384 floats** for all of T5-base.)

> **How a lookup table gets trained**: [supplementary/03_t5_relative_bias_code.md](supplementary/03_t5_relative_bias_code.md) walks the real implementation. Short version: `r` is literally an `nn.Embedding(num_buckets, n_heads)`, and a table is differentiable **in its entries, not its index** — `bucket(i − j)` is a discrete function of fixed integers, so no gradient is ever needed with respect to it. Since `r` enters the logit *additively*, `∂score/∂r = 1` and the gradient is a pure scatter-add: each scalar collects `∂L/∂score(i,j)` from every position pair that used it. The supplementary file has the verified bucketing function, the exact injection point, and the measured gradient identity.

### Why bucketing

You can't afford (and don't need) a distinct learned scalar for every possible offset up to `S = 8192`. Language needs **fine resolution up close and coarse resolution far away** — the difference between offset 1 and 2 is linguistically huge; the difference between offset 500 and 520 is negligible. T5 **buckets** the offset logarithmically: exact buckets for small distances (0, 1, 2, 3, ...), then geometrically widening buckets (4–7, 8–15, ...) out to a capped maximum. This gives sharp local resolution with a handful of parameters and lumps all far distances into a few coarse bins.

Its extrapolation is **limited**: offsets beyond the training range all fall into the final "everything far" bucket, so the model can *run* on longer sequences without crashing (unlike a learned absolute table's hard wall), but it can't distinguish among the long distances it never learned. Better than absolute, not the end of the story.

## ALiBi — Attention with Linear Biases (Press et al., 2021)

ALiBi throws out the learned table entirely. The bias is a **fixed linear penalty** in the distance, with a per-head slope:

```
score(i, j) = (q_i · k_j) / √D_h  −  m_head · (i − j)      # for a causal model, i ≥ j, so (i − j) ≥ 0
```

- **No parameters.** The slope `m_head` is a fixed constant per head, set by a geometric schedule (e.g. `m = 1/2, 1/4, 1/8, ...` across the `H` heads). Nothing is learned; nothing is added to the input.
- **A recency bias, baked in.** The penalty `−m·(i − j)` grows linearly with distance, so a query attends less to far-back keys and more to nearby ones — recency is hard-wired into the geometry rather than learned. Different heads get different slopes, so some heads are sharply local (steep slope) and others near-global (shallow slope).

### Why ALiBi's headline is extrapolation

Because the penalty is a *smooth, unbounded linear function* of distance, it is defined and well-behaved at *any* offset — including offsets far larger than anything seen in training. A query at position 20,000 attending to a key at 19,000 gets penalty `−m·1000`, computed by the same formula that handled `−m·100` during training. There is no phase to wrap, no bucket to overflow, no table to run off the end. **Train on 1024 tokens, run on 4096+, with the perplexity barely moving** — that was ALiBi's headline result and the demonstration that a well-chosen positional scheme can genuinely train-short-run-long.

## The two relative schemes, side by side

| | T5 bias | ALiBi |
|---|---|---|
| Bias `b(i − j)` | learned scalar per (bucket, head) | fixed `−m_head · (i − j)` |
| Parameters | small learned table (often shared across layers) | **zero** |
| Distance resolution | log-bucketed (fine near, coarse far) | continuous linear |
| Recency bias | learned (can be any shape) | hard-wired (monotone decreasing) |
| Extrapolation | limited (far offsets collapse to one bucket) | **strong** (formula defined at any distance) |
| Where it enters | added to attention logits | added to attention logits |

Both are the same recipe — *bias the score by a function of the offset* — differing in whether that function is learned and shaped (T5) or fixed and linear (ALiBi).

## Setting up RoPE: why "bias the score" wasn't the last word

T5 and ALiBi are **additive**: they leave the `q·k` dot product alone and *add* a separate positional term on top. That works, but it's a bolt-on — position and content are combined by summation, and the positional term is a scalar that can't interact with the *direction* of `q` and `k`, only shift their score up or down. ALiBi in particular bakes in a fixed monotone recency prior, which is a strong assumption (great for language modeling, less flexible when you want non-monotone distance preferences).

The scheme that won — **RoPE** ([file 04](04_rope.md)) — is the *multiplicative / geometric* synthesis. Instead of adding a distance bias to the score, it **rotates** `q` and `k` by an angle proportional to their positions, so that the dot product `q_i · k_j` *itself* comes out depending only on `i − j`. Same relative property T5 and ALiBi engineer for, but achieved inside the geometry of the dot product rather than tacked on afterward — parameter-free like ALiBi, but without hard-wiring a fixed recency shape, and letting position modulate the full `q`/`k` interaction rather than just offsetting its score. That's the subject of the next file.

## Self-check

1. Both T5 bias and ALiBi are "relative." State the single recipe they share, and the one thing that differs between them.
2. Why does ALiBi extrapolate to unseen lengths while a learned absolute table cannot, and while T5 only partly can?
3. T5 buckets relative distances logarithmically instead of giving every offset its own learned scalar. What linguistic fact justifies coarse resolution at long range, and what does the bucketing cost you?

### Answers

1. **Shared recipe:** add a term that depends only on the offset `i − j` to the attention logit — `score(i,j) = q_i·k_j/√D_h + b(i−j)` — with position entering *inside attention*, never added to the input or carried in the residual stream. **What differs:** `b` is a **learned, bucketed, per-head scalar table** in T5 versus a **fixed linear penalty `−m_head·(i−j)`** in ALiBi (zero parameters, hard-wired recency).
2. ALiBi's bias is a smooth linear function of distance that is *defined and sensible at any offset*, including offsets far past training — the same formula that computed `−m·100` in training computes `−m·5000` at inference, no wall, no wrap. A learned absolute table has a fixed number of rows and simply has no vector for positions past `max_S` (hard failure). T5 can *run* at longer lengths but all far offsets collapse into its final "everything distant" bucket, so it can't *distinguish* long distances it never trained on — extrapolation without resolution.
3. Linguistically, local distance is high-stakes (offset 1 vs. 2 is the difference between adjacent and skip-one dependencies) while long distance is low-stakes (offset 500 vs. 520 is functionally the same "far away"), so you want fine bins near zero and coarse bins far out. The cost is exactly that lost resolution at range: any two offsets that share a bucket get the *same* bias, so the model literally cannot tell them apart — fine for "both far" but a real limitation if a task needs precise long-range distance.

## Exercise

Implement both biases as a function that produces an `(S, S)` additive matrix for `S = 16`, single head. For ALiBi use slope `m = 1/4` and the causal convention (mask `i < j` to `−∞`, penalty `−m·(i−j)` for `i ≥ j`); visualize the matrix as a heatmap and confirm the penalty deepens linearly as you move away from the diagonal. For a crude T5 stand-in, log-bucket the offset `i − j` into ~6 buckets and assign each bucket an arbitrary learned-stand-in scalar; visualize and note the flat "staircase" plateaus where multiple far offsets share a bucket. Then, in one sentence each, predict what happens to each matrix if you extend to `S = 64` — which one keeps producing meaningful new distinctions and which one saturates.
