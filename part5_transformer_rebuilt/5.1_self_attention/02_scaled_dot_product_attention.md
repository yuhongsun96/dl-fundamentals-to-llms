# Scaled Dot-Product Attention

Given `Q, K, V` from file [01](01_qkv_projections.md), the attention operation itself is one line:

```
Attention(Q, K, V) = softmax( Q K^T / √d_k ) V
```

Everything interesting is in the shapes, the `√d_k`, and the direction the softmax runs. This file walks all three, contrasts the operation with the Bahdanau/Luong attention you know from Part 4, and flags the two things later parts optimize (causal mask → file [05](05_causal_and_bidirectional_masking.md); the `(B,H,S,S)` memory → FlashAttention, Part 7.2).

**Convention:** row-vector (`Y = X W`), repo default. Dims from [NOTATION.md](../../NOTATION.md); 8B config `H=32, D_h=d_k=128, S ≤ 8192` from [ARCHITECTURE.md](../../ARCHITECTURE.md).

## Walk the shapes

Batched, all `H` heads at once. `Q, K, V` are `(B, H, S, D_h)` after projecting and reshaping (file [01](01_qkv_projections.md)):

```
Q          (B, H, S, D_h)
K          (B, H, S, D_h)
K^T        (B, H, D_h, S)          transpose last two axes
scores  =  Q K^T / √d_k            (B, H, S, S)   ← one S×S score matrix per head
A       =  softmax(scores, axis=-1)(B, H, S, S)   ← softmax over the KEY axis
out     =  A V                     (B, H, S, D_h) ← weighted blend of value rows
```

Read the `(B, H, S, S)` score tensor as: for each batch item and head, an `S × S` matrix where **row `i`** holds token `i`'s affinity to every token `j`. `scores[b,h,i,j] = q_i · k_j / √d_k`. That row is what softmax turns into a distribution over "who token `i` reads from."

For the 8B config at full context, one head's score matrix is `8192 × 8192 ≈ 67M` floats; times `H=32` heads times batch — this is the quadratic-in-`S` blowup (last section).

## The √d_k scaling — state the result, don't re-derive

The scale factor `1/√d_k` is not a tuning knob; it is a **variance calibration** derived in full in [1.2/03](../../part1_math_foundations/1.2_calculus_probability/03_softmax_logsumexp.md). The result, stated:

> At init, with `q, k` components ~unit-scale, `q · k = Σ_{i=1}^{d_k} q_i k_i` is a sum of `d_k` roughly-independent mean-zero terms, so it has **mean 0 and variance `d_k`** — standard deviation `√d_k`. Dividing by `√d_k` makes the scores have **variance 1**, independent of head size.

For `d_k = 128`, unscaled scores have std `≈ 11`, and the max over `S=8192` keys scales like `√(2 log S)·√d_k ≈ ±50`. Feed those into softmax and you are deep in the saturated regime.

### What breaks without it

Softmax saturation, and then dead gradients — the mechanism is in [1.2/03](../../part1_math_foundations/1.2_calculus_probability/03_softmax_logsumexp.md)'s saturation section:

- Scores of `±50` mean softmax outputs `~1` at the argmax and `~0` everywhere else — a near one-hot, before the model has learned anything.
- The softmax Jacobian `diag(s) − s s^T` vanishes as `s → one-hot`. So the gradient into `Q` and `K` is `~0`. The attention pattern is frozen at whatever random pattern init produced, and it **can't learn out of it**.
- In fp16, scores that large also risk `exp` overflow → NaN (again [1.2/03](../../part1_math_foundations/1.2_calculus_probability/03_softmax_logsumexp.md)).

`1/√d_k` is a fixed "temperature" of `√d_k` (the temperature framing is in [1.2/03](../../part1_math_foundations/1.2_calculus_probability/03_softmax_logsumexp.md)) that keeps init in the *soft* regime where gradients flow. Note it fixes variance **at init**; once the model learns to align `Q` and `K`, logits grow again — which is the separate instability QK-norm addresses (Part 5.1 aside in [ARCHITECTURE.md](../../ARCHITECTURE.md), Part 3.2 / 7).

Why `√d_k` and not, say, `d_k`? Std scales as `√d_k`, not `d_k`; dividing by the std is exactly the normalization that yields unit variance. Dividing by `d_k` would over-shrink and flatten scores toward uniform.

## Softmax over keys = a convex combination of values

The softmax runs along the **last axis (keys)**, so **each row of `A` sums to 1** and is non-negative. Therefore output row `i` is:

```
out_i = Σ_j A[i,j] · V[j]      with  A[i,j] ≥ 0,  Σ_j A[i,j] = 1
```

a **convex combination** (weighted average) of the value vectors. Token `i`'s new content is a blend of everyone's values, weighted by how well its query matched their keys. This is the single most important structural fact:

- The output lives in the **convex hull** of the value rows — attention can only *interpolate* among values present, never extrapolate outside them.
- "Attending to nothing" isn't an option — the weights must sum to 1, so every query commits its full unit of attention somewhere. (This is why models learn **attention sinks** / BOS dumping grounds — a place to park attention when a query has nothing relevant to read. Forward: file [04](04_multi_head_attention.md).)
- Because it's a weighted average of `H`-many `D_h`-vectors, one head's write into the stream is low-rank (rank ≤ `D_h`) — the multi-head expressivity argument in file [04](04_multi_head_attention.md).

## Contrast with the RNN/seq2seq attention you know (Part 4)

Bahdanau (2015) / Luong attention bolted an alignment module onto an encoder-decoder RNN. Same *idea* — a softmax-weighted read over source states — but structurally different:

| | Bahdanau/Luong (Part 4.2) | Transformer scaled dot-product |
|---|---|---|
| Score function | learned MLP `v^T tanh(W_a[h_dec; h_enc])` (additive) | pure dot product `q·k` (multiplicative) |
| Who queries whom | decoder state queries encoder states (**cross**) | every token queries every token (**self**) |
| Backbone | attached to a **sequential** RNN | attention **is** the whole layer — no recurrence |
| Scaling | none (small dims, additive score) | `/√d_k` (large dims, dot product would saturate) |
| Heads | single alignment | `H` parallel heads |
| Cost | `O(S_dec · S_enc)` on top of `O(S)` recurrence | `O(S²)` and that's the whole layer |

The conceptual leap of "Attention Is All You Need": drop the RNN entirely and let *self*-attention (queries and keys from the same sequence) do all the token mixing, stacked `L` times. The additive score became a plain dot product because it's cheaper as one big matmul — and the `√d_k` scaling exists precisely *because* the dot product over large `d_k` saturates in a way the small additive score never did.

## The two things later parts optimize

Flagging forward-pointers so you don't mistake this file's version for the final one:

- **Causal mask.** For decoder LMs a mask is added to `scores` *before* softmax so a token can't attend to the future. It slots in exactly at the `scores` line: `scores += mask` where `mask[i,j] = −∞` for `j > i`. Full treatment in file [05](05_causal_and_bidirectional_masking.md).
- **The `(B,H,S,S)` memory.** Materializing the full score/probability tensor is `O(S²)` memory per head — the dominant training-activation cost and the thing **FlashAttention** (Part 7.2) avoids by never writing the full matrix to HBM (it tiles and recomputes, using the online-softmax running-max from [1.2/03](../../part1_math_foundations/1.2_calculus_probability/03_softmax_logsumexp.md)). This file computes *exact, full* attention; FlashAttention computes the *same numbers* more cheaply.

## Self-check

1. `scores` has shape `(B, H, S, S)`. Along which axis does softmax run, and what does the "sums to 1" property guarantee about each output vector `out_i`?
2. You remove the `/√d_k`. Trace what happens at initialization: to the scores, to the softmax output, to the gradient into `Q` and `K`.
3. Why is the score a plain dot product here when the Bahdanau attention you remember used a learned additive MLP? What made the switch both possible and necessary?

### Answers

1. Softmax runs along the **last axis (the key axis, `j`)**, so each **row** of `A` is a distribution. "Sums to 1" means `out_i = Σ_j A[i,j] V[j]` is a **convex combination** of the value rows — it lies in their convex hull, so attention interpolates among existing values and cannot produce content outside their span. It also means every query must spend its full unit of attention weight somewhere (motivating attention sinks).
2. Scores keep variance `d_k` (std `≈ √128 ≈ 11`), so typical logits are `±11` and the max over `S` keys reaches `±50`. Softmax on `±50`-scale logits is a near one-hot — saturated. The softmax Jacobian `diag(s) − s s^T` → 0 there, so the backward signal into `Q, K` vanishes; the (random) attention pattern from init is frozen and can't be learned away. In fp16 the large `exp` can also overflow to NaN. Net: training stalls or blows up.
3. **Possible**: with `Q, K` already being learned projections of the stream (file [01](01_qkv_projections.md)), a dot product is a perfectly good learnable similarity — and it's one big matmul, far cheaper than an additive MLP evaluated per pair. **Necessary**: the dot product over large `d_k` has variance `d_k`, so it saturates softmax; the small additive score never got large enough to need scaling. The move to dot-product attention is *why* the `√d_k` correction had to be invented.

## Exercise

Implement `attention(Q, K, V)` for a single head on `(S, D_h) = (5, 16)` random tensors. (1) Compute it *with* and *without* the `/√d_k` and print the max attention weight per row — the unscaled version should already be near-one-hot at random init. (2) Scale `Q` up by `10×` (simulating learned alignment) and watch the scaled version also start saturating — this previews why QK-norm exists. (3) Verify each row of `A` sums to 1 and that `out` rows lie in the convex hull of `V` rows (each `out_i` component is between the min and max of that component across `V`).
