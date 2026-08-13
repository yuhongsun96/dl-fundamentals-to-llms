# Supplementary: T5's Relative Position Bias in Code

Companion to [`../03_relative_positions.md`](../03_relative_positions.md). The primary file states the mechanism:

```
score(i, j) = (q_i · k_j) / √D_h  +  r[ bucket(i − j), head ]
```

This file answers the mechanical questions that raises: **how is a lookup table differentiable, and how does the gradient actually reach it?** Every code block here is faithful to the HuggingFace implementation, and the bucketing function below is verified to match it exactly.

## The short answer

A table is differentiable **in its entries, not in its index**, and only the entries are parameters:

| Quantity | Differentiable? | Needed? |
|---|---|---|
| `bucket(i − j)` — *which* row | **No** — integer-valued, discrete jumps | **No.** `i` and `j` are fixed integers, never learned |
| `r[b, h]` — the value *in* that row | **Yes** — an ordinary float parameter | **Yes.** This is the whole learnable object |

You never differentiate which row you looked up; you differentiate the number that was in it. This is exactly `nn.Embedding`, where token IDs are discrete and the table still trains.

And `r` is not a "constant" — it's constant with respect to the *input* but variable with respect to *training*, the same status as the bias in `nn.Linear`. **T5's positional scheme is a bias term on the attention logits, indexed by relative distance.**

## The table is literally an nn.Embedding

Straight from `transformers/models/t5/modeling_t5.py`:

```python
self.relative_attention_bias = nn.Embedding(
    self.relative_attention_num_buckets,   # 32 by default
    self.n_heads,                          # one scalar per (bucket, head)
)
```

For T5-base (`H = 12`): **32 × 12 = 384 parameters** for the model's entire positional scheme, shared across every layer.

## Computing the bias

Also from the HF source, lightly annotated:

```python
def compute_bias(self, query_length, key_length, device=None):
    context_position = torch.arange(query_length, device=device)[:, None]   # (S, 1)
    memory_position  = torch.arange(key_length,   device=device)[None, :]   # (1, S)
    relative_position = memory_position - context_position                  # (S, S) ints

    relative_position_bucket = self._relative_position_bucket(
        relative_position,
        bidirectional=(not self.is_decoder),   # encoder sees both ways; decoder only back
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )                                                                       # (S, S) ints in [0, 32)

    values = self.relative_attention_bias(relative_position_bucket)         # (S, S, H)  <- the gather
    values = values.permute([2, 0, 1]).unsqueeze(0)                         # (1, H, S, S)
    return values
```

Note `relative_position` is built from `torch.arange` — plain integers, no `requires_grad`. The bucket tensor depends only on the sequence length, so it can be computed once and cached.

## The bucketing function

Verified to match `T5Attention._relative_position_bucket` **exactly** for distances −600…600, in both directional modes, at `(num_buckets, max_distance)` of `(32,128)`, `(32,512)`, and `(16,64)`.

```python
import math, torch

def relative_position_bucket(relative_position, bidirectional=True,
                             num_buckets=32, max_distance=128):
    """Map integer relative distances to bucket ids. Faithful to T5 / Mesh-TensorFlow.

    Half the buckets cover small distances exactly; the rest are logarithmically
    spaced out to max_distance. Everything beyond max_distance shares one bucket.
    """
    ret = 0
    if bidirectional:
        num_buckets //= 2                                    # split the budget by direction
        ret += (relative_position > 0).long() * num_buckets  # sign selects which half
        n = relative_position.abs()
    else:
        # causal: future positions all collapse to 0 (they get masked anyway)
        n = -torch.min(relative_position, torch.zeros_like(relative_position))

    max_exact = num_buckets // 2                             # exact buckets for n < max_exact
    is_small = n < max_exact

    # log-spaced for the rest, then clamped into the final bucket
    large = max_exact + (
        torch.log(n.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long()
    large = torch.min(large, torch.full_like(large, num_buckets - 1))

    return ret + torch.where(is_small, n, large)
```

### What the buckets actually do

Causal mode, `num_buckets=32`, `max_distance=128`:

| distance | 0 | −1 | −2 | −7 | −15 | −16 | −31 | −32 | −63 | −127 | −128 | −500 | −5000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bucket** | 0 | 1 | 2 | 7 | 15 | 16 | 21 | **21** | 26 | 31 | **31** | **31** | **31** |

Read off the two design decisions:

- **Distances 0–15 get their own bucket each** — full resolution where it matters most (previous token, two back, same clause).
- **Beyond that, buckets widen logarithmically, and everything from ~127 on collapses into bucket 31.** Distances −31 and −32 already share a bucket; −128, −500, and −5000 are *indistinguishable* to the model.

The cost is real: T5 cannot tell distance 200 from 2000. The benefit is that there's no per-position parameter to run out of, so the scheme degrades gracefully past training length instead of hard-failing the way a learned absolute table does ([../02](../02_sinusoidal_and_learned_absolute.md)).

## The injection point, and the gradient

A minimal but faithful decoder-style attention. The bias is added **pre-softmax**, which is what puts the table inside the autograd graph:

```python
import torch, torch.nn as nn

S, H, Dh, NB = 16, 4, 8, 32
table = nn.Embedding(NB, H); nn.init.normal_(table.weight, std=0.02)
q = torch.randn(1, H, S, Dh); k = torch.randn(1, H, S, Dh); v = torch.randn(1, H, S, Dh)

pos    = torch.arange(S)
rel    = pos[None, :] - pos[:, None]                        # memory - query
bucket = relative_position_bucket(rel, bidirectional=False, num_buckets=NB)
causal = torch.tril(torch.ones(S, S, dtype=torch.bool))

bias   = table(bucket).permute(2, 0, 1).unsqueeze(0)        # (1, H, S, S)
scores = q @ k.transpose(-1, -2) / Dh**0.5 + bias           # <-- the entire injection
scores = scores.masked_fill(~causal, float("-inf"))
loss   = (scores.softmax(-1) @ v).pow(2).mean()
loss.backward()

print(table.weight.grad.shape)     # torch.Size([32, 4]) == (num_buckets, num_heads)
```

Backprop walks `loss → out → attn → softmax → scores → bias → table.weight`. Nothing custom is needed.

### Why the gradient is the simplest one possible

`r` enters **additively**, so `∂score(i,j) / ∂r[b,h] = 1`. Therefore:

```
∂L/∂r[b,h]  =  Σ  ∂L/∂score(i,j)      over every unmasked (i,j) with bucket(i−j) = b
```

A pure **scatter-add**: each scalar collects the logit-gradient of every position pair that used it. Verified against a hand-computed sum:

```
dL/dr == scatter-add of dL/dscore over unmasked pairs:  True  (max diff 9.3e-10)
unmasked pairs per bucket (S=16):  [16, 15, 14, 13, 12, 11, 10, 9, ...]
grad is exactly zero for unreachable buckets:            True
```

Two things worth reading off that output:

- **Massive weight sharing is why 384 parameters train fast.** At `S = 512` there are ~130,000 unmasked `(i,j)` pairs per head per layer funnelling into 32 buckets — thousands of gradient contributions per scalar per example, multiplied again because the table is shared across all layers.
- **Unreachable buckets get exactly zero gradient.** At `S = 16` only 16 of 32 buckets are reachable, so the long-distance entries are untouched. Their values are whatever initialization left, and they only start learning once training sees sequences long enough to reach them — a real thing to keep in mind when fine-tuning on short sequences and then deploying on long ones.

## Sharing across layers

Only the first block constructs the table (`has_relative_attention_bias=True`); every later block receives the already-computed `(1, H, S, S)` tensor as `position_bias` and just adds it. So:

- **Parameters:** one table for the whole stack, 384 floats for T5-base.
- **Compute:** the gather and bucketing happen once per forward pass, not per layer.
- **Gradient:** accumulates from every layer into the same 384 scalars.

The trade-off is that all layers are forced to share one distance-preference profile per head. Later designs (ALiBi, RoPE) drop the learned table entirely, and are compared in [../03](../03_relative_positions.md).

## Self-check

1. `bucket(i − j)` is a step function of `i − j`, so it has zero derivative almost everywhere and is undefined at the jumps. Why doesn't that break training?
2. Why is `∂L/∂r[b,h]` a *sum* rather than a single term?
3. At `S = 16` with 32 buckets, half the table receives exactly zero gradient. Why, and when does that matter in practice?
4. What can T5 *not* learn about distance, no matter how long you train it?
5. The table is shared across all layers. Name one advantage and one cost.

### Answers

1. Because nothing needs the derivative with respect to the index. `i` and `j` are integer positions supplied by the data loader, not learned quantities — there is no parameter upstream of `bucket(i − j)` for a gradient to flow to. The index only *selects* which parameter participates; the parameter itself is a plain float with a well-defined derivative. Identical to how `nn.Embedding` trains despite token IDs being discrete.
2. Because one scalar is reused by many position pairs. Every `(i, j)` whose relative distance falls in bucket `b` adds the same `r[b,h]` to its logit, so `r[b,h]` has many paths to the loss, and the multivariate chain rule sums over all of them. That's the same accumulation that makes `.grad` add up rather than overwrite, and it's why weight sharing produces large, low-variance gradients.
3. Buckets 16–31 encode distances beyond ~16 tokens, and a 16-token sequence never produces such a pair — so those entries appear in no forward computation and receive exactly zero gradient (verified above). It matters when your training distribution is shorter than your inference distribution: the long-distance biases stay at their initialization values, and the model will apply untrained parameters to distances it never practiced on. Train on sequences that actually exercise the buckets you intend to use.
4. **Where the bucket boundaries are.** The bucketing function is hand-designed and frozen — only the 32 values are learned, not the partition. So T5 can never distinguish distance 200 from 2000 (both bucket 31), nor distance 31 from 32 (both bucket 21), regardless of training. It also can't reallocate resolution — e.g. decide it needs fine-grained buckets at distance 50 — because that would require changing the function, not the parameters.
5. **Advantage:** parameter and compute efficiency — 384 floats total, computed once per forward pass, and every layer's gradient trains the same scalars, so they see enormous effective batch size. **Cost:** every layer is forced into the same per-head distance profile, even though early layers plausibly want local attention and later layers longer-range. A per-layer table would cost `L ×` more parameters (still trivial) but was found unnecessary; the constraint is nonetheless a real loss of expressiveness.

## Exercise

1. Copy `relative_position_bucket` above and verify it against `T5Attention._relative_position_bucket` for distances −1000…1000 in both directional modes. Then plot bucket id against distance and identify the exact-to-logarithmic transition.
2. Run the gradient demo and confirm `table.weight.grad` equals the scatter-add of `scores.grad` over unmasked pairs of each bucket. Then remove the causal mask and explain why bucket 0's gradient changes so much (hint: check where future distances map in causal mode).
3. Sweep `S ∈ {8, 32, 128, 512}` and report how many of the 32 buckets receive nonzero gradient at each. At what sequence length is the table fully exercised?
4. Set `num_buckets=8` and re-run. Inspect which distances now collide, and predict which attention behaviors from the primary file's list would become impossible to express.
