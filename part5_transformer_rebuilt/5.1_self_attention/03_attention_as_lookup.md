# Attention as Soft Lookup / Kernel Smoothing

File [02](02_scaled_dot_product_attention.md) gave the operation: `softmax(QK^T/√d_k) V`. This file gives the three mental models that make it *click* and, more usefully, that tell you what attention can and can't do. All three describe the same arithmetic; pick whichever unlocks a given problem.

**Convention:** row-vector (`Y = X W`), repo default. Dims from [NOTATION.md](../../NOTATION.md).

## Model A — a soft, differentiable dictionary lookup

Start with a hard Python dict. You have keys `k_j` with associated values `v_j`, and a query `q`. A hard lookup returns the value whose key best matches:

```
hard:   out = value[ argmax_j (q · k_j) ]        pick the single best-matching entry
soft:   out = Σ_j softmax_j(q · k_j / √d_k) · v_j   blend all entries by match strength
```

Two upgrades over an ordinary dict:

- **Content-based addressing.** A dict matches on *exact key equality*. Attention matches on *learned similarity* (`q·k`) — the query retrieves entries that are *close* in the learned key space, not identical. This is soft, associative, fuzzy retrieval.
- **Differentiable.** `argmax` has zero gradient almost everywhere — you can't learn through it. Softmax is a smooth relaxation of `argmax` (temperature `√d_k`, file [02](02_scaled_dot_product_attention.md)): as scores sharpen it approaches the hard lookup, but everywhere it has a usable gradient. So the *keys, values, and query projections are all learnable*. Attention is "the dictionary you can backprop through."

This is why attention replaced the RNN's fixed hidden state for context: instead of compressing all history into one vector and hoping the relevant bit survives, attention keeps every past token as a `(key, value)` entry and *retrieves* the relevant ones on demand. It's random-access memory over the sequence.

## Model B — kernel smoothing / Nadaraya-Watson regression

If you've seen non-parametric regression, this is the exact same object. Nadaraya-Watson (1964) estimates a function at query point `q` as a kernel-weighted average of observed values:

```
f̂(q) = Σ_j  κ(q, k_j) v_j  /  Σ_j κ(q, k_j)
```

Attention **is** Nadaraya-Watson with an exponential (softmax) kernel:

```
κ(q, k) = exp( q · k / √d_k )       →   out = Σ_j κ(q,k_j) v_j / Σ_j κ(q,k_j)
```

The `Σ κ` denominator is precisely softmax's normalizer. So attention is kernel regression where the kernel is learned (via `W_Q, W_K`) rather than fixed (a Gaussian bump at a chosen bandwidth). Consequences you get for free from this view:

- **Bandwidth = temperature.** `√d_k` is the kernel bandwidth: large → smooth/flat averaging, small → spiky/local. Same knob as file [02](02_scaled_dot_product_attention.md)'s temperature.
- **It's a weighted average, so it interpolates, never extrapolates** — the convex-hull point from file [02](02_scaled_dot_product_attention.md), restated in regression language.
- Attention having no learnable weights *in the averaging step itself* (only in the projections that form `q, k, v`) is exactly why non-parametric regression has no fit parameters — the "fitting" is choosing the feature map.

## Model C — information routing on the residual stream

The systems view, and the one that matters most for understanding the whole Transformer. In a decoder block there are exactly two mixing operations (file 5.2 [01](../5.2_the_full_block/01_assembling_the_block.md)):

| Operation | Mixes across... | Independent across... |
|---|---|---|
| **Attention** | **positions** (tokens talk to each other) | channels handled via heads |
| **FFN / SwiGLU** | **channels** (features within a token) | **positions** (per-token, no token talks to another) |

So **attention is the *only* token-mixing operation in the entire model.** The FFN is applied to each position independently — it can transform a token's own content but can never move information between tokens. Every cross-token dependency — agreement, coreference, copying, in-context retrieval — has to route through attention. When people call attention the "communication" step and the FFN the "computation/memory" step, this table is what they mean.

In residual-stream language (Part 3.1, [residual stream](../../part1_math_foundations/1.1_linear_algebra/supplementary/06_residual_stream.md)): each token's stream is a bus; attention lets a token **read from other tokens' buses** and **write the retrieved blend onto its own** (via `W_O`, file [01](01_qkv_projections.md)). It's a routing fabric over the sequence, and `L` stacked layers give `L` rounds of routing — enough to compose multi-hop dependencies ("the pronoun → its antecedent → *that* noun's attribute").

## What these views jointly explain

Two properties fall straight out, and they set up the next two files.

- **Generalizes across positions.** Because attention is content-based (Models A/B), the same head applies the *same* retrieval rule at every position — "attend to the previous token," "attend to the matching open-bracket" — regardless of *where* in the sequence that is. It's a position-agnostic rule, like a dict lookup that doesn't care about insertion order. This is what lets a head learned on short sequences fire correctly on long ones. (Multi-head specialization: file [04](04_multi_head_attention.md).)
- **Permutation-equivariant → position must be injected.** The flip side. Nothing in `softmax(QK^T/√d_k)V` references *order*: permute the input tokens and the outputs permute identically (`q·k` depends only on the vectors, not their indices). Attention sees a **bag of vectors**, not a sequence. That is a problem — "dog bites man" and "man bites dog" would be identical — so order has to be added back. That injection is the entire subject of [5.3](../5.3_positional_information/04_rope.md) (RoPE is the modern answer). Hold this thought: attention's position-blindness is not a bug to patch inside attention, it's a *design consequence* of content-based routing.

## Self-check

1. A hard dictionary returns `value[argmax match]`. What two things does attention change about that, and why is each necessary for it to sit inside a trained network?
2. In the Nadaraya-Watson view, what plays the role of the kernel, and what plays the role of the kernel bandwidth? What does shrinking the bandwidth do to the output?
3. The FFN has far more parameters than attention (≈80% of each block, [ARCHITECTURE.md](../../ARCHITECTURE.md)). Yet removing attention breaks the model in a way removing an FFN doesn't. What can attention do that the FFN structurally cannot?

### Answers

1. (a) **Soft instead of hard**: it blends *all* entries by softmax-weighted match strength rather than returning one — needed because `argmax` has no usable gradient, so a hard lookup couldn't be trained through. (b) **Content-based / learned similarity instead of exact-key equality**: it matches on `q·k` in a *learned* space (via `W_Q, W_K`), so retrieval is fuzzy and the addressing scheme itself is trained. Both together make it a differentiable, learnable associative memory — a dict you can backprop into.
2. The **kernel** is `κ(q,k) = exp(q·k/√d_k)` (the un-normalized softmax weight); the **bandwidth** is the temperature `√d_k`. Shrinking the bandwidth (smaller effective temperature, e.g. after the model learns to align `q,k` and inflate their norms) sharpens the kernel toward a spike — the weighted average concentrates on the single best-matching value, approaching the hard-lookup / argmax limit. Widening it flattens toward a uniform average.
3. Attention is the **only token-mixing operation** — it moves information *between positions*. The FFN acts on each position independently and can never transfer information from one token to another. So any cross-token dependency (coreference, copying, in-context recall, agreement) *must* go through attention; no amount of FFN capacity can substitute, because the FFN literally never sees more than one token's vector at a time.

## Exercise

In a notebook, build a tiny "associative memory" demo, no training. Create 6 keys `k_j` as random unit vectors in `R^8` and 6 values `v_j` as the one-hot rows of `I_6` (so you can read off *which* entry got retrieved). (1) Pick a query equal to `k_3` plus small noise; compute `softmax(q·k_j/√8)` and confirm the output is ≈ the one-hot for entry 3 — a soft lookup landing on the right key. (2) Sweep the bandwidth by scaling `q` by factors `{0.1, 1, 10}` and watch the retrieval go from near-uniform (wide kernel) to near-one-hot (narrow kernel) — you've reproduced the Nadaraya-Watson bandwidth knob. (3) Shuffle the order of the `(k_j, v_j)` pairs and confirm the retrieved value for the same query is unchanged up to the shuffle — a hands-on demonstration of permutation equivariance, and hence of why [5.3](../5.3_positional_information/04_rope.md) exists.
