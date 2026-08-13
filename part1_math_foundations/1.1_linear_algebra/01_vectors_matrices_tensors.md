# Vectors, Matrices, Tensors — Shapes and Broadcasting

## The mental model

Everything in DL is a tensor: an n-dimensional array with a shape. Scalars (0D), vectors (1D), matrices (2D), and higher-rank tensors (3D+) all obey the same rules.

In NLP, you will spend 90% of your life thinking about shapes like:

```
(batch, seq_len, d_model)
(batch, n_heads, seq_len, d_head)
(batch, seq_len, vocab_size)
```

If you can't predict the shape at every line of a forward pass, you don't understand the forward pass.

## Key conventions

- **Batch dimension is almost always first** (PyTorch default; JAX varies).
- **Row-major storage**: the last dim is contiguous in memory. Matters for performance.
- **Channels-last vs. channels-first** is a CNN concern — ignore it for NLP.

## Broadcasting

### Notation first: what do `B`, `S`, `D` mean?

These are the conventional one-letter names for the dimensions of an NLP activation tensor:

- `B` — **batch size**: how many independent sequences you're processing in parallel.
- `S` — **sequence length**: how many tokens in each sequence (also written `T` or `L` in some codebases).
- `D` — **model dimension** (a.k.a. `d_model`, hidden size, embedding dim): the size of the vector representing each token. Typical values: 768 (BERT-base), 4096 (Llama-7B), 12288 (GPT-3 175B).

So `(B, S, D)` is "a batch of `B` sequences, each `S` tokens long, each token a `D`-dim vector." This is *the* canonical shape of a hidden state inside a Transformer.

### Why the trailing comma in `(D,)`?

`(D,)` is Python tuple syntax for a **1-element tuple**. `(D)` without the comma is just `D` in parentheses — a scalar, not a tuple. So `(D,)` is the shape of a **1D tensor of length D**: it has exactly one dimension, no "omitted" second dim.

A `(D,)` tensor is a plain vector — length `D`, no batch, no sequence. You'll see it all over the place for things like bias vectors, per-feature scale/shift params, and running statistics.

### Why broadcasting `(D,)` across `(B, S, D)` is common

Because many operations apply **per-feature, identically across every token in every sequence**. The same `D`-length parameter vector is reused at every `(b, s)` position:

- **Bias terms**: `y = W @ x + b`. `b` has shape `(D,)` and gets added to every token's `D`-dim activation.
- **LayerNorm / RMSNorm params**: the learned `γ` (scale) and `β` (shift) are `(D,)` vectors applied pointwise to every token.
- **Positional bias offsets** (in some architectures).
- **Scaling by a learned per-feature gate** (e.g. in some MoE or gated-linear-unit variants).

If you had to explicitly tile `b` to `(B, S, D)` before adding, you'd waste `B × S` times the memory. Broadcasting is how the framework says "just read the same `(D,)` slice and add it wherever you need it" — no allocation, no copy.

### The broadcasting rules

When two tensors have different shapes, NumPy/PyTorch broadcast by:
1. Right-aligning the shapes.
2. Treating any size-1 or missing dim as repeatable.

```
(B, S, D) + (D,)        → OK, broadcasts to (B, S, D)      # bias/norm-param pattern
(B, S, D) + (S, D)      → OK                                # e.g. adding a per-position vector
(B, S, D) + (B, D)      → ERROR (middle dim mismatch)       # need to unsqueeze to (B, 1, D)
(B, S, D) + (B, 1, D)   → OK, the 1 broadcasts across S     # a per-sequence vector applied to every token
```

The rule for the first case: right-align `(B, S, D)` and `(D,)`. The missing left dims of `(D,)` are treated as `1`s, giving an effective `(1, 1, D)`, which then broadcasts to `(B, S, D)`.

**This is where bugs hide.** The dangerous case is when two shapes happen to be *compatible* under right-alignment but don't mean what you intended — the op runs and silently computes the wrong thing.

Canonical example — scaling per-sample activations by per-sample weights:

```
activations: (B, D)    # B=32, D=512
weights:     (B,)      # B=32, one scalar per sample
activations * weights  # you meant per-sample reweighting
```

Right-alignment of `(B,)` against `(B, D)` puts `B` where `D` is expected. If `B != D` (usually), you get a shape error — annoying but loud. If `B == D` by coincidence (e.g. B=512, D=512 in some debug config), it multiplies along the feature dim and trains through without complaint, producing nonsense. The fix: unsqueeze to `(B, 1)` so the `1` broadcasts over `D`.

Attention masks work the same way — you need to explicitly unsqueeze to the right shape. For scores of shape `(B, H, S, S)`, a padding mask of raw shape `(B, S)` must be reshaped to `(B, 1, 1, S)` so it broadcasts across heads (`H`) and queries (the third dim), masking by *key* position. Hand PyTorch the raw `(B, S)` and it will error on shape mismatch in the typical case — the *silent* bug only appears when your dims coincidentally line up.

## Reshape vs. view vs. permute

- `view` / `reshape`: change shape without moving data (must be contiguous).
- `permute` / `transpose`: swap dims; the result is **non-contiguous** until you `.contiguous()`.
- `expand`: broadcast without allocation; `repeat` allocates. Prefer `expand`.

Multi-head attention lives and dies by `(B, S, D) ↔ (B, S, H, D/H) ↔ (B, H, S, D/H)` gymnastics.

## Self-check

1. You have logits of shape `(B, S, V)` and labels `(B, S)`. What shape does cross-entropy expect, and how do you reshape to get there?
2. Why does `q @ k.transpose(-1, -2)` produce `(B, H, S, S)` given both are `(B, H, S, D/H)`?
3. What's the difference in memory layout between `.transpose(1, 2)` and `.reshape(B, H, S, D)` when going from `(B, S, H, D_h)`?

### Answers

1. CE wants `(N, V)` logits and `(N,)` labels. Reshape: `logits.reshape(-1, V)` → `(B*S, V)`, `labels.reshape(-1)` → `(B*S,)`. Equivalently `logits.flatten(0, 1)` and `labels.flatten()`. PyTorch's `F.cross_entropy` accepts 4D `(B, V, ...)` directly with class-dim as `dim=1`, but explicit flatten is the most transparent form.
2. Batched matmul: leading `(B, H)` are batch dims, last two are matrix dims. `(S, D_h) @ (D_h, S) → (S, S)` per `(b, h)` pair, all done in one fused kernel. Result: `(B, H, S, S)` — one S×S attention matrix per (batch, head).
3. `.transpose(1, 2)` on `(B, S, H, D_h)` → `(B, H, S, D_h)` by swapping strides only. Non-contiguous, zero memory move, free. `.reshape(B, H, S, D)` is impossible as a pure no-op: the reorder of S↔H plus the merge of `H, D_h` into `D` requires both data movement (the transpose) and a contiguous layout (for the merge). PyTorch will error or implicitly call `.contiguous()` (a copy). The standard idiom is `.transpose(1, 2).contiguous().view(B, H, S, D)` if you want the merged head form.

## Exercise

In a notebook, take a random `(2, 3, 4)` tensor. Reshape, permute, and broadcast it against various shapes until you can predict every output shape without running the code.

## Practice

Runnable self-quiz with 12 questions on tuple syntax, broadcasting, multi-head reshape, `view` vs `reshape` vs `permute`, mask broadcasting gotchas: [`supplementary/01_vectors_matrices_tensors_quiz.ipynb`](supplementary/01_vectors_matrices_tensors_quiz.ipynb).
