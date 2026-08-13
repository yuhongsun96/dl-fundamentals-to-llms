# Bias Terms — and Why Modern LLMs Drop Them

## What a bias does

In `y = Wx + b`, the bias `b ∈ R^(d_out)` is added to the projection. Geometrically: `W x` is a linear map (passes through the origin); `+ b` shifts the output by a fixed vector, making the map **affine** rather than purely linear.

Why you might want it:

- **Without `b`**, the layer can only produce outputs of the form `Wx` — outputs proportional to `x`. The output is *always* zero when `x = 0`, regardless of what `W` is.
- **With `b`**, the layer can produce a nonzero output even for zero input. The nonlinearity downstream can fire on inputs around zero.
- For a unit with ReLU, `b` shifts the kink. Without it, the kink is at the origin in input space and you have less control over when the unit activates.

In a 1-layer logistic regression, dropping bias is clearly bad: the decision boundary is forced through the origin. In a deep network, things are subtler.

## The classical default

For most of pre-2020 DL, you used bias on every linear layer by default. PyTorch's `nn.Linear` has `bias=True` as the default; same for `Conv2d`. The cost is `d_out` parameters per layer — small relative to `W`'s `d_in · d_out` — and the benefit was assumed to be unambiguous.

## Why modern LLMs drop them

Several large-model papers from 2020 onward observed that biases are essentially free to remove without quality loss, and often help stability. The trend started with PaLM, was confirmed by LLaMA, and is now standard:

- **PaLM (2022)**: dropped bias in all dense layers and LayerNorms. "We did not observe any quality degradation."
- **LLaMA 1, 2, 3**: no bias anywhere except in some token embedding / unembedding edge cases.
- **Mistral, Qwen, Gemma**: same — no bias.
- **GPT-2/3 (2019/2020)**: still had biases. The change happened across the architecture cleanup of the early 2020s.

### The reasons, ordered by how much they actually matter

**1. LayerNorm/RMSNorm absorbs the bias function.**

The most important reason. Modern Transformers apply LayerNorm or RMSNorm *before* each linear layer (pre-norm). A LayerNorm includes a learnable shift `β`. An RMSNorm doesn't, but the linear layer that follows can express any shift it needs because the normalized input is no longer mean-zero in any meaningful per-coordinate sense.

More precisely: if `x_norm = (x - μ)/σ · γ + β` (LayerNorm), then `W (x_norm) + b = W · γ · (x - μ)/σ + (W β + b)`. The `W β + b` term is a learnable affine shift — and `W β` already provides one. Adding a separate `b` is redundant.

**2. Optimization stability.**

Biases are unaffected by weight decay (you decay `W`, not `b`, conventionally). They can drift unboundedly. In long training runs at scale this drift correlates with loss spikes — biases catch noise and amplify it. Removing them removes a class of unstable parameters.

**3. Hardware efficiency, marginal but nonzero.**

A `Wx` matmul is one fused kernel call. `Wx + b` adds an elementwise add. With bias, the matmul and the add are typically fused; without bias the kernel is one operation simpler. Memory savings: trivial. Compute savings: ~1% of the matmul. Not the reason, but a tiebreaker if everything else is even.

**4. KV cache and quantization.**

In QKV projections specifically, having no bias means the projections are pure linear functions, which is easier to quantize precisely (no asymmetric-quantization considerations needed). Same applies to FFN matrices under post-training quantization.

## Where bias still lives

Not every bias is dead. A few survive:

- **Embedding tables sometimes have an effective bias** in the form of a learned token-frequency offset (e.g. ALiBi attention bias is a different thing — that's a bias on the *attention scores*, see Part 5.3).
- **The output unembedding** sometimes has a bias to model the unigram log-frequency of the vocabulary (rare in modern LLMs; usually folded into the embedding weights).
- **Conv layers in vision transformers** sometimes keep bias for the patch-embedding layer. Convention varies.
- **In some attention implementations**, the output projection `W_O` keeps a bias. Llama doesn't; OPT does; it's a style choice with no measured impact at scale.

## A subtler bias removal: QKV and attention

For the attention block specifically, removing bias from Q/K/V projections has a small additional benefit: it makes the attention computation **input-shift-equivariant** in a clean way. Adding a constant to every input token's representation doesn't change attention scores (because Q and K are both shifted by the same vector, and their dot product depends only on the projection). With bias, the shift would alter the scores. This invariance is usually not load-bearing, but it's clean.

## When to keep bias

If you're training a small model, fine-tuning, or working outside of the pre-normed Transformer template:

- **Definitely keep it** in the first linear layer after raw input (no preceding norm to provide shift capacity).
- **Definitely keep it** for the output head of a regression-style task where the target has a nonzero mean.
- **Definitely keep it** for any layer not preceded by a learnable affine norm.
- **Drop it** anywhere the layer is preceded by a normalization layer that already provides a learnable shift (LayerNorm with `β`) or an effective one (RMSNorm followed by a Linear).

The rule of thumb: ask "does the input to this layer already have a learnable affine offset upstream?" If yes, the bias is redundant.

## Self-check

1. Without bias, what's the only output a linear layer can produce when its input is the zero vector? Why does this not break Transformers in practice?
2. Why does weight decay typically exclude bias parameters? What would happen if you decayed them?
3. RMSNorm has no learnable shift `β` (unlike LayerNorm). Why don't modern LLMs add bias back to layers that follow RMSNorm?

### Answers

1. The only output is the zero vector: `y = W · 0 = 0`. In Transformers this never happens in practice because the residual stream never approaches the zero vector — token embeddings are nonzero, and each layer adds nonzero residuals. Pre-norm Transformers also normalize before each linear layer, and a normalized input is never identically zero (RMSNorm of zero is undefined; LayerNorm of zero is zero only if every coordinate is the mean, which it isn't in practice). The "zero input" pathology is a thought-experiment, not a real failure mode.
2. Biases shift the activation distribution and historically have small magnitudes; penalizing them would push them to zero with no clear benefit and potentially distort the learned representation. More importantly: weight decay's regularization story (e.g., shrinking weight magnitudes to reduce model complexity) applies to *multiplicative* parameters — biases are additive offsets and don't increase the function's Lipschitz constant or VC dimension. Decaying them confuses the regularizer's role. Modern implementations (`AdamW` typically with parameter groups) decay only `W`-shaped tensors and skip biases, embeddings, and norm parameters.
3. Because the linear layer that follows RMSNorm can produce any shift it wants implicitly. RMSNorm gives you `x · γ / RMS(x)` — magnitude-normalized but still in whatever direction the input had. The downstream `W` then projects this and produces some output `W x_norm`. To add a constant per-output-dimension shift, you'd need a `+ b`, but here's the trick: in pre-norm Transformers, the *output of every block is added to the residual stream*. The residual stream carries arbitrary offsets from earlier layers (embeddings, prior blocks). Any constant a `+b` would have provided is already present in the residual stream from upstream. Adding `b` would just double-count.

## Exercise

Build a 4-layer MLP (`R^10 → 64 → 64 → 64 → 10`) two ways: one with `bias=True` on every linear, one with `bias=False` and a LayerNorm *before* every linear layer. Train both on a random regression task. Compare:
- Final loss (should be comparable).
- Parameter count (no-bias version has fewer params).
- Per-layer activation distributions (should look similar).

This is the empirical case for the modern Transformer convention in miniature.
