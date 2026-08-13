# Pre-norm vs. Post-norm

A small architectural choice with outsized consequences. Two equally reasonable-looking ways to insert a normalization layer into a Transformer block — but only one lets you stack 50+ layers without elaborate stabilization tricks. Pre-norm won; understanding why illuminates how residual streams actually work.

## The two configurations

A Transformer block has two sublayers (attention and FFN), each wrapped in a residual connection and a norm. Two places to put the norm:

### Post-norm (original "Attention is All You Need", 2017)

```
h_attn = LayerNorm(h + Attention(h))
h_out  = LayerNorm(h_attn + FFN(h_attn))
```

Norm is **after** the residual add. The output of each sublayer is normalized as a whole.

### Pre-norm (GPT-2 onwards, Llama, everything modern)

```
h_attn = h + Attention(LayerNorm(h))
h_out  = h_attn + FFN(LayerNorm(h_attn))
```

Norm is **before** the sublayer. The residual `h` flows through *unchanged*; only the sublayer input gets normalized.

That's the whole change — the position of `LayerNorm` in the formula. The downstream consequences are huge.

## Why pre-norm wins for deep stacks

### The gradient path argument

In post-norm, the backward path from the loss to the input includes a `LayerNorm` after every residual add. The LN's Jacobian, while bounded, introduces multiplicative factors at every layer. After `L` layers, gradients have been pushed through `L` LN Jacobians — they can amplify in some directions and shrink in others, and the accumulated effect at large `L` makes training unstable.

In pre-norm, the residual `h_l` flows backward through a **pure identity path** (the `h_l +` term in `h_{l+1} = h_l + f_l(LN(h_l))`). The gradient w.r.t. `h_l` from the residual side is exactly the upstream gradient, no Jacobian factor. The contribution from `f_l(LN(h_l))` adds to it. Over `L` layers, the gradient accumulates additively along the residual stream — no multiplicative compounding of LN Jacobians.

The forward-pass analog: in pre-norm, the residual stream `h_l` is the sum of contributions from layers 0 through `l-1`, undisturbed by any norm — a clean additive "highway." In post-norm, the residual stream is repeatedly normalized, erasing the magnitude information that earlier layers contributed.

### The empirical evidence

The 2019 "On Layer Normalization in the Transformer Architecture" paper (Xiong et al.) demonstrated:
- Post-norm Transformers at large depth require careful learning-rate warmup and are still unstable past ~24 layers.
- Pre-norm Transformers train smoothly to 100+ layers with much less warmup.

GPT-2 (2019) was the first major model to use pre-norm. Every successor (GPT-3, BERT-large-cased variants, T5, OPT, Llama, Mistral, ...) followed.

The original "Attention is All You Need" model was only 6 layers deep — at that scale, post-norm works fine and they didn't notice the depth limitation. The problem only surfaces past ~24 layers.

## What pre-norm trades away

Pre-norm isn't a strict improvement. The downside:

- **The residual stream's variance grows with depth.** In pre-norm, each block adds a contribution to the residual stream. Without explicit damping, the stream's variance at layer `L` is roughly `L × σ_block²`. This means the LayerNorm at the *final* output layer has to normalize against a much larger magnitude than the LN at layer 1 sees. The final logits depend on a stream that has had `L` accumulated contributions.
- **The block's input is normalized, but its output (which gets added to the stream) is not.** So while attention/FFN computations operate on unit-scale inputs, the residual stream they're added to drifts in magnitude as you go deeper.

Two consequences:
- The first reason to scale output projections by `1/√L` at init (file `02`): keeps residual-stream variance constant at depth.
- The reason for the **final LayerNorm before the output head** in pre-norm models: tames the accumulated residual stream right before the logit projection. Post-norm doesn't need this — its blocks already terminate in a norm.

## A third option: DeepNorm (2022)

Post-norm with a per-layer scaling trick to stabilize it at depth:
```
h_out = LayerNorm(α · h + f(h))    where α = (2L)^(1/4) for decoder-only
```

The `α` scaling on the residual amplifies the skip connection at depth, allowing post-norm Transformers to train at hundreds of layers. Used in some specific 1000-layer experiments (DeepNet); not adopted broadly because pre-norm already works fine and DeepNorm adds operational complexity.

## A fourth option: Sandwich-LN (2022 onwards)

Norm *both* before and after the sublayer:
```
h_out = h + LayerNorm(f(LayerNorm(h)))
```

Used in some Google models (PaLM 2-style) and Gemma. The "sandwich" pattern damps the sublayer's output magnitude before adding to the residual stream — gives extra stability at scale at marginal cost. Not universally adopted, but you'll see it.

## Pre-norm + the "final norm" pattern

A canonical modern decoder-only Transformer is structured:
```
x         = Embedding(tokens) + Positional(positions)
for l in 1..L:
    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))
x         = RMSNorm(x)                  # final norm
logits    = x @ W_U.T                   # unembedding projection
```

Two RMSNorms per block (one before attention, one before FFN) plus one **final RMSNorm** before the unembedding. The final norm is what handles the accumulated variance in the residual stream from all `L` blocks. Drop it and the logits become unstable at deep models.

You'll occasionally see slight variants:
- **No final norm** (rare; only stable with very specific init).
- **`x_pre_norm = norm(x); x = x_pre_norm + ...`** (an explicit pre-norm pattern that's mathematically equivalent to the above).
- **Sandwich-norm** as discussed above.

But the canonical "norm before each sublayer + final norm" pattern is what 95% of modern open models use.

## Where the norm-placement matters at inference

For inference (no backward needed), pre-norm and post-norm produce *different outputs* even for the same parameters — the architectures are different. You can't swap one for the other without retraining. This is sometimes confusing when porting model weights between codebases (HuggingFace's Llama vs. an alternative implementation, say): the same weights mean different things in different architectures.

## Self-check

1. In pre-norm, the residual `h` flows backward without going through any LN. Why does this matter for trainability at depth? What specifically breaks in post-norm at `L = 100`?
2. Pre-norm requires a final LayerNorm before the output head. Why? What's wrong with reading logits directly off the unnormalized residual stream?
3. Post-norm was the original convention but is now nearly extinct. Are there *any* scenarios where it's still preferable, or is it strictly dominated?

### Answers

1. The gradient backward through pre-norm has an identity path at every layer (the residual `h_l`), so the gradient signal at any depth equals the upstream gradient plus the layer's contribution — no multiplicative shrinkage or growth. In post-norm, the backward through LN at every layer introduces a Jacobian factor. Even when the LN's Jacobian has norm ≈ 1 on average, the product of 100 such factors can have very large or very small singular values in some directions (the LN Jacobian isn't isotropic). At `L = 100` this compounding makes gradient magnitudes layer-dependent and unstable. Empirically: post-norm beyond ~24 layers needs heroics (long warmup, low LR, careful init) to converge; pre-norm just works. The identity path is the difference.
2. The residual stream in pre-norm has accumulated contributions from `L` blocks, each potentially adding O(1) magnitude. Without the final norm, the logits `x @ W_U.T` see a residual stream whose scale depends on `L` and on training dynamics — not a stable distribution. The cross-entropy / softmax expects logits at a controlled scale (file 1.2/03). The final norm bounds the residual stream magnitude before logit projection, stabilizing both forward (no softmax saturation from extreme logits) and backward (no extreme gradient scales). Without it, the model can train but is much more brittle, especially during warmup and at the end of training where logit scales matter most for sampling.
3. Mostly strictly dominated. Two minor exceptions: (a) **Very shallow stacks** (< 12 layers) where the depth-stability concern doesn't apply — post-norm's slight benefits (output-of-block always at controlled scale, no need for final norm) can be cleaner. Not relevant for any modern LLM. (b) **Encoder-only models for some specific downstream tasks** where the post-norm output is directly used as a feature representation — though even here, modern usage has shifted to pre-norm. In practice: no, post-norm is not preferable for any new from-scratch design today.

## Exercise

Implement two versions of a 12-layer Transformer with `D = 256`:
- **Post-norm**: `h = LN(h + Attn(h)); h = LN(h + FFN(h))`.
- **Pre-norm**: `h = h + Attn(LN(h)); h = h + FFN(LN(h))`, with a final `LN` before the output head.

Train both on a tiny LM task with the *same* learning rate (no warmup). The post-norm version may NaN or stall; the pre-norm version should train smoothly. Then add a 1000-step warmup schedule to the post-norm version and try again — it'll start working. The cost of post-norm is paid in operational complexity (warmup, LR scheduling, careful init), not just in final loss.
