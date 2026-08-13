# Linear + Nonlinearity: The Whole Game

> **Convention note.** Row-vector throughout this subsection: `Y = X W + b`, with `X ∈ R^(B·S, d_in)`, `W ∈ R^(d_in, d_out)`. Matches the activations-as-rows form used in modern code.

## The MLP atom

An MLP layer is exactly two things stacked:

```
h = σ(X W + b)
```

A **linear map** (`X W + b`) followed by a **pointwise nonlinearity** (`σ`). That's it. Every fully-connected network in DL — including the FFN inside every Transformer block — is a stack of these.

The names you'll see used interchangeably: **MLP** (multi-layer perceptron), **FFN** (feed-forward network), **FC layer** (fully connected), **dense layer**. Same thing.

## Why both pieces are necessary

Stack two linear layers without a nonlinearity:
```
Y = (X W_1) W_2 = X (W_1 W_2) = X W'
```

You get **one** linear map with a composed weight `W' = W_1 W_2`. The depth bought you nothing — you could have learned `W'` directly with a single layer. The nonlinearity is what makes depth meaningful: each layer can transform the geometry of the representation in ways the next layer can build on.

The universal approximation theorem says a single hidden layer of sufficient width can approximate any continuous function. True but useless — "sufficient" can mean exponential in input dimension. Depth is the trick that turns universal approximation into *efficient* approximation: a deep network can represent functions that a shallow one would need exponentially more units to match. This is the entire reason "deep" is in "deep learning."

## What the linear part does

`X W + b` is the only place this layer has **parameters** and the only place information **mixes across features**. The nonlinearity is pointwise — coordinate `i` of `σ(z)` depends only on `z_i` — so it can't move information between coordinates. All mixing is in the matmul.

Three jobs in one operation:
- **Mixing**: every output feature is a learned weighted sum of every input feature.
- **Projection**: maps `d_in → d_out`. The output dim is a design choice that controls capacity at this layer.
- **Affine offset**: `+ b` shifts the result. The bias lets the nonlinearity activate even when the inputs are zero. (More on whether you actually need it: file `03`.)

## What the nonlinearity does

Bends the linear function into something non-flat. Each nonlinearity has a shape, and the shape determines:
- How information passes through (the forward signal).
- How gradients flow back (the slope, i.e. the derivative).

Without bending, no curvature; without curvature, no expressive function class beyond hyperplanes. The choice of nonlinearity affects training dynamics — see file `02`.

## Depth vs. width tradeoffs

Given a fixed parameter budget `N`, how should you spend it: more layers or wider layers?

**Width buys parallel features.** A wider layer has more independent neurons, each potentially specializing in a different feature. The cost grows roughly as `width²` per layer (the matmul). Wider is closer to a "lookup table" of features.

**Depth buys composition.** Each additional layer can apply another nonlinear transformation, so the function class becomes exponentially richer per added layer. The cost grows linearly with depth (more matmuls in series). Deeper is closer to a "program" — successive refinements of the representation.

Empirical regularities for Transformers:

- **Deeper helps more, up to a point.** Going from 6 to 12 layers buys more than going from `D=512` to `D=1024`, at matched param count. Until ~50–100 layers, depth tends to win.
- **Past ~100 layers, depth gives diminishing returns**, and width starts winning. Training becomes unstable, gradients are harder to propagate (even with residuals and pre-norm), and the marginal layer adds less.
- **For Transformers specifically, `D` (model width) is more frequently scaled than `L` (depth)** in the GPT-2 → GPT-3 → Llama progression. Roughly: layer count `L ~ 12 · log10(N / 10^7)`, while `D` grows much faster. Llama-3 70B uses `L=80, D=8192`; Llama-3 405B keeps `L=126` and pushes `D=16384`. Width scales roughly with `√N`, depth roughly with `log N`.

The intuition: an LLM's "thoughts" are vectors in `R^D`. Each layer is one rewrite step on that vector. Past a certain depth, more rewrite steps stop helping — but a higher-dimensional residual stream lets each rewrite carry more information.

For the residual-stream picture of this — why Transformer depth is best read as "iterative refinement of a single vector" rather than "deep stack of independent transforms" — see Part 3.

## Width allocations inside a Transformer block

Two width parameters matter inside one block:

- `d_model` (or `D`): the residual-stream width. Everything that enters or leaves the block is `(B, S, D)`.
- `d_ff`: the hidden width of the FFN's intermediate layer. The FFN expands from `D → d_ff → D`.

The classic Transformer used `d_ff = 4 D`. Modern gated FFNs (SwiGLU, GeGLU — see file `02`) compensate for having two parallel projections by using `d_ff ≈ (2/3) · 4D ≈ 2.67 D` to keep parameter count comparable.

Why FFN is wider than the residual stream: the FFN is where most of the model's **storage** lives. Roughly two-thirds of an LLM's parameters are in FFN weights, not attention. The expansion lets the layer act as a key-value lookup over a larger memory than the residual width would allow.

## What an FFN actually computes in modern LLMs

In a Transformer FFN (e.g. Llama), the operation is no longer just `σ(X W_1) W_2`. It's a **gated** variant:

```
FFN(x) = (silu(x W_gate) ⊙ (x W_up)) W_down
```

Two projections of `x` — one passed through `silu`, one not — multiplied elementwise, then projected back down. That's SwiGLU (file `02`).

But strip away the gating and you're left with the same atom: linear → nonlinearity → linear. The MLP is still the unit.

## Self-check

1. Why does removing the nonlinearity collapse a deep MLP to a single linear layer? Be precise — what *exactly* breaks?
2. For a Transformer with `D = 4096`, `d_ff = 11008` (the SwiGLU 2.67× ratio), how many parameters live in one FFN block? Compare to the attention block at the same `D` (4 weight matrices, each `D × D`).
3. Why does universal approximation (one hidden layer is enough) not imply that depth is unnecessary?

### Answers

1. `Y = σ(X W_1) W_2` with `σ` linear means `σ(z) = α z + β`. Then `Y = (α X W_1 + β) W_2 = X (α W_1 W_2) + β W_2 = X W' + b'`. The composition of two linear maps is one linear map (matrix multiplication is associative and closed). The hidden representation `h_1 = α X W_1 + β` lives in a `d_hidden`-dim space, but the *function class* the network expresses is just the set of affine maps `X ↦ X W' + b'` — same as a single linear layer with `(W', b')`. Nothing nonlinear can be expressed regardless of how many layers you stack.
2. SwiGLU FFN has three weight matrices: `W_gate, W_up ∈ R^(D, d_ff)` and `W_down ∈ R^(d_ff, D)`. Params: `3 · D · d_ff = 3 · 4096 · 11008 ≈ 135M`. Attention has four `D × D` matrices (Q, K, V, O): `4 · D² = 4 · 4096² ≈ 67M`. The FFN is ~2× the attention block at this scale, which is typical. This is why "most of an LLM is FFN weights" — and why MoE (file 7.4) puts the expert sparsity on the FFN, not on attention.
3. The theorem guarantees existence at infinite (or merely exponential) width. It says nothing about how *efficiently* you can represent the target function. A function expressible by a depth-`L` network with `O(N)` units may require `O(2^L)` units in a depth-1 network. Depth gives compositional efficiency: layer `k` can reuse the features layer `k-1` built. A shallow net has to "rediscover" all intermediate features in its single hidden layer. Empirically, this exponential efficiency advantage is real — depth is what made modern DL practical.

## Exercise

Take a tiny 2D-input MLP: `R² → 64 → 64 → 1` with ReLU. Train it to fit `f(x, y) = x² + y²` (a paraboloid). Now do the same with the ReLUs replaced by the identity. Plot both predicted surfaces and compare to the target. The linearized version should fit a flat plane — the best linear approximation — and miss the curvature entirely. This is "no nonlinearity = no curvature" made visible.
