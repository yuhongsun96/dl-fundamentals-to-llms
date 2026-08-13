# Backprop Through a Linear Layer: The Outer-Product Gradient

A deeper look at the three gradient rules for `y = Wx + b` (column-vector convention, matching the primary file). These come up in the self-check of `08_jacobians_chain_rule.md`. If only one set of formulas in all of backprop gets memorized, make it these — every other layer's backward pass is built out of them.

## The setup

```
y = W x + b
```

| Symbol | Shape | Role |
|--------|-------|------|
| `x` | `(n,)` | input column vector |
| `W` | `(m, n)` | weight matrix |
| `b` | `(m,)` | bias |
| `y` | `(m,)` | output column vector |
| `L` | scalar | loss |
| `∂L/∂y` | `(m,)` | upstream gradient (given by what comes next) |

## The three rules

```
∂L/∂W = (∂L/∂y) · x^T        shape (m, n) — outer product
∂L/∂x = W^T · (∂L/∂y)        shape (n,)   — transposed matmul
∂L/∂b = ∂L/∂y                shape (m,)   — passthrough
```

Each has a different structural flavor. Understanding *why* each one has the form it does is more useful than memorizing the formulas.

---

## Rule 1: `∂L/∂W = (∂L/∂y) · x^T` — the outer product

### Shape sanity check

`(∂L/∂y)` is `(m, 1)`, `x^T` is `(1, n)`. Their product is `(m, n)`, same as `W`. Gradients always match the shape of the parameter they're for — this is the deepest invariant in autograd, and it's already telling you the answer must be an outer product.

### Where the formula comes from

Key observation: **each weight `W_{ij}` is used in exactly one place** in the forward pass. It multiplies `x_j` and contributes the result to `y_i`. That's it.

```
y_i = W_{i1} x_1 + W_{i2} x_2 + ... + W_{in} x_n + b_i
∂y_i / ∂W_{ij} = x_j        ← bump W_{ij} by ε, y_i changes by ε·x_j
∂y_k / ∂W_{ij} = 0          ← W_{ij} doesn't touch any other output
```

Apply the chain rule for a single weight:
```
∂L/∂W_{ij} = Σ_k (∂L/∂y_k) · (∂y_k/∂W_{ij})
           = (∂L/∂y_i) · x_j          ← only k=i contributes
```

Stack that over all `(i, j)` and you get a matrix whose `(i, j)` entry is `(∂L/∂y)_i · x_j` — which is exactly the **outer product** `(∂L/∂y) x^T`.

### The one-line intuition

> Weight `W_{ij}` is the conduit from input `j` to output `i`. Its gradient is **"how much output `i` wants to change" × "how strong input `j` was."**

In symbols:
```
(∂L/∂W)_{ij}  =  (upstream gradient at output i)  ·  (input j that this weight saw)
```

A weight that connects a *large input* to an *output the loss is complaining loudly about* gets a big gradient. A weight where either side is small gets a small gradient.

### Why it's an *outer* product, not a matmul

An outer product is what you get when you multiply an `(m, 1)` column by a `(1, n)` row: every entry of one vector pairs with every entry of the other, producing an `(m, n)` matrix of all pairwise products.

That's the right structure because **every output gets a gradient signal, every input was used by every output, and `W_{ij}` is the unique parameter responsible for the (output `i`, input `j`) interaction.** Pair them all up → outer product.

A regular matrix multiplication contracts (sums over) a shared index. There's no shared index to contract here — we're producing one number per `(i, j)` pair, not collapsing them.

### Hebbian-flavored gloss

If you've ever seen the slogan "neurons that fire together wire together," this is the math version of it. `(∂L/∂y)_i` is the postsynaptic error signal, `x_j` is the presynaptic activity, and the weight update is their product. Backprop generalizes that local rule with a global error signal — but for a single linear layer, the update is literally **pre-activation × post-error**.

---

## Rule 2: `∂L/∂x = W^T · (∂L/∂y)` — backward routing

Same chain-rule machinery, different free variable:
```
y_i = Σ_j W_{ij} x_j + b_i
∂y_i / ∂x_j = W_{ij}
∂L/∂x_j = Σ_i (∂L/∂y_i) · W_{ij}  =  Σ_i W_{ij} (∂L/∂y_i)  =  (W^T · ∂L/∂y)_j
```

The transpose appears because in the forward pass `W` routes input dim `j` to output dim `i`, and in the backward pass we need to route the signal at output `i` back to input `j` — the connectivity is the same matrix, but read the other direction.

### Why this matters

`∂L/∂x` is what gets handed to the **previous** layer as *its* upstream gradient. So `W^T (∂L/∂y)` is the engine of backprop: every linear layer, on the way back, applies `W^T` to whatever came in from above. That's all "gradients flow backward by `J^T`" actually means for the most common layer type.

### Role swap with Rule 1

Notice the symmetry between the two:

| Gradient | Uses... |
|---|---|
| `∂L/∂W` | the *input* `x` from forward, paired with the upstream gradient |
| `∂L/∂x` | the *weights* `W` (transposed), applied to the upstream gradient |

This is why a `Linear` layer's backward pass needs to **cache `x` from the forward pass**. Without it, you can't compute the weight gradient. The weights are already in memory; the inputs aren't, unless explicitly retained. This caching is the dominant memory cost of training and the thing gradient checkpointing trades against compute.

---

## Rule 3: `∂L/∂b = ∂L/∂y` — passthrough

The bias adds elementwise: `y_i = (Wx)_i + b_i`, so `∂y_i/∂b_i = 1` and `∂y_i/∂b_j = 0` for `j ≠ i`. The chain rule gives `∂L/∂b_i = ∂L/∂y_i` directly. The Jacobian of "add `b`" is the identity, so its transpose is also the identity — the upstream gradient passes through unchanged.

---

## Putting the three together

| Gradient | Formula | Shape | Structure | Mnemonic |
|---|---|---|---|---|
| `∂L/∂W` | `(∂L/∂y) x^T` | `(m, n)` | outer product | "upstream × input" |
| `∂L/∂x` | `W^T (∂L/∂y)` | `(n,)` | transposed matmul | "backward through `W^T`" |
| `∂L/∂b` | `∂L/∂y` | `(m,)` | identity passthrough | "bias = upstream" |

These three are the complete backward rule for a Linear layer. Stack many such layers with nonlinearities between them and you have backprop for an MLP.

## Batched version (what code actually computes)

In the row-vector convention (what most other files use), with `X ∈ R^(B, n)`, `Y = X W`, `W ∈ R^(n, m)`, the gradients become:

```
∂L/∂W = X^T (∂L/∂Y)             shape (n, m), same as W
∂L/∂X = (∂L/∂Y) W^T             shape (B, n), same as X
∂L/∂b = sum over batch of ∂L/∂Y shape (m,)
```

The outer-product story still holds, but with a twist: each sample in the batch contributes its own rank-1 outer product to `∂L/∂W`, and the `X^T` on the left does the outer-product *and* sums across the batch in one matmul. (Concretely: `(X^T)_{j, b} · (∂L/∂Y)_{b, i}` summed over `b` gives `(∂L/∂W)_{j, i}` — for each batch element you form `x_b ⊗ (∂L/∂y_b)` and add them up.)

The bias gradient sums across the batch because the same `b` was used for every sample, so each sample's upstream gradient contributes additively.

## Why these rules are the whole backprop story

Almost every layer in a modern Transformer is built from linear layers and elementwise nonlinearities:

- `Linear`, `Embedding` (which is a linear layer with a one-hot input shortcut), the `Q/K/V` projections in attention, the output projection, the up- and down-projections in MLP/SwiGLU — all linear.
- LayerNorm, RMSNorm, softmax, activations — elementwise or near-elementwise.
- Attention itself is matmul + softmax + matmul, no parameters in the body.

Master the three rules above + element-wise VJP rules + softmax Jacobian and you can derive backprop for any Transformer-style architecture from scratch. Everything else (FlashAttention's recomputation tricks, gradient checkpointing, mixed-precision casts) is engineering on top of these mathematical primitives.

## Self-check

1. For a Linear layer with `n = 4`, `m = 3`, `B = 8`: what shape is `∂L/∂W`? What is the rank of any single sample's contribution to it?
2. Why does the bias gradient sum across the batch but the weight gradient *also* sum across the batch (implicitly via the matmul)? What would happen if you used the bias gradient formula `∂L/∂Y` directly without summing on a batched input?
3. In `∂L/∂W = X^T (∂L/∂Y)`, identify which factor came from "things the forward pass had to save" and which came from "things passed in from the next layer's backward."

### Answers

1. `(4, 3)` — matches `W`'s shape `(n, m)`. Each single sample contributes a rank-1 matrix (an outer product of two vectors), so the batch sum is at most rank `min(B, n, m) = 3` here. In general, the accumulated weight gradient has rank ≤ batch size, which is one (informal) reason small batches give noisier, lower-rank updates.
2. The weight gradient implicitly sums across the batch because the matmul `X^T (∂L/∂Y)` contracts over the batch dim `B` — that's the shared index `b` in `Σ_b X_{b,j} · (∂L/∂Y)_{b,i}`. The bias gradient must sum *explicitly* because it isn't part of any matmul — `b` is just broadcast-added in the forward, so its backward is broadcast-summed (summing over the dim that was broadcast). If you forgot to sum, you'd get `(B, m)` instead of `(m,)` — a per-sample bias gradient that doesn't fit the parameter's shape.
3. `X` came from the cached forward-pass activations (this is the memory cost of training). `∂L/∂Y` came from the upstream layer's backward. The product mixes them: forward-pass state × backward-pass signal = parameter gradient. This pattern — "the gradient of a layer's parameters is the convolution of what flowed through it forward with what flowed in backward" — generalizes to every parameterized layer.
