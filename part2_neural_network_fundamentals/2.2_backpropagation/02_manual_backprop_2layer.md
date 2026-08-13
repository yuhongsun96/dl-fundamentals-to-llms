# Manual Backprop Through a 2-Layer MLP

> **Convention note.** Column-vector throughout this file (`y = W x + b`, gradients are columns, Jacobians multiply on the left). Matches the chain-rule conventions in Part 1.2/01. The row-vector form is in Part 1.1/supplementary/08.

## Setup

A 2-layer MLP for a regression task:
```
z1 = W1 x + b1             # pre-activation, shape (h, 1)
a1 = σ(z1)                  # hidden activation, shape (h, 1)
z2 = W2 a1 + b2            # output pre-activation, shape (o, 1)
ŷ  = z2                     # linear output head
L  = ½ ‖ŷ - y‖²            # squared-error loss, scalar
```

Shapes:
- `x ∈ R^(d, 1)` — input.
- `W1 ∈ R^(h, d)`, `b1 ∈ R^(h, 1)` — first layer.
- `W2 ∈ R^(o, h)`, `b2 ∈ R^(o, 1)` — second layer.
- `y ∈ R^(o, 1)` — target.

Five learnable tensors: `W1, b1, W2, b2`. We need `∂L/∂W1, ∂L/∂b1, ∂L/∂W2, ∂L/∂b2`, plus `∂L/∂x` (occasionally needed; freely computed as a byproduct).

## Do this once on paper

The point of this file isn't to read it. It's to redo this derivation with pen and paper before you trust any framework. The mechanics never change; only the shapes get bigger. Once you've manually backpropped through one 2-layer MLP, you'll never mis-derive a Transformer gradient.

## The chain rule, applied node by node

We walk backward from `L`, computing one local gradient at each step. The right column tracks shape.

### Step 1: `∂L/∂ŷ`

`L = ½ Σ_i (ŷ_i - y_i)²`. Partial w.r.t. `ŷ_i` is `(ŷ_i - y_i)`. Stacked:
```
∂L/∂ŷ = ŷ - y                                       shape (o, 1)
```

This is the **residual** — the error vector. Memorize this: for squared-error loss with a linear output, the upstream gradient is just `(prediction - target)`.

### Step 2: `∂L/∂z2`

`ŷ = z2` (linear head, no activation). Identity Jacobian.
```
∂L/∂z2 = ∂L/∂ŷ = ŷ - y                              shape (o, 1)
```

If there were a softmax + CE here instead, this step would still simplify to `ŷ - y` (see Part 1.2/03). It's a beautiful coincidence — and the GLM theory says it's not a coincidence at all.

### Step 3: `∂L/∂W2` and `∂L/∂b2`

Now `z2 = W2 a1 + b2`. The recipe from Part 1.2/01:
- **Gradient w.r.t. matrix weight** = upstream gradient outer-producted with input activation.
- **Gradient w.r.t. bias** = upstream gradient (sum over batch).
- **Gradient w.r.t. input** = `W^T` times upstream gradient.

So:
```
∂L/∂W2 = (∂L/∂z2) · a1^T = (ŷ - y) · a1^T          shape (o, h)
∂L/∂b2 = ∂L/∂z2          = (ŷ - y)                  shape (o, 1)
∂L/∂a1 = W2^T · (∂L/∂z2) = W2^T · (ŷ - y)          shape (h, 1)
```

Each shape works out exactly. `∂L/∂W2` is `(o, 1) · (1, h) = (o, h)` — same as `W2`. `∂L/∂a1` is `(h, o) · (o, 1) = (h, 1)` — same as `a1`.

### Step 4: `∂L/∂z1`

`a1 = σ(z1)` is **elementwise**. Its Jacobian is the diagonal matrix `diag(σ'(z1))`. Applied to the upstream gradient, it becomes a Hadamard product:
```
∂L/∂z1 = (∂L/∂a1) ⊙ σ'(z1)                         shape (h, 1)
        = (W2^T (ŷ - y)) ⊙ σ'(z1)
```

For ReLU, `σ'(z1)` is 1 where `z1 > 0`, else 0 — a binary mask. For GELU/SiLU, it's a smooth scalar per coordinate. Either way, an elementwise multiply.

### Step 5: `∂L/∂W1`, `∂L/∂b1`, `∂L/∂x`

Same recipe as step 3, with the new upstream gradient `∂L/∂z1`:
```
∂L/∂W1 = (∂L/∂z1) · x^T                            shape (h, d)
∂L/∂b1 = ∂L/∂z1                                    shape (h, 1)
∂L/∂x  = W1^T · (∂L/∂z1)                           shape (d, 1)
```

`∂L/∂x` is the gradient passed further upstream — if `x` came from another layer, this is what that layer would receive as its `∂L/∂a_prev`. In our 2-layer net, `x` is the input, and `∂L/∂x` is usually discarded.

## The two universal recipes

Compress the above into two rules you should never have to re-derive:

**Rule 1 — linear layer `z = W a + b`:**
```
∂L/∂W = (∂L/∂z) · a^T            # outer product → shape of W
∂L/∂b = ∂L/∂z                    # identity → shape of b
∂L/∂a = W^T · (∂L/∂z)            # transpose → shape of a
```

**Rule 2 — elementwise nonlinearity `a = σ(z)`:**
```
∂L/∂z = (∂L/∂a) ⊙ σ'(z)          # Hadamard with derivative
```

Every backward pass in every feed-forward stack — including the FFN inside a Transformer — is just Rule 1 and Rule 2 alternating.

## Batched version (very small change)

In practice you process a batch of `B` samples. With row-vector convention, activations are `(B, d)`, weights are `(d_in, d_out)`:
```
Forward:  Z = X W + b              # X: (B, d_in), Z: (B, d_out)
                                    # b broadcasts (d_out,) → (B, d_out)

Backward (let G = ∂L/∂Z):
  ∂L/∂W = X^T · G                  # (d_in, B) · (B, d_out) = (d_in, d_out) — same shape as W
  ∂L/∂b = G.sum(axis=0)            # collapse batch dim → (d_out,)
  ∂L/∂X = G · W^T                  # (B, d_out) · (d_out, d_in) = (B, d_in)
```

The bias gradient **sums over the batch** because the same `b` was added to every sample's pre-activation — the chain rule sums contributions over all paths it influenced.

The matrix weight gradient is itself a sum: `X^T · G` is the sum, over the batch, of outer products `x_i · g_i^T`. Each sample contributes one rank-1 update; the matmul accumulates them.

## Cached activations (recap)

To compute the backward pass above, you needed:
- `x` (for step 5's outer product)
- `z1` (for step 4's `σ'(z1)` — though some implementations store `a1` if `σ'` can be expressed in terms of the output, e.g. for `tanh`, `σ'(z) = 1 - σ(z)²` = `1 - a²`)
- `a1` (for step 3's outer product)
- `ŷ` and `y` (for step 1)

In autograd these are kept alive automatically as long as the `grad_fn` chain holds them. The memory cost is one `(B, d_l)` tensor per layer per saved activation — the dominant memory cost in training.

## Common bugs caught early

- **Forgot to multiply by `σ'(z)`** in step 4 → gradients are too large by a factor that varies per element. The model trains but slowly and unstably.
- **Used `W^T` instead of `W`** (or vice versa) in `∂L/∂a` → shape mismatch, immediate error.
- **Forgot to sum over batch** for `∂L/∂b` → gradient shape is `(B, d)` instead of `(d,)` → optimizer step errors out.
- **Forgot to zero gradients** between steps (PyTorch accumulates) → gradients are sum of last K steps' gradients → optimizer takes runaway steps. Always `.zero_grad()` before `.backward()`.

## Self-check

1. In step 4, the Jacobian of `a = σ(z)` is `diag(σ'(z))` — a full diagonal matrix. Why is the implementation `g_a ⊙ σ'(z)` (an elementwise multiply) and not a matrix multiply? What does this save?
2. The bias gradient sums over the batch. The weight gradient is `X^T G`, which is implicitly a sum of outer products. Why don't we also sum over the batch for `∂L/∂X`?
3. If you swapped the squared-error loss for `L = ½ ‖ŷ - y‖²` per-sample-averaged (so `L_total = (1/B) Σ_i L_i`), how would every gradient above change?

### Answers

1. Multiplying by a diagonal matrix is equivalent to multiplying each row of the result by the corresponding diagonal entry. For `D · v` with `D = diag(d)`, the result is `[d_1 v_1, ..., d_n v_n]` — the Hadamard product `d ⊙ v`. The save: instead of an `O(n²)` matmul (most of whose entries are zero), it's an `O(n)` elementwise multiply. For `n = 4096`, that's 4096 multiplies instead of 16M. Generalizes to any pointwise op: its Jacobian is diagonal, so its VJP is a Hadamard product. This is why elementwise ops are cheap in backward.
2. Because `∂L/∂X` is **per-sample** — there's a separate gradient for each row of `X` (each sample's input). The bias is shared across samples, so its gradient is the sum. The weight is also shared across samples, so its gradient is the sum (`X^T G` is exactly Σᵢ xᵢ gᵢ^T). The input isn't shared — each row of `X` is a distinct quantity — so its gradient stays per-row. Rule of thumb: **shared parameters get summed gradients**; **per-sample tensors get per-sample gradients**.
3. Every gradient above (from step 1 through step 5) would be scaled by `1/B`. That's it. Mean reduction in the loss is equivalent to averaging the gradient over the batch instead of summing it. This is why most training code reports mean loss and uses mean-reduction implicitly — gradient magnitudes stay batch-size-independent, which decouples learning rate from batch size. (Note: this isn't quite true at scale — see file 04 in 2.4 on the relationship between learning rate and effective batch size.)

## Exercise

Implement the forward and backward pass above in pure NumPy. Train it on a toy regression task (e.g. fit `f(x) = sin(x)` on 1D data with a 2-layer MLP, `h = 32`, ReLU activation). Compare gradient updates step-by-step against an identical PyTorch implementation that calls `.backward()`. They should agree to ~1e-6.

When they agree, you've earned the right to trust autograd. When they disagree, your derivation is the bug.
