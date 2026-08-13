# Jacobians and the Chain Rule in Matrix Form

This file is the bridge into backprop. If only one file in Part 1 sticks, make it this one.

## Jacobian

> **Convention note.** This file uses the **column-vector** convention (inputs and outputs are column vectors, `J` acts on the left as `J δ`) because Jacobian math reads more cleanly that way. Most other files in the repo use the row-vector convention (`Y = X W`). Translating: the row-vector Jacobian of `Y = X W` w.r.t. `X` is just `W` on the right; here we'd write it with `W^T` on the left. Stay aware of which side things sit on, but the content is the same.

For `f: R^n → R^m`, the Jacobian `J ∈ R^(m×n)` has entries:
```
J[i, j] = ∂f_i / ∂x_j
```

It's the matrix of all partial derivatives. It's the best linear approximation of `f` at a point:
```
f(x + δ) ≈ f(x) + J δ
```

### Reading the definition

`f` takes an `n`-dimensional input and returns an `m`-dimensional output — really `m` scalar functions `f_1(x), ..., f_m(x)` bundled together, each depending on all `n` input coordinates. The Jacobian is the table that answers, for every (output, input) pair: **if I nudge this input a tiny bit, how much does this output change?** That number is the partial derivative `∂f_i / ∂x_j`.

- **Row index `i`** = which output you're looking at (`1..m`).
- **Column index `j`** = which input you're wiggling (`1..n`).

Shape `(m, n)` falls out: `m` rows (one per output), `n` columns (one per input). Two useful ways to read it:

- **Across row `i`**: how output `i` responds to each input. That row is the gradient of the scalar function `f_i`.
- **Down column `j`**: how every output responds when you wiggle input `j`. That column is the directional derivative of `f` along `e_j`.

The "best linear approximation" line says: locally, `f` looks affine, and `J` is its linear part. Perturb the input by a small `δ` and `J δ` is a matrix–vector product where each output coordinate is `(row i of J) · δ` — a weighted sum of input nudges, weighted by sensitivities. This is the multi-input, multi-output generalization of `f'(x) · δ` from 1D.

Shape sanity check: `δ` lives in input space (`n`-dim), the result lives in output space (`m`-dim), so `J` must be `(m, n)` for `J δ` to typecheck. The convention isn't arbitrary — it's forced by "inputs in, outputs out."

### "Best linear approximation" is only interesting when `f` is nonlinear

The whole point of forming a Jacobian is to **linearize a nonlinear function** at a point. If `f` is already linear — i.e. `f(x) = Wx` for some matrix `W` — there's nothing to approximate. The "best linear approximation" of `Wx` is `Wx`, exactly, everywhere. And the Jacobian works out to:

```
∂(Wx)_i / ∂x_j = W_{ij}      ⟹      J = W
```

So for a linear map, the Jacobian *is* the matrix itself. That's the row in the special-cases table. It also collapses two ideas that look distinct in textbooks: in DL, the weight matrix of a `Linear` layer **is** its Jacobian — no derivative needs to be computed at runtime, the "derivative" is already sitting in memory.

The Jacobian becomes a genuinely new object as soon as `f` is **nonlinear** — `σ(x)`, `softmax(x)`, `‖x‖`, an MLP, attention. There, `f(x + δ) ≈ f(x) + Jδ` is an actual approximation that's only valid locally, and `J` varies with `x`. That's the regime the chain rule lives in.

### What if the function takes or produces matrices?

In DL this comes up constantly — a loss `L(W)` takes a matrix; attention's `Q = X W_Q` is a matrix-valued function of a matrix. The Jacobian definition generalizes, but it stops fitting in a 2D table.

- **Matrix in, scalar out** (`L: R^(m×n) → R`): one partial per input entry, so the "Jacobian" is shape `(m, n)` — same shape as the input. This is what we call the **gradient** `∇_W L`, and the matched shape is what makes `W ← W - η ∇_W L` typecheck.
- **Vector in, matrix out** (`f: R^n → R^(m×p)`): partials indexed by `(i, j, k)` → 3-index tensor of shape `(m, p, n)`.
- **Matrix in, matrix out** (`f: R^(a×b) → R^(c×d)`): partials indexed by `(i, j, p, q)` → 4-index tensor of shape `(c, d, a, b)`.

You can force these into ordinary matrices by **flattening** (`vec(·)`): stack input and output into long vectors and pretend it's `R^(ab) → R^(cd)` with a `(cd, ab)` Jacobian. Correct but ugly, and you lose the shape that made the original problem readable.

In practice **autograd never materializes these tensors.** It computes vector-Jacobian products that respect the original tensor shapes, with the invariant that **the gradient of a scalar loss w.r.t. a parameter has the same shape as the parameter.** For a linear layer `Y = X W`:

- `∂L/∂W` is shaped like `W`: it's `X^T (∂L/∂Y)`.
- `∂L/∂X` is shaped like `X`: it's `(∂L/∂Y) W^T`.

The "true" 4-index object `∂Y_{ij} / ∂W_{pq}` exists conceptually but is never instantiated. Keep this in mind when we hit attention — `Q, K, V` are all matrix-valued functions of matrix inputs, and the only sane way to reason about gradients there is "match the shape of the parameter," not "write the full Jacobian."

## Special cases you should recognize

| Function | Input | Output | Jacobian shape | Jacobian form |
|----------|-------|--------|----------------|---------------|
| Linear `f(x) = Wx` | `R^n` | `R^m` | `(m, n)` | `W` |
| Linear `f(x) = x^T W` | `R^n` | `R^m` | `(m, n)` | `W^T` |
| Element-wise `f(x) = σ(x)` | `R^n` | `R^n` | `(n, n)` | `diag(σ'(x))` |
| Softmax `f(x) = softmax(x)` | `R^n` | `R^n` | `(n, n)` | `diag(s) - ss^T` where `s = softmax(x)` |
| Norm `f(x) = ‖x‖` | `R^n` | `R` | `(1, n)` | `x^T / ‖x‖` |

Know these cold. They're the building blocks of every backward pass.

## Chain rule in matrix form

For composed functions `h = g ∘ f`, i.e. `h(x) = g(f(x))`:
```
J_h = J_g(f(x)) · J_f(x)
```

**Jacobians multiply.** A deep network is a composition, so its total Jacobian is a product of layer Jacobians. Backprop is this product, computed right-to-left.

## Gradient of a scalar loss

In DL the final output is a scalar loss `L`. For `L = g(f(x))` where `f: R^n → R^m`, `g: R^m → R`:
```
∇_x L = J_f^T · ∇_{f(x)} L
        ↑ (n×m)     ↑ (m,)
      = (n,) vector
```

**Gradients flow backward by multiplying the upstream gradient by the Jacobian transpose of each layer.** This is all of backprop in one sentence.

## Why we never materialize Jacobians

For a layer `R^n → R^m` with `n, m ~ 10^4`, the Jacobian is `10^8` entries — unworkable. Instead we compute **vector-Jacobian products** (VJPs): given upstream gradient `v`, compute `v^T J` directly from the forward-pass intermediates.

For `f(x) = Wx`: the Jacobian is `W`, so `v^T J = v^T W`. We don't need to "form" `W` — it's already stored.

For `f(x) = σ(x)` (element-wise): `v^T J = v ⊙ σ'(x)`. Pointwise multiply.

Autograd is a systematic way of composing VJPs. When you `.backward()` a scalar, PyTorch walks the graph from output to inputs, each node applying its VJP rule.

## JVP vs. VJP (preview of 1.2)

- **VJP** (reverse-mode AD): `v^T J`. Cheap when output is scalar (gradients). Default for DL.
- **JVP** (forward-mode AD): `J v`. Cheap when input is scalar. Used for directional derivatives, Hessian-vector products.

For a scalar loss, reverse-mode is ~free (one backward pass = same FLOPs as forward). Forward-mode would require one pass per input dim — billions of passes. That's why DL is reverse-mode.

## The chain-rule gotcha

Matrix multiplication is associative but **not** commutative. `J_g J_f ≠ J_f J_g`. Getting the order right is the whole game:

```
input → f → g → h → L    (forward)
∇x  ← J_f^T J_g^T J_h^T ∇L    (backward)
```

The transposes and the reversed order are both correct and both necessary.

## Self-check

1. Given `y = Wx + b` and a loss `L(y)`, write `∂L/∂W` and `∂L/∂x` in terms of `∂L/∂y` and `x`, `W`.
2. Derive the softmax Jacobian: starting from `s_i = e^{x_i} / Σ e^{x_j}`, show `∂s_i/∂x_j = s_i(δ_{ij} - s_j)`.
3. Why is reverse-mode AD O(forward-pass cost) while forward-mode would be O(n × forward-pass cost) for an `n`-dim input?

### Answers

1. **`∂L/∂W = (∂L/∂y) · x^T`** — outer product of upstream gradient and input. Shape matches `W`. **`∂L/∂x = W^T · (∂L/∂y)`** — backward flow through the layer is just multiplication by `W^T`. (And `∂L/∂b = ∂L/∂y` directly, since `b` adds elementwise.) These two rules — outer-product gradient w.r.t. weight, transposed-matmul gradient w.r.t. input — are the entire content of backprop through a Linear layer. See `supplementary/08_linear_layer_gradients.md` for a full derivation, the role-swap intuition between `∂L/∂W` and `∂L/∂x`, the Hebbian gloss, the batched form, and why these three rules are essentially all of backprop for a Transformer.
2. Quotient rule on `s_i = e^{x_i} / S` where `S = Σ_j e^{x_j}`:
   - `∂e^{x_i}/∂x_j = δ_{ij} e^{x_i}`. `∂S/∂x_j = e^{x_j}`.
   - `∂s_i/∂x_j = (δ_{ij} e^{x_i} · S - e^{x_i} · e^{x_j}) / S²`
   - `= (e^{x_i}/S) · (δ_{ij} - e^{x_j}/S) = s_i (δ_{ij} - s_j)` ✓
   - Matrix form: `J = diag(s) - s s^T`.
3. Reverse-mode propagates **one** cotangent (the seed `1.0` at the scalar output) backward through the graph. Each layer's VJP runs in time ~equal to the forward op. Total: one forward + one backward ≈ 2× forward cost, **independent of input dim**. Forward-mode propagates a tangent (a directional perturbation) **forward**, but only one direction at a time — to get the full gradient w.r.t. `n` inputs, you need `n` forward passes (one per coordinate). Hence O(n × forward cost). For DL with `n` ≈ billions of params, only reverse-mode is feasible.

## Exercise

By hand: a 2-layer MLP `y = W_2 σ(W_1 x + b_1) + b_2`, with MSE loss against a target `t`. Write out `∂L/∂W_1`, `∂L/∂W_2`, `∂L/∂x`. Do it without peeking. This exercise is the price of admission to Part 2.
