# Gradients, Directional Derivatives, Chain Rule

> **Convention note.** This file uses the **column-vector** convention (`y = W x`, gradients are column vectors, Jacobians multiply on the left). The row-vector counterpart of the linear-layer rule (`Y = X W` with `∂L/∂W = X^T (∂L/∂Y)`) is in `supplementary/08_linear_layer_gradients.md`.

## Gradient

For `f: R^n → R`, the gradient `∇f ∈ R^n` is the vector of partial derivatives:
```
∇f = [∂f/∂x_1, ..., ∂f/∂x_n]^T
```

Geometric meaning: `∇f(x)` points in the direction of **steepest ascent** at `x`, with magnitude equal to the slope in that direction.

Gradient descent: step in `-∇f` to reduce loss. That's it. Everything else in optimization is variants on this.

## Directional derivative

The rate of change of `f` in direction `u` (unit vector) at point `x`:
```
D_u f(x) = <∇f(x), u>
```

The max over all unit `u` is `‖∇f‖`, achieved at `u = ∇f / ‖∇f‖`. The min is `-‖∇f‖`. Along any direction orthogonal to the gradient, `f` doesn't change (to first order) — that's the level set.

## Chain rule (scalar version)

For `h(x) = g(f(x))`:
```
h'(x) = g'(f(x)) · f'(x)
```

In DL, every layer is nested inside every later layer. The chain rule unrolls the nesting into a product.

## Chain rule (multivariate)

For `h = g ∘ f` where `f: R^n → R^m`, `g: R^m → R`:
```
∇_x h = J_f(x)^T · ∇_y g(y=f(x))
```

where `J_f` is the `m×n` Jacobian of `f`. The transpose is essential — gradients are column vectors, Jacobians multiply on the left, so the transpose flips the direction.

**What the formula is actually doing.** You want to know how wiggling `x_i` changes the final scalar. It can take many paths: `x_i` perturbs every intermediate `y_j` (that's column `i` of `J_f`), and each `y_j` then pushes the scalar by `(∇_y g)_j`. Summing over paths gives `(∇_x h)_i = Σ_j (∂y_j / ∂x_i)(∂g / ∂y_j)` — exactly the dot product of column `i` of `J_f` with `∇_y g`. Stacking over all `i` is `J_f^T ∇_y g`. The transpose is just the bookkeeping that turns "rows = outputs" into "rows = inputs."

**Forward vs. backward, same map.** Forward, `J_f` sends an input-perturbation to an output-perturbation. Backward, you pull a scalar's sensitivity back through the same linear map in reverse — which is exactly its transpose (adjoint). Same matrix, opposite direction.

**This is autograd.** Every node in a computational graph is some local `f`. The forward pass caches whatever's needed to know `J_f`. The backward pass receives an incoming gradient `∇_y g` from downstream and left-multiplies by `J_f^T` to produce the gradient w.r.t. the node's inputs, which it hands further upstream. The trick is that nobody ever materializes `J_f` — each op implements a **vector-Jacobian product (VJP)** that returns `J_f^T v` directly. For `y = Wx`: VJP w.r.t. `x` is `W^T v`; VJP w.r.t. `W` is `v x^T`. No Jacobian built, same answer.

## The gradient of a loss w.r.t. weights

For a single linear layer `y = Wx`, loss `L(y)`:
```
∂L/∂W = (∂L/∂y) x^T     (outer product — result is (m, n), same shape as W)
∂L/∂x = W^T (∂L/∂y)     (a vector of shape (n,))
```

These two gradients do completely different jobs:

- **`∂L/∂W` is the gradient you actually use to learn.** It's what the optimizer consumes — SGD literally does `W ← W - η · ∂L/∂W`, and Adam does the same with momentum and second-moment scaling. This is the *only* place this layer's `W` shows up in the entire training loop's update rule. Don't compute it, and this layer never learns.
- **`∂L/∂x` isn't for this layer — it's the baton you hand to the previous layer.** `x` is the *output* of the layer before this one, so `∂L/∂x` is exactly what that layer needs as *its* incoming `∂L/∂y`. Every backward node has the same two-output shape: one stream lands on each parameter (terminal — goes to the optimizer), one stream flows back through each input (continuing — becomes the next backward step's incoming gradient). That continuation is the *propagation* in back-propagation. At the very first layer, `∂L/∂x` has no upstream consumer and is usually discarded — unless you're doing adversarial examples, input attribution, or learning the input itself (e.g. an embedding table, where `x` *is* a parameter).

Two rules you should never forget:
- **Gradient w.r.t. weights** involves an **outer product** of upstream gradient and input activation.
- **Gradient w.r.t. input** involves `W^T` times upstream gradient — the "backward flow" through the layer.

This is *why* activations must be cached during the forward pass: the backward needs them for the outer product. It's also why activation memory dominates training memory.

## A warning about "gradient" vs "derivative"

In ML papers, "gradient" often loosely means "vector of partial derivatives w.r.t. parameters", which for a matrix `W` is actually a matrix of the same shape (technically a Jacobian-shaped object). Don't get hung up on the math-purist distinction — follow the shapes.

## Second-order: Hessian (briefly)

`H = ∇²f ∈ R^(n×n)` — matrix of second derivatives. For a scalar loss, `H[i, j] = ∂²L/∂θ_i ∂θ_j`. Tells you local curvature.

You will almost never compute the Hessian directly. But:
- **Hessian-vector products** `Hv` are computable at ~forward-pass cost via a trick (autograd of `∇L · v`).
- Large-eigenvalue-of-Hessian → "sharp" minimum → often worse generalization.
- Second-order optimizers (K-FAC, Shampoo, Sophia) approximate the Hessian to precondition gradients.

## Self-check

1. For `L = ‖Wx - y‖²` (squared error), derive `∂L/∂W` from scratch.
2. Why is the gradient w.r.t. `W` the same shape as `W`? (This is obvious but articulate it.)
3. In the chain rule `∇_x h = J_f^T ∇_y g`, why is the Jacobian transposed? What would happen if you forgot the transpose?

### Answers

1. Let `r = Wx - y`. Then `L = r^T r`. Chain rule: `∂L/∂r = 2r`. For `∂L/∂W`, observe that `r` is linear in `W`, so the gradient is the outer product of the upstream gradient with the input: `∂L/∂W = 2 (Wx - y) x^T = 2 r x^T`. Shape: same as `W`. This is the same outer-product rule from file 07 — every Linear-layer gradient is `(error) · (input)^T`.
2. Because the optimizer's update rule is `W ← W - η · ∇W`, which requires `∇W` to be elementwise subtractable from `W` — same shape. More fundamentally: the gradient stores the partial derivative `∂L/∂W_{ij}` for every entry of `W`, and arranging those partials in the same `(m, n)` layout gives a tensor of shape `(m, n)`. PyTorch's autograd convention: the gradient of a scalar w.r.t. a tensor has the same shape as that tensor.
3. Differentials: `dh = (∇_y g)^T dy` and `dy = J_f dx`. Combining: `dh = (∇_y g)^T J_f dx`. But the gradient is defined by `dh = (∇_x h)^T dx`, so `(∇_x h)^T = (∇_y g)^T J_f`, giving `∇_x h = J_f^T (∇_y g)`. The transpose is essential for shapes to align: `J_f` is `(m, n)`, `∇_y g` is `(m,)`, so `J_f^T (∇_y g)` is `(n,)` — matching `∇_x h`. Forget the transpose: shape mismatch error, or worse, silently-wrong gradients.

## Exercise

Given `z = tanh(Wx + b)` and `L = ½‖z - t‖²`, compute `∂L/∂W`, `∂L/∂b`, `∂L/∂x` on paper. You should get:
- `∂L/∂z = z - t`
- `∂L/∂pre = (z - t) ⊙ (1 - z²)`   (since `tanh'(u) = 1 - tanh²(u)`)
- `∂L/∂W = (∂L/∂pre) x^T`
- `∂L/∂b = ∂L/∂pre`
- `∂L/∂x = W^T (∂L/∂pre)`
