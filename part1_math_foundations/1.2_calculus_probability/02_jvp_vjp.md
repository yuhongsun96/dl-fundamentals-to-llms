# JVP and VJP — The Two Flavors of Autodiff

## The setup

For `f: R^n → R^m` with Jacobian `J ∈ R^(m×n)`:

- **JVP (Jacobian-vector product)**: given `v ∈ R^n`, compute `Jv ∈ R^m`
- **VJP (vector-Jacobian product)**: given `u ∈ R^m`, compute `u^T J ∈ R^n` (or equivalently `J^T u`)

Both avoid materializing `J`. Both run in ~forward-pass cost (specifically, 2–3× for VJP).

## Forward-mode AD = JVP

Propagate derivatives **alongside** the forward pass. Given a perturbation direction `v` at the input, compute the output perturbation by pushing tangent vectors forward through each op:

```
x → v_x
↓    ↓
f    J_f v_x
↓    ↓
y → v_y = J_f v_x
```

Cost: one forward pass *per input dimension* to get the full Jacobian. For a neural net with millions of inputs, infeasible.

**Good for:** small input dims, directional derivatives, Hessian-vector products (via nested AD).

## Reverse-mode AD = VJP

Propagate derivatives **backward** from a scalar output. Given upstream gradient `u = ∇L` at the output, pull cotangent vectors backward through each op:

```
x ← u_x = J_f^T u_y
↓    ↑
f    
↓    ↑
y ← u_y = ∇L
```

Cost: one forward + one backward pass to get gradients w.r.t. *all* inputs simultaneously. Independent of input dim.

**Good for:** scalar losses with millions of parameters. **This is why DL uses reverse-mode.**

## When to use which

| Scenario | Use |
|----------|-----|
| Scalar loss, many params (normal DL) | VJP (reverse-mode) |
| Many outputs, few inputs (rare) | JVP (forward-mode) |
| Full Jacobian needed (rare) | JVP if `n < m`, VJP if `m < n` |
| Hessian-vector product `Hv` | JVP of a VJP (or VJP of a JVP) |
| Per-sample gradients | VJP, but `vmap`'d (vectorized) |

## Memory: the cost of reverse-mode

VJP requires caching **all intermediate activations** from the forward pass, because each layer's backward needs its input. This is why:

- Training memory >> inference memory.
- Activation memory scales with depth × batch × sequence length × hidden dim.
- **Gradient checkpointing** trades compute for memory: don't cache intermediates; recompute them during the backward pass. Typically cuts activation memory by √(depth) at ~1.5× compute cost.

JVP has no such memory cost — it runs alongside the forward — which is why some people are reviving forward-mode AD for memory-constrained training ("forward gradients").

## In PyTorch

- `loss.backward()` → VJP under the hood, populates `.grad` on leaf tensors.
- `torch.autograd.grad(output, inputs, grad_outputs=v)` → compute `v^T J` directly.
- `torch.func.jvp(f, primals, tangents)` → JVP.
- `torch.func.vjp(f, primals)` → VJP as a callable.
- `torch.func.vmap` → vectorize over a batch axis (for per-sample grads).

You rarely need these explicitly — `.backward()` covers 99% — but knowing the machinery helps read papers on efficient training.

## Self-check

1. Why does reverse-mode AD cost the same regardless of the number of parameters, while forward-mode cost scales with input dim?
2. If I want the gradient of a single scalar loss w.r.t. a 70B-parameter model, which mode do I use and why?
3. How would you compute `Hv` (Hessian-vector product) for a loss `L(θ)` using standard AD tools?

### Answers

1. **Reverse-mode** propagates a single cotangent (the seed `1.0` at the scalar output) backward through the graph. Each layer's VJP costs roughly the same as the forward op. Total cost ≈ 1 forward + 1 backward = ~2× forward cost, **independent of input/param count**. **Forward-mode** propagates a tangent (a perturbation along one input direction) forward — but only one direction per pass. To get the full gradient over `n` inputs you need `n` separate forward passes (one per coordinate basis vector). Hence O(n × forward cost).
2. **Reverse-mode** (`.backward()`). 70B params is a 70B-dim input to the forward → loss map. Reverse-mode is O(forward cost) ≈ 2× forward regardless of param count. Forward-mode would be O(70B × forward cost) — completely infeasible. Every modern training loop is reverse-mode.
3. Trick: `Hv = ∇_θ ((∇_θ L) · v)`. Two backward passes:
   - First: compute `g = ∇_θ L` (standard backward, retain graph).
   - Compute the scalar `(g · v).sum()`.
   - Second backward: `∇_θ ((g · v).sum()) = Hv`.
   Cost: ~2× a normal backward pass. Used by some second-order optimizers (Sophia, K-FAC variants), curvature analyses, and natural-gradient methods. PyTorch: `torch.autograd.grad(loss, params, create_graph=True)` to retain graph, then a second `torch.autograd.grad` on `(g · v).sum()`.

## Exercise

In PyTorch, for `f(x) = sum(sin(W @ x))` with random `W ∈ R^(3×5)`:
1. Use `.backward()` to get `∂f/∂x`.
2. Use `torch.autograd.grad` to compute the same.
3. Use `torch.func.jvp` to compute `J · e_i` for each `i` and verify you can stack them into the Jacobian.
4. Confirm both approaches give consistent Jacobians.
