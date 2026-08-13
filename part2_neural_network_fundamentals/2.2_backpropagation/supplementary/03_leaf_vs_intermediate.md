# Leaf vs. intermediate: why a trainable weight is a leaf

This expands the "Leaf tensors vs. intermediate tensors" section of [`03_autograd_mechanics.md`](../03_autograd_mechanics.md). It exists to kill one specific confusion: *"if a weight lives deep in the network, isn't it intermediate — downstream of all the earlier layers?"* No. The answer turns on separating two unrelated meanings of "intermediate."

## Two different meanings of "intermediate"

The word gets overloaded:

- **Intermediate in the model architecture** — layer 5 comes after layers 1–4. The *activations* flow forward through that stack.
- **Intermediate in the computation graph** — a tensor that was *produced by an operation* from other tensors.

These are not the same axis, and leaf-vs-intermediate is **only** about the second one.

## What "leaf" actually means

A leaf is defined by **how the tensor came into existence**, not by where it sits in your model.

A tensor can be born in exactly two ways:

1. **You made it directly.** `torch.randn(...)`, the weight `nn.Linear` allocates, data loaded from disk. No PyTorch op produced it from other tensors. → **leaf**
2. **An op produced it.** Output of `x @ W`, `relu(z)`, `a + b`. PyTorch built it by combining existing tensors. → **intermediate** (non-leaf); it carries a `.grad_fn` recording the op that made it.

That is the whole distinction. "Leaf" literally means *a node at the edge of the graph with nothing feeding into it* — the graph terminates there.

## Why a trainable weight is a leaf, regardless of depth

Take weight `W₅` in layer 5. The temptation: "layer 5 is after layers 1–4, so `W₅` is downstream of them — intermediate."

The fix: **`W₅` is not computed from the earlier layers.** Activations flow `1 → 2 → 3 → 4 → 5`. But `W₅` itself just sits there. It was created once (initialized or loaded) and depends on no other tensor. The earlier layers feed **data into the operation** `a₄ @ W₅` — they do not *produce* `W₅`. The data flows *past* `W₅`, not *into* it.

In the graph, weights hang off the side as starting points; activations are the chain strung between them:

```
input_data (leaf) ──┐
                     ├─► z₁ ─► a₁ ─► z₂ ─► a₂ ─► ... ─► L
W₁ (leaf) ───────────┘
W₂ (leaf) ─────────────────────┘
```

The `zᵢ`/`aᵢ` are graph-intermediates (each has a `.grad_fn`). Every `Wᵢ` is a leaf — no matter how deep its layer.

> **Architecture depth ≠ graph position.**
> Depth (layer 1 vs. layer 5) is the order activations flow through.
> Leaf-vs-intermediate is whether a tensor was *computed from* other tensors.
> Weights are never computed during the forward pass — they are inputs to it. So every trainable parameter is a leaf.

## Why gradients accumulate only on leaves

`backward()` *computes* a gradient for every tensor in the graph — it has to. The chain rule for `∂L/∂W₁` routes through `a₁, z₂, a₂, …`, so those intermediate gradients are computed in transit. They are simply not **stored**.

Why not store them?

- **Memory.** Activations are large (`B × S × D`). A `.grad` for each would roughly double activation memory for no benefit.
- **Nothing needs them.** Training updates *parameters*: the optimizer reads `param.grad` and steps. Nobody steps an activation — it's recomputed next forward pass. So the intermediate gradient is used to keep the chain rule moving backward, then discarded.

Leaves are exactly the tensors where a gradient is the *endpoint* of a backward path — nothing further upstream to flow into. Leaves with `requires_grad=True` (every trainable parameter) are precisely the ones you want to update. Those, and only those, accumulate into `.grad`.

`.retain_grad()` is the escape hatch: it tells autograd to keep one specific intermediate's gradient — for debugging or for inspecting how gradient magnitude changes with depth — without paying the storage cost everywhere.

## One-sentence summary

A trainable weight is a leaf because **nothing computes it** — it is an input to the graph, not a product of it — and that holds at any depth; "intermediate" in the autograd sense means "an op produced this tensor," which is the activations, never the parameters.

## Self-check

1. Is the input data tensor a leaf? Is it the same *kind* of leaf as a weight? (Hint: both are leaves; what differs is `requires_grad`.)
2. You have `z = x @ W`. Which of `x`, `W`, `z` have a `.grad_fn`? Which accumulate `.grad` by default?
3. A colleague says "the gradient of an activation isn't computed during backprop, only the gradient of the weights." Correct them in one sentence.
4. Why would `retain_grad()` on an early-layer activation help you diagnose vanishing gradients?

## Exercise

Without running code, sketch the graph for a 2-layer MLP `L = loss(relu(x @ W₁) @ W₂, target)`. Mark every tensor as leaf or intermediate, mark which carry a `.grad_fn`, and mark which hold a populated `.grad` after `backward()` (assume `W₁, W₂` require grad and `x` does not). Then state where in the graph the chain rule *computes but discards* a gradient.
