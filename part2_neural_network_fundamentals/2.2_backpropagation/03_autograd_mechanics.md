# Autograd Mechanics

How PyTorch turns "call `loss.backward()`" into the gradient computation you derived by hand in the previous file. You don't need to read the source to use it, but a clear mental model prevents an entire category of bugs.

## The graph is built during forward

Every differentiable op in PyTorch, when called on a tensor that has `requires_grad=True` (or any tensor whose ancestors do), does three things:

1. Compute the output tensor.
2. Create a **`grad_fn` node** capturing what op this was and what it needs for backward.
3. Attach the `grad_fn` to the output tensor.

The graph is the chain of `grad_fn` references. You can inspect it:
```
>>> x = torch.randn(3, requires_grad=True)
>>> y = (x * 2).sum()
>>> y.grad_fn
<SumBackward0>
>>> y.grad_fn.next_functions
((<MulBackward0>, 0),)
```

Each `grad_fn` knows the op, the inputs it needs to compute the local Jacobian, and pointers to the upstream nodes (`next_functions`).

This happens **eagerly** — there's no separate "graph construction" phase. The graph is a byproduct of forward.

## `.backward()` walks the graph

When you call `loss.backward()`:

1. Seed: gradient at the root is `1.0` (a scalar, since `loss` is scalar).
2. Iterate nodes in **reverse topological order**. For each node:
   a. Apply its local VJP using the cached inputs and the incoming gradient.
   b. Add the result to `.grad` of each leaf tensor it touches.
   c. Pass the gradient further upstream via `next_functions`.

When backward completes, every leaf tensor with `requires_grad=True` has its `.grad` field populated.

## Leaf tensors vs. intermediate tensors

- **Leaf**: created by the user (not by an op). Examples: model parameters, input data. By default, intermediates don't accumulate gradients — only leaves do, in `.grad`.
- **Intermediate**: created by an op. Has a `.grad_fn`. Its gradient flows through to upstream leaves but isn't stored.

If you want to inspect an intermediate's gradient, call `.retain_grad()` on it before backward.

A trainable weight is a leaf no matter how deep its layer — nothing *computes* it, so it sits at the edge of the graph as an input, not as a product of earlier layers. For why "intermediate in the architecture" and "intermediate in the graph" are different axes, and why gradients accumulate only on leaves, see [`supplementary/03_leaf_vs_intermediate.md`](supplementary/03_leaf_vs_intermediate.md).

## Gradient accumulation (the default)

`.grad` on a leaf is **additive**. Calling `.backward()` a second time without clearing gradients adds to the existing `.grad`. This is why training loops always start with:
```
optimizer.zero_grad()        # or model.zero_grad()
loss.backward()
optimizer.step()
```

This is also intentional behavior, not a misfeature. Gradient accumulation across mini-batches (without `zero_grad`) is the standard way to simulate a larger effective batch on memory-constrained hardware:
```
for i, (x, y) in enumerate(loader):
    loss = model(x, y) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Forget the `/ accum_steps` and your effective learning rate is `accum_steps` times what you intended. Common bug at scale.

Runnable walk-through of the four classic `.backward()` gotchas — grad on a leaf, the no-`requires_grad` error, the "backward through the graph a second time" error, and how `retain_graph=True` exposes `.grad` accumulation (`[1,1,1]` then `[2,2,2]`) — each with a prediction before the code: [`supplementary/03_backward_semantics.ipynb`](supplementary/03_backward_semantics.ipynb).

## The `requires_grad` propagation rule

A tensor produced by an op has `requires_grad=True` iff **any** of its inputs do. This is how PyTorch decides whether to build the graph for that op:
- If yes, build the graph node and track activations.
- If no, skip graph construction — operate as if in `no_grad` mode for that op.

This means: in a frozen pretrained model (parameters with `requires_grad=False`), if you pass an input that doesn't require grad, no graph is built. Want to fine-tune just a head? Freeze the base (`p.requires_grad=False` for base params), and the input to the head is the base's output — which has no grad because none of its inputs do. So you need to either set the input's `requires_grad=True` or only freeze part of the way and let some intermediate require grad.

In practice for LoRA, QLoRA, and any adapter setup: you set `requires_grad=False` on base parameters and `requires_grad=True` on adapter parameters; activations downstream of the adapters then require grad and the graph is built only where it's needed. This is what makes adapter fine-tuning memory-efficient.

## `torch.no_grad()` and `inference_mode()`

Two ways to disable autograd:

- **`torch.no_grad()`**: forces all ops in the context to skip graph construction. Outputs are detached (no `grad_fn`).
- **`torch.inference_mode()`**: stricter and faster. Also disables some version-counter tracking that `no_grad` retains. Use this for inference; can't be exited halfway.

Standard pattern:
```
model.eval()                       # affects dropout, batchnorm behavior
with torch.inference_mode():       # disables autograd
    outputs = model(inputs)
```

Forgetting `inference_mode()` during eval is a common source of "eval is slow / using too much memory" problems — autograd is silently keeping all activations alive.

## `.detach()` and `.data`

Both produce a tensor sharing storage with the original but without a `grad_fn`:
- **`.detach()`**: official, returns a new tensor with no autograd connection.
- **`.data`**: legacy, same effect but bypasses autograd's version tracking — can produce silently wrong gradients if you modify in place. **Don't use `.data` in new code.**

Common use: cutting the gradient flow somewhere you don't want it to propagate. Example: target-network updates in RL, where you want the target value to be treated as a constant.

## Higher-order gradients

Add `create_graph=True` to `.backward()` and the backward pass itself becomes part of a new graph, which you can differentiate again. Used for:
- Meta-learning (MAML).
- Hessian-vector products (compute `Hv` via grad of `(grad · v)`).
- Some adversarial training and influence-function methods.

The cost roughly doubles per order of differentiation. Rare in standard LLM training.

## The hooks API

You can register Python functions that fire on forward or backward:
```
def hook(module, grad_input, grad_output):
    print(grad_output[0].norm())

layer.register_full_backward_hook(hook)
```

Used for:
- Inspecting / logging gradients during training.
- Modifying gradients (e.g. gradient surgery in multi-task learning).
- Activation patching for interpretability work (Part 11.2).

Hooks are a power-user feature; in 95% of training code, you don't touch them.

## Functional autograd (functorch / `torch.func`)

The recommended modern API for *programmatic* gradient computation (e.g. per-sample gradients, JVPs, Hessian-vector products):
```
from torch.func import grad, vmap, jacrev, jacfwd
```

`grad(f)` returns a function that computes `∇f`. `vmap` lets you vectorize over a batch. Together they let you compute per-sample gradients without a Python loop. Used in some optimization research and in influence-function code.

Not needed for standard training — `.backward()` handles that.

## A few things autograd does *not* do for you

- **Numerical stability.** Autograd computes the gradient of whatever you wrote. If you wrote `log(softmax(x))` instead of `log_softmax(x)`, autograd will faithfully compute the gradient of the unstable version. The math is correct; the numerics aren't.
- **Layer fusion.** Each op is a separate node. Doesn't fuse `mul + add + ReLU` into a single kernel unless you use `torch.compile` or hand-write a fused kernel.
- **Choosing what to checkpoint.** Activation memory is your problem — autograd will dutifully cache *everything* needed for backward unless you tell it not to (see file `04`).

## Self-check

1. After calling `loss.backward()`, what's the state of the graph? Can you call `loss.backward()` again on the same `loss` tensor?
2. Why do you set `model.eval()` *and* use `torch.inference_mode()` separately? Don't they overlap?
3. You freeze a pretrained model (`p.requires_grad = False` for all params) and add a small trainable head. During the forward pass, are activations being cached for backward? Why or why not?

### Answers

1. By default, the graph is **freed** after backward — intermediate buffers (cached activations) are released. Calling `loss.backward()` again raises an error: "Trying to backward through the graph a second time...". To prevent this freeing, call `loss.backward(retain_graph=True)` — but you usually don't want to; the standard pattern is one forward, one backward, one step, then a new forward. Multiple backwards on the same graph come up in second-order methods and in some RL/meta-learning setups.
2. Different concerns. `model.eval()` changes the **forward behavior** of certain modules: dropout becomes identity, batchnorm uses running statistics instead of batch statistics. `inference_mode()` disables **autograd tracking**: no graph is built, no activations cached. You need both for correct + fast inference: `eval()` for correctness (otherwise dropout corrupts predictions), `inference_mode()` for speed and memory (otherwise you're carrying around the training-shaped activation memory).
3. The base's parameters don't require grad, but its **activations are computed normally**. Whether they're cached depends on whether the *output* of the base requires grad — which it doesn't, since none of its inputs (input data + frozen params) require grad. So the base's activations are **not cached**, and its forward is essentially in `no_grad` mode automatically. The head's input is the base's output — without grad. To get gradients into the head's input you have one of two options: (a) set `requires_grad=True` on the bridge tensor manually, or (b) let some intermediate of the base require grad (e.g. LoRA adapters injected into the base). LoRA picks (b): inject low-rank trainable matrices, so the activations downstream of them require grad and get cached, while everything upstream of any adapter still runs in effective no_grad mode. This is the memory advantage of adapter fine-tuning.

## Exercise

In PyTorch, do the following and predict the output of each:
1. `x = torch.randn(3, requires_grad=True); y = x.sum(); y.backward(); print(x.grad)`
2. Repeat without `requires_grad=True`. What error or behavior?
3. Now: `x = torch.randn(3, requires_grad=True); y = x.sum(); y.backward(); y.backward()`. What happens on the second `.backward()`?
4. Now: `x = torch.randn(3, requires_grad=True); y = x.sum(); y.backward(retain_graph=True); y.backward(); print(x.grad)`. What's in `x.grad`?

Each is one line and one minute. Worth doing once.
