# Forward / Backward as Computational Graph Traversal

## The mental model

A neural network's forward pass is a **directed acyclic graph (DAG)**:
- **Nodes** = operations (matmul, add, ReLU, softmax, ...).
- **Edges** = tensors flowing from one op's output to another op's input.
- **Leaves** = inputs and parameters.
- **Root** = the scalar loss `L`.

Forward pass: walk the DAG from leaves to root, evaluating each op. Backward pass: walk it root-to-leaves, computing gradients.

Both passes visit every node exactly once. Both passes are linear in the number of ops. Backprop is just a structured second walk of the same graph.

## A concrete example

Take `L = ½ ‖σ(W x + b) - t‖²` with one hidden unit. The graph:

```
       x                  W
        \                /
         \              /
          ↘            ↙
            (matmul) → z1
                       |
                       + b        → z2
                       |
                       σ(·)       → h
                       |
                       - t        → r
                       |
                       ‖·‖²·½     → L  (scalar)
```

Each arrow is one op. Each node stores:
- Its output (forward).
- A reference to its inputs (so it knows where to send gradients).
- Whatever's needed to compute its local Jacobian (often the inputs themselves, sometimes the output).

## Forward pass: build and evaluate

In autograd frameworks, the forward pass does **two** things simultaneously:
1. Compute the output tensor.
2. **Record the op into the graph**, with pointers to its inputs.

The graph is built *as a side effect of the forward pass*. PyTorch's term: the graph is **dynamic** — rebuilt every forward, allowing control flow (if/while) to vary per batch. TensorFlow 1.x's term: **static** — graph is defined once and reused (faster, less flexible).

In modern PyTorch (eager mode by default), every tensor produced by a differentiable op has a `.grad_fn` attribute pointing to the op that produced it. Walk `.grad_fn` backward to traverse the graph.

## Backward pass: the chain rule, mechanized

Start at `L` with seed gradient `∂L/∂L = 1` (a scalar 1).

At each node, given the incoming gradient `∂L/∂(node output)`, compute:
1. The gradient w.r.t. each **input** of the node (so it can be passed further upstream).
2. The gradient w.r.t. each **parameter** of the node (so it can be accumulated for the optimizer).

Both are computed via a **vector-Jacobian product (VJP)**: the node has a local Jacobian `J` describing how its output depends on its inputs, and the gradient w.r.t. the inputs is `J^T · (∂L/∂output)`. See `02_jvp_vjp.md` in Part 1.

The key property: **each op only needs to know its local Jacobian** and the upstream gradient. It doesn't need to know what comes after it in the graph. This locality is what makes autograd composable.

## Why activations have to be cached

For `y = W x`, the gradient w.r.t. `W` is `(∂L/∂y) x^T` — needs `x`. So during the forward pass, before moving on, you have to keep `x` around. This is **activation memory**, and at scale it dominates training memory usage.

Per layer in a Transformer training step, you cache:
- The input activations to every matmul.
- The attention weights (or enough state to recompute them).
- Intermediate values needed by other ops (e.g. `exp(z)` for softmax derivatives).

For a forward pass with `B` sequences of length `S` through an `L`-layer model with `D`-dim residual stream, activation memory scales as `O(B · S · D · L)`. For long context this can dwarf the parameter memory. Gradient checkpointing (file `04`) trades compute for activation memory by *not* caching some activations and recomputing them in backward.

## Topological ordering

For the backward pass to be correct, each node must receive **all** its downstream gradients before it can compute its own backward. In other words, you process nodes in **reverse topological order**: a node is visited only after every node that depends on it.

In a feed-forward network the order is obvious — process layer `L`, then `L-1`, etc. But in graphs with branches (residual connections, attention with multiple heads) a single tensor can feed into multiple downstream ops. Its gradient is the **sum** of the gradients from each downstream consumer. Autograd implements this by:
- Counting how many downstream ops a tensor feeds into.
- Accumulating gradient contributions as they arrive.
- Continuing backward through that node once all contributions are in.

This sum-over-paths is just the multivariable chain rule applied to the graph.

## A worked symbolic example

For `L = (σ(W x + b) - t)² / 2`, scalar `x`, scalar `W`, scalar `b`, scalar `t`:

Forward:
```
z1 = W * x
z2 = z1 + b
h  = σ(z2)
r  = h - t
L  = r² / 2
```

Backward (seed: `dL = 1`):
```
dr  = r              # since L = r² / 2, dL/dr = r
dh  = dr             # r = h - t, dr/dh = 1
dz2 = dh * σ'(z2)    # h = σ(z2), local Jacobian is σ'(z2)
db  = dz2            # z2 = z1 + b, dz2/db = 1
dz1 = dz2            # z2 = z1 + b, dz2/dz1 = 1
dW  = dz1 * x        # z1 = W x, dz1/dW = x
dx  = dz1 * W        # z1 = W x, dz1/dx = W
```

Six lines. Each line is one local Jacobian applied to the upstream gradient. Same structure as the matrix-valued case; only the shapes change.

## Why this is "backpropagation" and not something else

Two alternative ways to compute gradients:

- **Numerical (finite differences)**: `(L(W + ε) - L(W)) / ε` for each parameter. Cost: one forward pass per parameter. For a million-parameter model, this is a million forward passes per gradient step. Useless for training; useful for gradient checking small models.
- **Forward-mode autodiff (JVP)**: walk the graph forward, propagating perturbations alongside values. Cost: one forward pass per *input* dimension to perturb. Efficient when you have few inputs and many outputs. **Wrong direction for ML**: we have many inputs (parameters) and one output (scalar loss).
- **Reverse-mode autodiff (VJP) = backprop**: one forward pass + one backward pass, total cost ≈ 2× a forward pass, gradient w.r.t. **all** parameters. Efficient when you have many inputs and few outputs. **Right direction for ML**.

This asymmetry — "training is reverse-mode autodiff because the loss is a scalar" — is the single fact that makes deep learning computationally tractable. If we had to compute gradients via finite differences, no model with > 10K parameters could be trained.

## Self-check

1. In a graph where tensor `h` is consumed by two downstream ops, what is `∂L/∂h`? Why must autograd wait for both downstream paths before propagating?
2. Why does a forward pass cost roughly `F` FLOPs, a backward pass cost roughly `2F` FLOPs, and a full training step cost `3F`?
3. What does it mean for a tensor in PyTorch to have `requires_grad=True`? What does it cost?

### Answers

1. `∂L/∂h = (∂L/∂h via path 1) + (∂L/∂h via path 2)` — the multivariable chain rule sums contributions over all paths. Autograd must wait because it needs the **total** upstream gradient before it can apply the local VJP and pass gradient further upstream. If it propagated path 1's contribution prematurely, path 2's contribution would be missed and the resulting gradient would be wrong by the path-2 amount. Implementations track an "in-degree counter" per node; backward through a node only fires when all in-edges have delivered their gradient.
2. Forward: one matmul per layer, cost `F`. Backward: two matmuls per layer — one to compute the gradient w.r.t. the input (`W^T · upstream`, cost `F`) and one to compute the gradient w.r.t. the weight (`upstream · input^T`, cost `F`). So backward is `2F`, total step is `3F`. The `3F` rule of thumb is the basis of training-FLOP estimates: Chinchilla scaling laws and the like all use `3F` per token per parameter pair as the unit.
3. `requires_grad=True` tells autograd: "track ops applied to this tensor and record them in the graph so we can compute its gradient later." Cost: every op consuming a `requires_grad=True` tensor builds a graph node, which holds references to inputs. This pins the inputs in memory (no garbage collection until backward runs) and adds per-op metadata overhead. For inference, set `requires_grad=False` (or use `torch.no_grad()`) to skip graph construction and free activation memory immediately. This is why inference uses far less memory than training at the same model size.

## Exercise

Take the symbolic example above (`L = (σ(W x + b) - t)² / 2`) and:
1. Pick concrete values: `x = 1.5`, `W = 0.8`, `b = 0.3`, `t = 0.4`, `σ = tanh`.
2. Compute the forward pass by hand. Get `L`.
3. Compute the backward pass by hand. Get `dL/dW`, `dL/db`, `dL/dx`.
4. Verify against PyTorch by setting up the same computation with `requires_grad=True` and calling `.backward()`. Compare to ~1e-6.

You will not believe how clarifying this is until you've done it once.
