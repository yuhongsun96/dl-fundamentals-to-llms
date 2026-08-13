# Hidden-State Recurrence and BPTT

You know this cold, so this file restores the *two structural facts* about RNNs that matter for the rest of the curriculum — the fixed-size state and the product-of-Jacobians backward — and drops the rest.

**Convention:** column-vector (`h = W x`), matching the derivation files in 1.1/1.2. Hidden state `h_t ∈ R^D`, input `x_t ∈ R^{d_x}`, per timestep `t = 1…S`.

## The recurrence

A vanilla RNN folds a sequence into a running state, one step at a time:

```
h_t = φ(W_h h_{t-1} + W_x x_t + b)          φ = tanh (usually)
y_t = W_y h_t                                (optional per-step output)
```

Read it as a loop that walks the sequence one token at a time, updating a running "memory vector" `h`:

- **`x_t`** — the input at step `t` (e.g. the embedding of the `t`-th token).
- **`h_{t-1}`** — the state from the *previous* step: the model's memory of everything seen so far, *before* reading `x_t` (a vector of `D` numbers).
- **`h_t`** — the *new* state after folding in `x_t`; it becomes the `h_{t-1}` of the next step. That feedback is the loop.
- **`W_x x_t`** — transform the new input into the state's `D`-dim space: "what the new token contributes."
- **`W_h h_{t-1}`** — transform the old memory (`W_h` is `D×D`): "what to carry over from the past."
- **`+ b`** — a constant bias offset.
- **`φ(·)`** — squash the sum through a nonlinearity (`tanh`), keeping the state bounded and adding expressiveness.
- **`y_t = W_y h_t`** — an *optional* per-step output (e.g. a next-token prediction) read off the current memory by another matrix `W_y`. Optional because some tasks use only the final state.

So each step is: **new memory = squash( carried-over old memory + contribution of the new input )**. Start with `h_0` (usually zeros) and repeat for `t = 1…S`. Mental picture: a reader going through a sentence word by word, keeping a single running "gist" in their head and updating it after each word — `h_t` is the gist after token `t`.

Two things to notice, because both echo forward through the curriculum:

1. **The state `h_t` is a fixed-size summary of all history `x_1…x_t`.** Everything the model remembers about the past is compressed into `D` numbers. This is a *lossy bottleneck by construction* — and the exact idea that returns, deliberately, in state-space models / Mamba (Part 7.3). The Transformer will make the opposite choice: keep *all* past tokens accessible (the KV cache), unbounded in size (Part 9.2).
2. **The same weights `W_h, W_x` are reused at every step.** An RNN is not `S` layers with `S` weight sets; it's *one* cell applied `S` times. Weight-sharing across time is what makes it a "recurrent" net and what shapes its gradient.

## BPTT = unroll, then it's just backprop

To train, **unroll** the recurrence into a feed-forward graph of depth `S` — timestep `t` becomes "layer" `t`, all sharing weights — and run ordinary backprop over it. That's **backpropagation through time**. Two consequences fall straight out of the structure above:

- **Shared weights → gradients sum over time.** Because `W_h` is used at every step, its gradient is the *sum* of the contributions from all `S` steps (the fan-out → add rule from [2.2](../../part2_neural_network_fundamentals/2.2_backpropagation/01_forward_backward_compgraph.md)). One weight, many uses, gradients accumulate.
- **The recurrence → gradients chain as a product across time.** The gradient of the loss at step `t` w.r.t. an earlier state `h_k` runs back through every intermediate step:
  ```
  ∂h_t/∂h_k = ∏_{i=k+1}^{t} ∂h_i/∂h_{i-1} = ∏_{i=k+1}^{t} diag(φ'(z_i)) · W_h
  ```
  Reading the pieces:
  - **`∂h_t/∂h_k`** — a `D×D` sensitivity matrix: "if I nudged the memory at step `k`, how would the memory at step `t` move?" This is the pathway gradient travels across `t−k` steps.
  - **`∏_{i=k+1}^{t}`** — a **product** (matrix, so ordered) over the `t−k` intermediate steps: chain rule composes the one-step sensitivities `h_k → h_{k+1} → … → h_t`.
  - **The single-step factor `diag(φ'(z_i)) · W_h`** is the Jacobian of one update `h_i = φ(z_i)`, `z_i = W_h h_{i-1} + …`, in two links: `∂z_i/∂h_{i-1} = W_h` (the reused recurrent matrix) and `∂φ/∂z_i = diag(φ'(z_i))` (tanh is elementwise, so its Jacobian is diagonal — a per-coordinate slope that scales/gates each dimension, `= 1 − tanh² ∈ (0,1]`, collapsing to 0 when saturated). `z_i` differs at each step, hence the `(·)`.

  So `∂h_t/∂h_k` is roughly `(typical φ' × spectral norm of W_h)^(t−k)` — **exponential in the time distance**: per-step factor `< 1` → gradient **vanishes**, `> 1` → **explodes**. This is the *same* multiplicative-chain math as the depth story in Part 3 — only here the chain runs along the **sequence** instead of along **layers** — and it's the reason gradients vanish or explode over long ranges (file `02`).

**Truncated BPTT** is the practical patch: unrolling all `S` steps costs `O(S)` memory (every `h_t` cached for backward — the activation-memory problem again, [2.2/04](../../part2_neural_network_fundamentals/2.2_backpropagation/04_gradient_checkpointing.md)), so you chop the sequence into windows and only backprop within each. It bounds memory but also bounds how far gradient — and thus learnable dependency — can reach.

## Why it matters in modern LLM work

- **The sequential dependency is the fatal flaw.** `h_t` cannot be computed until `h_{t-1}` exists, so training and generation are inherently `O(S)` *sequential* steps — no parallelism over the sequence. This is the single biggest thing the Transformer removed (file `03`), and reclaiming it *without* losing parallel training is the entire pitch of Mamba (Part 7.3).
- **The fixed-size state is a design axis, not a mistake.** "Compress history into a bounded state" (RNN/SSM) vs. "keep everything and attend" (Transformer) is a live tradeoff — bounded state = cheap, constant-memory inference but a hard recall ceiling; full attention = perfect recall but quadratic cost. Part 7 is largely this tension.
- **BPTT's product-of-Jacobians is the through-line.** Vanishing/exploding across time (here), across depth ([2.2/05](../../part2_neural_network_fundamentals/2.2_backpropagation/05_gradient_pathologies.md)), and the fix in both cases (an *additive* path that turns the product into `identity + correction`) — LSTM gating (file `02`) and residual connections ([3.1](../../part3_residual_connections_deep_networks/3.1_skip_connection/02_gradient_highway.md)) are the same idea in two settings.

## Self-check

1. An RNN processes a length-`S` sequence. Why can't the `S` timesteps be computed in parallel during training, when a Transformer's `S` positions can?
2. `W_h` appears at every timestep. State, in one sentence each, how that fact shapes (a) the *forward* computation and (b) `W_h`'s *gradient*.
3. Truncated BPTT bounds memory. What does it simultaneously bound, and why is that a modeling limitation, not just an engineering one?
4. Where does the "product of Jacobians across time" reappear later in the curriculum as a "product across depth"?

### Answers

1. The recurrence `h_t = φ(W_h h_{t-1} + …)` makes `h_t` depend on `h_{t-1}`, a strict sequential chain — you must finish step `t−1` before starting `t`. A Transformer computes all positions' representations from the (already-known) inputs with no position-to-position recurrence, so the whole sequence goes through each layer in one parallel batched matmul. (Generation is still sequential for the Transformer — but *training*, with teacher forcing, is fully parallel; that's the win.)
2. (a) Forward: one shared cell is applied `S` times, so the network's depth-in-time is `S` but its parameter count is constant in `S`. (b) Gradient: because `W_h` is reused every step, `∂L/∂W_h` is the *sum* over all `S` timesteps of each step's local contribution (shared-parameter fan-out → accumulate).
3. It bounds the *temporal reach of the gradient* — dependencies longer than the truncation window get no learning signal, so the model can't learn them even in principle. That's a cap on modeling capability (effective context), not merely on memory.
4. In Part 3's gradient-highway / pathologies story: backprop through `L` stacked layers is `∏ J_layer`, exactly analogous to BPTT's `∏ J_time`. Both vanish/explode for the same reason and are fixed by the same additive-path trick (gating in time, residuals in depth).

## Exercise

On paper: take the scalar RNN `h_t = w·h_{t-1}` (drop the nonlinearity and inputs). (a) Write `∂h_t/∂h_0` in closed form. (b) For `w = 0.9` and `w = 1.1`, evaluate it at `t = 50` and `t = 200`. (c) In one sentence, connect the two numbers to "vanishing" and "exploding," and name the single scalar that decides which happens. (d) Now replace the recurrence with `h_t = h_{t-1} + g_t` (an additive carry). What is `∂h_t/∂h_0` now, and why does that preview both the LSTM cell state (file `02`) and the residual connection (Part 3)?
