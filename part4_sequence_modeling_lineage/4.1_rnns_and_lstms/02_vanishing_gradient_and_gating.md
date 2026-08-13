# Vanishing Gradients and the Gating Fix

The file `01` product-of-Jacobians is where vanilla RNNs go to die. This file says precisely why, then shows that the LSTM's fix is the **same additive-path trick** as the residual connection — just across time instead of depth. If you internalize that equivalence, you've collapsed three "separate" topics (LSTM gates, ResNets, Mamba's gating) into one idea.

**Convention:** column-vector; cell/hidden state `∈ R^D`; `⊙` is elementwise product; `σ` is sigmoid (gate), `tanh` the candidate nonlinearity.

## Why vanilla RNNs can't reach

From file `01`, the gradient across time is a product of Jacobians:

```
∂h_t/∂h_k = ∏_{i=k+1}^{t} diag(φ'(·)) · W_h
```

Two multiplicative killers stack up:

1. **`W_h` repeated.** Multiplying by the *same* matrix `t−k` times drives the gradient like its spectral radius to the power of the distance: `< 1` → **vanishes** exponentially, `> 1` → **explodes** exponentially. Only a razor's-edge `≈ 1` survives, and training won't hold you there.
2. **`tanh'` ≤ 1, and usually ≪ 1.** The activation derivative saturates toward 0 whenever the pre-activation is large (the [saturation](../../part1_math_foundations/1.1_linear_algebra/supplementary/03_scaling_and_saturation.md) valve), multiplying in another sub-1 factor at every step.

The result: gradient from step `t` to step `k` decays to nothing within ~10–20 steps. The network *cannot* learn dependencies longer than that — "the clouds are in the ___" is fine, "I grew up in France … [50 tokens] … I speak fluent ___" is not. Exploding is the mirror image (fixed bluntly by gradient clipping, [2.4/03](../../part2_neural_network_fundamentals/2.4_optimization/03_gradient_clipping.md)); vanishing is the deeper problem because you can't clip your way out of a zero.

## The fix: an additive, gated carry

The LSTM (Hochreiter & Schmidhuber, 1997) adds a second state, the **cell state `c_t`**, whose update is *additive*, not multiplicative:

```
f_t = σ(W_f · [h_{t-1}, x_t])         forget gate   — how much of c_{t-1} to keep
i_t = σ(W_i · [h_{t-1}, x_t])         input gate    — how much new info to write
g_t = tanh(W_g · [h_{t-1}, x_t])      candidate     — the new info
o_t = σ(W_o · [h_{t-1}, x_t])         output gate   — how much of c_t to expose

c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t     ← the cell state: ADDITIVE update
h_t = o_t ⊙ tanh(c_t)
```

Drawn as data flow, the shape of the cell is two tracks:

```
  MEMORY  (cell state c) — an additive conveyor belt; the gradient highway:

     c_{t-1} ──►(⊙ f_t)──►(⊕ i_t⊙g_t)───────────────────►  c_t ──► (to next step)
                  │             │                            │
             forget:        write new info:                 tanh
          keep how much     how much (i_t) × what (g_t)       │
           of old memory                                   (⊙ o_t)  ── output gate:
                                                             │        how much memory to expose
  WORKING VIEW  (hidden state h):                            ▼
     h_{t-1}, x_t ──►[ compute f_t, i_t, g_t, o_t ]         h_t ──► output y_t  &  next step's gates
                     (each = σ or tanh of W·[h_{t-1}, x_t])
```

Reading it: the cell state `c` runs straight across the top, only *multiplied by the forget gate* and *added to* — no squashing, no weight matrix in its path. The hidden state `h` is produced at the end as a gated, `tanh`-squashed **read-out** of `c`, and it (with `x_t`) is what computes the four gates for the next step.

## Why two states: a memory vs. a working view

The single-state RNN forced one vector `h_t` to do **two jobs that pull in opposite directions**:

1. **Be a stable long-term memory** — persist information across many steps, which (from file `01`) demands *near-identity* dynamics so the gradient product doesn't vanish. A good memory wants to *not change*.
2. **Be the active working representation** — drive this step's output and the next step's decisions, which demands changing every step and being nonlinearly transformed and bounded. A good working state wants to *change a lot*.

These conflict directly: the `tanh` squashing and `W_h` multiply that make `h_t` a good *current representation* are exactly what destroy it as a *long-term memory* (that's the vanishing product). One vector can't be both.

The LSTM's insight is to **split the two jobs into two states**:

- **Cell state `c_t` = the memory.** Updated *only* additively (`f_t ⊙ c_{t-1} + i_t ⊙ g_t`) — never pushed through `tanh`, never multiplied by a full weight matrix on its way across time. So it can carry information and gradient across many steps nearly intact (the additive path → `∂c_t/∂c_{t-1} = diag(f_t) ≈ I`). It's the protected long-term channel, and it's allowed to be unbounded (accumulate) because nothing squashes it in transit.
- **Hidden state `h_t` = the working view.** `h_t = o_t ⊙ tanh(c_t)` — a gated, squashed *read-out* of the memory. It's what the rest of the network sees: it produces the output `y_t` and feeds the next step's gate computations. It's bounded (`tanh`) and free to change every step, because it's a *derived view*, not the thing carried across time.

Analogy: `c_t` is a **long-term notebook** — you selectively write and erase, but otherwise entries persist untouched; `h_t` is **what you say out loud right now** — a filtered, presentable summary of the notebook relevant to this moment, chosen by the output gate. The model uses its current working view (`h`) to decide what to remember, forget, and expose — but the memory's gradient highway lives on `c`, kept clear of the lossy `h` computation. That separation — *unbounded additive memory* + *bounded squashed read-out* — is the whole trick, and it's exactly the residual-stream vs. sublayer split you'll see next.

The load-bearing line is `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`. Look at the gradient it creates:

```
∂c_t/∂c_{t-1} = diag(f_t)
```

When the forget gate `f_t ≈ 1`, this is **≈ the identity** — the gradient flows back across that step multiplied by ~1, not by a sub-1 `W_h·tanh'`. Chain many such steps and you get `∏ diag(f_t) ≈ I` instead of a vanishing product. This is the **"constant error carousel"**: the cell state is a protected channel along which gradient (and information) travels across time almost undamped, as long as the gates hold it open. **GRU** (Cho, 2014) is the same idea with fewer gates (merges forget/input into one update gate, drops the separate cell state) — cheaper, usually comparable.

## The equivalence you should carry forward

Put the LSTM cell state next to a residual block and the pre-attention/FFN sum:

```
LSTM cell (across time):   c_t   = f_t ⊙ c_{t-1} + i_t ⊙ g_t        f_t ≈ 1  →  c_t ≈ c_{t-1} + (new)
Residual block (depth):    h_{l+1} =        h_l     +   f_l(h_l)                  h_{l+1} =  h_l    + (new)
```

Same move: replace a **multiplicative** transform (which makes gradient a vanishing product) with an **additive** carry (which makes it `identity + correction`, a sum). The gate just makes the LSTM's carry *learnable* — it can choose to forget. A ResNet hard-wires the carry fully open (coefficient 1); a Highway network keeps it learnable like the LSTM. [3.1/01](../../part3_residual_connections_deep_networks/3.1_skip_connection/01_degradation_problem.md) states this lineage from the ResNet side; this is the same claim from the LSTM side. The additive path is *the* recurring cure for the product-of-Jacobians disease, whether the product runs over time or over depth.

### Why not drop the forget gate entirely — pure addition, like a residual stream?

Natural question given the analogy: if a residual stream is *pure* addition (`h_{l+1} = h_l + f_l`, no gate), why doesn't the LSTM just use `c_t = c_{t-1} + i_t ⊙ g_t` and drop the forget gate? For *gradients* it would be even better — `∂c_t/∂c_{t-1} = I` exactly, a perfect highway. And this isn't hypothetical: it's **the original 1997 LSTM.** The forget gate was added in 2000 (Gers et al., "Learning to Forget") precisely because pure accumulation *failed* on long/continual sequences — the cell grew unbounded and the `tanh(c_t)` read-out saturated.

The reason is functional, not about gradients: a bounded memory over an **unbounded** timeline must be able to *evict* stale information (a clause ends, the subject changes, a list resets), and pure addition can only ever *add* — old content piles up as interference forever in a fixed-`D` space. The forget gate is **eviction**, and multiplicative gating makes "clear this slot" cheap and exact (`f_t → 0`), whereas pure addition could only erase by writing an `i_t ⊙ g_t` that *exactly cancels* `c_{t-1}` — a brittle target (same asymmetry as identity-is-easy-for-residual, file [3.1/01](../../part3_residual_connections_deep_networks/3.1_skip_connection/01_degradation_problem.md)).

So why does the residual stream get away with no forget gate? Two disanalogies: depth is **bounded** (fixed `L` ~ tens-to-100, so variance stays finite and the pipeline *ends* — no runaway accumulation), and a layer *refines one token* rather than *maintaining an evolving memory* where relevance expires. Forgetting is the price of compressing history into a **bounded, fixed-size state over unbounded time**: the LSTM pays it (the gate), the residual stream avoids it (bounded depth), and the Transformer avoids it differently (it refuses to compress — the KV cache keeps every token, so nothing needs evicting, at `O(S)` memory cost, Part 9.2). Mamba (Part 7.3) keeps a bounded state and so brings forgetting back as a state-space **decay**.

## Why it matters in modern LLM work

- **It's the direct ancestor of the residual stream.** The intuition "keep an undamped additive channel and let components add to it" — the whole of Part 3 — is the LSTM cell state generalized to depth. Your LSTM knowledge *is* residual-stream knowledge.
- **Gating came back.** Mamba / selective SSMs (Part 7.3) are essentially input-dependent gated recurrences — the LSTM's `f_t`, `i_t` reborn as data-dependent state-space parameters, but with a training pass that parallelizes (the thing the LSTM couldn't do, file `03`). SwiGLU's gate ([2.1/02](../../part2_neural_network_fundamentals/2.1_mlp_building_block/02_activations.md)) is the same multiplicative-gate motif inside the FFN.
- **Gating didn't fully solve long range.** Even LSTMs degrade over hundreds-to-thousands of tokens — the carousel leaks, and a fixed-size `c_t` still can't store unboundedly much. That residual failure is reason #2 they lost (file `03`), and the reason attention's *direct* access won.

## Self-check

1. In `∂h_t/∂h_k = ∏ diag(φ') W_h`, name the two factors that each push the product toward zero, and why only a knife-edge avoids both vanishing and exploding.
2. Which single LSTM quantity creates the near-identity gradient path across time, and what value must it take for the path to stay open?
3. Write the LSTM cell update and a residual block side by side and state, in one sentence, the structural move they share.
4. LSTMs fixed vanishing gradients but still lost to Transformers on long-range tasks. Give the reason gating *doesn't* fully solve long-range dependence.

### Answers

1. `W_h` raised to the step-distance (spectral radius `<1` vanishes, `>1` explodes) and `tanh' ≤ 1` (further sub-1 shrinkage, worse under saturation). Surviving requires the product of all these factors to stay near 1 across many steps — an unstable knife-edge SGD won't maintain.
2. The forget gate `f_t`, via `∂c_t/∂c_{t-1} = diag(f_t)`. It must be `≈ 1` for the cell state to carry gradient/information across the step nearly undamped (the constant error carousel).
3. `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t` vs. `h_{l+1} = h_l + f_l(h_l)`: both replace a multiplicative transform with an **additive carry**, turning a vanishing product of Jacobians into an `identity + correction` sum — the LSTM does it across time with a learnable gate, the ResNet across depth with the gate hard-wired open.
4. The carry is gated (can leak/forget) and, crucially, the state `c_t` is **fixed-size** — it must compress *all* relevant history into `D` numbers, so distant information competes for bounded capacity and gets overwritten. Attention sidesteps both by keeping every past state directly addressable, with no compression and an `O(1)`-length gradient path to any position.

## Exercise

Consider the cell-state gradient `∂c_T/∂c_0 = ∏_{t=1}^{T} f_t` (scalar, one dimension). (a) If the model needs to carry a signal 100 steps with no loss, what must every `f_t` be, and what is the product? (b) Suppose each `f_t = 0.95` instead (a slightly leaky gate). Compute the product at `T = 100`. (c) Contrast with a vanilla RNN whose per-step factor is `0.95` for the *same* reason — is the LSTM actually better here, and what does that tell you about *why* the win is the gate being able to learn `f_t ≈ 1` rather than the number 0.95 itself? (d) One sentence: how does this motivate attention's `O(1)` path over *any* gated recurrence?
