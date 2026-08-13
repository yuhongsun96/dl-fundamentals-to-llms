# The Gradient Highway

The previous file argued the skip connection reshapes the *optimization landscape* (identity becomes free). This file gives the second, complementary view: what the skip connection does to *gradients*. The two are the same phenomenon seen from opposite ends of the backward pass. [2.2/05](../../part2_neural_network_fundamentals/2.2_backpropagation/05_gradient_pathologies.md) already stated the punchline in its self-check; here we make it the whole point and connect forward and backward into one picture.

**Convention note:** we work with the per-token residual vector `h_l ∈ R^D` (one position, one layer). `f_l` is a sublayer (attention or FFN, with its norm folded in). Jacobians are `D × D`; gradients `∂L/∂h_l` share `h_l`'s shape ([NOTATION.md](../../NOTATION.md) invariant).

## The two-line derivation

A residual block, forward:

```
h_{l+1} = h_l + f_l(h_l)
```

Differentiate the block output with respect to its input:

```
∂h_{l+1}/∂h_l = I + ∂f_l/∂h_l
                ▲       ▲
             identity   the sublayer's Jacobian
```

That `I` is the entire story. Now push a loss gradient backward through the block (chain rule, VJP form — [1.2/02](../../part1_math_foundations/1.2_calculus_probability/02_jvp_vjp.md)):

```
∂L/∂h_l = ∂L/∂h_{l+1} · (I + ∂f_l/∂h_l)
        = ∂L/∂h_{l+1}  +  ∂L/∂h_{l+1} · ∂f_l/∂h_l
          ▲                ▲
     copied straight through   routed through the sublayer
```

The upstream gradient `∂L/∂h_{l+1}` arrives at `h_l` **unchanged**, plus a correction term routed through `f_l`. Even if the sublayer's Jacobian `∂f_l/∂h_l` is tiny (a saturated nonlinearity, a near-zero weight matrix), the gradient still reaches `h_l` at full strength via the `I` path.

## Why "highway"

Unroll the identity term across all `L` layers. The gradient from the loss to the *first* block's input, keeping only the pure-identity path, is:

```
∂L/∂h_0  ⊇  ∂L/∂h_L · I · I · ⋯ · I  =  ∂L/∂h_L
```

There is a path from the loss to *every* layer that is a product of identity matrices — **no attenuation, no amplification, at any depth.** Contrast the plain network, whose only path is the full product of Jacobians:

```
plain:     ∂L/∂h_0 = ∂L/∂h_L · J_L · J_{L−1} · ⋯ · J_1     # geometric: r^L → 0 or ∞
residual:  ∂L/∂h_0 = ∂L/∂h_L · Π(I + J_l)                   # expands to 1 + (sum of paths)
```

Expanding `Π(I + J_l)` gives `I` (the clean highway) plus every product of *some subset* of the `J_l`'s. The `I` term guarantees the gradient never vanishes; the subset-products are the corrections. This expansion is also exactly the "ensemble of paths" that the [next file](03_ensemble_of_shallow_paths.md) reads *forward* instead of backward — same algebra, two directions.

## Forward and backward are the same picture

The highway runs both ways, and it's worth holding both in your head at once:

| Direction | What flows on the identity path | Consequence |
|---|---|---|
| **Forward** | The accumulated activation `h_l` — sum of all earlier contributions, undisturbed | Layer 50 can *read* what layer 5 wrote; information isn't forced through 45 transforms ([04](04_residual_stream_as_abstraction.md)) |
| **Backward** | The loss gradient `∂L/∂h`, copied to every depth without a Jacobian factor | Layer 1 gets a training signal as strong as layer `L`; no vanishing across depth |

The forward highway is *why the residual stream exists* as a coherent object; the backward highway is *why the whole thing trains*. Both are the single `+ h_l` term. This is the payoff of the [degradation-problem](01_degradation_problem.md) fix: not just "identity is representable" but "identity is the gradient's default route home."

## What the highway does *not* fix

The identity path solves vanishing gradients across depth. It does **not** by itself solve:

- **Exploding gradients / activation growth.** The `Π(I + J_l)` expansion has `2^L` terms; if the `J_l` contributions are large and aligned, the sum can still blow up. And on the forward side, each block *adds* to the stream, so the stream's magnitude grows with depth — the variance-`~L` problem that [3.2/02](../3.2_normalization_and_depth/02_scaling_the_residual_stream.md) is entirely about. Residuals trade "vanishing" for "controlled growth," and the growth still needs managing (norm + init + output scaling).
- **The Jacobian of `f_l` itself.** Inside the sublayer, a saturated softmax or a bad weight matrix still produces poor local gradients. The highway routes *around* that damage for the deep-credit-assignment signal, but the sublayer still has to learn.

So residuals are necessary, not sufficient. They're paired with normalization *placement* ([3.2/01](../3.2_normalization_and_depth/01_normalization_placement_recap.md)) precisely because the placement determines whether that clean identity path stays clean — pre-norm keeps `h_l` off the norm's Jacobian; post-norm routes the highway *through* a norm at every layer and partially spoils it.

## Why this matters for modern NLP/LLM work

- It's the reason "grad norm stable and O(1) across all layers" is the healthy signature in [2.2/05](../../part2_neural_network_fundamentals/2.2_backpropagation/05_gradient_pathologies.md)'s debugging table — the highway is what makes deep-layer and shallow-layer gradients comparable.
- The `I + J` structure is exactly what pre-norm preserves and post-norm degrades — you can't reason about norm placement without this file's picture ([3.2/01](../3.2_normalization_and_depth/01_normalization_placement_recap.md)).
- "Gradient reaches every layer directly" is why you can fine-tune *any* layer of a pretrained LLM, why LoRA adapters at any depth get signal, and why very deep stacks (100+ layers) became routine rather than heroic.

## Self-check

1. Write the backward pass through one residual block and identify the term that prevents vanishing gradients. What is that term's value, exactly?
2. In the plain network the gradient to layer 1 is a product of `L` Jacobians; in the residual network it's `Π(I + J_l)`. Expand the residual product conceptually — what are the two kinds of terms, and which one is the "highway"?
3. Residuals fix vanishing gradients but not exploding gradients / activation growth. Explain why the same `+ h_l` that saves the backward pass *creates* a forward-pass magnitude problem.
4. Forward and backward "highway" are the same `+ h_l` term. State what rides the highway in each direction and the architectural payoff of each.

### Answers

1. `∂L/∂h_l = ∂L/∂h_{l+1} · (I + ∂f_l/∂h_l) = ∂L/∂h_{l+1} + ∂L/∂h_{l+1}·∂f_l/∂h_l`. The vanishing-preventing term is `∂L/∂h_{l+1}` (the first summand) — it's the upstream gradient copied through with coefficient exactly `I` (the identity), independent of the sublayer's Jacobian. Even if `∂f_l/∂h_l = 0`, the gradient still passes.
2. Expanding `Π_l (I + J_l)` gives `I` plus every product of a non-empty subset of the `J_l`'s. The `I` term — the product where you pick the identity at *every* layer — is the highway: a Jacobian-free path from loss to every depth. The subset-products are the "correction" paths through various combinations of sublayers. (This is the ensemble-of-paths decomposition read backward.)
3. The `+ h_l` copies the upstream gradient back with no attenuation — good for backward. But on the forward pass the *same* structure means each block's output `f_l(h_l)` is *added* to `h_l` rather than replacing it, so magnitudes accumulate: `Var(h_L) ≈ Var(h_0) + Σ Var(f_l)` grows roughly linearly in `L`. Residuals convert a *multiplicative* vanishing problem into an *additive* growth problem — milder, but not zero, hence norm + `1/√(2L)` output scaling ([3.2/02](../3.2_normalization_and_depth/02_scaling_the_residual_stream.md)).
4. **Forward**: the accumulated activation `h_l` (the running sum of all prior contributions) rides the highway, so any later layer can read any earlier layer's output directly — the residual stream. **Backward**: the loss gradient rides the highway, reaching every layer at full strength — deep credit assignment without vanishing. Same `+ h_l`, both payoffs.

## Exercise

Take a 3-block residual chain `h_3 = h_2 + f_3(h_2)`, `h_2 = h_1 + f_2(h_1)`, `h_1 = h_0 + f_1(h_0)`. (a) Write `∂h_3/∂h_0` by the chain rule and expand the product `(I+J_3)(I+J_2)(I+J_1)` into its 8 terms. (b) Identify the single term that survives when all `J_l → 0`. (c) Now suppose you insert a LayerNorm *after* each add (post-norm): `h_{l+1} = LN(h_l + f_l(h_l))`. Explain in words why the clean `I·I·I` term from (b) no longer appears — what replaces each `I`? Relate your answer to why post-norm degrades at depth ([3.2/01](../3.2_normalization_and_depth/01_normalization_placement_recap.md)).
