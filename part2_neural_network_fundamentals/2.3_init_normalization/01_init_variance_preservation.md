# Initialization: Why Variance Preservation Matters

## The setup

A linear layer maps `x ∈ R^(d_in) → y ∈ R^(d_out)` via `y = W x + b`. The question: **what should the entries of `W` be at step 0**?

You can't initialize to all zeros — every neuron would compute the same thing forever (no symmetry breaking). You can't initialize to large random values — activations explode. You can't initialize to tiny random values — activations vanish. The right answer is "random but at exactly the magnitude that preserves the activation variance across the layer."

This is the entire content of every initialization paper. Different schemes (Xavier, Kaiming, μP) differ in *how* they preserve variance, but the goal is always the same.

## What "variance preservation" means

Treat the input components `x_i` as i.i.d. random variables with mean 0 and variance `σ_x²`. We want the output components `y_j` to also be mean-0 and variance `σ_x²` — same scale as the input.

Compute `Var(y_j)` for `y_j = Σ_i W_{ji} x_i`. Assuming `W_{ji}` i.i.d. with mean 0 and variance `σ_W²`, independent of `x_i`:
```
Var(y_j) = Σ_i Var(W_{ji} x_i) = d_in · σ_W² · σ_x²
```

For `Var(y_j) = σ_x²` to hold, we need:
```
σ_W² = 1 / d_in
```

That's the rule. **Initialize `W` with variance `1 / fan_in`.** Then the forward pass preserves activation variance layer by layer.

In words: the more inputs each neuron sums (`d_in`), the more those random terms pile up, so make each weight smaller — variance `1/d_in` — to exactly cancel the pile-up. A layer with 1000 inputs uses smaller weights than one with 100. (The terms pile up like `d_in` rather than `d_in²` because they're mean-0: positives and negatives partially cancel, so a sum of `n` random terms grows like `√n` in magnitude, i.e. like `n` in variance.)

Equivalently, sample `W_{ji} ~ N(0, 1/d_in)` (Gaussian) or `~ Uniform(-√(3/d_in), √(3/d_in))` (uniform with matched variance).

## Why this matters

Without it, the activation variance compounds across layers:
- Variance > 1 per layer → activations grow exponentially with depth → overflow, saturation in any squashing nonlinearity, exploding pre-activations into softmax.
- Variance < 1 per layer → activations shrink exponentially → vanishing signal, dead nonlinearities, no useful gradient.

For a 50-layer network, even a 2× per-layer variance miss compounds to `2^50 ≈ 10^15` — astronomically bad. Init has to be approximately right; "approximately" is within ~2×, not within 10%.

The same logic in the **backward** direction. The gradient w.r.t. the input is `∂L/∂x = W^T (∂L/∂y)`. By the same calculation, `Var(∂L/∂x) = d_out · σ_W² · Var(∂L/∂y)`. To preserve gradient variance, we'd want `σ_W² = 1 / d_out`.

Forward says `1/fan_in`, backward says `1/fan_out`. These agree only when `fan_in = fan_out`. For non-square layers, you have to compromise — see file `02`.

## Accounting for activations

The variance-preservation derivation above assumed the layer was pure linear. Once you add a nonlinearity, the activation's effect on variance has to be folded in.

For a centered nonlinearity (`tanh` near origin), the slope is ≈ 1 and the variance is roughly preserved. For ReLU, half the inputs are killed (`x < 0 → 0`), and the variance of the output is **half** the variance of the pre-activation — a 2× shrinkage per layer. To compensate, you double the weight variance:
```
σ_W² = 2 / d_in     (Kaiming / He init, for ReLU)
```

For GELU and SiLU, the shrinkage factor is similar to ReLU (they suppress negative inputs by *most* of their magnitude), so the same `2/d_in` heuristic is used.

## What "broken" init looks like in practice

If you scale weights too small at init:
- Activations vanish — every layer's output is closer to zero than its input. By the time you reach the loss, the logits are tiny, the loss is near `log V`, and gradients are likewise tiny.
- Slow or no learning. Looks like "the model isn't training" with no obvious error.

If you scale weights too large at init:
- Activations explode — outputs grow with depth. Pre-activations into softmax (output layer, attention) saturate to one-hot. Gradients through the saturated regions are zero.
- Loss is huge at step 0 (model is confidently wrong on every token). May train through it; may not.

If you initialize bias to nonzero or weights to constants:
- All units compute the same function. Forever. The gradient is symmetric and stays symmetric.
- Symmetry-breaking failure. Fix: random init.

## The crucial role of residuals

In a residual network, `h_{l+1} = h_l + f_l(h_l)`. The forward pass adds layer contributions; the activation variance at the residual stream **grows additively** with depth, not multiplicatively.

If each `f_l` produces an output with variance `σ_f²`, then after `L` layers the residual-stream variance is roughly `σ_x² + L · σ_f²` (assuming independence of layer outputs from the stream, which is approximately true at init). For this to be O(1), you want either:
- `σ_f² = O(1)` and accept linear growth in depth, OR
- `σ_f² = O(1/L)` to keep total variance O(1).

Modern stabilization tricks (μP, DeepNorm, "scaled init") choose the second route: initialize the output projection of each block (`W_O` in attention, `W_down` in FFN) with extra `1/√L` damping. This keeps the residual stream at unit variance at init regardless of `L`. Without this, deep stacks at init have residual-stream variance proportional to `L`, which then has to normalize against — and as `L` grows, init becomes more delicate. The `1/√L` trick decouples that.

## What modern LLMs actually do

The common modern recipe:
- **Linear layers**: `W ~ N(0, σ²)` with `σ² = 2/d_in` (Kaiming-flavored) or sometimes the simpler `0.02` (GPT-2 / NanoGPT default for small embedding dim).
- **Output projections of residual sublayers** (`W_O`, `W_down`): scaled extra by `1/√(2L)` to dampen residual-stream variance growth.
- **Embeddings**: `W_E ~ N(0, σ²)` with `σ ≈ 0.02`. Crucial that this matches the per-token scale the model expects downstream — embeddings that are too large flood the residual stream, too small leave the model relying on positional info.
- **Bias and norm parameters** (`γ`, `β`): set to constants. `γ = 1`, `β = 0` for norms. Bias = 0 if you use it at all.

These are heuristics that work. The full theory (Maximal Update Parameterization, μP) prescribes init scales that allow you to transfer hyperparameters from a small "proxy" model to a large target model — a major operational win at frontier scale.

## μP in one paragraph

**Maximal Update Parameterization** (Yang et al., 2022) is a more careful init + LR scheme designed so that **hyperparameters tuned at a small width transfer to a large width without re-tuning**. The standard init scheme breaks at scale because the optimal LR shifts with width; μP scales LR per layer by `1/width` (more precisely, per parameter group) such that the effective update is width-independent. The practical impact at frontier scale: tune on a 1B-parameter model, deploy hyperparameters directly to a 1T-parameter model. Saves enormous amounts of compute.

## Self-check

1. Derive the `1/fan_in` rule from the assumption that input components are mean-0 i.i.d. with variance `σ²` and weights are mean-0 i.i.d. independent of inputs.
2. Why does Kaiming init use `2/fan_in` instead of `1/fan_in` for ReLU networks?
3. The output projection of each Transformer sublayer is sometimes initialized with extra `1/√L` damping. What pathology does this prevent, and why is the standard `1/fan_in` init not enough?

### Answers

1. `y_j = Σ_i W_{ji} x_i`. By independence, `Var(y_j) = Σ_i Var(W_{ji} x_i)`. By independence and zero means, `Var(W x) = E[W²] E[x²] - 0 = σ_W² σ_x²`. So `Var(y_j) = d_in · σ_W² · σ_x²`. Setting `Var(y_j) = σ_x²` gives `σ_W² = 1/d_in`. The key assumption is independence (between weights and inputs, and among weights). It holds at init when weights are freshly random — but breaks during training as weights become correlated with input distributions. The init scheme is still right because we only need it to hold at *step 0*.
2. Because ReLU kills half the input variance. `Var(ReLU(x))` for `x ~ N(0, σ²)` is `σ²/2` (the negative half contributes 0, the positive half contributes the full distribution's second moment scaled). To compensate, double `σ_W²` so that after the activation, variance is back to `σ²` at the next layer's input. Without this 2× compensation, signal halves per layer and you get vanishing activations through any deep ReLU stack. For GELU and SiLU, the variance shrinkage is similar (those activations also suppress negative inputs heavily), so `2/fan_in` is used there too.
3. Without damping, the residual stream's variance grows linearly with depth: `σ_stream² ≈ L · σ_layer²` at init. For deep stacks (50+ layers) this means the residual stream at later layers has 50× the variance of the input — which destabilizes any norm layer downstream (the LayerNorm's running scale has to adapt to a much larger input than it'll see later in training), and worse, breaks the assumption of unit-variance everywhere that the rest of the init relies on. The `1/√L` damping makes each sublayer's contribution `O(1/√L)`, so the sum over `L` layers is `O(1)` regardless of depth. This is the "init for depth" trick that GPT-2 and most successors apply.

## Exercise

Train a 30-layer MLP (ReLU, no residuals, no normalization) on MNIST-flat with three init schemes:
1. `W ~ N(0, 1)` — too big.
2. `W ~ N(0, 0.01)` — too small.
3. `W ~ N(0, 2/fan_in)` — Kaiming.

For each: at step 0, log the L2 norm of every layer's activation. Plot the norms vs. layer index on a log scale. You should see (1) exponentially exploding, (2) exponentially vanishing, (3) roughly flat. Then train each and compare convergence. Only the Kaiming-init model will actually learn.
