# Common Gradient Pathologies

Backprop is just the chain rule. The pathologies all come from the same place: a multiplicative chain of `L` numbers, each contributed by one layer's local Jacobian (its weight matrix's singular values, its activation's slope). If the typical factor is < 1, the product collapses to 0; if > 1, it explodes. Modern architectures fix this with residuals + normalization, but understanding the failure modes is what tells you when something has gone wrong in training.

## Vanishing gradients

**Symptom**: gradients at deep layers (far from the loss) are exponentially smaller than gradients at shallow layers. Early layers don't learn — they sit at their init values while late layers train.

**Cause**: at each backward step through a layer, the gradient is multiplied by the layer's local Jacobian. If `‖J‖ < 1` consistently, the gradient shrinks geometrically:
```
‖∂L/∂h_l‖ ~ ‖J_l‖ · ‖J_{l+1}‖ · ... · ‖J_L‖ · ‖∂L/∂h_L‖
```
For `L` layers with average per-layer norm `r`, the gradient at layer 1 is `r^L` times the gradient at the loss. With `r = 0.5` and `L = 50`, that's `10^{-15}`. Numerically zero.

**Where it comes from**, historically:
- **tanh and sigmoid activations**: their derivative is bounded by 1 (tanh: max 1 at origin) or 0.25 (sigmoid). Stacking through many such layers guarantees shrinkage. This is what killed pre-2010 deep networks.
- **Poorly initialized weights**: if weight matrices have `‖W‖ < 1` (say all small), Jacobians shrink even with ReLU.
- **No residual connections**: every backward path is forced through every layer's full nonlinearity.

**What fixes it in modern architectures**:
- **Residual connections** (Part 3): the gradient has a direct path that just copies the upstream signal, avoiding the multiplicative chain entirely.
- **Normalization layers** (file 2.3/03): keep activations at unit variance, which keeps Jacobians at roughly unit norm.
- **ReLU / GELU / SiLU**: derivative is 1 (ReLU) or close to 1 (GELU, SiLU) on the active region — no shrinkage along that path.

In modern Transformers, vanishing gradients are mostly a solved problem — but residual *streams* still develop weird gradient distributions when one layer's contribution dominates; that's where pre-norm vs. post-norm comes in (file 2.3/04).

## Exploding gradients

**Symptom**: gradient norm spikes — sometimes to NaN — at a particular step. Loss becomes Inf or NaN. Training trashed.

**Cause**: opposite of vanishing. Local Jacobians with norm > 1, stacked across layers, give an exponentially large gradient. Single trigger events:
- A particularly bad batch (outlier inputs, mislabeled data).
- An attention softmax saturating in a weird way that produces sharp gradients.
- A learning-rate transient during warmup.

**Why Transformers are especially susceptible**: the attention matmul `Q K^T / √d` is unbounded above. Even with normalization, the QK product can produce large scores that interact badly with softmax. Some attention heads can develop very large weights at scale. RMSNorm/LayerNorm bounds the *input* magnitude but not the *output* magnitude after the matmul.

**What fixes it**:
- **Gradient clipping** (file 2.4/03): hard ceiling on the global gradient norm. Non-optional for Transformer training.
- **bfloat16** instead of float16 (file 2.4/05): wider dynamic range tolerates larger transient magnitudes without overflow.
- **Warmup** at the start of training (file 2.4/02): start with a tiny learning rate so early-step transients can't trigger blow-ups.
- **QK normalization**: some recent LLMs apply RMSNorm to Q and K before the attention matmul, bounding the score magnitudes.

## Dead ReLUs (already covered in file 2.1/02, recapped here)

**Symptom**: a fraction of hidden units output exactly 0 for every input in the dataset. They contribute nothing forward and receive zero gradient backward.

**Cause**: ReLU's derivative is exactly 0 for `x < 0`. If a unit's pre-activation drifts below 0 across all inputs, the gradient through it is 0, the weights don't update, the pre-activation stays negative. Permanent.

**Triggers**: too-large learning rate (a single bad step can flip many units into the dead region), bad init (units start dead), aggressive weight decay.

**Modern mitigations**: GELU / SiLU have nonzero derivatives everywhere, so this pathology is essentially gone in modern Transformer FFNs. Still see it in older code or small networks using ReLU.

## Loss spikes

**Symptom**: loss flatlines or trends down for thousands of steps, then suddenly jumps by 10× or more. May recover, may not. Often associated with a gradient norm spike just before.

**Cause family**: a perfect storm of (a) the optimizer state (Adam's second moment) becoming stale, (b) the loss landscape having a sharp cliff in some direction, (c) a single batch with unusually large gradients pushing the parameters off a cliff. Adam-family optimizers' second-moment EMA can be slow to react to sudden gradient growth, so the update is taken before the second moment has caught up — overshooting and landing in a bad region.

**At frontier scale this is a major operational concern.** Multi-week training runs of 100B+ parameter models have loss spikes; engineers maintain checkpoints every few hours so they can roll back, lower the LR, and continue. There's a whole subgenre of papers on stabilizing pretraining (μP, DeepNorm, Sandwich-LN, embedding norm tricks).

**Mitigations**:
- Gradient clipping (catches the immediate spike).
- Skip the bad batch (some training stacks detect anomaly-loss-batches and discard them).
- Lower max learning rate.
- More aggressive warmup.
- Better init (μP — see file 2.3).

## Gradient noise scale / poor signal-to-noise

**Symptom**: large gradient variance across minibatches relative to gradient magnitude. Training requires very large batches or many gradient accumulation steps to get a stable signal.

**Concept**: McCandlish et al. (2018) defined the **critical batch size** — beyond which adding more samples per batch gives diminishing returns. Below it, you're computing a noisy estimate of the true gradient; above it, you're past the point where averaging more samples helps.

Modern frontier training uses batch sizes in the millions of tokens — well past most models' critical batch size — because larger batches enable more parallelism, even when per-token compute efficiency drops.

This is more of a scaling concern than a training-stability concern, but it's part of the same "gradient quality" story.

## Symptoms ↔ causes cheat sheet

| Symptom | Most likely cause | First fix to try |
|---|---|---|
| Loss not decreasing at all | Bad init, vanishing gradients, dead units | Check init scheme; check activation distributions; check `requires_grad` |
| Loss decreasing very slowly in deep layers | Vanishing | Add residuals, switch to pre-norm, check init |
| Loss → NaN suddenly | Exploding gradient, fp16 overflow, log(0) somewhere | Gradient clipping, switch to bf16, check loss formula |
| Loss → NaN gradually | Numerical instability in softmax / log | Use log-softmax, log-sum-exp tricks |
| Loss plateaus then spikes | Adam second-moment lag, sharp landscape | Lower max LR, more warmup, gradient clipping |
| Some activations always 0 | Dead ReLUs | Switch to GELU/SiLU, smaller LR, better init |
| Tiny gradient norm globally | Vanishing gradient through stack | Pre-norm, residuals, init |

## What "healthy" gradient norms look like

For a well-trained Transformer mid-training:
- Global gradient L2 norm: typically O(1) to O(10), roughly stable across training.
- Per-parameter gradient distributions: roughly bell-shaped, no extreme outliers.
- Activation norms: bounded across layers, roughly constant in magnitude through the residual stream.
- No layer dominating the gradient norm by 10×+.

Anything wildly off these is a sign something is wrong upstream — init, normalization placement, or numerical precision.

## Debugging gradients in practice

The fastest checks to run when training looks wrong:

1. **Print `grad_norm`** at every step. Sudden spikes → exploding gradients. Steady decay → vanishing or dead units. Healthy: stable around some O(1) value.
2. **Per-layer gradient norms**. If layer-1 grad is `10^{-10}` and layer-32 grad is `10^{-2}`, you have vanishing. If layer-1 is 1000× layer-32, something's wrong with normalization.
3. **Activation statistics per layer**. Mean ≈ 0 and std ≈ 1 (or ≈ `γ` for normed layers) at every layer is the healthy baseline. Drift indicates instability.
4. **Loss curve smoothness**. Smooth monotone decrease = healthy. Plateau + spike = optimizer/landscape issue. Slow descent that never speeds up = optimization is stuck.

## Self-check

1. Why are residual connections (Part 3) such a powerful fix for vanishing gradients? Be specific about what the gradient path looks like with vs. without them.
2. You see loss → NaN at step 4500 of a Transformer training run. Walk through the diagnostic checks you'd run, in order.
3. Why does pre-norm (file 2.3/04) help with gradient flow more than post-norm in deep stacks, and why does this matter for the "exploding gradient" failure mode specifically?

### Answers

1. With residuals, the forward is `h_{l+1} = h_l + f_l(h_l)`. The backward through this is `∂L/∂h_l = ∂L/∂h_{l+1} · (I + ∂f_l/∂h_l)`. The `I` term means the upstream gradient flows directly back to `h_l` *unchanged* — no multiplicative shrinkage. Even if `∂f_l/∂h_l` is tiny or zero (vanishing through `f_l`), the `I` gives a direct path. Without residuals, every layer's backward goes through the full local Jacobian and the gradient is a `L`-deep product of those Jacobians — geometric decay. With residuals, you can stack 100+ layers and gradients still reach layer 1 with O(1) magnitude. This is the single most important architectural change of the post-2015 era.
2. (a) Re-run with grad-norm logging and find which step's grad-norm spiked. (b) Look at the batch at that step — outlier inputs? bad labels? (c) Look at per-layer grad norms — is one layer dominating? (d) Check if you're using fp16 (consider switching to bf16). (e) Verify gradient clipping is on. (f) Confirm warmup completed correctly. (g) Re-check loss formula for `log(0)` or `div by 0` cases (label smoothing helps here). (h) Reduce LR by 2× and resume from a checkpoint pre-spike. At frontier scale, (h) is what people end up doing — diagnosing exactly why is often not possible, but rolling back and continuing with lower LR usually works.
3. **Post-norm** applies LayerNorm *after* the residual add: `h_{l+1} = LN(h_l + f_l(h_l))`. The norm operates on the residual stream's output, so the gradient backward through it is shaped by the LN Jacobian — which is bounded but introduces multiplicative factors at every layer. **Pre-norm** applies LayerNorm *before* the sublayer: `h_{l+1} = h_l + f_l(LN(h_l))`. The residual `h_l` flows backward without passing through any LN — a clean identity path. So the cumulative gradient signal stays close to O(1) regardless of depth. For exploding gradients specifically: post-norm at large depths can produce gradient norms that grow with depth (each LN's Jacobian can amplify in some direction), making spikes more likely. Pre-norm bounds this by design. Hence: deep stacks (50+ layers) reliably train only with pre-norm.

## Exercise

In a notebook, train two small Transformers on a tiny LM task: one with 4 layers, one with 32 layers. For each, log:
- Per-layer gradient L2 norms at every step.
- Activation L2 norms at every layer at every step.

For the 4-layer model both should be stable. For the 32-layer model, depending on your init and norm choices, you should see one of three things:
- Vanishing: layer-1 grad shrinks toward 0.
- Exploding: layer-1 grad grows toward ∞.
- Healthy: stable across depth (this is what you want — and what modern init + pre-norm gives you).

Try toggling pre-norm vs. post-norm and watch the gradient profile change.
