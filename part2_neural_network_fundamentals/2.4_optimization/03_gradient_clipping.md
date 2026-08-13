# Gradient Clipping

A single line of code that turns "Transformer training sometimes NaNs" into "Transformer training basically never NaNs." Universal in modern LLM stacks. Skipping it is reckless at any nontrivial scale.

## What it does

After computing all parameter gradients but **before** the optimizer step:
```
total_norm = √(Σ_p ‖∂L/∂p‖²)                  # L2 norm of the full flattened gradient
if total_norm > max_norm:
    scale = max_norm / total_norm
    for p in params: p.grad *= scale
```

If the gradient is too large, scale it down so its L2 norm equals `max_norm`. If it's already smaller, leave it alone.

PyTorch one-liner:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Typical `max_norm`: **1.0** for LLM pretraining. Some setups use 0.5 or even 0.1; rarely above 5.0.

## Why this matters

The fundamental issue: Adam's second moment can't react fast enough to a sudden gradient spike.

In normal training, `v_t` reflects typical gradient magnitudes, and `m_t / √v_t` produces a unit-ish-scale update direction. Then a single batch happens to produce a 50× larger gradient than usual (an outlier batch, a sharp region of the landscape, an instability in attention). The new big gradient gets added to `m_t` and `v_t`, but with EMA weights of `(1-β1) = 0.1` and `(1-β2) = 0.001` — only a small fraction shows up immediately. So the update `m_t / √v_t` becomes much larger than usual, the parameters take a huge step, and you land somewhere catastrophic.

Clipping shorts this out. The gradient is hard-capped at `max_norm` before going into the optimizer; the EMA only ever sees normal-scale gradients; the update stays in the normal range. The bad batch's signal is preserved (the gradient direction is still that of the bad batch), just at controlled magnitude.

This is why it's clipping by **norm** (preserves direction) rather than **value** (clips each coordinate separately, which distorts direction). Direction is the useful signal; magnitude is what causes blowups.

## Global vs. per-parameter

The standard is **global** clipping: compute the norm of the entire flattened gradient (concatenation of all parameter gradients), and apply the same scale to everything.

Why global, not per-parameter:
- The gradient is a single vector in the high-dimensional parameter space. Its direction encodes a *coordinated* update across all parameters. Per-parameter clipping would distort this direction by scaling some coordinates and not others.
- Per-parameter clipping has been tried; it's strictly worse empirically. Mostly a non-starter today.

`clip_grad_norm_` does global by default. `clip_grad_value_(model.parameters(), clip_value=1.0)` does element-wise clipping — don't use this for LLMs.

## When clipping fires (and when it doesn't)

In a healthy training run, clipping should fire **rarely** — say <5% of steps. Most steps the natural gradient norm is below `max_norm` and clipping is a no-op.

If clipping is firing on most steps, your `max_norm` is too low — you're systematically clipping the natural gradient magnitude, which slows learning. Either raise `max_norm` or your LR is too high.

If clipping never fires, your `max_norm` is too high — clipping isn't doing anything. Either lower it or your training is exceptionally well-behaved.

Logging `total_norm` (before clipping) every step is one of the most useful debug signals during training. Sudden spikes are early warnings of impending NaN.

## Why clipping by norm is the right choice (vs. other options)

You could imagine several ways to bound gradients:
- **Hard cap on per-coordinate value** (`clip_grad_value_`): distorts direction, weakly bounds total magnitude.
- **Soft saturation** (`tanh(g / s) · s`): smoothly limits magnitude, but distorts direction non-linearly and is harder to reason about.
- **Norm clip** (the standard): one threshold, preserves direction exactly, doesn't act unless needed.

Norm clip is the only one with the property "if the gradient is in the normal range, do nothing; if too big, scale to the boundary." This minimal-intervention property is what makes it safe to deploy as default behavior.

## Where it lives in the training loop

```python
optimizer.zero_grad()
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Order matters: clipping must be **after** backward (so gradients exist) and **before** `step()` (so the optimizer sees clipped gradients). The return value `grad_norm` is the pre-clipping norm — log it.

With mixed precision (file `05`) and a `GradScaler`:
```python
optimizer.zero_grad()
with autocast(dtype=torch.bfloat16):
    loss = model(x, y)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)                       # unscale before clipping!
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

`unscale_` is essential — otherwise you'd be clipping the **scaled** gradients (which are huge by design), and clipping would over-clip everything.

## Clipping ≠ regularization

Clipping is a **safety** mechanism, not a regularizer. It changes training dynamics only on the rare steps that would otherwise blow up. It does not, by itself, improve generalization or constrain model complexity. Don't conflate it with weight decay or dropout.

## When you might tune `max_norm`

For most LLM training, `max_norm = 1.0` works. Cases where you'd tune:

- **Very small models** (< 10M params): natural gradient norms are smaller, so `max_norm = 1.0` may never fire. Drop to 0.5 or 0.25, or just leave at 1.0 — doesn't matter.
- **Very large batches**: large batches → smaller-scale per-token gradients but the *total* gradient norm depends on your reduction (mean vs. sum). With mean reduction, the gradient norm is roughly batch-size-independent; with sum, it grows with batch size and you'd need a bigger `max_norm`. Always check.
- **Training instability**: if you're seeing loss spikes despite clipping, lower `max_norm` to 0.5 or 0.25. This more aggressively dampens transients at the cost of slightly slower learning on benign steps.

## Related but different: gradient noise scaling

Some setups add Gaussian noise to gradients (`g += ε · noise`). This is a **regularizer**, not a clip. Used in some privacy-preserving training (DP-SGD) and some experimental work. Not standard for LLMs.

## Self-check

1. Why is gradient clipping by norm preferred over per-coordinate clipping for Transformer training? What property would per-coordinate clipping break?
2. With mixed-precision training and a `GradScaler`, why must you call `scaler.unscale_(optimizer)` before clipping? What goes wrong if you don't?
3. You're training a Transformer and notice clipping is firing on 80% of steps. What does this tell you, and what should you change?

### Answers

1. The gradient is a single direction in the model's parameter space — it tells the optimizer *which way* to move and *how far*. Norm clipping scales the entire gradient by a single scalar, which preserves the direction exactly. Per-coordinate clipping clips each parameter's gradient independently, distorting the direction: parameters whose individual gradients were large get clipped, parameters whose individual gradients were normal don't. The resulting "clipped gradient" no longer points where the unclipped gradient pointed. For Transformers (and most DL), this directional distortion produces measurably worse training. Norm clipping is "scale only when needed, preserve direction always" — minimal intervention.
2. With a `GradScaler`, gradients in the backward pass are scaled up by a large factor (e.g. 2^16) to prevent underflow in fp16. The actual gradient w.r.t. the unscaled loss is what you want to clip — that's at the natural scale. If you clip the *scaled* gradients with `max_norm = 1.0`, you're effectively clipping the true gradients to `1.0 / scale_factor ≈ 1.5e-5` — astronomically small — every step. Training would never make progress. `unscale_` reverses the scale-up, restoring gradients to their true magnitude, *then* you clip, *then* the optimizer step uses the clipped (true-scale) gradients. With bfloat16, `GradScaler` isn't needed (bf16 has the same exponent range as fp32), and the unscale dance is skipped.
3. The natural gradient norm is consistently above `max_norm`. Either: (a) `max_norm` is set too low for your batch size / LR — try raising it (e.g. to 2.0 or 5.0); or (b) your learning rate is too high, and the gradients are systematically large because the model is in a regime where they should be large but the LR can't handle it — lower the LR. Either fix is reasonable. Avoid the situation: training where clipping fires on most steps is essentially training with a fixed effective step size unrelated to the gradient — the optimizer signal is being constantly distorted, and the model is learning under unfavorable conditions. Healthy training has clipping firing on < 5% of steps.

## Exercise

Train a 6-layer Transformer on a tiny LM task with three setups:
1. No gradient clipping.
2. `max_norm = 1.0`.
3. `max_norm = 0.01` (artificially aggressive).

Log `grad_norm` (pre-clipping) every step. Plot histograms.

Expect: (1) occasional spikes, possible NaN; (2) histogram has a hard tail at 1.0 — clipping fires occasionally; (3) histogram piles up at 0.01 — clipping always fires, training slow. Setup (2) is the right call. This makes "clipping is a safety net, not an active intervention" concrete.
