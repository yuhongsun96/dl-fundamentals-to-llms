# Loss Landscapes, Sharpness, and Large-Batch Training

The loss `L(θ)` is a scalar function of millions or billions of parameters. Optimization is the search for a low-loss region. The *shape* of the loss landscape — how flat, how sharp, how connected — controls whether a given optimizer can find a good solution and whether the solution generalizes.

This file is the lightest part of the curriculum's optimization section. Most of the theory here is hand-wavy intuition. The empirical observations, however, are load-bearing for understanding why frontier-scale training looks the way it does.

## What the loss landscape looks like

Old intuition (pre-2015): "the loss landscape is full of local minima; optimization is about avoiding bad ones."

Modern picture (post-2017): for over-parameterized neural networks, the loss landscape has:
- **Many flat regions** with similar loss values, connected by smooth paths.
- **Few sharp local minima**; most stationary points are **saddle points** (flat in some directions, curved in others).
- The "global minimum" isn't a single point — it's a high-dimensional manifold of solutions with identical (or near-identical) loss.

This shift was empirically driven: people found that with enough overparameterization, SGD just *works* — it doesn't get stuck in bad local minima, because the bad local minima largely don't exist. The optimization problem in modern DL isn't "avoid local minima"; it's "find a wide flat basin and stay there."

## Flat vs. sharp minima

Two minima with the same loss value can differ in their **curvature**:
- **Flat minimum**: the loss varies slowly in all directions around the minimum. Small parameter perturbations cause small loss changes.
- **Sharp minimum**: the loss spikes up rapidly in some directions. Small perturbations cause large loss changes.

Mathematically, this is the maximum eigenvalue of the Hessian `H = ∇²L(θ*)`. Small `λ_max` = flat; large `λ_max` = sharp.

**The empirical claim**: flat minima generalize better than sharp ones. The intuition: training data and test data come from slightly different distributions; the test loss landscape is a slight perturbation of the train loss landscape. At a flat minimum, the perturbation barely moves the loss — train loss and test loss agree. At a sharp minimum, even a small perturbation moves you up the cliff — train loss and test loss diverge (= bad generalization).

This isn't a theorem; it's a robust empirical regularity. There are pathological counterexamples (sharp minima that also generalize well due to scale invariance) but in practice, flat = good.

## SGD's implicit preference for flat minima

Stochastic gradient noise acts like a "thermal jiggle" in parameter space: each mini-batch's gradient is the true gradient plus noise. The noise prevents the optimizer from settling into sharp minima — small perturbations bounce it out — while flat minima are stable under perturbation.

This is one explanation for why SGD's solutions generalize well despite the absence of any explicit regularization toward flatness. The noise is the regularizer.

Adam has a similar implicit bias, though weaker because the per-coordinate normalization in Adam reduces the effective noise scale relative to SGD. This is one (weak) explanation for why models trained with Adam sometimes generalize slightly worse than SGD-trained models on small tasks — though for LLMs, AdamW is preferred for other reasons (file `01`).

## Large-batch training and sharpness

The catch: **large batches have less noise**, so they push toward sharper minima.

- With batch size 1, each step uses one noisy gradient — huge noise.
- With batch size 1M, the gradient is a very precise estimate of the true gradient — almost no noise.
- The optimizer in the large-batch regime is almost-deterministic: it follows the true gradient flow and converges to whatever minimum gradient flow leads to. This may well be a sharp one.

So you'd predict: **large-batch training generalizes worse than small-batch training, all else equal**. This was the "large batch problem" of the 2017-2019 era — large-batch training (used to exploit data parallelism) gave demonstrably worse test accuracy than small-batch training.

Fixes that emerged:
- **Larger LR with larger batches** (linear scaling rule up to a critical batch size): introduces back some of the missing noise via larger steps.
- **Longer warmup**: prevents the early-step instability that's worse with large batches.
- **LARS / LAMB**: layer-wise adaptive LR schemes that handle the large-batch regime better.
- **Sharpness-Aware Minimization (SAM)**: explicitly seek flat minima by computing the gradient at a perturbed point.

For LLMs specifically, the large-batch problem is less acute because the loss landscape of language modeling seems to be flatter than image classification's. Frontier LLMs use batches of millions of tokens routinely without obvious generalization penalty. But it's not nothing — Llama 3's 16M-token batch is at the edge of what's considered safe.

## Critical batch size

McCandlish et al. (2018) formalized: there's a **critical batch size** `B_crit` for each model + dataset combination. Below `B_crit`, doubling the batch size roughly halves the number of steps to convergence (good — more parallel, no slower wall clock). Above `B_crit`, doubling the batch barely reduces steps (bad — more compute per step, no benefit).

`B_crit` correlates with the **gradient noise scale** — how noisy individual gradient estimates are. Larger models, longer training, and more diverse data all increase `B_crit`. For a frontier LLM, `B_crit` is in the millions of tokens.

Operationally: pick `batch_size` just past `B_crit`. Anything larger wastes compute per step; anything smaller leaves parallelism on the table.

## Loss landscape geometry for Transformers

Some empirical observations specific to Transformers:

- **Loss surfaces look smooth at large scale.** The 1B+ parameter regime seems to have a connected basin structure where most reasonable trajectories converge to similar-loss solutions.
- **Linear mode connectivity**: two models trained from different random seeds can often be linearly interpolated in parameter space without crossing a high-loss barrier (in their "merge well" basins). This underlies model merging / weight averaging techniques used in some post-training pipelines.
- **Loss spikes happen.** Even in flat regions, individual batches can produce gradients that push you off a cliff. Hence gradient clipping (file `03`).
- **Late-stage training is in a flat region.** By the end of cosine decay, the model is in a wide flat minimum; further training doesn't change loss much but does shift the parameters along the flat directions.

## Practical implications

For training:
- **Don't be afraid of large batches** for LLM training, up to your model's `B_crit`. The classical "sharp-minimum" concern is less severe for LLMs than for vision classifiers.
- **Use AdamW + warmup + cosine/WSD + clipping**. Each component addresses a specific landscape pathology.
- **Logging loss spikes is important**. They're early warning of sharper-than-expected regions.

For post-training:
- **Weight averaging across checkpoints / runs** (e.g. EMA of weights, model merging) can produce flatter minima than any single training run, often improving evals. Used in some open-model post-training stacks.

## Self-check

1. Why do flat minima typically generalize better than sharp minima? Be precise about the train/test distribution argument.
2. The "linear scaling rule" says you can double your LR when you double your batch size. Why does this work, and what's the breakdown point?
3. Why are LLMs less susceptible to the large-batch generalization problem than image classifiers?

### Answers

1. Test data comes from a slightly different distribution than train data — call this a perturbation of the loss landscape: `L_test(θ) ≈ L_train(θ + δ)` for some small `δ` representing the train-test shift. At a flat minimum (loss varies slowly in the parameter direction), `L_train(θ* + δ) ≈ L_train(θ*)` — the test loss is close to the train loss. At a sharp minimum, the loss spikes for small `δ` — test loss is much higher than train loss, i.e. poor generalization. The "test perturbation" view is a heuristic but lines up with empirical findings: training schemes that find flatter minima (small batch, SGD, SAM, weight averaging) consistently generalize better than those that find sharper minima.
2. Linear scaling holds in the regime where the gradient is noise-dominated. Each minibatch's gradient is `true_grad + noise`. The noise scales as `1/√batch_size`. The signal (true_grad) is batch-size-independent. To take a step of similar effective signal-to-noise ratio with a larger batch, you can take a larger step — proportional to batch_size, because the same step travels the same "true gradient" distance but with less noise riding on top. Breakdown: beyond the critical batch size, the gradient is already nearly noise-free; further increases don't reduce noise (it's already negligible) but do increase the step size, which destabilizes training. Past `B_crit`, you need sublinear LR scaling (e.g. `√batch`).
3. Several reasons. (a) Language modeling has a smoother loss landscape than image classification — more equivalent local minima, fewer sharp ones. The supervision signal is dense (every token) and well-behaved. (b) Transformers' use of pre-norm, residuals, and LayerNorm contributes to landscape smoothness. (c) The frontier-scale regime has lots of data; the train/test distribution is closer (test data is "more of the same"), so the "flat minimum advantage" is less pronounced. (d) Empirically, even very large batches (millions of tokens) work for LLMs because the per-token gradient is small and the per-step gradient noise is still nontrivial. The classical vision benchmarks where sharpness was a clear concern were small-data regimes with very specific failure modes that LLMs largely don't share.

## Exercise

This one is more conceptual than runnable. Take an LLM training log (NanoGPT or any small-scale training run you have access to). Compute the gradient norm at every step. Plot:
- The full distribution (histogram).
- The trajectory over training (line plot).

You should see: a few-step ramp during warmup (gradient norm grows), a stable middle phase (centered around some value), and possibly occasional spikes. The spikes are sharpness encounters — moments where the loss landscape has high curvature in the current step's direction. Gradient clipping caps them. This is the loss-landscape geometry made visible through gradient norms.
