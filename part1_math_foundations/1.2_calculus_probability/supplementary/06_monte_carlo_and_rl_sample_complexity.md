# Monte Carlo Estimation and Why RL Is Sample-Hungry

Standalone deep-dive on the Monte Carlo estimator mentioned in `06_expectation_variance_reparam.md`. Walks through what the estimator is doing, where the `O(1/√N)` rate comes from, and why this becomes the structural bottleneck of RL training.

## What we're trying to compute

`E_{x~p}[f(x)]` reads as: "the average value of `f(x)` when `x` is drawn from distribution `p`." Formally:

```
E_{x~p}[f(x)]  =  ∫ f(x) · p(x) dx       (continuous)
               =  Σ  f(x_i) · p(x_i)     (discrete)
```

You're integrating `f` weighted by how often each input occurs under `p`. The answer is a single number — the *true mean* of `f` under `p`.

Why this is often "hard to compute directly":

- The space of `x` might be enormous or continuous (e.g., all possible 1000-token sequences a language model could emit — `vocab^1000` outcomes).
- `p(x)` might not be available in closed form — it's defined implicitly by your model's forward pass.
- The integral may have no analytical solution.

So the integral exists in principle but is intractable to compute analytically.

## The Monte Carlo trick

Instead of evaluating `f` against *every* possible `x` weighted by `p`, do this:

1. Draw `N` samples `x_1, ..., x_N` from `p`. You don't need `p` in closed form — you just need to *sample* from it. For an LM, this means generating text.
2. Evaluate `f` on each sample.
3. Average: `θ̂ = (1/N) Σ f(x_i)`.

That's it. `θ̂` is a random number (depends on which samples you happened to draw), but it approximates the true expectation.

**Why it works (LLN intuition).** Drawing samples from `p` naturally weights each region of `x`-space by how often `p` visits it — exactly the weighting in the integral. By sampling and averaging, you let nature do the integration for you.

Two formal properties:

- **Unbiased**: `E[θ̂] = E_{x~p}[f(x)]`. Averaged across all possible random samples, you get the right answer. No systematic over- or under-shoot.
- **Variance**: `Var(θ̂) = Var(f(x)) / N`. The spread of your estimate around the true value shrinks linearly in `N`.

## The √N convergence — the bad news

Standard deviation is the square root of variance:

```
std(θ̂)  =  std(f(x)) / √N
```

This is `O(1/√N)`. To shrink the error by 10×, you need **100× more samples**. To shrink it by 100×, **10,000× more samples**. The estimator converges to the true value, but agonizingly slowly compared to deterministic computation.

Concretely: if `f` has standard deviation 1, getting an estimate accurate to ±0.01 needs ~`1/0.01² = 10,000` samples. ±0.001 needs a million. **Each extra digit of precision costs 100× more samples.**

This rate is a *theorem* (Central Limit Theorem). No amount of cleverness breaks `1/√N` — you can only reduce the *constant* in front (`std(f(x))`), not the rate.

## Why this matters: supervised CE vs. RL

**Supervised learning with cross-entropy** doesn't suffer this. The gradient on a minibatch is a deterministic sum over labeled examples — there's no random variable whose mean you're estimating. Minibatch noise comes from subsampling a fixed dataset, which is mild; even 256 examples give a very informative gradient.

**RL is fundamentally different.** The policy-gradient objective is:

```
J(θ)   =  E_{τ ~ π_θ} [ R(τ) ]
∇_θ J  =  E_{τ ~ π_θ} [ R(τ) · ∇_θ log π_θ(τ) ]
```

The trajectory `τ` is high-dimensional (a whole sequence of states/actions, possibly thousands of tokens). The distribution `π_θ(τ)` is the policy itself, defined implicitly by the model's forward pass. The reward `R(τ)` often comes from a black-box verifier (a unit test, an answer checker) you can only evaluate by actually running things.

So you estimate the gradient with Monte Carlo: roll out `N` trajectories, compute `R(τ_i) · ∇_θ log π_θ(τ_i)` for each, average. **Every RL gradient step is a Monte Carlo estimate of the true gradient direction.**

And now `√N` slowness is the bottleneck. Each rollout:

- Generates a long sequence (potentially thousands of tokens of reasoning).
- Requires a forward pass through a multi-billion-parameter model for each token.
- Possibly runs external code, tools, or a verifier.
- Produces one scalar reward at the end.

So one sample is *expensive*. And to get a gradient with low variance you need *many* of them — hundreds to thousands per step in practice. Supervised training: one forward-backward on 256 examples → informative gradient. RL: hundreds of long expensive rollouts → still-noisy gradient. **RL needs orders of magnitude more compute per parameter update just to keep the gradient direction reliable.** That's the precise meaning of "sample-hungry."

## Why this is mitigated, not fixed

The `√N` rate is mathematically locked. What people fight against is the constant — per-sample variance `Var(f(x))`:

- **Baselines** (subtract average reward from `R(τ)`): keeps the estimator unbiased, lowers variance. The advantage `A(τ) = R(τ) − b` is unbiased whenever `b` doesn't depend on `τ`.
- **GAE (Generalized Advantage Estimation)**: trades a bit of bias for much lower variance using a learned value function.
- **Importance sampling**: lets you reuse samples from an older policy, effectively multiplying `N` (with caveats — variance can explode if policies diverge).
- **Control variates, antithetic sampling, stratified sampling**: classical variance-reduction techniques from Monte Carlo theory.
- **Per-token / process rewards (PRMs)**: rather than one reward per trajectory, assign rewards per step — many more "samples" per rollout, denser learning signal.
- **Reparameterization trick** (for differentiable distributions): instead of `∇ E_p[f]` via the log-trick (high variance), express `x = g(ε, θ)` for some noise `ε` independent of `θ`, then use `∇ E_ε[f(g(ε, θ))]` — pathwise gradient. Dramatically lower variance when applicable. This is why VAEs use reparameterization, and why policy gradients (where the action distribution typically isn't reparameterizable) remain stuck with the noisier estimator.

All of these reduce the constant in `√(Var(f)/N)`. None break the `1/√N` itself.

## Why RL only became practical recently

The structural cost of Monte Carlo estimation is *the* reason RL stayed in research labs for decades and only became a frontier-LM tool around 2023–2024. Two enabling shifts:

1. **Compute got cheap enough.** Frontier-scale RL needs hundreds of rollouts per step × thousands of steps. The total cost is enormous — only feasible once GPU compute hit the scale it did.
2. **Verifiable rewards arrived.** For most of RL history, the reward signal was either dense-but-shaped (Atari) or sparse-and-noisy (real-world tasks). Math/code with automatic verifiers (test suites, answer checkers, formal proof systems) gave RL a clean, scalable reward source — first time it was practical to scale RL on a *real* task with *real* rewards.

The Monte Carlo `√N` rate didn't change; the constant got smaller (better methods, denser rewards) and the budget got larger (more compute). Together they crossed the threshold where RL on language models became worthwhile.

## TL;DR

- **Monte Carlo** is the trick of approximating an intractable expectation by averaging over samples drawn from the distribution. Unbiased, simple, universal.
- It **converges**, but at the painfully slow rate of `1/√N` standard deviation. Each extra digit of precision costs 100× more samples. This rate is a theorem, not a temporary limitation.
- **Supervised CE training** doesn't pay this cost because gradients are deterministic sums, not Monte Carlo estimates.
- **RL training** does pay it because the policy-gradient objective is fundamentally an expectation over high-variance trajectory rewards. Each rollout is expensive, and `√N` means doubling reliability quadruples cost.
- Variance reduction (baselines, GAE, PRMs, reparameterization where possible) lowers the *constant*, never the rate. RL has only become practical at frontier scale once compute was cheap enough and rewards were dense enough to absorb the `√N` cost.
