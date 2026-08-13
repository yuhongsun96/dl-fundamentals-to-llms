# Expectation, Variance, and the Reparameterization Trick

> **Prereq refresher**: if variance, std dev, Gaussians, or `N(μ, σ²)` notation feel hazy, read [`supplementary/06_gaussians_basics.md`](supplementary/06_gaussians_basics.md) first. This file assumes those are already fluent.

## Expectation and variance

```
E[X] = Σ x p(x)    (discrete)  or  ∫ x p(x) dx    (continuous)
Var(X) = E[(X - E[X])²] = E[X²] - E[X]²
```

Basic properties you should never need to look up:
- `E[aX + b] = a E[X] + b`
- `Var(aX + b) = a² Var(X)`
- `E[X + Y] = E[X] + E[Y]` (always)
- `Var(X + Y) = Var(X) + Var(Y)` iff X, Y independent

The variance rule is why random projections and init schemes scale by `1/√d` — to preserve unit variance when you sum over `d` independent terms.

## Monte Carlo estimation

If `θ* = E_{x~p}[f(x)]` is hard to compute directly, sample `x_1, ..., x_N ~ p` and use:
```
θ̂ = (1/N) Σ f(x_i)
```

Unbiased, variance `Var(f(x))/N`. Convergence is `O(1/√N)` — slow. This is why RL-based training is sample-hungry.

For the full unpacking — what "sample from `p` and average" is actually doing, why `1/√N` is a theorem (not a fixable inefficiency), the precise contrast between supervised CE gradients (deterministic sums) and RL gradients (Monte Carlo estimates of expectations), and why RL only became practical at frontier scale once compute was cheap enough and rewards dense enough to absorb the `√N` cost — see `supplementary/06_monte_carlo_and_rl_sample_complexity.md`.

## Gradient of an expectation — the core problem

Frequently in ML you want:
```
∇_θ E_{x ~ p_θ(x)}[f(x)]
```

i.e. the gradient of an expectation where the **distribution depends on θ**. This is one of the most important objects in modern ML — let's unpack what it represents, why it's hard, and why the rest of this section exists to solve it.

### What the expression means

- **`θ`** are parameters we control — model weights, policy parameters, the knob the optimizer turns.
- **`p_θ(x)`** is a probability distribution *that depends on `θ`*. Change `θ` and the distribution itself shifts: different outcomes become more or less likely. The distribution is something we're *shaping* by choosing `θ`, not a fixed reference.
- **`f(x)`** scores an outcome — a reward, a loss, a quality measure. Takes a sample, returns a number.
- **`E_{x ~ p_θ}[f(x)]`** is a single number: the average value of `f` when `x` is drawn from the current `p_θ`. It tells us **how good the current parameter setting is, in expectation, when outcomes are stochastic.**

The gradient `∇_θ E[f(x)]` answers: **if I nudge `θ`, how does the expected score change?** Move `θ` along this vector → expected score goes up. This is the gradient we need for any optimization where outcomes are random.

### Where it shows up

This single expression is the load-bearing object behind a long list of modern methods:

- **Policy gradient in RL**: `θ` = policy weights, `p_θ(τ)` = distribution over trajectories the policy generates, `f(τ)` = reward. "How do I change the policy to generate higher-reward trajectories?"
- **VAE encoder training**: `θ` = encoder weights, `p_θ(z|x)` = variational posterior, `f(z)` = ELBO integrand. "How do I shape my posterior to make the ELBO bigger?"
- **Variational inference more broadly**: `q_θ` approximates an intractable posterior; optimize `θ` so `q_θ` matches reality.
- **Evolution strategies / black-box optimization**: `p_θ` is a search distribution over candidates; `f` is fitness.
- **Diffusion, GFlowNets, energy-based models**: anywhere you sample from a learned distribution and score the samples.

Master this gradient and you understand the core update rule of all of them.

### Why you can't naively swap ∇ and E

Instinctively you might write:
```
∇_θ E_{x ~ p_θ}[f(x)]  =?=  E_{x ~ p_θ}[∇_θ f(x)]
```

This is **wrong**, and the reason is the heart of the whole problem. Expand the expectation:
```
E_{x ~ p_θ}[f(x)]  =  ∫ f(x) · p_θ(x) dx
```

The expectation is an integral of `f(x)` *weighted by* `p_θ(x)`. So `θ` lives in **two distinct places**:
1. Possibly inside `f` (if `f` depended on `θ`).
2. **Inside `p_θ` — the weighting itself shifts when `θ` moves.**

In the standard setup `f` doesn't depend on `θ`, so place (1) contributes nothing. **All of the gradient signal lives in the weighting.** The naive swap captures place (1) and misses place (2) — i.e. it misses the entire effect.

**Concrete intuition.** In RL, `f` is a fixed reward function and `p_θ` is your policy. The naive swap differentiates the reward by `θ` — but the reward function has no `θ` in it, so the answer is zero. Obviously wrong: you absolutely can improve your expected reward by changing your policy. What you're changing isn't *how actions are scored*, it's *which actions get tried more often*. The naive swap is blind to this.

The correct gradient has to capture: **as `θ` shifts, `p_θ` reweights toward different `x`'s, and that reweighting changes the average score.**

### The honest derivative — and why it's stuck

Bring the gradient inside the integral (a routine swap of `∇` with `∫`, almost always legal):
```
∇_θ ∫ f(x) · p_θ(x) dx  =  ∫ f(x) · ∇_θ p_θ(x) dx
```

The gradient lands on `p_θ(x)`. Now we have an integral of `f(x)` weighted by `∇_θ p_θ(x)`. **But `∇_θ p_θ(x)` is not a probability distribution** — it's the gradient of a density. It can be negative, doesn't integrate to 1, and you can't sample from it. So you can't Monte Carlo this integral directly.

We know the gradient exists in closed form. We just can't estimate it via samples, which is the only tool available when `x` is high-dimensional. **This is the impasse the two tricks below are built to resolve.** They both rewrite the gradient into something we *can* estimate — score-function turns it back into an expectation under `p_θ`, reparameterization moves the θ-dependence out of the sampling distribution entirely.

Two main solutions:

### 1. Score function / REINFORCE

```
∇_θ E_{x~p_θ}[f(x)] = E_{x~p_θ}[f(x) ∇_θ log p_θ(x)]
```

Pros: works for discrete distributions, any `f`.
Cons: **high variance**. You need baselines, advantage estimators (GAE), and many samples.

This is the gradient used in policy gradient / PPO / RLHF / GRPO.

### 2. Reparameterization trick

Rewrite `x ~ p_θ(x)` as `x = g_θ(ε)` with `ε ~ p_0` (some distribution not depending on θ). Then:
```
∇_θ E_{x~p_θ}[f(x)] = ∇_θ E_{ε~p_0}[f(g_θ(ε))] = E_{ε}[∇_θ f(g_θ(ε))]
```

Gradient goes through `g_θ` via the chain rule like any other computation.

**Example**: `x ~ N(μ, σ²)` → `x = μ + σ · ε` where `ε ~ N(0, 1)`. Gradient w.r.t. `μ, σ` flows through directly.

Pros: **much lower variance** than REINFORCE.
Cons: requires differentiable `g` — only works for continuous distributions with nice parameterizations.

This is the trick that made VAEs work. It's also what enables some diffusion training objectives and continuous latent-variable models.

## Where both show up in LLM-land

- **PPO / RLHF**: REINFORCE-style, because language is discrete. Variance-reduction techniques (value function, GAE, KL penalty, clipping) are all about taming the score-function estimator.
- **DPO**: sidesteps sampling entirely by deriving an analytical loss from preferences.
- **Gumbel-softmax / straight-through estimator**: discrete-variable reparameterization hacks used in some MoE routing and quantization work.
- **Diffusion**: reparameterization of the noise schedule is at the heart of the training objective.

## Gumbel-softmax in one paragraph

To sample from a categorical `softmax(logits)` and *still* flow gradients through, use:
```
y_i = softmax((logits_i + g_i) / τ)    where g_i ~ Gumbel(0, 1)
```

As `τ → 0`, `y` approaches a one-hot sample. For small `τ` you get an approximately discrete sample with pathwise gradients. Shows up in MoE training (e.g. Switch Transformer) and in hard-attention variants.

## Law of total variance (handy intuition)

```
Var(X) = E[Var(X|Y)] + Var(E[X|Y])
```

"Total variance = average noise given Y + spread of conditional means." Useful framing for:
- Why conditioning (giving the model more context) reduces prediction variance.
- Why ensembling reduces variance.
- Why control variates in RL work.

## Self-check

1. Why does REINFORCE's gradient estimator have higher variance than the reparameterization gradient?
2. Can you use the reparameterization trick on a Bernoulli sample? What about a Gumbel-softmax relaxation?
3. In RLHF with PPO, the KL penalty `β KL(π_θ ‖ π_ref)` — is that expectation estimated by sampling or computed in closed form? (Answer: typically sampled per-token.)

### Answers

1. **REINFORCE** estimates `∇E[f(x)] = E[f(x) · ∇log p_θ(x)]` — the gradient is `f(x_i) · score(x_i)` for each sample. Variance comes from three places stacked: (a) `f(x)` varies across samples; (b) the score function `∇log p` is near-zero in some regions and huge in others (especially for outliers in the tails); (c) `f` and `score` **multiply**, so their noise compounds. **Reparam** estimates `∇f(g_θ(ε))` directly via the chain rule. The randomness `ε` is **fixed** while you take the gradient w.r.t. `θ`; the estimator is just `∂f/∂θ` evaluated at one sampled `ε`. No score function multiplier, gradient flows through the deterministic `g_θ`. Much lower variance — usually 10–100× depending on the problem.
2. **Bernoulli**: not directly. It's discrete; sampling is non-differentiable (you can't take a derivative through "did we get heads?"). Reparam needs a differentiable sample-as-function-of-noise pathway. **Gumbel-softmax**: yes, with a relaxation. `y = softmax((log p + g)/τ)` with `g ~ Gumbel(0, 1)` gives a continuous "soft" sample. As `τ → 0`, approaches a one-hot draw from the categorical; for finite `τ`, you get pathwise gradients. Used in MoE training, VQ-VAE codebook training tricks, and some discrete-latent papers. The "straight-through estimator" is a related hack: forward uses the discrete sample, backward uses the gradient of the soft relaxation.
3. **Sampling**, per-token. You can't enumerate over all possible response sequences, so the KL is estimated as `(1/N) Σ_i log(π_θ(y_i | x) / π_ref(y_i | x))` for tokens drawn from rollouts. Implementations often compute a **per-token** KL (sum across tokens in a response) rather than a sequence-level KL — this gives lower-variance per-step gradients but isn't an unbiased estimator of the true sequence-level KL. There are several valid estimator variants (`k1`, `k2`, `k3` from Schulman's notes); modern RLHF stacks usually use `k3` for unbiasedness with positive variance.

## Exercise

Simulate in Python: estimate `E_{x~N(μ,1)}[x²]` and its gradient w.r.t. `μ` at `μ = 2`, using:
1. Score function: `(1/N) Σ x_i² · ∇_μ log p(x_i)`
2. Reparameterization: `x = μ + ε`, then differentiate `(μ + ε)²` w.r.t. `μ`.

Run with `N = 100` samples, 1000 trials. Compute empirical variance of the gradient estimate for both. Reparam should be dramatically lower variance. This is *why* VAEs prefer it and *why* policy gradients are sample-hungry.
