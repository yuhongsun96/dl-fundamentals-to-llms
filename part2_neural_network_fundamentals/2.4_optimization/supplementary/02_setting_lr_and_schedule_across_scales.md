# Supplementary: Setting LR and schedule across scales — from sweeps to μTransfer

Companion to [`../02_lr_schedules.md`](../02_lr_schedules.md). The primary file tells you *what* the warmup → main → decay shape is and *why* warmup is mandatory. This sidecar answers a different question: **how do you actually pick the peak LR and the schedule length** — and how that answer changed as models outgrew the ability to sweep them directly.

The throughline: in every era you *measure* hyperparameters empirically rather than derive them from first principles. What changed is *where* you measure — small models you can run hundreds of times — and what machinery carries the answer up to a model you only get to train once.

---

## The small-model era (BERT and friends): sweep, then copy

For sub-100M models, the guidance was overwhelmingly empirical: take a known-good recipe, run a small LR grid on the *actual* target model, pick the winner. The recipes were folklore passed paper-to-paper.

| Model (~size) | Optimizer | Peak LR | Warmup | Decay |
|---|---|---|---|---|
| Original Transformer (2017, ~65M) | Adam, `β2=0.98` | set by formula (below) | 4000 steps | inverse-sqrt (`1/√t`) |
| GPT-1 (2018, 117M) | Adam | 2.5e-4 | 2000 steps (linear) | cosine to 0 |
| BERT pretraining (2018, 110M/340M) | Adam, `β2=0.999`, wd 0.01 | 1e-4 | 10000 steps (~1%) | linear to 0 |
| BERT fine-tuning | Adam | grid `{2e-5, 3e-5, 5e-5}` | 10% of steps | linear to 0 |

Patterns worth absorbing: Adam pretraining LRs cluster around `1e-4`–`3e-4`; warmup is `~1%` of *pretraining* but `~10%` for the short fine-tuning runs (the absolute-floor-vs-fraction asymmetry from the primary file, in the wild); and BERT fine-tuning was a literal three-point grid that everyone reused.

**Why pure empiricism was fine.** Sub-100M models train in hours-to-days. You could afford to sweep LR over a log-grid (`{1e-4, 3e-4, 5e-4, 1e-3}`) on the real model, watch which diverged, and pick the best. The hyperparameter was *measured*, not predicted — and the measurement was cheap.

### The few semi-principled rules of the era

1. **The Noam schedule's width rule.** The original Transformer's LR was a formula:
   ```
   η(t) = d_model^(−0.5) · min(t^(−0.5),  t · W^(−1.5))      W = 4000
   ```
   The `d_model^(−0.5)` factor is the era's one widely-used **LR-vs-size** rule: wider model → proportionally smaller LR. A heuristic that worked, not a theorem — but the closest thing to "compute the LR from the architecture." The `min(...)` just splices linear warmup into inverse-sqrt decay.
2. **The linear batch-scaling rule** (Goyal et al., 2017): if LR `η` works at batch `B`, then `kB` wants `~kη` (with warmup to survive the larger early steps). It didn't give a starting LR — it let you *port* a known-good one to a new batch size.
3. **The LR range test** (Smith, ~2015–17): ramp LR up over a few hundred steps, plot loss-vs-LR, pick a peak just below divergence. A cheap *empirical* max-LR finder; used by some, never standard in NLP pretraining.

---

## The problem at scale

When a single run costs millions of dollars and you get *one shot*, you can no longer sweep LR on the target model. For sub-100M models this never mattered (just sweep it); at frontier scale it becomes the central difficulty. The post-Chinchilla era produced two ways to *set* the LR without a target-scale sweep, plus two refinements to the *schedule*.

## Setting the LR value without sweeping the big model

### 1. μP / μTransfer — transfer the LR across width

**Yang et al., "Tensor Programs V: μTransfer" (2022, around Chinchilla).** Under a specific **Maximal Update Parametrization (μP)** — particular rules for how init variance, LR, and per-layer multipliers scale with width — the *optimal LR becomes invariant to width*. The recipe:

1. Put the model in μP.
2. Sweep LR (and other HPs) on a **small proxy** model — cheap, run it hundreds of times.
3. **Reuse the same LR** on the giant model; μP guarantees the optimum doesn't move as you widen.

This is the cleanest answer to "you can't sweep at scale." Cerebras-GPT (2023) used it publicly across a model family; GPT-4's report describes predicting large-model behavior from small runs in the same spirit. Caveat: μP controls transfer across **width** cleanly, depth and other axes less perfectly — in practice it's μP plus an empirical correction.

### 2. Hyperparameter scaling laws — fit a power law, extrapolate

Chinchilla showed you can fit power laws for *loss* vs. compute; the natural extension fits the *optimal hyperparameters* themselves. The most explicit public version is the **DeepSeek LLM scaling-laws paper (2024)**, which ran a grid of small models and fit optimal LR and batch size against the compute budget `C`:

```
η_opt  ≈  0.31 · C^(−0.125)      # optimal LR slowly decreases with compute
B_opt  ≈  0.29 · C^(+0.327)      # optimal batch size grows with compute
```

(constants illustrative; the *shape* is the point). Measure the exponents where you can afford to, then plug in the `C` of the run you can't sweep. Two robust qualitative findings, consistent across the literature:

- **Optimal LR decreases with scale, but slowly** (small negative exponent) — bigger models want somewhat smaller LRs.
- **Optimal batch size grows with scale** — related to the older **critical batch size** (McCandlish et al., 2018), which itself grows *during* a run as the loss falls.

μP and HP scaling laws are independent levers: one transfers across **width**, the other extrapolates across **compute**.

## Setting the schedule without a target-scale sweep

### 3. Chinchilla's cosine-length rule

An easily-missed point from Hoffmann et al. (2022): the **cosine decay length must match the actual number of training steps.** Set the cosine to decay over (say) 2× more steps than you train, and the LR is still too high when you stop → meaningfully worse final loss. The rule for large models: *decay to your minimum LR exactly at the token budget* — which forces you to commit to the horizon up front.

### 4. WSD — decouple the decay from the horizon

The "cosine must know the horizon" requirement is painful when you can't sweep and may want to train longer. **WSD (Warmup-Stable-Decay, MiniCPM 2024)** warms up, holds LR **constant** indefinitely, then decays sharply over only the last ~10–20%. Consequences:

- You don't fix total length in advance (you can decide to stop later).
- It's *cheaper for scaling studies*: run one long stable trunk, branch a short decay at several points, and harvest multiple "fully-trained" checkpoints to fit laws against — instead of one full cosine run per data point.

This is why WSD spread fast for large-scale and scaling-law work: it makes the schedule horizon-agnostic, exactly what you want when sweeping is off the table. (See the WSD section of the primary file for the schedule mechanics.)

---

## Synthesis

| Era | LR value | Schedule |
|---|---|---|
| Small models (≤~100M) | Sweep a small grid on the real model; anchor with Noam `d_model^(−0.5)` and batch scaling | "Adam, ~1% warmup, decay to near-zero" — shape mattered less than having warmup + some decay |
| Post-Chinchilla, frontier | **μP** transfer from a small proxy, and/or fit a **HP scaling law** over small runs and extrapolate to target `C` | **Cosine with decay length = token budget** (Chinchilla), or **WSD** to stay horizon-agnostic |

The constant across eras: hyperparameters are still *measured*, not derived. The shift is measuring them where it's cheap (small width, small compute) and relying on a **parametrization (μP)** or a **fitted law** to carry the answer up to a scale you only run once. Connects forward to scaling laws (Part 6.3).

## Self-check

1. Before μP, why was it acceptable to have no way to *predict* the optimal LR for a given model size?
2. μP and HP scaling laws both let you avoid a target-scale sweep, but they extrapolate along different axes. Which axis does each handle?
3. Why does WSD make fitting hyperparameter/loss scaling laws cheaper than cosine does?

### Answers

1. Because sub-100M models were cheap enough to *sweep directly* — you measured the LR on the real model in hours. Prediction-from-size only becomes necessary when a single run is too expensive to repeat, which didn't happen until frontier scale.
2. μP transfers the optimal LR across **width** (tune small-width, reuse at large-width). HP scaling laws extrapolate across **compute budget `C`** (fit exponents on small runs, plug in the target `C`). They're complementary, not redundant.
3. Cosine bakes the training horizon into the curve, so each "fully-trained" data point needs its *own* full cosine run. WSD's constant stable phase lets you run one long trunk and branch a short decay off it at several lengths, yielding many fully-decayed checkpoints from largely shared compute.
