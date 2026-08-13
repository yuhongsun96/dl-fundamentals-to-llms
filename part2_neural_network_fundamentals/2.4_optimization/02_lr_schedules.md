# Learning Rate Schedules: Warmup, Cosine, WSD

The optimizer (file `01`) tells you the *direction* of the parameter update. The learning rate tells you the *step size*. Constant LR is rarely optimal for deep learning — different phases of training benefit from different sizes. A schedule is a function `η(step)`.

The dominant pattern for LLM pretraining is: **warmup → main schedule → optional anneal**. Each part exists for a specific reason.

## Why constant LR is wrong

Three failure modes a constant LR runs into:

1. **Early training**: parameters are at init, which is far from any good solution. Optimizer state (especially Adam's second moment) is uninformed. Large steps in random directions destabilize training. You want **small LR early**.
2. **Mid training**: the model has found a useful region; you want to make rapid progress. **Large LR** to traverse the landscape efficiently.
3. **Late training**: the model is close to the (local) optimum; large steps overshoot. You want **small LR late** to fine-tune.

A schedule encodes this three-phase structure. Constant LR can only optimize for one phase; you lose progress in the others.

## Warmup

Linearly ramp the LR from 0 (or a small value) to the peak LR over `W` steps:
```
η(t) = η_peak · min(1, t / W)
```

Typical `W`: 500–5000 steps, or ~1% of total training. For frontier LLMs sometimes 2000–10000 steps over a multi-month run.

### Why warmup is essentially mandatory for Transformers

Two reasons, both load-bearing:

1. **Adam's second-moment estimate `v_t` is unreliable at small `t`.** With `β2 = 0.999`, it takes ~1000 steps for `v_t` to be a well-conditioned estimate of true gradient magnitudes. In the first few hundred steps, `v_t` underestimates the gradient variance (it's been averaging with a zero-initial), so `m_t / √v_t` is too large in some coordinates. Taking full-LR steps with bad `v_t` overshoots and can NaN the run.
2. **The loss landscape at init is full of sharp directions** — small parameter changes can produce big loss changes. Large early steps land in regions where gradients explode. Smaller early steps let the optimizer find a smoother basin first.

Try training a Transformer without warmup: the loss will jump or NaN in the first few hundred steps. Add warmup and the same setup trains fine.

Some recent work (e.g. some μP variants) claims to remove the need for warmup with careful init scaling. In practice, every standard LLM training stack still uses warmup, because it's cheap insurance.

### How `W` is chosen before training

There's no closed-form formula — `W` is picked from heuristics keyed to the hyperparameters you fix in advance, floored by an optimizer timescale. Think of it as **a floor pushed upward by the factors that make the early landscape more dangerous:**

- **The `β2` floor (~a few hundred to ~1000+ steps).** Reason 1 above sets a hard lower bound: `v_t`'s memory is `~1/(1-β2) ≈ 1000` steps, so warmup much shorter than that holds you at peak LR while `√v_t` is still noisy. This is the minimum. Lowering `β2` to `0.95` (common in LLM runs) shortens the floor.
- **Peak LR — the biggest lever.** Warmup's job is to safely *reach* `η_peak`; a more aggressive peak needs a longer ramp to get there without overshooting. Higher peak → longer `W`.
- **Batch size.** Large batches use a larger LR (linear/√ scaling, file `04`) and change the early gradient-noise structure — both push `W` longer. Frontier runs combine huge batches *and* aggressive LRs, which is why their warmups stretch to 2000–10000 steps.
- **Model size / depth.** Deeper, wider models have more fragile early dynamics (more residual additions compounding, more attention layers to stabilize), so bigger models warm up longer.

The `~1%` convention then expresses `W` as a fraction of planned steps so it scales loosely with run length. Note the asymmetry: the floor (optimizer + settling timescale) is roughly **fixed in absolute steps**, while the 1% rule **scales with run length** — so short runs spend a larger *fraction* warming up, and multi-month runs spend a tiny fraction but a large absolute number. In practice: satisfy the floor, set `W ≈ 1%` of total, then optionally confirm with a short sweep. It's forgiving — there's a wide plateau where any reasonable value works, which is why the typical value is quoted as a range, not a number.

> For how the *peak LR* and *schedule length* were chosen across eras — BERT-style direct sweeps and the Noam `d_model^(−0.5)` rule, then μP/μTransfer, hyperparameter scaling laws, and WSD for frontier models too large to sweep — see [`supplementary/02_setting_lr_and_schedule_across_scales.md`](supplementary/02_setting_lr_and_schedule_across_scales.md).

## Cosine decay

After warmup, decay LR following a cosine curve down to a minimum:
```
η(t) = η_min + 0.5 · (η_peak - η_min) · (1 + cos(π · (t - W) / (T - W)))
```

For `t = W`, `η = η_peak`. For `t = T`, `η = η_min`. Smooth monotone decrease in between.

The whole schedule (linear warmup, then cosine decay) looks like:

```
  η(t)
 η_peak ┤       _.-‾‾‾‾‾‾-._
        │     ./           ‾·._
        │     /                ‾·._
        │    /                     ‾·._
        │   /                          ‾·-._
        │  /                                ‾·-.____
 η_min  ┤ /                                       ‾‾‾‾
        │/
      0 ┤
        └──┬───┬───────────────────────────────────┬──→  t (steps)
           0   W                                    T
           │←─→│←──────────── cosine decay ────────→│
           warmup     (not to scale: W ≈ 1% of T)
```

Read the shape: a steep **linear ramp** from 0 up to `η_peak` over the first `W` steps (warmup is a thin sliver at the far left — drawn wide here for legibility), then a **cosine fall** to `η_min`. The cosine is **flat near the top** (so you linger in the high-LR exploration phase), **steepest in the middle**, and **flat again near the bottom** (so you settle gently onto the floor) — that flat-steep-flat profile is exactly why cosine is preferred over a straight linear decay, which would spend less time at high LR.

Typical setup: `η_peak = 3e-4` for a 7B model, `η_min = η_peak / 10` (sometimes `η_min = 0`), `T = total training steps`.

Why cosine specifically? Empirically it works well — beats linear decay and exponential decay slightly. There's no strong theory, but the intuition is that cosine decays slowly at first (preserving the high-LR phase) and slowly at the end (gently approaching the floor), spending more time at moderately large LRs than a linear schedule would.

GPT-3 used cosine. Llama 1/2 used cosine. Most pretraining still uses cosine.

### A practical pitfall

The cosine schedule is parameterized by `T` (total training steps). If you stop training early, you don't get the benefit of the late-stage low-LR phase. If you train past `T`, the LR stays at `η_min` (or wraps around, depending on impl) — usually unhelpful.

This means **cosine schedules commit you to a specific training length**. You can't easily "train for one more month at full LR" without designing a new schedule.

## WSD (Warmup-Stable-Decay)

A response to the cosine commitment problem. WSD = Warmup → Stable → Decay:

```
η(t) = η_peak · min(1, t / W)               # warmup
     = η_peak                                # stable phase, no decay
     = η_peak · decay(t - (T - D))           # decay phase (linear or sqrt) over the last D steps
```

The model spends most of its training at full LR (`η_peak`). Only the last ~10–20% of training decays down. The "decay" function is often linear or `1/sqrt(t)` — simpler than cosine.

### Why WSD is gaining ground

- **Decoupled from total length.** You can decide to keep training during the stable phase without re-planning the schedule. Just don't enter the decay yet.
- **Better for multi-stage training.** If you want to add a new dataset midway, start instruction tuning, switch to long-context phase — the stable phase makes this clean. Cosine's monotone decay locks you in.
- **Empirically competitive.** Recent papers (MiniCPM, others) showed WSD matches or beats cosine at fixed compute. Llama 3 used a WSD-like schedule.

If you're doing serious LLM pretraining today, WSD is increasingly the right choice.

## Other schedules you'll see

- **Linear decay**: simple, slightly worse than cosine empirically. Some fine-tuning setups use this.
- **Step decay**: drop LR by 10× at specific epochs. Common in ResNet-era vision training. Not used for LLMs.
- **Inverse-sqrt** (`1/√t` after warmup): used in the original Transformer paper. Decays too fast in modern long-training regimes. Not used anymore.
- **Cyclical / one-cycle**: LR goes up and down repeatedly. Some experimental evidence it helps; not standard for LLM training.
- **Constant after warmup**: appropriate for some fine-tuning. Bad for pretraining (model never "settles" into a local optimum).

## Fine-tuning schedules

LLM **fine-tuning** uses different conventions:
- Much smaller peak LR (5e-5 to 2e-4 for a 7B-30B model, vs. 3e-4 for pretraining).
- Short warmup (100–500 steps).
- Linear decay or even constant LR through training.
- Total training is much shorter (1–10K steps vs. 100K+ for pretraining), so the schedule design matters less.

### Why warmup is short even though the optimizer state is fresh

Fine-tuning resets the AdamW moments (`m = v = 0`), so the "`v_t` is unreliable early" problem (warmup reason 1) is back at step 0 just as in pretraining. By the β2-floor logic alone you'd expect the same ~1000-step warmup. The reason you don't need it: warmup's **dominant** job was reason 2 — *surviving the treacherous random-init landscape*, where a full-LR step near sharp curvature diverges. **A fine-tuned model has no random init to survive** — it's a converged network already sitting in a smooth, low-curvature basin (activation scales, residual-stream magnitude, attention patterns all settled). With benign curvature *and* a small peak LR, a step taken through a still-rough `v_t` is recoverable, not catastrophic, so you only need `v_t` *roughly* calibrated (a few hundred steps), not the full β2 settling that supports aggressive full-LR pretraining steps. (Two forces push the same way: warmup scales with peak LR, which is 5–10× smaller here; and a 1–10K-step run can't spend 1000 steps warming up anyway.) The short warmup that remains mostly eases the **distribution-shift shock** — not hammering the pretrained weights with a big update on new data in the first few steps.

For LoRA / QLoRA fine-tuning specifically, peak LRs are *higher* (1e-4 to 1e-3) because only the LoRA parameters are updated and they start from zero.

### Why "starts from zero" raises the LoRA LR — but lowers the pretraining LR

"Starts from zero" can't be the whole story, since pretraining *also* starts from near-zero random init yet uses a *smaller* LR. The difference is mechanism plus landscape:

- **The cold-start mechanism.** LoRA sets `ΔW = B·A` with `A` random and **`B = 0`**, so `ΔW = 0` at init — the adapter is a deliberate no-op. The gradients of the product are lopsided at step 0: `B` gets a gradient (it sees `A`), but `A` gets *none* (it sees `B = 0`). The adapter wakes up sequentially, and the effective update to `ΔW` is *second-order* near the origin (a product of two things, one starting at zero) — so the effective step is **suppressed**, and you raise the raw LR to compensate. On top of that, only a tiny rank-`r` slice is trainable, so each parameter must move more to change the effective weight meaningfully.
- **Why pretraining is the opposite.** Pretraining learns **full-rank** weights in front of a **random-init, high-curvature** network, where a large LR diverges (the init variance-preservation story, file 2.3) — it *can't afford* a big LR. LoRA bolts a **low-rank** correction onto a **frozen, already-trained, smooth** base, so a large LR is *safe* (benign landscape) and *needed* (cold, suppressed update). "Starts from zero" means *small random **active** values near a cliff* for pretraining, but *an **inert** adapter that must be switched on, atop a stable model* for LoRA — opposite implications for LR.

## Learning rate vs. batch size

For a fixed compute budget, you can trade off batch size and LR (within limits). Larger batches → less noisy gradient → can sustain larger LR. Smaller batches → noisier gradient → smaller LR.

Rough heuristic (the "linear scaling rule"): if you double the batch size, double the LR. This works up to a critical batch size (file `04`); past it, LR has to grow sublinearly or training destabilizes.

For LLMs, the relationship is more like `LR ∝ √batch_size` in the regime where batch is already past the noise-dominated phase. Frontier-scale training uses batches in the millions of tokens, with LR in the `1e-4` to `5e-4` range.

## Self-check

1. Why is warmup essentially mandatory for Adam-trained Transformers? What specifically goes wrong with no warmup?
2. The cosine schedule commits you to a specific total training length. What happens to the LR if you train past `T`? Why does this make cosine awkward for staged training?
3. WSD's stable phase keeps LR at peak for most of training. Why doesn't this destabilize training the way "constant high LR forever" would in the limit?

### Answers

1. Adam's second-moment `v_t` is an EMA with `β2 = 0.999`, so it takes ~1000 steps to "fill up" with a representative average. At step `t = 10`, `v_t` is dominated by the bias-correction factor — it's an estimate based on 10 gradients, not 1000. The per-coordinate `√v_t` denominator is then noisy: some coordinates get a tiny denominator (and thus a huge effective LR) by random chance. Combined with the unfriendly init-time loss landscape, a step at full LR can land somewhere catastrophic — overflow into NaN or just into a bad region the optimizer can't recover from. Warmup gives `v_t` time to stabilize and gives the optimizer time to find a smooth basin before opening the throttle. Try removing warmup from a Transformer training script: you'll see NaN within ~100 steps maybe half the time.
2. After step `T`, the cosine formula either repeats (some implementations) or stays at `η_min` (most). Either way, the schedule was designed assuming you stop at `T`; training longer enters undefined territory. The awkwardness for staged training: if you finish pretraining and want to start instruction tuning with a new schedule, you can't smoothly continue from cosine's end state — you have to re-warmup or accept a discontinuity. If midway through pretraining you decide to add a new dataset and extend training, the cosine curve at the extension point is unlikely to match what you'd want for the next phase. WSD's stable phase makes both transitions clean.
3. Two reasons. First, by the end of warmup the optimizer state (Adam's `v_t`) has stabilized and the parameters have moved into a reasonably smooth region — the model is no longer in the "any step can NaN" regime. Constant high LR is dangerous *early*, not because it's high but because the optimizer state isn't ready for it. Second, the stable phase is bounded by a decay phase that does the final annealing. Without the decay phase, the model never gets to fine-tune around a local optimum; with it, you collect the best of both worlds — long phase of high-LR exploration, then fine-tuning. Empirically WSD with no decay is worse than WSD with decay, which is comparable to cosine. The stable+decay structure has all the components a good schedule needs.

## Exercise

Plot four schedules on the same axes for `T = 100K` steps, `W = 2K`, `η_peak = 3e-4`:
1. Linear decay (warmup, then linear to 0).
2. Cosine (warmup, then cosine to `η_peak/10`).
3. WSD (warmup, stable, linear decay over last 20% to 0).
4. Constant at `η_peak` after warmup.

Then train a small Transformer with each. Track loss at end. Cosine and WSD will be similar; constant will plateau higher; linear decay will be in between. The shape of the schedule matters less than (a) you have warmup, and (b) you decay at the end.
