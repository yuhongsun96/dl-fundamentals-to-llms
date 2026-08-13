# Batches, Steps, and Tokens — the Units of a Pretraining Run

The BERT-era mental model of a batch — "32 examples" — doesn't survive contact with a pretraining run. At scale, **every quantity is denominated in tokens**: the budget ([6.3](../6.3_scaling_laws/02_chinchilla.md)), the schedule ([2.4/02](../../part2_neural_network_fundamentals/2.4_optimization/02_lr_schedules.md)), and the batch. This file is the dimensional analysis of a real run — short, concrete, and the difference between reading a training report and merely skimming it.

## The batch is measured in tokens

A pretraining batch is `(sequences per step) × (sequence length)` tokens. Real numbers: GPT-3 trained at a **3.2M-token** batch; Llama-3-405B staged its batch from **4M → 8M → 16M tokens** over the run. At `S = 8192`, a 16M-token batch is ~2,000 packed sequences per optimizer step — spread across thousands of GPUs via data parallelism (Part 12), with **gradient accumulation** making the arithmetic batch independent of what fits in memory (accumulate micro-batch gradients, step once — mathematically identical to one big batch, since gradients of a mean decompose into a mean of gradients).

The run's arithmetic then has one master identity:

```
steps = token budget / batch size
```

15T tokens at a 16M batch = **937,500 optimizer steps** — under a million weight updates to produce a frontier model. That number sets everything downstream: the LR schedule's horizon is those steps ([2.4/02](../../part2_neural_network_fundamentals/2.4_optimization/02_lr_schedules.md) — and mis-matching schedule to horizon is exactly the Kaplan flaw, [6.3/01](../6.3_scaling_laws/01_power_laws_kaplan.md)), checkpoint cadence, and the resolution of every ablation.

## Why batches are that size: the critical batch size

Two forces set the batch, and their crossover has a name:

- **Systems push up:** more data parallelism = more GPUs usable = shorter wall-clock. If doubling the batch halves the steps at equal quality, you finish in half the time on twice the hardware.
- **Optimization pushes back:** a gradient estimated from `B` tokens has noise falling as `~1/√B`. Small `B`: doubling it nearly halves the noise — each step is almost twice as useful, so doubling the batch is *free* (same total tokens, half the steps, same loss). Large `B`: the estimate is already nearly exact; doubling it buys almost nothing per step, so you're spending twice the compute per step for the same trajectory — pure waste.

The crossover is the **critical batch size** (McCandlish et al., 2018), estimable from the **gradient noise scale** — roughly, the batch size at which gradient noise stops dominating. Below it, "perfect scaling" (batch × 2 ⇒ steps ÷ 2); above it, diminishing returns. Two practical corollaries seen in every modern run:

- **Batch warmup / staging** (the Llama 4M→8M→16M pattern): the noise scale *grows* during training — early on, gradients across data agree (everything reduces loss), so noise is low and the critical batch small; later, the remaining signal is subtler and noisier, supporting larger batches. Staged batches track the critical batch upward. (Bonus: small early batches = more early steps = gentler on the not-yet-stable network, complementing LR warmup.)
- **Large batch ⇒ retune the LR.** Batch size and learning rate are coupled (bigger batch ⇒ less noise ⇒ larger stable steps); the Part 2.4 loss-landscape cautions about large-batch training are this coupling's sharp edge.

## The quantities you monitor, in these units

A pretraining run is flown on ~four gauges, all per-token or per-step ([2.2/05](../../part2_neural_network_fundamentals/2.2_backpropagation/05_gradient_pathologies.md) covered the pathologies they detect):

| Gauge | Healthy | The failure it catches |
|---|---|---|
| Loss (nats/token) | starts at `ln V`, falls smoothly ([6.1/01](../6.1_pretraining_objectives/01_causal_lm.md)) | spikes (data poison, instability), plateaus, *too-fast descent past the unigram floor* (leakage) |
| Global grad-norm | O(1), stable band | explosions → clipping engaged ([2.4](../../part2_neural_network_fundamentals/2.4_optimization/)) |
| Tokens/sec (MFU) | steady | stragglers, dataloader stalls — the systems side, Part 12 |
| Eval loss at checkpoints | tracks train loss | divergence = repetition beyond budget ([6.2/06](../6.2_data/06_the_data_wall.md)) or contamination artifacts |

The unit discipline pays immediately: "loss spike at step 700k" locates the *tokens* responsible (step × batch = position in the stream), which is how bad data shards actually get found and skipped in practice.

## Why it matters in modern LLM work

- **Reading training reports:** "batch 16M, 940k steps, WSD with 600B-token decay" is a complete run specification once you hold the identities in this file — and gibberish otherwise.
- **The token denomination is the bridge** between 6.3's laws (budgets in tokens), 6.2's mixtures (weights over token streams), and 2.4's schedules (horizons in steps = tokens ÷ batch).
- **Critical-batch reasoning** is the principled answer to "why not just use more GPUs?" — the parallelism ceiling for a *single* run is an optimization property, not a hardware one (Part 12 picks up from there).

## Self-check

1. Convert: a 12T-token run at batch 8M — how many steps? At `S = 4096`, how many sequences per step?
2. Why is gradient accumulation exactly equivalent to a larger batch, and what does it decouple from what?
3. Explain "free scaling below the critical batch size" in one sentence about noise, and what changes above it.
4. Why does the critical batch size *grow* during training, and what run-design pattern is the response?
5. A run's first logged loss is 8.9 with `V = 128,256`. Diagnose — and say what the *actual* signature of target leakage would be.

### Answers

1. `12e12 / 8e6 = 1.5M` steps; `8e6 / 4096 ≈ 1,950` sequences per step.
2. The batch gradient is a mean over examples, and a mean can be computed in chunks: summing micro-batch gradients then stepping once gives bit-for-the-same-math the large-batch update. It decouples the *statistical* batch (chosen for optimization) from the *memory* batch (what fits on a device) — so batch size is a free design parameter, not a hardware constraint.
3. Below critical, the gradient is noise-dominated, so doubling `B` roughly halves the variance and each step does ~double the work — same tokens, half the steps, same final loss. Above it the gradient is already signal-dominated; extra samples refine an estimate that's good enough, so steps stop getting better and compute-per-step is wasted.
4. Early training's gradient signal is huge and consistent across data (every example agrees: fix the unigram stats), so relative noise is low and small batches suffice; as training progresses the consensus direction shrinks and per-example gradients disagree more, raising the noise scale and hence the batch size worth paying for. Response: **batch-size staging/warmup** — 4M → 8M → 16M as the run matures.
5. `ln(128256) ≈ 11.76` is the step-0 value, and a random network **cannot** start below it — its logits are independent of the targets, so `E[L] ≈ ln V + σ²/2 ≥ ln V`. So 8.9 says the logged value **isn't step 0**: almost always logging lag or an average over the first window, during which the model rapidly banks part of the unigram distribution (entropy only ~5–6 nats on real text). Benign — worth confirming, not chasing. **Leakage has a different signature entirely:** it doesn't move the starting value (a leaking model also starts at `ln V`), it changes the *slope* — loss plunges *through* the unigram floor toward 0 within hundreds of steps. Watch the trajectory, not the first point; the threshold is "well below ~5 nats far too early" ([6.1/01](../6.1_pretraining_objectives/01_causal_lm.md) works the numbers).

## Exercise

Specify the arithmetic skeleton of a run: 3B params, 6T tokens, `S = 4096`, WSD schedule, hardware that fits a 0.25M-token micro-batch per accumulation chunk. (a) Choose a staged batch plan (state your stages and when they switch, in tokens), compute total optimizer steps, and the gradient-accumulation factor at each stage. (b) Place the WSD decay phase ([6.2/04](../6.2_data/04_mixtures_and_midtraining.md)) in *steps* and in *tokens*. (c) Compute `C = 6ND` and, from [6.3/03](../6.3_scaling_laws/03_beyond_chinchilla.md)'s table, name the model this run most resembles in tokens-per-param. (d) Your loss spikes at step 1.1M — give the two-line procedure to locate and handle the offending data, and the gauge that should have fired first.
