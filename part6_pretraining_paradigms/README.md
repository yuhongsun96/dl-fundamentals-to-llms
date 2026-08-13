# Part 6 — Pretraining Paradigms

Part 5 built the machine; this part is about the run that turns it into a language model. The 2018 mental model you're updating from is *"pick an objective (probably MLM), grab a corpus, train until the loss flattens."* Every clause of that has changed: the objective race is over (causal LM won, and the *why* is worth owning), the corpus is now the single most engineered artifact in the pipeline, the budget is set by scaling laws rather than intuition, and the run itself is a staged curriculum — bulk web early, precious data in the final anneal — not one homogeneous pass.

The outline's three headings (objectives, data, scaling laws) understate where the 2024–25 leverage actually is, so this part adds the material practitioners hit first: **training mechanics** (batches measured in tokens, sequence packing, document boundaries, loss masking), **multi-stage pipelines and mid-training**, **synthetic data**, and **the data wall** (what happens when tokens, not FLOPs, are the binding constraint).

## Structure

- **6.1 Pretraining Objectives** — what the loss is, and why one choice swept the field:
  - `01` Causal LM — next-token prediction stated precisely: the autoregressive factorization, loss at every position, why one sequence is `S` training examples.
  - `02` MLM & denoising — BERT's masked LM and T5/BART span corruption; what bidirectional buys and what it costs.
  - `03` Why decoder-only won — the file [5.1/05](../part5_transformer_rebuilt/5.1_self_attention/05_causal_and_bidirectional_masking.md) promised: supervision density, task unification, serving economics — and where encoders still live.
- **6.2 Data** — the highest-leverage, least-published part of pretraining:
  - `01` Corpora, and what "one epoch" means — Common Crawl → C4 → Pile → RedPajama → FineWeb; token accounting.
  - `02` Deduplication — exact/near/substring dedup; memorization; why dupes waste compute.
  - `03` Quality filtering — heuristic → classifier → model-based (FineWeb-Edu, DCLM); filtering as a scaling-curve shifter.
  - `04` Mixtures & mid-training — domain weights, upsampling code/math, and the stable→anneal staged run (MiniCPM, Llama 3).
  - `05` Synthetic data — textbooks (Phi), rephrasing (WRAP), distillation; model-collapse caveats.
  - `06` The data wall — how much text exists, multi-epoch training (repeating is cheap up to ~4 epochs), and the pressure valves.
- **6.3 Scaling Laws** — the quantitative machinery behind "bigger is better":
  - `01` Power laws (Kaplan) — the original exponents and what they *don't* say.
  - `02` Chinchilla — compute-optimal allocation, ~20 tokens/param, and the fine print (including the published fit that contradicts the headline).
  - `03` Beyond Chinchilla — inference-aware overtraining (Llama's 200–1900 tokens/param), and what scaling laws are used for now.
  - `04` The emergence debate — abrupt capabilities vs. metric artifacts.
- **6.4 Training Mechanics at Scale** — the unglamorous details every run lives or dies on:
  - `01` Batches, steps, and tokens — batch size measured in tokens, critical batch size, the arithmetic of a 15T-token run.
  - `02` Packing, boundaries, and loss masking — how documents become fixed-length training sequences, cross-document attention, which positions get loss.

## How to use

6.1 and 6.3 are the conceptual core — read them in order; 6.3 leans on [1.4 group D](../part1_math_foundations/1.4_optional_deeper_knowledge/09_approximations_and_orders_of_magnitude.md) for log-log fluency. 6.2 is a survey you can read non-linearly, but do `02`–`03` before `04`–`05` (the pipeline builds in that order). 6.4 is short and concrete; read it right before (or while) looking at any real pretraining codebase — it's the part that maps to actual code.

**Symbol note for 6.3:** the scaling-laws literature uses `N` = parameter count and `D` = dataset size in tokens. That collides with this repo's `D` = model width, so within 6.3 the model width is always written `d_model`. [NOTATION.md](../NOTATION.md) records the clash.

## Target time

4–6 days (it shares "weeks 4–5" with Part 7). 6.1/03, 6.2/04, and all of 6.3 are the load-bearing files; 6.2's survey files can be skimmed and revisited.

## What's deliberately omitted

- **Distributed training systems** (data/tensor/pipeline parallel, ZeRO, failure recovery) — Part 12. Here "the run" is described at the level of tokens and losses, not GPUs.
- **LR schedules and optimizers** — owned by [2.4](../part2_neural_network_fundamentals/2.4_optimization/); 6.2/04 only adds the *data-side* view of the WSD anneal.
- **Post-training data** (instruction tuning, preferences) — Part 8. The boundary is fuzzy in practice (mid-training already sneaks in instruction-formatted text), and 6.2/04 flags exactly where.
- **Long-context extension mechanics** — [5.3/05](../part5_transformer_rebuilt/5.3_positional_information/05_context_length_extension.md) and Part 7.5 own the *how*; 6.2/04 notes only *when* it happens in the pipeline.
- **Contamination's effect on reported evals** — the eval-side story is Part 11.1; 6.2/02 covers the data-side process.
