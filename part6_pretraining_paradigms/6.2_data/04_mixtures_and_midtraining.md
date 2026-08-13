# Mixtures, Upsampling, and Mid-Training

Two decisions turn a shelf of filtered corpora into a training run: **what fraction of the stream each source gets** (the mixture), and — the part the classic mental model misses entirely — **when in the run each source appears** (the schedule). The second is the big post-2023 shift: pretraining stopped being one homogeneous pass and became a *staged curriculum*, with a name for the late stage: **mid-training**.

## The mixture: weights are value judgments

A mixture assigns each source a sampling weight; weight ÷ source size = effective epochs ([file 01](01_corpora_and_epochs.md)). The canonical public example, from the Llama-1 paper:

| Source | Weight | Effective epochs |
|---|---|---|
| CommonCrawl | 67.0% | 1.10 |
| C4 | 15.0% | 1.06 |
| GitHub | 4.5% | 0.64 |
| Wikipedia | 4.5% | **2.45** |
| Books | 4.5% | **2.23** |
| arXiv | 2.5% | 1.06 |
| StackExchange | 2.0% | 1.03 |

Read the epochs column, not the weights: Wikipedia and books are deliberately repeated while GitHub is *undersampled* — every row is a judgment that a token from source X is worth `k×` a token from source Y. Three notes on how those judgments get made:

- **Mostly: ablation runs.** Train small models on candidate mixtures, evaluate, extrapolate — the same methodology as filtering ([file 03](03_quality_filtering.md)). Principled alternatives exist (**DoReMi** learns weights via a small proxy model that upweights domains where excess loss is reducible) and data-mixing scaling laws are an active area, but the frontier practice is still closer to "measured taste" than derivation.
- **Domain upsampling is capability engineering.** Code and math get weights far above their natural share of the crawl because they transfer: code correlates with reasoning benchmarks, math with math. Modern mixtures run 10–30% code (vs. Llama-1's ~4.5%) — a decision downstream of the observation, made everywhere, that "code helps everything."
- **Multilinguality is a weights war.** Every point of English you give up buys other languages; where the line sits (Llama-3: ~5% non-English; Qwen: far more) is a product decision expressed as a sampling ratio.

## The schedule: order matters

The homogeneous-pass assumption — mixture fixed from token 1 to token 15T — was quietly false at the frontier by 2024. Two facts drive the staging:

1. **The LR schedule creates phases with different characters** ([2.4/02](../../part2_neural_network_fundamentals/2.4_optimization/02_lr_schedules.md)): what the model sees during the final low-LR decay is consolidated with less catastrophic overwriting — the decay phase is when the model *keeps* things.
2. **Your best data is scarce** ([file 03](03_quality_filtering.md)): spending 50B tokens of textbook-grade data uniformly across a 15T stream dilutes it to noise; spending it *at the end, at low LR*, is measurably better.

Put together: **bulk web early, precious data late.**

- **MiniCPM (2024)** made the recipe explicit with WSD (warmup–stable–decay, [2.4/02](../../part2_neural_network_fundamentals/2.4_optimization/02_lr_schedules.md)): stable phase on the ordinary mixture; decay phase on a mixture spiked with high-quality and even SFT-style data. The WSD shape and the data schedule are one design — the "D" phase exists partly *to be* the high-quality phase.
- **Llama 3 (2024)** reports **annealing**: the final stage of pretraining upweights very high-quality (notably math/code) data as LR decays to zero, with clear benchmark gains at 8B scale.

  > **The term.** Borrowed from metallurgy: heat a metal, then cool it *slowly*, and its structure settles into a strong, stable state (cool it fast and you freeze in defects). In training, the **learning rate is the temperature** — "annealing" is the final slow-cooling leg of the LR schedule, and by point 1 above, whatever the model sees during that cooldown settles in with little subsequent disturbance. So "annealed on dataset X" means: run the low-LR final phase *while feeding X* — deliberately using the schedule's consolidation window to lock in your best data. (Same metaphor as *simulated annealing* in optimization; here it names the LR decay leg, not a separate algorithm.) The paper also discloses the clever corollary: **annealing as a measurement instrument** — to value a candidate dataset, anneal a checkpoint on a mix containing it and read the benchmark delta. Cheaper than full ablation runs, and now standard practice.
- **Long-context extension lives here too:** train the bulk run at moderate `S`, then a late stage on long sequences with adjusted RoPE base ([5.3/05](../../part5_transformer_rebuilt/5.3_positional_information/05_context_length_extension.md)) — because `O(S²)` attention makes long sequences expensive, you buy length only for the tokens that need it.

## "Mid-training" — the stage that got a name

The industry term for everything between "the bulk pretraining run" and "post-training proper" (Part 8). Typical contents: the high-quality anneal, long-context extension, large doses of instruction-*formatted* data (not preference-tuned — just Q&A/textbook shaped), domain infusions (math, code), sometimes early synthetic data ([file 05](05_synthetic_data.md)). Three reasons it earned a name:

- **Different data contract.** Pretraining tolerates noise at scale; mid-training uses small, curated, sometimes synthetic corpora where single-source quality matters enormously.
- **Different failure modes.** The risk isn't underfitting the web; it's *forgetting* (the anneal overwrites breadth with the spike's distribution) and *contamination* (instruction-formatted and benchmark-adjacent data concentrates here — [file 02](02_deduplication.md)'s hygiene matters most at exactly this stage).
- **It blurs into post-training by design.** A base model that ends pretraining on instruction-shaped text is already half-aligned to the chat format; Part 8's SFT then has less work to do. The old clean line — "pretraining = raw text, post-training = formatted tasks" — is now a gradient, and "base model" on a 2025 model card already includes these choices.

## Why it matters in modern LLM work

- **Reading model cards:** "annealed on X," "mid-trained for long context," "two-stage data schedule" are now standard vocabulary — this file is their decoder.
- **The mixture table plus the schedule** predict a model's personality (code-strong? chat-ready out of the base?) better than any architecture detail.
- **The anneal-as-probe trick** (Llama 3) is the practical answer to "how do labs value data without burning full runs" — worth knowing as methodology, not just trivia.

## Self-check

1. Llama-1 gave GitHub 0.64 effective epochs and Wikipedia 2.45. Express what that pair of numbers claims, in one sentence about token value.
2. Why is the same 50B tokens of textbook-quality data worth more at the end of training (low LR) than spread uniformly?
3. What two properties make mid-training's contamination risk higher than bulk pretraining's?
4. Describe the "annealing as measurement" trick and why it's cheaper than a from-scratch ablation.
5. A model card says "base model." Post-2024, what can you no longer assume?

### Answers

1. That a Wikipedia token is worth roughly four GitHub tokens *to this model's goals* — enough to pay for repetition on one while leaving a third of the other unread; mixture weights are per-source value judgments made operational.
2. Two compounding reasons: **dilution** — uniformly mixed, it's 0.3% of the stream, and its gradient signal is swamped by web noise; **consolidation** — during LR decay, updates are small and less destructive, so what's learned late is retained rather than overwritten by 10T subsequent web tokens. The decay phase is the run's long-term memory window, so you spend your best tokens there.
3. (a) Its data is exactly the kind that overlaps benchmarks — instruction-formatted Q&A, curated math/code — so verbatim and paraphrase overlap with eval sets is far likelier per token than in raw crawl; (b) it happens *last and at low LR*, i.e., in the consolidation window, so anything contaminated is retained unusually well. High-risk content × high-retention phase.
4. Take one pretrained checkpoint; run a short LR-decay phase on a mixture spiked with the candidate dataset; read benchmark deltas against annealing on the control mixture. It reuses the sunk cost of the full pretraining run and only spends the anneal tokens (a few tens of billions), instead of training a model from scratch per candidate — turning data valuation from a capital expense into an experiment.
5. That it's "just next-token prediction on raw web text." A 2025 base model has typically already seen a staged curriculum: quality-annealed data, instruction-formatted text, long-context extension, possibly synthetic textbooks — so base-model behavior (e.g., answering questions in a tidy format unprompted) already reflects curriculum choices, and comparisons of "base" models are partly comparisons of their mid-training.

## Exercise

Design the data schedule for a hypothetical 3B-model, 6T-token run, given: 5T filtered web, 800B code, 60B textbook-grade synthetic, 40B instruction-formatted Q&A, 100B long-document corpus, and a WSD schedule with a 600B-token decay phase. Specify (a) the stable-phase mixture with per-source effective epochs, (b) the decay-phase mixture, (c) where long-context extension goes and why, and (d) the one dataset you'd decontaminate most aggressively and against what. Then answer: if the textbook data unexpectedly grew 10× (600B), which of your choices flip, and which are insensitive?
