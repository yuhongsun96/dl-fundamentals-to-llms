# The Data Wall — Running Out of Tokens, and What Repetition Buys

Scaling laws ([6.3](../6.3_scaling_laws/02_chinchilla.md)) tell you to grow data with compute. This file is about the constraint they take for granted: **fresh tokens are finite**. It quantifies the stock, then covers the result that changed the default response — repeating data is nearly free for a few epochs — and closes with the pressure valves.

## How much text is there?

The Epoch AI estimates (Villalobos et al., *Will we run out of data?*, 2022/2024 revision) are the standard reference points, order-of-magnitude reliable:

- **Public human-generated text:** ~**300T tokens** (indexed web, after basic quality/dedup considerations). The *high-quality* slice — the part that survives serious filtering ([file 03](03_quality_filtering.md)) — is an order of magnitude smaller.
- **Frontier consumption:** 15–18T tokens per run *already* ([file 01](01_corpora_and_epochs.md)), growing with compute budgets.
- **Their projection:** training runs fully utilize the public-text stock somewhere around **2026–2032**, depending on growth rates and how much of the stock is actually usable.

The wall isn't a cliff on a date — it's the *gradient* that already binds: each extra trillion tokens of genuinely new, high-quality text costs more than the last (deeper crawls, more repair, licensing), while compute keeps getting cheaper. Chinchilla-optimal scaling wants `D` to grow with `√C` ([6.3/02](../6.3_scaling_laws/02_chinchilla.md)) — data demand grows polynomially while supply grows at roughly "the rate humanity writes."

## The result that reframed it: repetition is cheap (up to a point)

Muennighoff et al. (2023, *Scaling Data-Constrained Language Models*) asked the right question: fixing the *unique*-token supply, how does loss behave as you re-read? Their fitted answer, from hundreds of runs:

- **Up to ~4 epochs, repeated tokens are worth almost as much as fresh ones.** The loss curve tracks the infinite-data curve closely.
- **Value decays smoothly after that**; by ~**16 epochs**, additional repeats contribute approximately **zero** — the model has extracted what the corpus has to give, and further compute on it is wasted (they model this as an exponentially decaying effective-token value).

Two immediate consequences:

1. **The wall moves ~4× out, for free.** A 300T stock supports ~1.2Q effective tokens before repetition costs become material. Panic postponed, not canceled.
2. **The quality-vs-quantity trade tilts toward quality.** The strict FineWeb-Edu cut (1.3T tokens, [file 03](03_quality_filtering.md)) looks small against a 15T budget — but at 4 cheap epochs it covers 5.2T effective tokens, and 4 epochs of excellent data beats 1 epoch of mediocre data at equal compute. Aggressive filtering plus repetition is a *package*; this file supplies the second half of the argument that [file 03](03_quality_filtering.md)'s economics started. (Precondition: dedup — repetition counts are meaningless over a corpus with hidden internal multiplicities, [file 02](02_deduplication.md).)

The nuance to keep: the 4-epoch shoulder is a fitted regularity for bulk pretraining loss, not a law of nature — small curated sets in mid-training routinely see more epochs ([file 04](04_mixtures_and_midtraining.md): Llama-1 already gave Wikipedia 2.4), and the tolerable repetition presumably varies with data type. Treat "≤4 free, ~16 dead" as the calibrated default, not scripture.

## The pressure valves

Ranked roughly by how much relief each is actually delivering:

- **Synthetic data** ([file 05](05_synthetic_data.md)) — the big one. For anything a generator+verifier can produce, supply is no longer the constraint; verification and diversity are. This is the field's principal bet, and the reason the 2026–2032 projections read more like "end of the found-text era" than "end of scaling."
- **Better extraction from the same stock** — quality filtering ([03](03_quality_filtering.md)) and rephrasing-as-repair (WRAP, [05](05_synthetic_data.md)) raise value-per-crawled-byte; more of the 300T becomes usable rather than discarded.
- **Non-public text** — licensing deals (news archives, publishers, code hosts) and proprietary user data. Real but incremental: the private stock isn't orders of magnitude beyond the public one, and it's legally gated.
- **Other modalities** — video/audio/images dwarf text in raw bytes (Part 10), and multimodal training does consume them; how much *language* capability transfers per non-text token is still an open empirical question.
- **Post-training compute** — the deepest response: if pretraining data saturates, spend compute where data isn't the input — RL on verifiable tasks generates its own experience (Part 8.3), and inference-time compute scales capability with zero new corpus ([6.3/03](../6.3_scaling_laws/03_beyond_chinchilla.md)). The o1/R1 turn is partly *caused* by the wall: compute kept growing, found text didn't, and the compute had to go somewhere.

## Why it matters in modern LLM work

- **It explains the 2024–25 research agenda.** Synthetic pipelines, data-efficiency papers, multi-epoch recipes, licensing deals, and the RL/inference-compute turn are all responses to the same constraint — this file is the "why now" behind half of Parts 8–9's material.
- **Multi-epoch is respectable again.** Post-2023 model cards state effective epochs without embarrassment; "we never repeat data" stopped being a brag when repetition was shown to be nearly free.
- **Budget reasoning changes:** below ~4 epochs of your best available data, more compute should buy more *passes*; beyond it, more compute should buy better *data* (synthesis, filtering) or move to post-training. That's a genuinely useful decision rule.

## Self-check

1. Why is the data wall better described as a rising marginal cost than as a date?
2. State the Muennighoff result as two numbers, and what each implies for a lab holding 2T unique high-quality tokens and a 12T-token compute budget.
3. Why is deduplication a *precondition* for reasoning about repetition budgets?
4. Rank synthetic data and licensing deals as pressure valves, and justify the gap between them.
5. Connect the data wall to the rise of reasoning-RL models in one causal sentence.

### Answers

1. Because nothing happens on the date — instead, each marginal fresh token already costs more than the last (deeper crawls, repair, licensing, lower quality at the frontier of the stock) while demand grows with compute; "the wall" is the point where marginal fresh tokens cost more than their alternatives (repeats, synthesis), and that crossover is already behind us for parts of the mixture.
2. ~**4** epochs at near-full value, ~**16** epochs at ~zero marginal value. For the lab: 12T budget ÷ 2T unique = 6 epochs — mostly inside the cheap zone; expect slightly worse than 12T fresh tokens but far better than stretching to lower-quality data, and past ~8T of that budget the marginal pass is weakening — the last few T are better spent on synthesis or mid-training data than on epochs 7+.
3. Because a non-deduped corpus already contains hidden, *skewed* repetition — some documents effectively at 1 epoch, some at 1,000 ([file 02](02_deduplication.md)). "Train for 4 epochs" is only a meaningful, uniform quantity when the baseline multiplicity is 1; otherwise the popular documents are at 4,000 effective epochs (deep in the dead zone, driving memorization) while you believe you're at 4.
4. Synthetic is the larger valve: its supply scales with compute (generate more) rather than with a fixed stock, and verification can push quality above found text for checkable domains. Licensing adds a one-time, bounded increment — valuable for specific gaps (news, books), but the private text stock is within an order of magnitude of the public one and cannot grow with demand. One is a tap; the other is a bucket.
5. With compute budgets still growing but fresh pretraining tokens increasingly scarce and expensive, the marginal FLOP shifted to places that don't consume found text — RL on verifiable tasks and inference-time reasoning — so the data wall is a direct economic cause of the o1/R1-style allocation of compute (Part 8.3).

## Exercise

Work the decision problem. You have: 3T unique tokens of good filtered web, 200B of excellent curated data, a generator+verifier pipeline that can produce math/code synthetic data at quality ≥ your web data, and a 20T-token compute budget. Using the ≤4-cheap / ~16-dead repetition model: (a) allocate the 20T among web epochs, curated epochs (in the anneal, [file 04](04_mixtures_and_midtraining.md)), and synthetic generation, stating effective epochs for each source; (b) identify which allocation the *Chinchilla-optimal* frame ([6.3/02](../6.3_scaling_laws/02_chinchilla.md)) would have naively recommended and why it's infeasible here; (c) state the single measurement you'd run first to de-risk the plan (hint: [file 04](04_mixtures_and_midtraining.md)'s anneal-as-probe).
