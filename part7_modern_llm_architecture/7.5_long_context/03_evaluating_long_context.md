# Evaluating Long Context

The fifth wall from [file 01](01_where_quadratic_bites.md), and the one that isn't computational: **a model that accepts 128K tokens may not be able to use them.** Advertised context length is a statement about memory allocation, not capability, and the gap between the two is large enough that measuring it became its own subfield. This file is about why the obvious metrics fail and what replaced them.

## Why perplexity fails

The instinct is to measure loss on long documents. It doesn't work, and the reason is structural rather than a matter of insufficient care.

Perplexity averages over **all** tokens, and the overwhelming majority of next-token predictions are locality-dominated — determined by the last few dozen tokens. A model with a 4K effective window scores nearly as well on a 128K document as one that genuinely uses all 128K, because the tokens where long-range information matters are a **small minority of the average**. The signal you want is diluted by roughly the ratio of local to long-range-dependent tokens.

Two verified illustrations already in the curriculum: a sliding-window model with `w = 4096` has near-unchanged perplexity but fails retrieval at 100K ([7.1/04](../7.1_attention_variants/04_local_and_sparse_patterns.md)), and [5.3/05](../../part5_transformer_rebuilt/5.3_positional_information/05_context_length_extension.md) makes the same point about context-extension methods that look fine on perplexity and break on retrieval. This is the same **metric-shape** problem as the emergence debate ([6.3/04](../../part6_pretraining_paradigms/6.3_scaling_laws/04_emergence_debate.md)): an average over a mostly-easy population hides the behavior on the hard minority. The fix there and here is identical — measure the thing you care about directly.

## Needle in a Haystack, and its limits

The response was **NIAH**: hide a specific fact ("the best thing to do in San Francisco is eat a sandwich at Dolores Park") at a controlled depth inside a long filler document, ask for it, and plot accuracy over the grid of (context length × insertion depth). It became the standard long-context marketing artifact — those green-and-red heatmaps in every model release.

NIAH did real work: it exposed models whose advertised context was fiction, and it made the **"lost in the middle"** pattern visible (accuracy high at the start and end of context, sagging in the middle — Liu et al., 2023). But it is a weak test, in specific ways worth knowing:

- **It's a single exact-match lookup.** The needle is lexically distinctive and semantically alien to the filler, so it's findable by nearly-keyword matching — closer to `grep` than to comprehension.
- **One fact, one hop.** No aggregation, no multi-fact reasoning, no tracking state across the document.
- **It's trivially gameable** — and it's benchmarked so heavily that optimizing for it is a rational lab strategy, which makes it a saturated metric ([Part 11.1](../../review_outline.md) on benchmark rot).
- **Passing it means "can retrieve one distinctive string,"** which is close to the minimum viable long-context capability, not evidence of using the context.

## RULER and the effective-context idea

**RULER** (Hsieh et al., 2024) is the standard replacement and its design fixes NIAH's specific weaknesses by making the task **synthetically generatable at any length with controllable difficulty**, across four families:

- **Retrieval** — multi-needle variants: many needles, needles to be distinguished from similar distractors, needles requiring the *k*-th match rather than any match.
- **Multi-hop tracing** — variable tracking: `X1 = 5`, later `X2 = X1`, later `X3 = X2`… then query `X3`, forcing a chain through the context.
- **Aggregation** — common/frequent-words extraction, which requires reading *all* of the context rather than finding one span.
- **Question answering** — real QA with distractor documents inserted to lengthen the context.

RULER's headline finding is the one to carry: **almost every model's *effective* context is far shorter than its *advertised* context.** Defining effective length as "the longest length at which the model still beats a strong short-context baseline," most models evaluated fell dramatically short of their claimed windows, and performance degraded steadily with length rather than holding until a cliff.

That gives the vocabulary the field now uses:

| Term | Meaning |
|---|---|
| **Advertised context** | What the config permits — a memory-allocation fact |
| **Effective context** | Where the model still performs acceptably on real tasks |

Always ask for the second. And note the asymmetry in incentives: the first is free to increase (change a number, rescale RoPE), the second costs training tokens on genuinely long documents ([file 02](02_training_at_long_context.md)).

Related evals worth knowing: **LongBench** (realistic bilingual long-document tasks), **∞Bench**, **LOFT** (long-context as a retrieval-system replacement), and NIAH's harder descendants. Also worth flagging: **aggregation tasks are the honest hard case** — a model can pass any number of retrieval probes while failing "count how many times X appears," because retrieval needs one successful lookup and aggregation needs *complete* coverage.

## What to actually do

If you're evaluating a model for long-context use:

1. **Never rely on perplexity** for this question.
2. **Treat NIAH as a smoke test** — necessary, nowhere near sufficient. Passing means "not broken."
3. **Run RULER-style tasks at your target length**, including multi-hop and aggregation, and find the length where it degrades — that's your effective context.
4. **Test at your own depth distribution.** "Lost in the middle" means position matters; if your documents put critical information at 60% depth, test at 60% depth.
5. **Match the task shape.** Retrieval-flavored (find the clause) and aggregation-flavored (summarize all sections) workloads fail differently and need separate measurement.

## Why it matters in modern LLM work

- **It's the correct skepticism to apply to every context-length claim**, including ones backed by NIAH heatmaps.
- **It closes Part 7's loop.** All the machinery in 7.1–7.5 exists to make long context *possible*; this file is the check on whether it's *real*. A 1M-token model with a 32K effective context has spent its engineering budget on a number.
- **The metric-dilution lesson generalizes** — averages over easy-dominated populations hide minority behavior, which is the same trap as exact-match emergence ([6.3/04](../../part6_pretraining_paradigms/6.3_scaling_laws/04_emergence_debate.md)) and as judging separation by raw cosine ([1.4/04](../../part1_math_foundations/1.4_optional_deeper_knowledge/04_linear_readout_and_identifiability.md)).

## Self-check

1. Explain mechanically why perplexity can't distinguish a 4K-window model from a true 128K model on long documents.
2. What did NIAH genuinely accomplish, and name three specific ways it's a weak test.
3. What is "lost in the middle," and what practical evaluation rule follows from it?
4. Name RULER's four task families and say which one a retrieval-capable-but-not-comprehending model would fail first.
5. Distinguish advertised from effective context, and explain why the incentives around them differ.
6. A vendor shows a solid-green NIAH heatmap at 1M tokens. What two tests do you run before believing the capability claim?

### Answers

1. Perplexity is a mean over all token positions, and the great majority of next-token predictions depend only on nearby context. The tokens whose prediction genuinely requires distant information are a small minority, so their contribution to the average is swamped — the 4K model pays a tiny penalty on a few tokens and is otherwise identical. The metric is diluted by the local-token majority, exactly like exact-match metrics being dominated by threshold effects ([6.3/04](../../part6_pretraining_paradigms/6.3_scaling_laws/04_emergence_debate.md)).
2. It **exposed fictitious context windows** and made the "lost in the middle" positional pattern visible — real contributions. Weaknesses: (a) it's a single exact-match lookup of a lexically distinctive string, solvable by near-keyword matching; (b) one fact, one hop — no aggregation or multi-fact reasoning; (c) it's heavily optimized against and saturated, so passing is uninformative about anything beyond minimum viability.
3. Accuracy on retrieving information is higher when the target sits near the **beginning or end** of the context and sags for material in the **middle** — a U-shaped curve in insertion depth. Rule: **evaluate at the depth distribution your application actually produces**, and never report a single depth-averaged number, because it hides the worst region.
4. Retrieval (multi-needle), multi-hop tracing (variable chains), aggregation (frequent-word extraction), and QA with distractors. Such a model fails **aggregation** first: retrieval needs one successful lookup while aggregation requires reading *all* of the context completely, so partial attention over the window passes the former and fails the latter.
5. **Advertised** = the maximum sequence the config accepts, a memory-allocation fact changeable by editing a number and rescaling RoPE ([5.3/05](../../part5_transformer_rebuilt/5.3_positional_information/05_context_length_extension.md)). **Effective** = the length at which the model still performs acceptably on real tasks, which requires training tokens on genuinely long documents with long-range structure ([file 02](02_training_at_long_context.md)). Advertised length is nearly free and highly marketable; effective length is expensive and only visible under adversarial evaluation — so the two diverge systematically, and always in the same direction.
6. (a) A **RULER-style suite at 1M** including multi-hop tracing and aggregation, to find where performance actually degrades — NIAH's single-lookup task is the one most likely to be optimized for. (b) A test at **my own depth and task distribution**, since "lost in the middle" plus task-shape sensitivity means their grid may not cover my case. (Bonus: check whether the long-context extension stage used genuinely long documents, per [file 02](02_training_at_long_context.md).)

## Exercise

Build a miniature RULER and measure an effective context. (a) Pick any model you can run and implement four probes at several lengths (say 2K, 8K, 32K, and its advertised max): single-needle NIAH, multi-needle with similar distractors, variable-chain multi-hop (`X1=5; …; X2=X1; …; X3=X2` then query), and an aggregation task (count occurrences of a word inserted `n` times). (b) For each, sweep insertion depth in 10% increments and plot the heatmap — try to reproduce "lost in the middle." (c) Define a short-context baseline (the same model at 2K with the needle guaranteed in-window) and report the **effective context** as the longest length where each task stays within 5% of baseline. Report four numbers, one per task family. (d) One paragraph: which task family degraded first, how far the effective context fell short of advertised, and what you'd tell a colleague planning to feed this model 500K-token documents.
