# The Corpora — and What "One Epoch" Means Now

Pretraining data is the least-published, highest-leverage part of the recipe: architectures converged ([ARCHITECTURE.md](../../ARCHITECTURE.md)) while data pipelines diverged into the actual competitive moat. This file maps the public corpus lineage and fixes a piece of vocabulary that quietly changed meaning since the BERT era: the **epoch**.

## The raw substrate: Common Crawl

Nearly every web-scale corpus is a filtered view of **Common Crawl** — a nonprofit that has crawled the web roughly monthly since 2008, publishing raw page archives (WARC) and extracted text (WET). Two facts set the whole field's shape:

- **It's enormous and mostly garbage.** Each snapshot is billions of pages: boilerplate, SEO spam, machine-translated sludge, porn, duplicated templates. Raw CC is unusable; *every* corpus below is defined by **what it deletes** ([file 03](03_quality_filtering.md)).
- **It's common.** Everyone starts from the same crawls, so differences between models' data are differences in *pipeline* — filtering, dedup, mixing — not in access to some secret internet.

## The public lineage

| Corpus | Year | Size (approx.) | What it contributed |
|---|---|---|---|
| **C4** | 2019 | ~156B tokens | T5's cleaned CC snapshot; the canonical *heuristic* filter (see [file 03](03_quality_filtering.md)) |
| **The Pile** | 2020 | 825 GiB (~300B tokens) | The *curation* thesis: 22 deliberate sources (arXiv, GitHub, StackExchange, books, PubMed) — diversity as a design goal |
| **RedPajama v1** | 2023 | 1.2T tokens | Open reproduction of Llama-1's recipe; made "the mixture" public vocabulary |
| **RefinedWeb** | 2023 | ~600B released (of ~5T claimed) | The counter-thesis: *filtered web alone* matches curated mixtures — filtering rivals curation |
| **Dolma** | 2023 | ~3T tokens | AI2's fully documented open pipeline (used for OLMo) |
| **FineWeb** | 2024 | **15T tokens** | HuggingFace's CC distillation with published ablations for every pipeline stage |
| **FineWeb-Edu** | 2024 | 1.3T (strict) / 5.4T | Model-based *educational-value* filtering; the current open state of the art ([file 03](03_quality_filtering.md)) |

Read the column of years against training budgets: Llama 1 trained on 1.0–1.4T tokens, Llama 2 on 2T, Llama 3 on **15T**, Qwen 2.5 on ~18T, DeepSeek-V3 on 14.8T. The public corpus frontier and the frontier training budget are the same order of magnitude — which is why [file 06](06_the_data_wall.md) (the data wall) is a live concern and not a thought experiment.

Two structural notes on the table: the Pile-vs-RefinedWeb tension (curate many sources vs. filter one source harder) was effectively resolved as *"both"* — modern mixtures are filtered-web *plus* deliberately upsampled specialty domains ([file 04](04_mixtures_and_midtraining.md)). And token counts are tokenizer-dependent (~3.5–4 chars/token for English with modern vocabularies), so cross-corpus comparisons are approximate by construction.

## What "one epoch" means now

In the BERT era an epoch was a real unit: pass over the dataset, shuffle, repeat — dozens of epochs were normal. At LLM scale the word survived but the referent changed, and it's worth being precise:

**The budget is tokens, not passes.** A run is specified as "train for `D` tokens" ([6.3](../6.3_scaling_laws/02_chinchilla.md) sets `D`), and the data pipeline's job is to *stream* that many tokens from the weighted mixture. Whether that constitutes one pass over anything is an output of the process, not an input.

**Each source has its own effective epoch count.** The mixture assigns each source a weight; weight ÷ source size = how many times that source is passed. From the Llama-1 paper's own table: CommonCrawl was seen ~1.1 times, but Wikipedia ~2.4 times and books ~2.2 times — the corpus as a whole has no single epoch count. "Epochs" became a *per-source* quantity, and choosing those numbers is exactly the mixing problem of [file 04](04_mixtures_and_midtraining.md).

**"One epoch" as a design norm ≈ "avoid repetition by default."** The scaling-era default (post-GPT-3) was to keep effective epochs near 1 for the bulk sources — fresh tokens beat repeated ones, and supply seemed infinite. Both halves of that are now qualified: repetition is measurably fine up to a few epochs, and supply is not infinite — [file 06](06_the_data_wall.md).

The practical takeaway: when a paper says "trained for 15T tokens," you know the *budget*; you know nothing about repetition until you see the per-source table — and whether a paper publishes that table is a good proxy for how seriously to take its data claims.

## Why it matters in modern LLM work

- **Corpus names are load-bearing context.** "Ablated on C4" vs. "on FineWeb-Edu" are experiments on different planets; reading data-curation papers requires knowing which generation of corpus is the baseline.
- **The mixture table is the model card's most informative section** — and the one most often missing. Per-source weights and effective epochs predict a model's flavor (code-heavy? multilingual?) better than any architecture detail.
- **All downstream data files** ([02](02_deduplication.md)–[06](06_the_data_wall.md)) operate on the objects defined here: a raw substrate, a filtered corpus, a weighted mixture, a token budget.

## Self-check

1. Why does "everyone starts from Common Crawl" imply that data pipelines, not data access, are the moat?
2. A model card says "trained on 12T tokens." What can you *not* conclude about epochs, and what table would you need?
3. State the Pile thesis and the RefinedWeb antithesis in one sentence each, and the synthesis actually used today.
4. Wikipedia is a fraction of a percent of Common Crawl's size, yet Llama-1 saw it ~2.4 times while seeing CC ~1.1 times. What does that assert about token value, and which file's machinery justifies the assertion?

### Answers

1. Because the input is shared, the output differences must come from processing: filtering thresholds, dedup aggressiveness, mixing weights, synthetic augmentation. Two labs with identical crawls and different pipelines produce measurably different models — so the pipeline is where the competitive information lives, which is also why it's the least-published part.
2. Nothing about repetition: 12T could be one pass over 12T distinct tokens or four passes over 3T. You need the per-source mixture table — weight and size per source — from which effective epochs per source follow. Without it, the headline number specifies the budget, not the data.
3. Pile: quality comes from *curating many deliberate sources*. RefinedWeb: quality comes from *filtering one huge source hard enough* — web is all you need. Synthesis: filtered web as the bulk carrier, plus deliberate upsampling of high-value domains (code, math, encyclopedic, books) on top — the modern mixture is both.
4. It asserts tokens are not equally valuable — a Wikipedia token is worth paying for twice while a marginal CC token is barely worth one pass — i.e., mixing weights encode a per-domain value judgment. [File 04](04_mixtures_and_midtraining.md) owns how those judgments are made, and [file 06](06_the_data_wall.md) covers why repeating good data beats acquiring more bad data.

## Exercise

Reconstruct the token accounting for a hypothetical 8B model trained for 10T tokens on: FineWeb-Edu-strict (1.3T), a 500B-token code corpus, 100B of "encyclopedic + books," and 50B of math. (a) Propose mixture weights that keep code at ~15% and math at ~3% of the stream, and compute the effective epochs each source undergoes. (b) Identify which source blows past 4 effective epochs — the threshold [file 06](06_the_data_wall.md) will justify — and propose two distinct remedies (change the weights; change the corpus). (c) One sentence: why is "just train on more FineWeb-Edu" not an available remedy here?
