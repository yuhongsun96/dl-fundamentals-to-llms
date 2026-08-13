# Deduplication — and Its Sibling, Decontamination

The web is massively self-copying: mirrors, syndication, templates, quotes, boilerplate. Deduplication is the pipeline stage that fights this, and it earns its own file because it defends against **three different failure modes at once** — wasted compute, memorization, and corrupted evaluation — and because the third one (contamination) is really the same operation pointed at a different target.

## The three levels of duplication, and the tool for each

Duplication comes in grades, and each grade needs different machinery:

**1. Exact duplicates — hashing.** Same document, byte-for-byte (mirrors, re-crawls). Hash every document (MD5/SHA over normalized text), keep one per hash. Cheap, exact, always done first.

**2. Near-duplicates — MinHash + LSH.** Same content, trivially different wrapper (timestamps, ads, "22 comments"). The standard tool is **MinHash**: shingle each document into overlapping n-grams, hash each shingle many ways, keep the minimum hash per function — the probability two documents share a minimum equals their **Jaccard similarity** (`|A∩B| / |A∪B|`) over shingle sets, so a small signature estimates set overlap without comparing sets. **Locality-sensitive hashing** (banding the signatures) then finds candidate pairs without the quadratic all-pairs comparison. Corpus-scale dedup (RefinedWeb, FineWeb) is MinHash-LSH at a chosen similarity threshold.

**3. Repeated substrings — suffix arrays.** Distinct documents sharing long verbatim passages (licenses, disclaimers, chain letters, quoted paragraphs). Lee et al. (2021, *Deduplicating Training Data Makes Language Models Better*) build a suffix array over the whole corpus and delete every repeated substring above a length threshold (they used 50 tokens) regardless of which documents contain it. This catches what document-level methods structurally cannot: a boilerplate paragraph appearing in a million otherwise-unique pages.

## Why bother: the three payoffs

**Compute.** A duplicated token buys the same gradient twice at full price. With mixtures already engineered token-by-token ([file 01](01_corpora_and_epochs.md)), uncontrolled duplication silently rewrites your mixture — whatever's most duplicated is most upweighted, and web duplication does not correlate with quality. Note the flip side: dedup is what makes *deliberate* repetition meaningful ([file 06](06_the_data_wall.md)) — you can't reason about "4 epochs of good data" if the corpus already contains an unknown, skewed number of hidden epochs.

**Memorization.** Carlini et al.'s extraction work established the dose-response: **the probability a sequence can be extracted verbatim grows roughly log-linearly with its duplicate count in training**, and larger models memorize more at every duplication level. Duplication is the single strongest *controllable* driver of verbatim regurgitation — which is a privacy problem (PII), a copyright problem (the exact issue in ongoing litigation), and a quality problem (parroting instead of generalizing). Lee et al. showed the direct fix works: substring-deduped training reduces emitted memorization ~10× while slightly *improving* perplexity — you lose nothing by deleting the copies.

**Evaluation integrity** — which is the bridge to the next section.

## Decontamination: dedup pointed at the benchmarks

**Contamination** = your evaluation data is in your training data. With web-scale crawls this is the *default state*, not an accident: benchmark test sets live on GitHub, in blog posts about the benchmark, in papers on arXiv. A contaminated eval measures recall, not capability.

**Decontamination is deduplication with the benchmark as the query.** Standard process: take each eval item, generate its n-grams (13-gram overlap is a common operating point, following GPT-3's report), scan the training corpus for matches, and drop (or flag) overlapping training documents *before* training. Understand what this is and isn't:

- It's a **process guarantee about the pipeline**, not a property you can fully verify from outside — which is why open-data models (OLMo, and FineWeb's published pipeline) matter for trustworthy comparisons.
- **N-gram matching catches verbatim overlap only.** Paraphrased test items, translated items, or the *answers discussed in prose* slip through — and LLM-generated rephrasings of benchmarks make paraphrase-contamination increasingly the common case. The eval-side consequences (benchmark rot, and detection methods) are Part 11.1's; the data-side lesson is that decontamination is necessary hygiene and nowhere near sufficient.
- There's an unfixable tension: the most aggressive decontamination starts deleting genuinely useful text (every math discussion resembling GSM8K), so labs choose an overlap threshold — one more unpublished judgment call in the pipeline.

## Why it matters in modern LLM work

- **Reading eval claims:** "we decontaminated against all reported benchmarks" is now table stakes in model cards — knowing it means 13-gram-style verbatim matching tells you exactly how much (and how little) the claim guarantees.
- **Memorization** is the mechanism behind extraction attacks, the copyright docket, and PII leakage — and duplication count is its main controllable knob.
- **Every serious pipeline runs all three levels.** FineWeb's ablations show dedup choices (how aggressive, at which granularity) measurably move downstream performance — dedup is a tuned stage, not a checkbox.

## Self-check

1. Why does document-level dedup (even perfect near-dup detection) fail to stop the memorization of a licensing paragraph that appears on a million pages?
2. What does MinHash actually estimate, and what does LSH add on top?
3. Explain "duplication silently rewrites your mixture" in one sentence, and why that framing makes dedup a *prerequisite* for the mixing decisions of [file 04](04_mixtures_and_midtraining.md).
4. Your model scores 92% on a benchmark, and n-gram decontamination was performed. Name two contamination paths that survive.
5. Dedup *removes* repetition; [file 06](06_the_data_wall.md) *advocates* multi-epoch repetition. Reconcile.

### Answers

1. Because each containing page is a distinct document — pairwise Jaccard similarity between two pages sharing one paragraph is low, so no document-level method flags them. The repeated *substring* is what recurs a million times, which is exactly the granularity suffix-array dedup operates at and document-level methods can't see.
2. MinHash signatures estimate the **Jaccard similarity** of two documents' shingle sets — the probability that a random hash function's minimum agrees equals the Jaccard index. LSH makes the search tractable: by banding signatures into buckets, near-duplicates collide with high probability while distant pairs don't, so you only compare within buckets instead of all `O(n²)` pairs.
3. Whatever is duplicated most is trained on most, so the *effective* mixture is the intended weights multiplied by an unknown per-document duplication factor — you cannot set meaningful per-source weights or epoch counts ([file 01](01_corpora_and_epochs.md)) over a corpus whose internal multiplicities you haven't normalized to one.
4. (a) **Paraphrase/translation contamination** — the item's content present in different words, invisible to n-gram matching; (b) **discussion contamination** — solutions and answers discussed in prose (forum walkthroughs of benchmark problems) that teach the answer without reproducing the text. Both are increasingly common as benchmarks age; detection from the eval side is Part 11.1.
5. They target different quantities. Dedup removes *uncontrolled, skewed* repetition baked into the corpus (some documents ×1, some ×10,000, uncorrelated with value); multi-epoch training applies *uniform, chosen* repetition to a cleaned corpus. Dedup is what converts repetition from a hidden corruption of the mixture into a controllable training-budget parameter.

## Exercise

Implement toy MinHash: shingle two documents into character 5-grams, compute true Jaccard similarity exactly, then estimate it with `k ∈ {16, 64, 256}` hash functions and compare the estimate's error against the `~1/√k` you'd expect ([1.4/09](../../part1_math_foundations/1.4_optional_deeper_knowledge/09_approximations_and_orders_of_magnitude.md)). Then construct the failure case from self-check 1: ten documents, each 90% unique text plus one shared 50-token paragraph. Confirm pairwise Jaccard stays below any sane dedup threshold, then find the shared passage with a (brute-force) common-substring scan — document-level blindness and substring-level visibility, both demonstrated.
