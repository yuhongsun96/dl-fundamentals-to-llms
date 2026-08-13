# Quality Filtering — Heuristics, Classifiers, and Models Rating Data

Filtering is where a corpus is actually *made* ([file 01](01_corpora_and_epochs.md): every corpus is defined by what it deletes). The methods have gone through three distinct generations — hand rules, small classifiers, LLM judgment distilled into classifiers — and the punchline of the third generation is important enough to state up front: **good filtering doesn't just save compute, it shifts the whole scaling curve.** A model trained on 1T well-filtered tokens can beat one trained on several times more raw tokens.

## The pipeline, in order

Filtering isn't one stage; it's a sequence, each pass cheaper-per-document than the next is valuable:

```
raw crawl → text extraction → language ID → URL/domain blocklists
         → heuristic rules → deduplication (file 02) → quality classifier
```

Order matters for cost (run cheap filters first, on everything) and for correctness — FineWeb's ablations found dedup and filtering *interact*, and each stage was validated by training small models on the result rather than by eyeballing documents. That methodology — **ablate data decisions with real training runs** — is the generation-three move, and it applies to every stage, not just the classifier.

## Generation 1 — heuristics (C4, Gopher)

Hand-written rules encoding "does this look like prose a human wrote for humans?" The canonical sets:

- **C4 (2019):** keep lines ending in terminal punctuation; drop pages with fewer than 3 sentences, any "bad word," the string `lorem ipsum`, or `{` (a crude JavaScript detector); drop lines containing "javascript"/cookie-notice boilerplate.
- **Gopher rules (2021):** the tightened, widely-reused version — word count in [50, 100k], mean word length in [3, 10], symbol-to-word ratio caps, bullet-point and ellipsis line-fraction caps, a minimum fraction of words containing an alphabetic character, and a "stop word presence" check.

Strengths: transparent, cheap, debuggable. Weaknesses: brittle and biased toward *tidy English prose* — C4's `{` rule deletes code wholesale, terminal-punctuation rules mangle poetry, lists, and transcripts, and every threshold quietly encodes English norms (a recurring theme: filtering choices are distribution choices, and they fall hardest on non-English text — the same inequity as tokenization, [5.4/04](../../part5_transformer_rebuilt/5.4_tokenization/04_tokenizer_pathologies.md)).

## Generation 2 — reference-corpus classifiers

Replace rules with a learned scorer: pick a corpus you *trust*, train a cheap classifier to distinguish it from raw crawl, keep crawl documents the classifier likes.

- **GPT-3's filter:** logistic regression on hashed features, positives = WebText/books/Wikipedia; crawl documents kept with probability increasing in their score (a soft keep, preserving some diversity).
- **CCNet-style perplexity filtering:** score documents with a small LM trained on Wikipedia; keep the low-perplexity (Wikipedia-like) buckets.

The structural weakness is the reference: "quality" now *means* "resembles Wikipedia/WebText," which is a proxy with failure modes in both directions — it admits fluent SEO spam (looks like prose) and rejects valuable non-prose (tables, forum Q&A, code comments). Perplexity filters add a subtle one: the *lowest*-perplexity text is repetitive boilerplate, so "too unsurprising" needs filtering as much as "too surprising."

## Generation 3 — model-based judgment (FineWeb-Edu, DCLM)

Use a strong LLM to *judge* documents against a semantic criterion, then distill that judgment into a classifier cheap enough for 15T tokens:

- **FineWeb-Edu (2024):** prompt Llama-3-70B to rate 500k documents 0–5 on **educational value**; train a small embedding-based regressor on those ratings; run it over all of FineWeb; keep score ≥ 3. Result: 1.3T tokens on which models markedly outperform same-compute models trained on the full 15T — most of the web is *worse than useless* relative to its best slice, given a fixed compute budget.
- **DCLM (2024):** same shape, different taste-source — a fastText classifier whose positives are instruction-formatted and high-engagement text (OpenHermes, ELI5). Its filtered corpus won that year's controlled data-competition benchmarks; notably, the *simple classifier with well-chosen positives* beat many fancier schemes.

Two things changed versus generation 2. First, the criterion is **semantic** ("is this educational/instructive?"), not distributional ("does this resemble corpus X?") — the judge generalizes across formats in a way n-gram statistics can't. Second, the loop is now **model-in-the-loop data curation**: today's model rates data that trains tomorrow's model. That loop is the opening move of [synthetic data](05_synthetic_data.md) — rating is to generation as a dimmer is to a lamp — and it inherits the same risk: the judge's taste (and its biases, and its blind spots) becomes the corpus's definition of quality.

## The economics: filtering as a scaling-curve shifter

The naive frame is "filtering saves compute by deleting junk." The correct frame, from [6.3](../6.3_scaling_laws/02_chinchilla.md)'s perspective: scaling laws are fit *per data distribution*, and better filtering gives a **better constant** — lower loss at every compute budget, not just a cheaper path to the same loss. FineWeb-Edu vs. FineWeb is the clean demonstration: same substrate, same models, different filter, visibly different capability-per-FLOP curve — with benchmark gains concentrated exactly where the filter's taste points (knowledge and reasoning evals like MMLU/ARC).

The tension that stops you filtering forever: aggressive filtering costs **tokens** (15T → 1.3T) and **diversity** (whatever the judge undervalues vanishes — code, informal registers, minority languages). When tokens are the binding constraint ([file 06](06_the_data_wall.md)), throwing away 90% of the corpus is a real price, and the strict-vs-loose threshold (1.3T vs 5.4T) becomes a budget decision, not a taste decision.

## Why it matters in modern LLM work

- **"Data quality" claims are checkable** once you know the three generations: ask *what* the filter is (rules? reference corpus? judged criterion?) and *how it was validated* (eyeballing? ablation runs?).
- **The filter defines the model's blind spots.** Whatever the pipeline deletes, the model never sees; C4's `{`-rule era is why code-capability required deliberate re-upsampling ([file 04](04_mixtures_and_midtraining.md)).
- **Model-in-the-loop curation is the present**, and its judge-bias problem is the same problem as LLM-as-judge in evaluation (Part 11.1) — one skill, two applications.

## Self-check

1. Why is the *order* of pipeline stages (heuristics → dedup → classifier) economically forced?
2. Give one failure in each direction for a "resembles Wikipedia" classifier, and name the generation-3 change that addresses both.
3. What makes "filtering shifts the scaling curve" a stronger claim than "filtering saves compute," and what would you plot to demonstrate it?
4. FineWeb-Edu keeps ~9% of FineWeb. Name the two costs of that aggressiveness and the situations where each binds.
5. What is the structural risk shared by FineWeb-Edu's filtering and LLM-as-judge evaluation?

### Answers

1. Cost per document rises down the pipeline (string rules ≈ free; MinHash cheap; classifier inference costs real compute per document), so each stage should only see what cheaper stages couldn't reject. Running the classifier on raw crawl would spend its budget mostly on documents a two-line heuristic kills.
2. **Admits:** fluent SEO/affiliate spam — prose-shaped by construction, worthless by content. **Rejects:** high-value non-prose — code, structured data, forum Q&A — that doesn't resemble the reference distribution. Generation 3 replaces the distributional criterion with a *semantic* one ("is this educational?"), which a strong judge can apply across surface formats, catching spam by meaning and sparing value by meaning.
3. "Saves compute" means reaching the *same* endpoint cheaper; "shifts the curve" means a *lower loss at every budget* — a change in the fitted constants of the scaling law, i.e., better capability-per-FLOP forever after. Demonstration: train matched models at several compute budgets on filtered vs. raw data and plot both loss-vs-compute curves on log-log axes ([1.4/09](../../part1_math_foundations/1.4_optional_deeper_knowledge/09_approximations_and_orders_of_magnitude.md)) — a curve shift, not a point win.
4. **Token supply:** 1.3T tokens supports far fewer total-token budgets before multi-epoch territory ([file 06](06_the_data_wall.md)) — binds when your budget is 10T+. **Diversity:** the judge's taste deletes registers and domains wholesale — binds when you care about capabilities the criterion undervalues (informal dialogue, niche languages, code — the last usually patched by adding a separate code corpus, [file 04](04_mixtures_and_midtraining.md)).
5. The judge's preferences become ground truth with no independent check: whatever the judging model systematically over- or under-values is silently baked into the corpus (or the eval). Same failure, two surfaces — and the mitigation is the same too: audit the judge against human ratings on a sample, and keep the judged criterion narrow.

## Exercise

Implement the Gopher-style heuristics (word count, mean word length, symbol ratio, alphabetic-word fraction) and run them on ~50 documents you construct: clean prose, a code file, a poem, a forum thread, SEO spam ("10 BEST protein powders REVIEWED"), and non-English text. Report the confusion matrix — which valuable documents die, which junk survives. Then write the generation-3 fix as a *prompt*: a rubric you'd give an LLM to rate these same documents 0–5, and note (a) which of your heuristic's mistakes the rubric fixes, and (b) one document where reasonable people would disagree with *any* single-axis score — that document is the diversity cost of scalar quality ratings.
