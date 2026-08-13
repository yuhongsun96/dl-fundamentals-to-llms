# Synthetic Data — Models Writing the Next Model's Corpus

The largest omission in the classic pretraining picture: by 2024–25, a substantial fraction of frontier pretraining tokens are **written by other models**. This file maps the three families that work, why they work (each is a different answer to "where does the new information come from?"), and the collapse risk that's real but routinely overstated.

## Why generate data at all

Three pressures, all already on the table:

- **Supply:** the high-quality slice of the web is small ([file 03](03_quality_filtering.md)) and finite ([file 06](06_the_data_wall.md)).
- **Quality ceiling:** even filtered web text is mostly *not* explanation-shaped — the web tells you facts sideways, buried in forums and comment threads.
- **Control:** a generator can be *aimed* — at underrepresented topics, difficulty levels, languages, formats — in a way no filter over found text can.

## Family 1 — textbook-style generation (Phi)

Prompt a strong model to write pedagogical content from scratch: explanations, worked examples, exercises, across a curriculum of topics.

- **Phi-1** (2023, *Textbooks Are All You Need*): a 1.3B code model trained on filtered code plus GPT-3.5-generated "textbook" explanations and exercises — competitive with far larger models on code evals. The series scaled the thesis; by **Phi-4** (2024), synthetic data is a large fraction of the pretraining mix, with the report detailing generation pipelines (multi-agent drafting, self-revision, seeded from curated real documents).
- The open replication is **Cosmopedia** (HuggingFace, 2024): tens of millions of synthetic textbooks/blogposts/stories (~25B tokens), seeded from web samples for coverage.

Why it works: it's **format transfer, not information creation**. The facts already exist diffusely in the generator's weights; generation re-serializes them into the densest learnable form — the explanation — with effectively infinite volume at a chosen difficulty. The catch is inherited limits: the generator's errors become the *ground truth* of the corpus, and its blind spots become the curriculum's.

## Family 2 — rephrasing real data (WRAP)

Don't invent content; **rewrite found content** in better form. WRAP (*Web Rephrase Augmented Pre-training*, 2024): prompt a model to rephrase crawl documents ("like Wikipedia," "as Q&A"), train on rephrasings mixed with the originals — several-fold pretraining efficiency gains (equal loss with ~3× fewer tokens/compute, per their runs).

Why this is the conservative option: **content is anchored to real text** — the facts come from the document, the model only restyles — so hallucination risk is bounded and diversity is inherited from the web rather than from the generator's imagination. It's [file 03](03_quality_filtering.md)'s logic completed: filtering keeps the best 10% of the web; rephrasing *repairs* the rest into usable form instead of discarding it. (The quiet cost: generating tokens costs inference compute, so "3× fewer training tokens" is partially offset by paying to write them.)

## Family 3 — distillation at pretraining scale

Train the small model on a big model's outputs, wholesale. Classical distillation (Part 9.2) matches logits on a fixed input set; the pretraining-scale version just treats strong-model generations as corpus. Most open small models of 2024–25 are substantially trained this way (many explicitly on GPT-4-class or DeepSeek-R1 outputs — the R1-distill series made it a product category). Two structural notes:

- Economically, this is **capability transfer**: the teacher's expensive training is amortized into text that cheaply reprograms students. It's why small-model quality tracked frontier quality with ~a year's lag.
- It caps you at the teacher: distillation moves capability *down* the scale ladder, never up. Families 1–2 share a weaker version of the same ceiling — which is why the frontier labs' synthetic pipelines lean on **verification** (below) to exceed the generator.

## The collapse question, calibrated

**Model collapse** (Shumailov et al., 2024, and kin): recursively training generation `n+1` on generation `n`'s outputs narrows the distribution — tails vanish first, diversity decays, eventually degeneration. Real, replicated, and the right null hypothesis for "just train on your own outputs."

But note what the experiments assume: **replacement** (synthetic replaces real, each generation) and **no selection** (outputs used as-is). Practice violates both:

- **Accumulation, not replacement:** real data stays in the mix; follow-up work shows accumulation largely averts collapse.
- **Selection/verification:** production pipelines filter generations — unit tests for code, checkable answers for math, judge models and dedup for prose. A verifier injects information the generator alone didn't have (only correct programs survive), which is both the practical anti-collapse mechanism and the conceptual bridge to RL with verifiable rewards (Part 8.3): *generate → verify → train* is the same loop, applied to weights via data instead of via reward.
- The engineering problem that remains is **diversity management** — seeding topics (Cosmopedia's web seeds), deduping generations ([file 02](02_deduplication.md)), monitoring distribution narrowness — and *provenance*: benchmark answers leaking through a teacher's outputs is contamination that decontamination-by-n-gram won't catch ([file 02](02_deduplication.md)).

Calibrated statement: collapse is what happens to *careless* recursive training; curated synthetic data with anchoring, mixing, and verification is currently a net, large win — and every frontier lab is acting accordingly.

## Why it matters in modern LLM work

- **Reading 2025 model cards:** "N tokens" no longer implies found text; the synthetic fraction and its generator are the questions to ask (and often the ones not answered).
- **It rewrites the data-wall math** ([file 06](06_the_data_wall.md)): token *supply* stops being the binding constraint for anything a generator+verifier can produce — the constraint moves to verification and diversity.
- **The generate→verify→train loop** is the shared skeleton of synthetic pretraining, RLVR (Part 8.3), and LLM-as-judge eval (Part 11.1) — one pattern, three surfaces.

## Self-check

1. For each family, answer: where does the information come from, relative to the generator?
2. Why is rephrasing strictly safer than textbook generation with respect to hallucination, and what does it give up?
3. State the two assumptions of the collapse experiments that production pipelines break, and what each break does.
4. Why does a verifier let synthetic data exceed the generator's own capability, and which post-training method is the same idea?
5. Your 7B model trained on teacher outputs aces a benchmark. Give the two distinct contamination paths, and which is invisible to n-gram decontamination.

### Answers

1. **Textbooks:** from the generator's weights (re-serialized latent knowledge — no new facts, better format). **Rephrasing:** from the *source document* (generator contributes style only). **Distillation:** from the teacher's weights (capability transfer down-scale). The families are ordered by how much trust you place in the generator as a source of truth.
2. Rephrasing conditions on a real document that supplies the facts; the generator's degrees of freedom are stylistic, so errors are mostly transcription-level rather than invented content. It gives up *aim*: you can't rephrase your way to content the web lacks (novel exercises, rare-topic curricula, chosen difficulty ramps) — that's exactly the territory where generation earns its risk.
3. **Replacement** (synthetic supplants real each generation) — broken by accumulation: keeping real data in the mix anchors the distribution and empirically prevents the degeneration spiral. **No selection** (use everything generated) — broken by verification/filtering: selecting survivors reinjects an outside signal, so the training distribution is generator × verifier, not generator alone.
4. The verifier is an information source independent of the generator: among many sampled attempts, only those passing tests/checks survive, so the surviving set has higher accuracy than the generator's unconditional output — best-of-N with the selection *baked into the corpus*. RLVR (Part 8.3) is the same generate→verify→reinforce loop expressed as rewards rather than as a dataset.
5. (a) Direct: benchmark items present verbatim in training text — n-gram decontamination can catch this. (b) **Teacher-mediated:** the teacher memorized benchmark answers and emits them (paraphrased, embedded in reasoning) in its generations — no verbatim overlap exists to match, so n-gram methods are blind to it. Provenance tracking and eval-side detection (Part 11.1) are the only defenses.

## Exercise

No GPU needed — this is a pipeline-design exercise. You must produce 100B tokens of synthetic math data for mid-training ([file 04](04_mixtures_and_midtraining.md)) using a strong teacher model. Specify: (a) the seeding strategy that guarantees topic and difficulty diversity (what's your analogue of Cosmopedia's web seeds?); (b) the verification stage — what's checkable in math, and what fraction of generations do you expect to discard; (c) the dedup you run on *generations* and why generator output needs it more than web text ([file 02](02_deduplication.md)); (d) your defense against teacher-mediated GSM8K/MATH contamination, given self-check 5b says n-grams won't save you. Then one sentence: which of the three families is your pipeline, and what single change would move it to a different family?
