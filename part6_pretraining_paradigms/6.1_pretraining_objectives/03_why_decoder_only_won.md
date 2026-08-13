# Why Decoder-Only Causal LM Won

The promise from [5.1/05](../../part5_transformer_rebuilt/5.1_self_attention/05_causal_and_bidirectional_masking.md), paid in full. In 2019 this was a genuinely open three-way race — BERT dominated the leaderboards, T5 had the most principled story, and GPT-2 looked like the weird one. By 2022 the race was over and everything at the frontier was a decoder. This file assembles the reasons, roughly in order of how much they mattered, and then honestly marks where the losers still win.

## Reason 1 — supervision density (the compute argument)

From [file 02](02_mlm_and_denoising.md)'s accounting: causal LM extracts a loss term from **every** position; MLM from ~15%. At equal compute, the decoder sees ~6.7× more supervised predictions. When the binding constraint became compute at scale — and [6.3](../6.3_scaling_laws/01_power_laws_kaplan.md) made everyone acutely aware it was — an objective that wastes 85% of its forward passes' supervision was carrying a structural handicap no architecture could offset.

## Reason 2 — the task-unification argument (zero-shot changes the product)

GPT-2's title was the thesis: *language models are unsupervised multitask learners*. If everything is next-token prediction, then **prompting is task specification**: translation is "French: … English:", QA is "Q: … A:", and no fine-tuning, task heads, or output parsers stand between the pretrained artifact and the task. GPT-3 turned this into in-context learning — examples in the prompt, weights untouched.

BERT structurally cannot do this — it has no way to *emit* text, so every new task means a head and a fine-tune. T5 can generate, but its sentinel-format pretraining puts a format gap between "what it learned" and "what you ask." The systematic comparison (Wang et al., 2022, *What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?*) found what the market found: **causal decoder-only, trained on plain next-token prediction, is the best zero-shot generalizer**. The product implication is the real point — zero-shot prompting made the *model itself* the product, and only one objective family could ship that product.

## Reason 3 — one stack, one loss, one recipe (the scaling argument)

Scaling rewards simplicity ruthlessly:

- **One hyperparameter recipe.** Decoder-only has a single stack to tune. Encoder-decoder doubles the design questions (relative sizes? shared embeddings? cross-attention where?) — every extra knob is another thing to get wrong at a scale where you get one shot ([2.4 supplementary](../../part2_neural_network_fundamentals/2.4_optimization/supplementary/02_setting_lr_and_schedule_across_scales.md)).
- **The loss is the metric.** One smooth scalar that scaling laws can be fit to ([6.3/01](../6.3_scaling_laws/01_power_laws_kaplan.md)). MLM's loss is noisier (15% of positions per batch) and doesn't correspond to a generative capability you can extrapolate.
- **Infrastructure mono-culture.** Every optimization in Parts 7 and 9 — KV caching, FlashAttention's causal tiling, continuous batching, speculative decoding — was built for the causal decoder. The compounding returns of the whole field optimizing one shape are themselves a moat; by 2023 a new objective didn't just have to be better, it had to be better than causal LM *plus five years of its tooling*.

## Reason 4 — serving economics

Generation from a decoder is incremental by construction: cache K/V for the prefix, pay one token's compute per new token (Part 9.2). An encoder-decoder serving a *conversation* must re-encode the growing history or maintain two caches with cross-attention; a bidirectional model can't cache incrementally at all, because every token's representation depends on the *whole* sequence — appending one token invalidates everything. The KV cache is a direct architectural dividend of the causal mask, and inference cost is where margins live.

## The honest caveats

**Bidirectionality is genuinely better for representations.** At matched scale, both-sided context wins for "what does this text mean as a vector." That's why encoders didn't die — they moved into retrieval. Embedding models and rerankers (E5/BGE — Part 8.5) are BERT descendants, and LLM2Vec-style conversions (take a causal LM, *remove* the mask, adapt) concede the point in the other direction.

**Encoder-decoder survives where the input is fixed or non-text** — Whisper (Part 10.4), translation, many VLM connectors (Part 10.3). When the "prefix" is an image or an audio clip that never grows, re-encoding costs nothing and bidirectional encoding of the input is free quality. Cross-attention as an *interface between modalities* outlived cross-attention as a *text architecture* — and the encoder-decoder shape survives *inside* every decoder anyway, as the prefill/decode split (Part 9.2): prefill is "encode the prompt in parallel," decode is autoregression against the cache.

**The margin wasn't enormous at fixed scale.** Head-to-head at 2020 sizes, objectives traded wins by task. The decisive gaps were the *system* properties — supervision density, zero-shot product, recipe simplicity, serving cost — which all *compound with scale*. That's why the race looked open at 350M parameters and settled at 100B.

## Why it matters in modern LLM work

- **Reading the family tree:** "decoder-only" on a model card is not a neutral fact but the residue of this argument — and when you meet a non-decoder (an embedder, Whisper, a diffusion LM), the first question is *which* of the four reasons above it's exempt from.
- **The pattern generalizes:** "simple objective × more compute beats clever objective" is the same bet as RLVR over process supervision (Part 8.3) and the bitter-lesson framing generally. This file is the canonical case study.
- **The prefill/decode and KV-cache material of Part 9** all flows from Reason 4; the scaling-law machinery of 6.3 assumes Reason 3's single smooth loss.

## Self-check

1. Rank the four reasons by when they bind: which matter at *training* time, which at *serving* time, and which at *product* time?
2. Why can't a bidirectional model use a KV cache for incremental generation, even in principle?
3. BERT beat GPT-1 soundly on understanding benchmarks in 2018–19 at similar scale. Reconcile that with this file's conclusion — what changed, the models or the question?
4. Give the strongest single piece of evidence that bidirectionality still beats causality for representations.
5. A 2025 startup proposes an encoder-decoder frontier chat model, arguing encoder quality on the prompt is worth it. Which reason above is the sharpest objection, and what deployment pattern would weaken that objection?

### Answers

1. Training: supervision density (1) and recipe simplicity (3). Serving: KV-cache economics (4), plus the tooling half of (3). Product: zero-shot unification (2). The composite is the point — each alone was survivable; a competitor had to lose on *all* of them simultaneously.
2. Bidirectional attention makes every position's representation a function of the entire sequence, so appending a token changes *all* previous representations — nothing is reusable. The causal mask is precisely the property "position `t`'s state is final once written," and that immutability is what a cache caches.
3. The question changed. 2018's question was "best frozen-representation factory for fine-tuning on my labeled task?" — bidirectional context wins, BERT was the right answer. The scaling era's question became "best artifact per FLOP that does arbitrary tasks from a prompt?" — where supervision density, generation, and zero-shot dominate. BERT didn't get worse; the market stopped asking its question.
4. That the modern embedding recipe *converts causal LMs into bidirectional ones* (LLM2Vec, and Part 8.5's lineage): practitioners with a state-of-the-art decoder in hand still remove the causal mask to get better vectors. If causality were free for representations, that step wouldn't exist.
5. Reason 4: chat history grows every turn, so the encoder re-encodes an ever-longer prefix per turn (or maintains an awkward incremental scheme), while a decoder pays once per token, ever. The objection weakens for **single-shot, fixed-input workloads** — translate-this-document, transcribe-this-audio — which is exactly the niche where encoder-decoders survive.

## Exercise

Collect the pretraining objective, architecture, and *primary use* for: GPT-4-class chat models, Llama 3, Whisper, E5 or BGE embeddings, a code-completion model with FIM, and T5 (still widely deployed). For each, write one line naming which of this file's four reasons predicts its shape — and flag the one where the *use* (not the era) is what exempts it from the decoder-only default. Then, in two sentences: if supervision density were the *only* force, what would the ecosystem look like, and what in your table proves it isn't?
