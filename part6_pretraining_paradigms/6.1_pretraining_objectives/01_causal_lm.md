# Causal Language Modeling — the Objective, Precisely

Everything downstream of this file — scaling laws, data pipelines, post-training — assumes one loss function. It's worth stating that loss with complete precision once, because half the design decisions in modern LLMs (the causal mask, the KV cache, the "every token is a training example" economics) are direct consequences of its exact form.

**Convention:** natural log / nats throughout ([1.2/05](../../part1_math_foundations/1.2_calculus_probability/05_entropy_perplexity.md) converts to perplexity). Shapes use `B, S, V` from [NOTATION.md](../../NOTATION.md).

## The factorization

A language model wants `p(x)` for a token sequence `x = (x_1, …, x_S)`. The chain rule of probability factorizes this **exactly** — no approximation, no modeling assumption:

```
p(x) = p(x_1) · p(x_2 | x_1) · p(x_3 | x_1, x_2) · … = Π_t  p(x_t | x_<t)
```

Causal LM is the decision to model each factor with one shared network: train `p_θ(x_t | x_<t)` to match the data. "Autoregressive" and "next-token prediction" both name this. Two things are worth pausing on:

- **The factorization is lossless.** Any distribution over sequences can be written this way. Left-to-right is a *choice of order*, not a restriction on what's representable — the restriction comes only from the network's capacity to represent the conditionals.
- **Everything is this one problem.** Translation, summarization, code, chat — once serialized into tokens, they're all "predict the next token given the prefix." That unification, not any architectural cleverness, is the deep reason one pretrained model transfers everywhere ([1.3/01](../../part1_math_foundations/1.3_information_theory/01_next_token_as_compression.md): the compression view of the same statement).

## The loss

Maximum likelihood on the factorization gives cross-entropy at **every position**:

```
L = − (1/S) Σ_t  log p_θ(x_t | x_<t)
```

Mechanically, per training sequence: the model maps `(B, S)` token IDs to `(B, S, V)` logits; position `t`'s logits are scored against target `x_{t+1}` (the **off-by-one shift** — logits at `t` predict token `t+1`); cross-entropy with integer targets, averaged. This is exactly the `F.cross_entropy(logits.view(-1, V), targets.view(-1))` line from the [capstone notebook](../../numpy_pytorch/05_capstone_transformer_block.ipynb).

Three properties carry the field:

**1. One sequence = `S` training examples.** The causal mask ([5.1/05](../../part5_transformer_rebuilt/5.1_self_attention/05_causal_and_bidirectional_masking.md)) makes position `t` blind to positions `> t`, so all `S` conditionals are computed **in one forward pass**, each a legitimate prediction problem with its own loss term. A 4096-token document contributes 4096 supervised examples for one pass of compute. This *supervision density* is the quiet engine of the scaling era, and the property MLM gives up ([file 02](02_mlm_and_denoising.md)).

**2. The supervision is free.** The target is the text itself — no labels, no annotation, no task design. The binding constraint moves from "how much labeled data can we afford" to "how much text exists" (which is why all of [6.2](../6.2_data/01_corpora_and_epochs.md) exists).

**3. The loss is a calibrated, smooth, single number.** It's the negative log-likelihood of real text in nats/token — directly comparable across runs, smoothly improving with scale (the raw material of [6.3](../6.3_scaling_laws/01_power_laws_kaplan.md)), and interpretable via information theory ([1.3/03](../../part1_math_foundations/1.3_information_theory/03_bits_per_byte.md)).

## Reading the loss curve

Because the loss is `−log p` of the true token, its landmarks have meanings:

| Loss value | Meaning |
|---|---|
| `ln V` at init | Uniform guessing: `11.76` for `V = 128,256`, `10.82` for GPT-2's 50,257. Step 0 should land here — see the floor note below for *why it can't start lower*. |
| First hours' cliff | **Unigram statistics** — raw token frequencies, no context. Worth ~6 of the ~9.8 total nats (detail below). |
| ~2 nats/token region | Where strong models on web text live (Chinchilla-era models fit `≈ 1.9–2.1` on their eval mix; [6.3/02](../6.3_scaling_laws/02_chinchilla.md)). |
| The irreducible floor | Text is genuinely uncertain — next tokens are often underdetermined ("entropy of language", `E ≈ 1.69` in Chinchilla's fitted law). Loss can never reach 0 and *shouldn't*. |

The gap between your loss and the irreducible floor is the only part the model can fix — a framing that matters when reading scaling-law plots, where the fitted `E` is subtracted before the power law appears.

**Why `ln V` is a floor at step 0, not just a typical value.** At init the logits carry no information about the target, so writing `L = −z_target + logsumexp(z)` and taking expectations over random logits with std `σ` gives

```
E[L] ≈ ln V + σ²/2        ≥  ln V,  always
```

Verified: measured `11.7618` at `σ = 0` (exactly `ln V`), `11.886` at `σ = 0.5`, `13.778` at `σ = 2`. Random logits can only *hurt* — they scatter probability mass onto tokens that are wrong as often as chance dictates. **A truly random network cannot start below `ln V`**, so if you see a first value like 8.9 it is almost never "leakage at init"; the overwhelmingly likely explanation is that **the value isn't step 0** — loggers commonly report after a few steps or average over the first window, and unigram learning is fast enough to cover that ground immediately. (The one legitimate exception: initializing the final bias to the log unigram frequencies, a real trick that starts you at the unigram entropy on purpose.)

### Why the cliff is the least interesting part of the curve

The near-vertical drop at the start is almost entirely the model learning **unigram statistics** — how often each token appears *overall*, with no reference to context (`the` ≈ 5%, `aardvark` ≈ 0.00001%). Three landmarks make the arithmetic stark:

| Stage | What the model is doing | Loss (nats) |
|---|---|---|
| Init | every token equally likely | `11.76` |
| Post-cliff | emit each token at its corpus base rate, context ignored | `≈ 5–6` |
| Trained | actually use the context | `≈ 2` |

So roughly **6 of the ~9.8 total nats come from the dumbest statistic available**. The remaining ~3.5 — grammar, facts, long-range context — consume essentially all of the compute.

It happens this fast for two compounding reasons: predicting base rates needs **no machinery** (no attention pattern, no circuit — just bias the output distribution), and it is the **loudest, most repeated signal in the data** (every token in every batch pushes the same way, unlike "in *this* context the next token is `Paris`," which is rare and points somewhere different each time).

Three practical consequences. The cliff is a **health check** — no drop means gradients aren't flowing or the LR is too low; a drop toward 0 means a data leak (self-check 3). It's why loss curves use a **log x-axis**: on a linear axis the cliff eats the plot and the meaningful progress looks flat. And **scaling laws fit the post-cliff grind**, not the plunge. Read as compression ([1.3/01](../../part1_math_foundations/1.3_information_theory/01_next_token_as_compression.md)), the curve retraces the field's own history: init is fixed-length codes, the cliff is frequency coding (Huffman), and everything after is context modeling — where all the intelligence lives.

## What the objective does *not* give you

Honesty about the raw artifact, because Part 8 exists to fix exactly this list:

- **It models the distribution of the corpus, not your intent.** The best next token after a question is whatever *the internet* usually puts there — sometimes an answer, sometimes another question, sometimes an ad.
- **Likelihood ≠ helpfulness.** Argmax-decoding a pure LM produces plausible continuations, not completed tasks; instruction-following is *latent* in the weights and surfaced later (SFT, Part 8.1).
- **Exposure bias.** Training always conditions on *real* prefixes (teacher forcing); generation conditions on the model's own possibly-wrong outputs. Mostly benign at scale, but it's the crack RL-based post-training ([8.3](../../review_outline.md)) later widens into a training signal.

## Why it matters in modern LLM work

- Every scaling law in 6.3 is a statement about *this* loss; every data decision in 6.2 is judged by what it does to *this* loss on held-out text.
- The mask/objective pairing from [5.1/05](../../part5_transformer_rebuilt/5.1_self_attention/05_causal_and_bidirectional_masking.md) is the reason the architecture and objective can't be chosen independently — [file 03](03_why_decoder_only_won.md) turns that into the full argument for why this objective won.
- The per-position loss layout is where practical loss-masking decisions live ([6.4/02](../6.4_training_mechanics_at_scale/02_packing_boundaries_loss_masking.md)) — and where SFT's "mask the prompt" convention (Part 8.1) plugs in later.

## Self-check

1. The chain-rule factorization is exact for *any* distribution over sequences. So what, precisely, is the modeling assumption in causal LM?
2. Why does one 4096-token document yield 4096 training examples in one forward pass — which two mechanisms make that true simultaneously?
3. Your logger's first reported loss for a fresh 8B model (`V = 128,256`) is 9.3. What are the two candidate explanations, which is far likelier, and what *would* target leakage look like instead?
4. Why can the loss never approach zero on real text, and what named quantity is the floor?

### Answers

1. Not the factorization but the **parameterization**: all `S` conditionals are computed by *one shared network* `p_θ` with finite capacity, reading the prefix through a fixed-width residual stream. The chain rule allows arbitrarily complex conditionals; the model approximates them. (Left-to-right order is also a choice, but a lossless one.)
2. (a) The **causal mask** guarantees position `t`'s prediction uses only `x_<t`, so each position is a valid prediction problem, and (b) **teacher forcing** — the full ground-truth sequence is available at train time, so all `S` problems share one forward pass instead of `S` sequential ones. Mask makes it *correct*; parallelism makes it *cheap* ([4.1/03](../../part4_sequence_modeling_lineage/4.1_rnns_and_lstms/03_why_they_lost.md)).
3. Since a random network cannot score below `ln V ≈ 11.76`, 9.3 is not a step-0 value. **Far likelier: it isn't step 0** — the logger reported after a handful of steps (or averaged its first window), and the model has already picked up part of the unigram distribution, whose entropy on real text is only ~5–6 nats. 9.3 sits between `ln V` and the unigram floor, exactly where "a few steps in" belongs. The rarer benign explanation: the output bias was initialized to log token frequencies deliberately. **Leakage looks nothing like this** — a model that can see its own target crashes *through* the unigram floor toward 0 (measured on a toy leaking setup: 10.96 at step 1, 7.6 by step 20, **0.74 by step 60**). Note it also *starts* at `ln V`, so the diagnostic is never the first value; it's the trajectory — any run that dives well below ~5 nats in its first few hundred steps is leaking, no matter how normal step 0 looked.
4. Because next tokens are genuinely underdetermined — many continuations are consistent with any prefix, so the true conditional has nonzero entropy. The floor is the **entropy (rate) of the text distribution** — the irreducible term `E` in fitted scaling laws — and only the gap above it is model error.

## Exercise

Take the tiny char-level GPT from the [capstone notebook](../../numpy_pytorch/05_capstone_transformer_block.ipynb). (a) Confirm the initial loss is `ln 65 ≈ 4.17`, then break the target shift on purpose (predict `x_t` from a window *including* `x_t`) and watch the loss collapse toward 0 — you've built a copy machine, not an LM. (b) Restore the shift and log the *per-position* loss across `t = 1…S` at several checkpoints: early positions (short prefixes) should stay high-loss while later positions improve — the conditional's difficulty falls as the prefix grows. (c) In one sentence, connect (b) to why long documents are worth more per token than short ones under this objective.
