# Entropy, Perplexity, and What LM Losses Mean

## Perplexity

```
PPL = exp(mean_NLL) = exp(H(p, q))
```

where the mean is over token positions. If your LM has CE loss `L` nats/token, perplexity is `e^L`.

**Intuition**: "the effective number of equally-likely next-token choices the model is torn between." A perplexity of 10 means the model is, on average, as uncertain as if it were choosing uniformly among 10 options.

Bounds:
- Uniform over vocab V: `PPL = V`
- Perfect prediction: `PPL = 1`
- Typical WikiText-103 for strong LMs: ~12–20 (depending on era and tokenizer)
- Typical pretraining corpus for a frontier LM: ~3–5 on clean text

## Why perplexity, not accuracy?

For next-token prediction:
- **Accuracy** (top-1) is a binary signal that ignores how confident/close-to-right the model was. Throws away gradient structure.
- **Perplexity** (via CE) rewards the model for putting *any* mass on the right token, with log-scaled partial credit. Has smooth gradients everywhere.

Accuracy is still useful as a human-interpretable metric (e.g. HumanEval pass rate), but it's never the training objective for LMs.

## Is perplexity actually the loss?

Subtle but worth nailing down: **CE is the loss; perplexity is `exp(CE)`, the human-readable form of the same number.** Backprop differentiates CE directly — perplexity is never fed into the optimizer.

But because `exp` is strictly increasing, `argmin(CE) = argmin(PPL)`. Same optimum, same optimal weights, same gradient *direction* — just different scale. That's why people say "trained to minimize perplexity" interchangeably with "trained with cross-entropy loss." They're describing the same optimization in different units.

Why CE, not PPL, in the backward pass:
- **Numerical stability.** CE lives in log-space (well-conditioned numbers like `2.3`); `exp(CE)` would have wildly varying magnitude.
- **Gradient scale.** Differentiating `exp(CE)` multiplies every gradient by `exp(CE)` — huge when loss is high, tiny when low. Working in CE keeps step sizes sane.

## Why perplexity matters (vs. just reporting CE)

`CE = 2.3 nats/token` is correct but meaningless to a human. `PPL = exp(2.3) ≈ 10` is interpretable: the model is *as uncertain as if it were choosing uniformly among 10 equally-likely next tokens*. That single sentence is why perplexity is the headline metric across the field — it converts an abstract log-likelihood into a concrete "effective branching factor." It also lets you reason about absolute progress: going from `PPL = 100` to `PPL = 10` means the model went from "uncertain among 100 choices" to "uncertain among 10" — a 10× reduction in effective uncertainty, regardless of vocab size or model details.

The catch (next section) is that this interpretation only holds within a fixed tokenization scheme.

## Perplexity is tokenizer-dependent

The same model evaluated with a coarser tokenizer will have *lower* perplexity than with a finer one — fewer tokens per document means fewer chances to be wrong. So perplexities across models with different tokenizers are **not comparable**.

Fixes:
- Report **bits per byte (BPB)** instead (normalizes away tokenization — see next file).
- Report **bits per character (BPC)** for historical comparison.
- Only compare perplexities within the same tokenizer family.

## Cross-entropy units

| Unit | Formula | When you see it |
|------|---------|-----------------|
| Nats/token | mean NLL in natural log | Training loss curves |
| Bits/token | nats/token × log₂(e) ≈ nats × 1.443 | Old NLP papers, compression framing |
| Bits/byte | bits/token × (tokens/byte) | Cross-tokenizer model comparison |
| Perplexity | exp(nats/token) | LM eval, historical |

All equivalent up to change of base and tokenization. Convert freely.

## Information content of a token

The "surprise" or information content of observing token `x` is:
```
I(x) = -log p(x)
```

In bits, this is how many bits you'd need to encode `x` under an optimal code for the distribution `p`. A loss of 2.3 nats/token = 3.3 bits/token means the model *could* compress its predictions to ~3.3 bits per token on average. This is the compression view we'll expand in Part 1.3.

## Entropy rate and language

Natural English has an entropy rate of roughly 1–1.5 bits per character (Shannon's original estimate, updated by modern LMs). GPT-4-class models achieve close to that on held-out text — the ceiling is genuinely low, not the model's fault.

This is why "just scale" worked: the task has a hard information-theoretic floor, and we're approaching it. It's also why *further* improvements on clean-text perplexity matter less now — the remaining bits are either noise or hard-to-model rare patterns.

For the full picture — *why* "approaching the entropy rate" is a precise claim (LM = lossless compressor via arithmetic coding, Shannon's no-compressor-can-do-better theorem), and *what it means* for scaling, data exhaustion, and the pivot toward post-training and test-time compute — see `supplementary/05_entropy_rate_and_scaling.md`.

## Conditional entropy and "in-context" improvement

LM perplexity on a document isn't constant per token — it's much higher for the first few tokens (no context) and drops as context accumulates. This reflects:
```
H(x_t | x_{<t})   goes down as t grows
```

That drop is "in-context learning" in the information-theoretic sense — the model uses prior tokens to reduce uncertainty about future ones. Papers sometimes plot **loss vs. token position in context** to demonstrate long-context utilization.

## Self-check

1. A model has mean NLL = 1.6 nats/token. What's the perplexity? Bits/token?
2. Why can't you compare BERT-tokenized perplexity to GPT-2-tokenized perplexity?
3. If PPL = 100 and the vocab is 50K, what does that say about the model? Is it useless?

### Answers

1. **PPL** = `e^{1.6} ≈ 4.95`. **Bits/token** = `1.6 × log₂(e) = 1.6 × 1.4427 ≈ 2.31`. (Convert nats to bits via the change-of-base factor `log₂(e)`.)
2. They use different tokenizers, so they chop the same text into different numbers of tokens. A coarser tokenizer (fewer, longer tokens) produces fewer prediction steps per document → naturally lower PPL even if the model is no better at compressing the underlying text. PPL is normalized per-token, not per-byte. Cross-tokenizer comparison: convert to **bits-per-byte (BPB)**, which divides total bits by raw byte count and is tokenizer-invariant.
3. PPL = 100 vs uniform baseline 50K → the model has reduced effective uncertainty by a factor of 500. Equivalently, it saves `log₂(50000/100) ≈ 9 bits/token` over uniform encoding. **Not useless**: it has clearly learned something — it's a weak language model rather than a non-model. **But not deployable**: a strong LM is at PPL 5–15. PPL 100 means the model is genuinely confused about the next token most of the time. Useful as a baseline; not for production generation.

## Exercise

Take a small LM (e.g. GPT-2 124M). Compute the per-position loss averaged across many documents, for positions 1 through 1024. Plot loss vs. position on log-log axes. You should see a clear power-law-ish decrease — this is the in-context learning curve. This pattern has been analyzed in many papers (e.g. Kaplan et al., Chinchilla).
