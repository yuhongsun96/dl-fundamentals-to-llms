# Entropy — The Natural-Language Picture

Companion to `04_cross_entropy_kl.md`. The primary file gives the formula; this one builds the intuition behind it.

Entropy measures **how uncertain a probability distribution is** — equivalently, **how much "surprise" you can expect on average when you draw a sample from it**.

## The "surprise" building block

The hidden building block is `−log(p_i)`, which you should read as **"how surprising is it when outcome `i` happens?"** Properties:

- If `p_i = 1` (certain event), `−log(1) = 0`. Zero surprise — you knew it would happen.
- If `p_i = 0.5` (coin flip), `−log(0.5) = log 2 ≈ 0.69` nats = `1` bit. One bit of surprise.
- If `p_i = 0.001` (a rare event), `−log(0.001) ≈ 6.9` nats ≈ `10` bits. Big surprise — that almost never happens.
- If `p_i → 0` (essentially impossible), `−log(p_i) → ∞`. Infinite surprise.

The `log` is what makes this work: surprise grows slowly for moderate-probability events and explodes for very rare ones. Doubling probability *halves* surprise. That's exactly how human intuition treats it — going from "1% chance" to "0.5% chance" feels like a much bigger jump than going from "50%" to "49.5%."

## Now: average surprise

Entropy is the **expected value of this surprise** — weighted by how often each outcome actually happens:

```
H(p) = Σ_i  p_i · ( − log p_i )
     = expected amount of surprise per sample drawn from p
```

You weight each outcome's surprise by `p_i` because you're averaging over actual draws from `p`. A super-rare event has huge per-occurrence surprise but almost never occurs, so its weighted contribution is moderate. A common event has tiny per-occurrence surprise but happens a lot, so it also contributes only moderately. The maximum total comes from spreading mass *across many outcomes of similar probability* — no outcome is rare enough to be very surprising on its own, but you also never know which one is coming.

## The two extremes

- **One-hot distribution** (one outcome with `p = 1`, everything else `0`): entropy is exactly **0**. There's no uncertainty — you know what's coming, so there's no surprise to average. Perfectly predictable.
- **Uniform distribution** over `n` outcomes (every outcome has `p = 1/n`): entropy is **`log n`** — the maximum possible for `n` outcomes. Every draw is maximally uncertain because you genuinely have no idea which of the `n` equally-likely options will come up.

Every other distribution sits between these extremes. Skewed/peaky distributions have low entropy (closer to certain). Spread-out/flat distributions have high entropy (closer to uniform).

## Concrete numbers

- **Fair coin** (50/50): `H = 1 bit`. One yes/no question's worth of uncertainty.
- **Heavily biased coin** (99/1): `H ≈ 0.08 bits`. Very predictable.
- **Fair 6-sided die**: `H ≈ 2.58 bits`.
- **Uniform over a 50,000-word vocab**: `H ≈ 15.6 bits ≈ 10.8 nats`. That's the "no idea what the next word is" baseline.
- **A trained LM on real English text**: typically around `2.5 nats` per token. Vastly less uncertain than uniform — the model has narrowed down each next-token choice from "50,000 equally likely options" down to "as if choosing among ~12 options" (this ratio is exactly what perplexity measures — see `05_entropy_perplexity.md`).

## The operational meaning

There's an actual physical interpretation (Shannon, 1948): entropy is **the minimum average number of bits per symbol needed to encode samples from `p`**. If you knew the distribution, you'd give short codewords to common symbols and long codewords to rare ones (think Morse code: `e` is short, `q` is long). The best you can possibly do, averaged over many samples, is `H(p)` bits per symbol. You cannot compress further without losing information.

This is why entropy isn't just a definition — it's a hard lower bound. It tells you the *information content* of a distribution in a precise, measurable sense.

## What it means in ML

A few uses you'll see throughout the curriculum:

- **A model's output distribution.** If a language model outputs probabilities `[0.9, 0.05, 0.03, 0.02, ...]` over the vocab, the entropy of that distribution is small — the model is *confident*. If it outputs `[0.01, 0.012, 0.009, ...]` spread across many tokens, entropy is large — the model is *uncertain*. Entropy of model output is sometimes used directly as a signal (e.g. entropy bonuses in RL exploration, entropy-based confidence calibration).
- **The "data distribution."** Real text has some intrinsic entropy — about 1 bit per character for English, by Shannon's famous experiments. That's the floor below which no language model can ever push its loss.
- **Bridge to cross-entropy.** Cross-entropy is "the surprise you'd experience if you assumed distribution `q` but the world was actually `p`" — always at least `H(p)`, with the gap being the KL divergence (the "tax" you pay for the wrong assumption).

## The short version

Entropy = expected surprise = average information content = how-uncertain-am-I, all the same quantity. Zero when you're certain, maximized when you're maximally confused (uniform over outcomes). It's the loss floor for any predictor, the compression limit for any encoder, and the confidence reading for any output distribution. Everything else in this chapter — cross-entropy, KL, perplexity, the LM loss — is built on top of this single quantity.
