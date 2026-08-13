# Next-Token Prediction = Compression = Learning

## The compression-learning equivalence

**Thesis**: a model that predicts next tokens well is exactly a model that compresses text well. Good compression requires *understanding*. Therefore next-token prediction is a learning objective.

This is not a metaphor. It's a mathematical identity.

## What this equivalence actually says

The mechanics below are precise, but they can obscure the conceptual punch. The reason "LM = compressor" is load-bearing in modern ML — not just a cute identity — comes down to a few claims that are each worth internalizing:

- **It converts something fuzzy into something measurable.** "Does this model understand English?" is unwinnable. "Does this model compress English to fewer bits than that one?" is a hard number. The equivalence says these are the *same question*. The unmeasurable becomes measurable.
- **Compression cannot be faked.** A memorization lookup table doesn't compress — it's bigger than the data, not smaller. To shrink the data, a system *must* find patterns, and patterns are exactly what we mean by "knowledge." This dissolves the eternal "is it really learning or just memorizing?" debate: measure the bits.
- **There's an absolute floor, not just a relative ranking.** Most ML metrics are comparative ("better than the previous model"). Compression has a Shannon-proven lower bound — the entropy rate of the data. We can say in *absolute* terms how close we are, and how much room is left. Almost nothing else in ML lets you do that.
- **It tells you what an LM mechanically *is*.** An LM is a probability distribution over sequences, which is the same object as a code for those sequences. Once you accept that, every capability (generation, scoring, embeddings, classification) is just a different query against that distribution, and every limitation (hallucination, sampling temperature, RLHF as reshaping) is a property of it.
- **It connects to a deeper claim about intelligence itself.** Solomonoff and Hutter formalized "inductive inference = finding the shortest explanation = compression." On that view, a near-optimal compressor of human-generated text has, by construction, recovered the implicit model of the world that generated it. You can reject the strong form of this claim, but the weak form is unavoidable: whatever structure is in the data, an optimal predictor has to find it.

### Why each of these matters

**The measurability point** is what makes the whole framing useful at all. AI is full of slippery claims — "this model reasons," "that one understands context," "the new version is smarter." Most are unfalsifiable. Compression gives us a single number you can put on a model and compare across architectures, training methods, and decades. Sutskever-style claims like "intelligence is just compression" sound glib until you realize they're an escape from an unwinnable definitional fight. We can measure compression to four decimal places. We can't measure understanding at all. If they're the same, the unmeasurable inherits the measurability.

**The non-fakeability point** is the deepest. You can game accuracy benchmarks by memorizing the test set. You can game perplexity-on-seen-data the same way. You cannot game compression of *held-out* data, because a memorization table is provably *larger* than the data, not smaller. So when a model compresses well, it has — by formal definition — found patterns that generalize. There is no "just memorized" alternative explanation. The bit count is the test.

**The absolute floor** changes how we talk about progress. Saying "perplexity improved from 12 to 10" is hard to interpret in absolute terms. Saying "we closed half the remaining gap to the entropy rate of English" is a precise, principled statement. It's also how we know the original scaling-laws axis is bending: it's bumping into a floor we can calculate, not vibes. (See `supplementary/05_entropy_rate_and_scaling.md` for the full version.)

**The "what an LM is" point** is clarifying every time you find yourself confused about model behavior. A model that confidently asserts a false fact isn't *malfunctioning* — it's assigning that sequence nonzero probability and sampling from its distribution. A model that's "creative" with temperature 1.5 isn't *being creative*, it's flattening its predicted distribution before sampling. RLHF doesn't *teach values*, it *tilts the distribution*. Once you internalize "the model is a probability distribution," surprising behavior stops being surprising and becomes "what would a distribution-over-sequences naturally do here?"

**The intelligence-as-compression connection** is the philosophical reach. Whether or not you buy that *all* intelligence is compression, the weak version is unavoidable: any system that predicts a complex source well must have absorbed the source's structure. So when frontier LMs approach the entropy rate of human-generated text, they have — in a measurable, formal sense — internalized whatever structure is in human language. The question becomes empirical: how rich is that structure, and what does it imply about cognition? Compression turns these from philosophy into measurement.

## Shannon's source coding theorem (informally)

For a random variable with distribution `p`, the minimum expected code length to encode a sample is `H(p)` bits (entropy). An arithmetic code with probabilities `q` achieves expected length `H(p, q)` (cross-entropy) — always ≥ `H(p)`, with equality iff `q = p`.

Translation: **cross-entropy = expected bits to encode data under your model's predictions**. Minimizing CE = building a better compressor.

## The LM-as-compressor construction

Given an autoregressive LM and a document `x_1, ..., x_T`:

1. At each step, the model produces a full next-token distribution `q_t(·) = p_model(· | x_{<t})` — a vector with one probability per vocab token, conditioned on the prefix. Evaluating it at the actually-observed token gives the scalar `q_t(x_t) = p_model(x_t | x_{<t})`.
2. Encode `x_t` using arithmetic coding with distribution `q_t`: takes `-log₂ q_t(x_t)` bits.
3. Total document length: `Σ_t -log₂ q_t(x_t)` bits. This is your compressed size.

Sum it up and you've compressed a document using `(mean CE in bits/token) × (num tokens)` bits.

And since `log₂` is just `log_e / log_e 2 = log_e × 1.443`:
```
compressed bits/token = (1.443) × (loss in nats/token)
```

A typical large LM with loss ~2 nats/token compresses natural text at ~2.9 bits/token. Gzip gets ~2.3 bits/**byte**. With a GPT-2 tokenizer (~4 bytes/token on English), 2.9 bits/token = ~0.73 bits/byte — dramatically better than gzip.

## Why this framing matters

**Learning = compression** explains a lot about modern LMs that "learning a conditional distribution" doesn't:

- **Why scaling works**: a bigger model is a better compressor; more data gives it more regularities to exploit. Scaling laws are compression-efficiency curves.
- **Why emergence (or not) happens**: certain patterns require a minimum capacity to be compressible at all. Below threshold → model treats them as noise; above → sharp gains. Whether this is "real emergence" is debated; the compression framing doesn't take a side.
- **Why next-token prediction beats masked prediction for generation**: arithmetic coding only works with a fully causal factorization. BERT-style objectives compress less cleanly and don't yield generators.
- **Why distillation works**: the teacher is an efficient codebook; the student learns to emulate its entropy.

## The "Chinchilla scaling laws" view

Parameters are **stored structure**; tokens are **observed data**. Compute-optimal training balances how much structure the model can absorb against how much data is available to reveal structure. Too few tokens → overparameterization that can't be identified. Too many → the model is starved for capacity to compress them.

Chinchilla's "~20 tokens per parameter" rule is the empirical sweet spot under a specific cost model (training FLOPs). When inference cost dominates (as in production), you overtrain past this ratio — Llama 3 was trained at ~75:1, GPT-OSS-like open models go even further.

## The Hutter Prize and "compression is AI"

Marcus Hutter formalized: the best text compressor *is* the most intelligent system, because compressing text arbitrarily well requires solving arbitrarily much of the text's generative process (which is: everything humans think and write). This is a maximalist view but it rhymes with the empirical reality of LM scaling.

## Important nuances

- **Lossless** compression only. LMs compress the original bits exactly (via arithmetic coding), unlike JPEG/MP3 which throw away bits.
- The **model parameters** are not counted in the compressed size unless you demand a truly self-contained codec. For a fixed model shipped separately, the per-document compressed size is just the arithmetic-coded output. For fair "total size" comparisons you'd add model bits.
- **Tokenization is part of the codec**. Different tokenizers yield different per-token losses but (roughly) similar per-byte compression. This is why bits-per-byte is the fair metric.

## Self-check

1. If a model has loss 1.4 nats/token on a 1M-token doc, roughly how many bits does it need to encode that doc?
2. Why is arithmetic coding essential to the equivalence — why doesn't Huffman coding work as cleanly?
3. Suppose you have two LMs with CE loss 1.8 and 2.0 nats/token. Which compresses better? By how much, in percent?

### Answers

1. Bits/token = `1.4 nats × log₂(e) = 1.4 × 1.4427 ≈ 2.02 bits/token`. For 1M tokens: `~2.02M bits ≈ 252 KB`. (For comparison: naive 16-bit-per-token encoding would be 2 MB; the model compresses ~8× better.)
2. **Huffman** assigns integer-bit codes per symbol, so it can't represent fractional probabilities efficiently. For high-probability tokens (where the optimal code is `< 1` bit), Huffman wastes the slack. **Arithmetic coding** represents the entire sequence as a single number in `[0, 1)` and can achieve fractional bits per symbol — within a few bits of the entropy bound regardless of the distribution. So arithmetic gives the **tight** equality "expected bits = cross-entropy"; Huffman has slack proportional to the rounding from continuous to integer bits. The compression-as-learning equivalence depends on having a code that achieves the entropy bound, which is arithmetic.
3. Lower CE = better compression. The ratio of compressed sizes is `1.8/2.0 = 0.9` — model A produces files 90% the size of model B's. Equivalently: A is **10% better** at compression. Not a huge win on a per-token basis, but compounds — over a 1B-token corpus, that's 200M nats ≈ 36 MB saved.

## Exercise

Read "Language Modeling Is Compression" (Delétang et al., DeepMind 2023). It literally uses Chinchilla as a gzip replacement and measures bytes saved. This is the cleanest empirical demonstration of the equivalence and worth an hour.
