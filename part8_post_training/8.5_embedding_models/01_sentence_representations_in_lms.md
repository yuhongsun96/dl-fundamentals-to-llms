# How Phrases and Sentences Are Represented Inside an LM

> Convention note: row-vector convention throughout — activations are rows, `Y = X W` with `W ∈ R^(d_in × d_out)`.

## There is no sentence vector

A Transformer LM never builds a single vector for "the sentence." A forward pass over a sequence of `S` tokens produces a matrix of contextual token states at every layer:

```
H_l ∈ R^(S, D)        one row per token position, at layer l
```

For a Llama-class model: `S` tokens × `D = 4096`, times `L = 32` layers to choose from. The meaning of the phrase or sentence is **distributed across these rows** — and organized for the pretraining objective, not for comparing two texts. Nothing in the architecture designates any row, or any combination of rows, as "the summary."

This is the single most important framing for embedding models: the checkpoint gives you a *set* of token vectors; an embedding model needs *one* vector per text whose dot products are meaningful. Everything in this subsection is about bridging that gap.

## What each token state actually contains

The residual-stream view (see [1.1 supplementary: residual stream](../../part1_math_foundations/1.1_linear_algebra/supplementary/06_residual_stream.md)): token `i`'s state at layer `l` is whatever the layers so far found useful to write there — and "useful" is defined entirely by the pretraining loss.

- **Causal LM (GPT/Llama family):** row `i` is optimized to be a good basis for predicting token `i+1`. It must encode the *prefix* `x_1..x_i` — but only the parts relevant to immediate continuation. Plenty of sentence-level meaning ends up there as a side effect (you can't predict the next token of a contract clause without representing the clause), but the geometry is tuned for next-token logits, not for semantic comparison.
- **Masked LM (BERT family):** row `i` is optimized for reconstructing tokens from *bidirectional* context, so every position sees the whole sentence. This is why encoder checkpoints were the natural starting point for the first sentence-embedding models.

**Depth matters.** Probing studies consistently find a division of labor across layers:

| Layers | What dominates |
|---|---|
| Early | Lexical identity, position, local syntax — close to the token embedding |
| Middle | The most *transferable* semantics — syntax trees, coreference, semantic roles peak here |
| Last few | Rotated toward the output head — the state increasingly looks like "next-token logits in waiting" (the logit-lens observation) |

Consequence: the final layer of a causal LM is often *not* the best layer for similarity. The last layers spend capacity specializing the representation for the vocabulary projection, discarding generality. Mid-to-late layers (~60–80% depth) frequently embed better, and several embedding recipes pool from there or learn a weighted mix of layers.

## The causal mask problem

In a decoder-only LM, position `i` attends only to positions `≤ i`. So:

- The state of an early token is computed **blind to the rest of the sentence**. In "The bank of the river...", the row for "bank" never gets to see "river" — disambiguation it would need for a good similarity vector.
- The **only** position whose state has seen the entire text is the **last token**.

A bidirectional encoder has no such problem — every row sees everything. This asymmetry drives most of the design differences between BERT-era and LLM-era embedding models (next file).

## Pooling: from `(S, D)` to `(D,)`

`pool(H)` is the bridge. The standard options:

| Pooling | How | When it works |
|---|---|---|
| **CLS** | Take the row of a special prepended token | Only if something *trained* that token to aggregate (BERT's NSP did, weakly). The CLS of a raw checkpoint with no sentence-level objective is poor. |
| **Mean** | Average rows over non-pad positions | Strong default for bidirectional encoders — every row saw full context, averaging denoises. Weaker for causal LMs: you're averaging rows that each saw only a prefix. |
| **Last-token** | Take row `S` (usually the EOS or an appended prompt token) | The natural choice for causal LMs — the one position that saw everything. Standard in E5-Mistral-style models, typically combined with an instruction so the model "knows" to aggregate (e.g. the input ends with a cue like `... Summarize the above in one word:`). |
| **Learned** | A small attention module queries the token states (NV-Embed's "latent attention") | Best quality, adds parameters; trained during contrastive adaptation. |

Pooling is cheap and differentiable, so during embedding training the gradient flows *through* the pooling into the whole network — the model learns to reorganize its token states so that the pooled vector is good, not just to cope with whatever pooling throws away.

## Anisotropy: why raw cosine similarity is broken

Even with sensible pooling, similarities from an unadapted checkpoint are poorly calibrated. Empirically (Ethayarajh 2019, and the "representation degeneration" line of work), LM representations are **anisotropic**: they occupy a narrow cone rather than spreading over the sphere.

Symptoms you can verify in ten lines of code:

- Cosine similarity between two *random, unrelated* sentences ≈ 0.8–0.99 — almost never near 0.
- A few high-variance directions (often correlated with token frequency) dominate every vector.
- Rankings are compressed: "cat sat on the mat" vs. "kitten rested on the rug" and vs. "quarterly earnings beat estimates" may differ by only a few hundredths of cosine.

The information needed to judge similarity is largely *present* — linear probes recover it — but it lives in low-variance directions, swamped by the dominant cone. The fix is not more pretraining knowledge; it is **recalibrating the geometry**, which is exactly what contrastive adaptation (next file) does. The cleanest evidence: SimCSE improves sentence-similarity scores dramatically with a few epochs of contrastive tuning using *no new information at all* (positives are just two dropout-noised copies of the same sentence).

## Why this matters in modern LLM work

Every RAG pipeline, semantic-search system, deduplication pass over pretraining data, and clustering-based eval rests on a text-embedding model — and nearly all competitive ones (E5, BGE, GTE, Qwen3-Embedding, NV-Embed) are checkpointed LMs with a pooling head and a contrastive post-train. Understanding *what the checkpoint's internal states do and don't give you* tells you why that recipe is shaped the way it is, why it is cheap, and where it fails (long documents, asymmetric query–document tasks, cross-lingual drift).

## Self-check

1. Why is the last-token state the only defensible single-position pooling choice for a causal LM, and what is the corresponding choice-of-position concern for mean pooling?
2. A colleague mean-pools the *final* layer of a raw Llama checkpoint and gets near-1.0 cosine similarity for every sentence pair. Name the two distinct problems at play.
3. Why do middle layers often transfer better for similarity than the last layer in an LM trained with a vocabulary-prediction head?

### Answers

1. Under the causal mask, position `i` only attends to positions `≤ i`, so the last token is the only state computed with the full text in view. Mean pooling over a causal LM averages rows that each saw only a prefix — early rows contribute context-blind vectors (and with many short-prefix rows, the average is biased toward sentence-initial content). For a bidirectional encoder this concern vanishes, which is why mean pooling is the encoder default.
2. (a) **Anisotropy** — raw LM states live in a narrow cone, so all cosines are high regardless of meaning; similarity signal exists but in low-variance directions. (b) **Wrong layer** — the final layer is specialized toward next-token logits (logit-lens view) and is generally a worse semantic summary than mid-to-late layers. The two are independent: fixing the layer choice still leaves the cone; contrastive adaptation fixes the cone.
3. The training signal enters through the output head: the last layers' job is to rotate the residual stream into a form whose dot products with the unembedding rows give the right logits. That is a *lossy specialization* toward the vocabulary basis — features useful for general semantic comparison but not for the immediate next token get de-emphasized. Middle layers sit at the peak of "abstract but not yet output-committed."

## Exercise

Take a small causal checkpoint (GPT-2 or Qwen-0.5B) and ~20 sentence pairs you write yourself: 10 paraphrase pairs, 10 unrelated pairs. For each layer `l`, compute mean-pooled and last-token embeddings, and record (a) the mean cosine of paraphrase pairs, (b) the mean cosine of unrelated pairs, (c) the gap between them. Plot the gap vs. layer for both poolings. You should see: near-saturated cosines everywhere (anisotropy), a hump in the gap around 60–80% depth, and last-token pooling beating mean pooling more clearly at deeper layers. Keep this script — the next file's exercise reuses it to measure what contrastive tuning changes.
