# Bits Per Byte — The Tokenizer-Independent Metric

## The problem with perplexity

Perplexity is reported per **token**, and different tokenizers chop text differently:
- Character-level: 1 token per byte (approximately, for ASCII) → many tokens
- Byte-pair encoding: ~3–5 bytes per token → fewer tokens
- Whole-word: ~5+ bytes per token → even fewer

A coarser tokenizer has fewer prediction steps per document and lower perplexity *trivially*, not because the model is better. Cross-tokenizer PPL comparison is meaningless.

## Bits per byte (BPB)

Normalize by raw bytes, not tokens:
```
BPB = (total_bits_to_encode_document) / (bytes_in_raw_document)
    = (mean_loss_nats_per_token × num_tokens × log₂(e)) / num_bytes
    = mean_loss_nats_per_token × tokens_per_byte × 1.4427
```

**Tokenizer drops out.** BPB is a property of the model's compression on the underlying text. You can now compare:
- GPT-2 with BPE to Llama with SentencePiece.
- An LLM to gzip to Hutter's compressor.
- A tokenized model to a byte-level model like ByT5 or MambaByte.

## Benchmark numbers to calibrate against

Approximate bits/byte on held-out English text:

| System | BPB |
|--------|-----|
| gzip | ~2.1–2.3 |
| LSTM (~2017 era, word-level) | ~1.1 |
| Transformer-XL | ~1.0 |
| GPT-2 1.5B | ~0.93 |
| GPT-3 175B | ~0.6–0.7 |
| Modern frontier LLMs | ~0.4–0.5 |
| Shannon's human estimate (English) | ~0.6–1.3 |

Lower is better. Every bit saved per byte is a 2× compression improvement.

### "Wait — frontier models beat Shannon's estimate. Isn't that the limit?"

No, and the confusion is worth clearing up, because Shannon's `~0.6–1.3` is **not a fundamental floor**. It's a measurement of *how well 1950s humans predicted English*, and that's a different thing from the true entropy of the text being benchmarked. Four reasons a model can go below it:

1. **Shannon measured human predictive skill, not the entropy floor.** His 1951 estimate came from people guessing the next letter of a sentence. That gives a bound tied to *human* prediction ability — and humans are not optimal predictors. The actual entropy rate of English can be lower than the rate at which a person can guess it. A better predictor legitimately achieves a lower cross-entropy; it isn't violating a law, it's just better at the guessing game than the humans were.

2. **Context length.** Shannon's subjects saw a sentence or two — whatever fits in working memory. A frontier LLM conditions on thousands of tokens. Long-range redundancy a human can't hold in their head — the document's topic, recurring names, established style, code structure, boilerplate — all lowers the *conditional* entropy of the next byte. More context → less surprise → lower BPB.

3. **Different corpora.** Shannon used clean literary prose. Modern BPB is reported on held-out web/code/mixed text, which is far more redundant: HTML scaffolding, repeated phrasing, templated formatting, near-duplicate passages. That text is genuinely more compressible than Dickens, so a low BPB partly reflects an easier (more predictable) corpus, not just a smarter model.

4. **Units aren't identical.** Shannon's figure is per *character* of English; BPB is per *byte*. For ASCII these nearly coincide, but the comparison is approximate, and small unit/encoding mismatches blur the line further.

**Bottom line:** beating `~1.0` just means today's models predict today's benchmark text better than mid-century humans predicted clean prose. The one genuine hard floor is the true entropy rate of the specific corpus you measure on — and nobody knows that number exactly; we only ever get *upper bounds* on it, which is precisely what every BPB figure in the table is.

## Bits per character (BPC)

Historically used for enwik8 / text8. Same idea as BPB but counted in characters. For ASCII text BPB ≈ BPC. For UTF-8 with multibyte chars they diverge slightly. When reading older papers (pre-2020), translate BPC mentally.

## Why "bits" instead of "nats"

Nats are natural for gradients (no base conversion). Bits are natural for humans — they correspond to yes/no questions, bit widths, and the real-world size of compressed files. Convert with:
```
bits = nats × log₂(e) ≈ nats × 1.4427
```

Training loss is almost always in nats. Eval reports often convert to bits.

## Cross-tokenizer evaluation in practice

The Pile / LM Eval Harness reports BPB for exactly this reason. When papers claim "our 7B model beats LLaMA on language modeling", check whether they're comparing:
1. Same tokenizer, same eval set → PPL is valid.
2. Different tokenizers → only BPB is valid.

A surprising number of ambiguous comparisons exist in the wild. Be skeptical of cross-family PPL claims.

## Byte-level models: the extreme case

ByT5, MambaByte, SpaceByte, BLT (Byte Latent Transformer, Meta 2024) operate directly on UTF-8 bytes. No tokenizer at all.

Pros:
- No tokenizer-induced pathologies (glitch tokens, SolidGoldMagikarp, tokenization of numbers/code).
- Cleanly multilingual.
- BPB comparisons are trivially fair.

Cons:
- Much longer sequences (4–5× for English text).
- Historically worse compute efficiency — most of your FLOPs go to short-range byte dependencies that BPE handles for free.
- Recent work (BLT) uses dynamic tokenization / patching to get the best of both worlds.

This is an active area and worth watching. If tokenizer-free becomes competitive on compute, it changes the whole stack.

## Self-check

1. Tokenizer A yields 4 bytes/token and loss 1.6 nats/token. Tokenizer B yields 2 bytes/token and loss 0.9 nats/token. Which model compresses better in bits/byte?
2. Why does reducing vocab size (making tokens shorter on average) typically *increase* per-token loss while leaving BPB roughly unchanged?
3. A byte-level model hits 0.95 BPB; a BPE model hits 0.95 BPB. Which is "better"? (Hint: think about compute and latency.)

### Answers

1. `BPB = (nats/token × log₂(e)) / (bytes/token)`.
   - **A**: `(1.6 × 1.4427) / 4 = 2.31 / 4 ≈ 0.577 BPB`.
   - **B**: `(0.9 × 1.4427) / 2 = 1.30 / 2 ≈ 0.649 BPB`.
   - **A compresses better** (lower BPB). Even though A has higher per-token loss, its tokens span more bytes, so the per-byte rate is lower. This is exactly why per-token PPL deceives across tokenizers.
2. Smaller vocab → tokens are shorter → more tokens per document. Per-token loss decreases proportionally (each prediction step has less to figure out, since each token carries less information), but the **total** bits to encode a document depends only on the underlying text's entropy — bounded below by the language's entropy rate, regardless of tokenization. So per-token loss and bytes-per-token scale together, leaving `BPB = bits/token / bytes/token ≈ constant`. **BPB is the tokenizer-invariant quantity**; per-token PPL is not.
3. Same compression quality (both 0.95 BPB), but **the BPE model is typically better in practice**: it processes ~4× fewer tokens per document (English averages ~4 bytes/BPE-token vs 1 byte/byte-token), so forward/backward passes per training step or inference call are 4× cheaper. Byte-level models are research-relevant (no tokenizer pathologies, multilingual without bias, fair cross-language eval) but slower for the same modeling quality. Newer architectures like BLT (Byte Latent Transformer, Meta 2024) try to close this gap with dynamic patching — the compute story is the active research frontier.

## Exercise

Take a GPT-2 tokenizer and a Llama tokenizer. Tokenize the same 1MB of English text with each. For each, run a same-size pretrained model (or a small scratch-trained one) and compute mean NLL. Convert both to BPB. You should get numbers that are comparable, even though the per-token PPLs will differ by a large factor. This is the point.

## Suggested reading

- "Language Modeling Is Compression" (Delétang et al., 2023) — direct compression benchmarks for LLMs.
- The Pile paper (Gao et al., 2020) — introduces BPB as a standard LM eval unit.
- BLT: Byte Latent Transformer (Meta, 2024) — modern case for going tokenizer-free.
