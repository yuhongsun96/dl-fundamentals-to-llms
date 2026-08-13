# Masked LM and Denoising — the Bidirectional Objectives

The other two classical objectives, and the ones you actually lived through: BERT's masked LM and the T5/BART denoising family. This file states each precisely, then does the accounting that [file 03](03_why_decoder_only_won.md) needs — because the interesting question isn't *how* MLM works (you know), it's *what it costs* relative to causal LM.

## Masked LM (BERT, 2018)

Corrupt, then reconstruct with both-sided context:

```
input:   The [MASK] sat on the [MASK] .        (15% of positions selected)
target:  predict "cat" and "mat" at the masked positions only
```

The recipe's fine print, each part load-bearing:

- **15% of positions are selected** for prediction. Of those: **80%** become `[MASK]`, **10%** a random token, **10%** left unchanged. The 10/10 split exists because `[MASK]` never appears at inference — if the model only ever predicted at `[MASK]`, its representations of *real* tokens would never train for the prediction task. The random/unchanged corruption forces every position to stay "alert."
- **Loss only at selected positions.** The other 85% of positions get no gradient from the prediction head — they participate only as context.
- **Bidirectional attention** ([5.1/05](../../part5_transformer_rebuilt/5.1_self_attention/05_causal_and_bidirectional_masking.md)): no causal mask, every position sees the whole sequence. That's the point — and the price, because it's exactly what makes generation impossible.

Why it made sense in 2018: for *understanding* tasks (classification, NER, QA-span), a token's meaning depends on both sides, and MLM representations with both-sided context beat causal ones at equal (small) scale. BERT was a representation factory, and fine-tuning was the product.

## Denoising / span corruption (T5, BART — 2019)

The encoder-decoder generalization: corrupt the input more aggressively, and *generate* the missing content rather than classify it per-position.

**T5 span corruption:** drop out contiguous **spans** (15% of tokens, mean span length 3), replace each span with a sentinel token, and train the decoder to emit the deleted spans in order:

```
input:   The <X> sat on <Y> mat .
target:  <X> cat <Y> the <Z>
```

**BART:** same shape, different noise menu — span infilling, sentence permutation, deletion — with a standard causal decoder reconstructing the *entire* original text.

What this family buys: an encoder for understanding *and* a decoder for generation in one model, and a corruption task closer to real seq2seq work (the decoder learns to produce fluent multi-token text conditioned on an encoded input). T5's "everything is text-to-text" framing was also the first serious unification pitch — the right idea, one architecture too early. **UL2** (2022) later showed you can mix denoising flavors (short spans, long spans, prefix-LM) as a single "mixture of denoisers" — a sign that the objectives are points on one corruption continuum, not different species.

## The accounting that decides everything

Line the three up on the dimensions that matter at scale:

| | Causal LM | MLM | Span denoising |
|---|---|---|---|
| Supervised positions per pass | **100%** | **15%** | ~15% (decoder targets) |
| Generation | native | no | yes (decoder) |
| Bidirectional context | no | **yes** | encoder side |
| Train/inference input mismatch | none | `[MASK]` never real | sentinels never real |
| Zero-shot prompting | natural (continue the text) | awkward | possible, clunky |
| Stacks/losses to maintain | one | one + task heads | two stacks, cross-attention |

The first row is the one to internalize. MLM extracts loss from 15% of positions — roughly **6–7× less supervision per token of compute** than causal LM, which supervises every position. You can raise the mask rate, but only so far: mask too much and there's no context left to condition on; the corruption *is* the context budget. Causal LM has no such tension — every position is simultaneously context (for later positions) and target (for its own prediction), because the causal mask separates the two roles by construction.

That, plus the unification story, is most of why the race ended the way it did — but the full argument (including serving economics and what happened to encoders) is [file 03](03_why_decoder_only_won.md).

## Where these objectives still live

Not a graveyard — a reassignment:

- **Encoder-only + MLM is alive everywhere text is *represented* rather than generated:** embedding models and rerankers for retrieval (E5/BGE/GTE lineage — Part 8.5), and classifiers. Bidirectional context genuinely is better for "what does this text mean as a vector," and Part 8.5's LLM2Vec trick — converting a causal LM to an embedder by *removing* its causal mask — is backhanded proof.
- **Encoder-decoder survives where the input is a different modality or genuinely fixed:** Whisper (audio → text, Part 10.4), translation systems, and T5-family models in production pipelines.
- **Span-corruption ideas recur** in code models' fill-in-the-middle training (predict a deleted middle given prefix + suffix — span corruption re-serialized for a causal decoder), and diffusion-style text models keep reinventing "corrupt and reconstruct."

## Self-check

1. Why does BERT replace only 80% of selected positions with `[MASK]`, rather than all of them?
2. State the supervision-density gap between MLM and causal LM, and explain why "just mask 90% instead of 15%" doesn't close it.
3. T5's targets contain sentinel tokens that never appear in real text — the same train/inference mismatch as `[MASK]`. Why is it less damaging for T5's use case than BERT's?
4. Code models train on fill-in-the-middle without an encoder. What is FIM, in this file's vocabulary?

### Answers

1. Because at inference the input contains no `[MASK]`. If prediction were only ever demanded at `[MASK]` positions, the model could learn "real tokens are never queried" and let their prediction-relevant features atrophy; the 10% random + 10% unchanged corruption keeps the predict-this-position machinery trained on realistic inputs too.
2. Causal LM gets a loss term at 100% of positions; MLM at 15% — ~6.7× fewer supervised predictions per sequence at equal compute. Raising the mask rate cannibalizes the conditioning signal: masked positions are predicted *from* the unmasked ones, so more targets means less context per target, and the task degrades toward unconditional guessing. Causal LM escapes the trade because the causal mask lets every position be *both* context and target at once.
3. T5's decoder consumes sentinels only as *span markers* in a seq2seq task it was explicitly trained and deployed on — fine-tuning and inference use the same text-to-text format, so the "mismatch" is part of the format. BERT's mismatch is between pretraining (inputs with `[MASK]`) and *every* downstream use (inputs without), so the gap is crossed at exactly the moment that matters.
4. Span corruption re-serialized for a decoder-only model: delete a middle chunk, move it to the end (`prefix, suffix → middle` with special tokens marking the pieces), and train with plain next-token loss. Bidirectional *conditioning* on prefix and suffix is achieved by reordering the text rather than by removing the causal mask — denoising's benefit at causal LM's supervision density.

## Exercise

Take the tiny GPT from the [capstone notebook](../../numpy_pytorch/05_capstone_transformer_block.ipynb) and convert it to a toy MLM: remove the causal mask, add a `[MASK]` token to the vocab, mask 15% of positions (80/10/10), and compute loss only at selected positions. Train it and the causal version for the same number of *steps* on the same corpus. Compare (a) loss-curve smoothness — MLM's is noisier; explain why from the number of loss terms per batch — and (b) what happens when you try to *sample* text from each. Then raise the MLM mask rate to 60% and report what happens to the loss and why.
