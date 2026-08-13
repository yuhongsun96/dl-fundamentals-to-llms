# The Tokenizer-Free Future: ByT5, MambaByte, BLT

Step back and notice what [04](04_tokenizer_pathologies.md) actually said. Glitch tokens, number-splitting, whitespace traps, multilingual inequity, the frozen vocabulary — *every one of those bugs is downstream of having a tokenizer at all.* They are not bugs in the model; they are bugs in the hand-designed compression layer we bolt on in front of it. So the obvious question: can we just delete the tokenizer and feed the model raw bytes?

This file surveys the frontier that's trying to. It's genuinely open research — no frontier-scale model has dropped tokenization yet — but the direction is important enough that you should understand the tradeoff and the leading approaches.

**Scope:** motivation and the current tokenizer-free / dynamic approaches. Forward-references Part 7.3 (SSMs) and Part 6.2 (data).

## What a tokenizer really is

Reframe it: **a tokenizer is a hand-designed compression scheme and prior.** BPE looks at a corpus and hard-codes "these byte sequences are common, give them short codes." That buys one thing — shorter sequences `S`, which matters enormously because attention is `O(S²)` ([01](01_granularity_tradeoffs.md)). And it costs everything in [04](04_tokenizer_pathologies.md): a frozen, English-biased, arithmetic-hostile, glitch-prone segmentation baked in before the model sees a single gradient.

Tokenizer-free models make the opposite bet: **feed raw bytes, spend more compute, get robustness back.** No `<UNK>`, no glitch tokens, fair multilingual treatment, consistent digit handling — all the [04](04_tokenizer_pathologies.md) pathologies dissolve because there's no learned vocabulary to be biased or frozen. The price is sequence length, and that price is steep.

## The core obstacle: O(S²) again

Feeding bytes directly means English text is ~4–5× longer in tokens ([1.3/03 bits-per-byte](../../part1_math_foundations/1.3_information_theory/03_bits_per_byte.md)). With standard attention at `O(S²)`, that's a ~16–25× compute/memory hit. That single fact is *why naive byte-level models lose on compute* even though they're more robust — and it's why the interesting recent work isn't "byte-level attention" but "byte-level input paired with something cheaper than quadratic attention."

That framing organizes the field into three responses.

## Response 1: Just use bytes with a Transformer — ByT5

**ByT5** (Google, 2021) is the honest baseline: a T5-style encoder-decoder that operates directly on UTF-8 bytes, no tokenizer.

- **Wins:** robust to noise/typos/rare scripts, cleanly multilingual, no tokenizer pathologies, trivially fair bits-per-byte comparison.
- **Loses:** long sequences → expensive. Slower training and inference for the same modeling quality, exactly the `O(S²)` tax above. Great for robustness-critical or heavily multilingual tasks; not how you'd build a frontier general model on a compute budget.

ByT5 proves tokenizer-free *works*; it also demonstrates precisely why you need to fix the sequence-length cost before it's competitive.

## Response 2: Keep bytes, swap attention for a cheaper mixer — MambaByte

The `O(S²)` obstacle is a property of *attention*, not of byte-level input. So: keep the bytes, replace attention with a sequence mixer that's **linear in `S`**.

**MambaByte** (2024) feeds bytes into a **state-space model (SSM / Mamba)** instead of a Transformer. SSMs mix sequence information in `O(S)` rather than `O(S²)` (the full treatment is Part 7.3), so the byte-level sequence-length blow-up stops being fatal — a 4–5× longer sequence is a ~4–5× linear cost, not a ~16–25× quadratic one. This makes byte-level modeling compute-competitive in a way ByT5 couldn't be. It's the cleanest illustration of the thesis: *tokenizer-free becomes practical once you pair it with sub-quadratic sequence mixing.*

## Response 3: Let the model learn its own chunking — BLT

The third response keeps the Transformer but makes the "tokenization" **dynamic and learned** instead of fixed and hand-designed.

**BLT — Byte Latent Transformer** (Meta, 2024). Instead of a fixed vocabulary, BLT groups bytes into **variable-length patches** based on **entropy**: a small byte-level model predicts the next byte, and where it's *confident* (low entropy — predictable continuation, e.g. the middle of a common word) it lumps many bytes into one patch; where it's *uncertain* (high entropy — a word boundary, a surprising character) it starts a new patch. A big Transformer then runs over the *patches*, not the bytes.

The elegant part: **compute is allocated by information content, not by a frozen token list.** Predictable regions get compressed hard (few patches → short `S` → cheap), surprising regions get fine-grained attention. This is what BPE was crudely approximating with static frequency, done dynamically per-input and end-to-end — and it does it without a fixed vocabulary, so [04](04_tokenizer_pathologies.md)'s frozen-vocab, glitch-token, and multilingual-bias problems don't arise.

## The tradeoff, stated plainly

| | Tokenized (BPE/byte-BPE) | Tokenizer-free (byte / patch) |
|---|---|---|
| Sequence length `S` | Short (hand-designed compression) | Long (bytes) — unless dynamically patched |
| Compute per document | Low | Higher (mitigated by SSM / patching) |
| OOV / glitch tokens | Possible (frozen vocab) | **Gone** — no vocabulary to be biased or dead |
| Multilingual fairness | Biased toward training-corpus languages | **Fair** — bytes are language-agnostic |
| Numbers / whitespace | Pathological ([04](04_tokenizer_pathologies.md)) | Uniform (every byte equal) |
| Adaptability | Frozen at pretraining | No vocab to freeze |
| Maturity | Battle-tested, entire ecosystem | Research frontier |

The honest summary: **tokenization trades robustness for a sequence-length (compute) saving.** Tokenizer-free trades compute back for robustness. The whole game at the frontier is making that compute trade cheap enough — via linear-time mixers (Part 7.3) or dynamic patching — that you get the robustness for free.

## Why we're not there yet at frontier scale

Despite the clean story, no flagship model has dropped the tokenizer as of this writing. Reasons:

- **Compute at scale.** Even sub-quadratic byte models carry overhead, and at frontier training budgets a few percent of efficiency is enormous money. The tokenizer's sequence-length saving is a *real* win that dynamic methods have to fully claw back before the switch pays off.
- **Established pipelines.** The entire stack — data ([Part 6.2](../../ARCHITECTURE.md)), eval harnesses, serving, KV-cache infra (Part 9) — assumes tokens. Ripping out the tokenizer means rebuilding a lot of mature machinery for an uncertain gain.
- **The pathologies are *tolerable*.** [04](04_tokenizer_pathologies.md)'s problems are annoying but mostly worked around (single-digit tokenization, bigger multilingual vocabs, careful prompting). None is fatal enough to force a rewrite yet.

So the frontier bet remains: keep the tokenizer, patch its worst pathologies, and grow the vocabulary. Tokenizer-free is a credible future — the BLT and MambaByte results are strong — but it wins only once the compute trade tips, and it hasn't quite, at scale, yet.

## Self-check

1. Every pathology in [04](04_tokenizer_pathologies.md) "dissolves" in a tokenizer-free model. What single property of byte/patch input makes *all* of them go away at once?
2. ByT5, MambaByte, and BLT are three responses to the same obstacle. Name the obstacle, and say which lever each one pulls to address it.
3. Given that tokenizer-free fixes real bugs, why hasn't a frontier model adopted it? Give the compute reason specifically, in terms of the `O(S²)` tradeoff.

### Answers

1. There is **no learned, frozen vocabulary.** Every [04](04_tokenizer_pathologies.md) pathology traces to the vocabulary being a separate, hand-designed, corpus-biased, frozen artifact: glitch tokens (untrained vocab rows), number-splitting (frequency-based grouping), whitespace traps (space folded into tokens), multilingual inequity (English-biased merges), and frozen-vocab rigidity. Remove the vocabulary — model raw bytes (or dynamically-formed patches) where every byte is equal and nothing is baked in ahead of training — and there's no artifact left to be biased, dead, or frozen.
2. The obstacle is **`O(S²)` attention cost on the ~4–5× longer byte sequences.** **ByT5** doesn't dodge it — it just pays the quadratic tax on bytes with a normal Transformer (baseline, shows robustness but is expensive). **MambaByte** swaps attention for a **linear-time SSM** (Part 7.3), so the longer sequence costs ~`O(S)` not `O(S²)`. **BLT** keeps the Transformer but **shortens `S` dynamically** via entropy-based patching, allocating compute by information content instead of a fixed vocab.
3. Tokenization's whole value is a *real* sequence-length reduction (~4–5× fewer tokens), and because attention is `O(S²)`, that saving is large. A tokenizer-free model has to fully recover that saving — via linear mixing or dynamic patching — before it breaks even on compute, and at frontier training budgets even a small efficiency gap is enormous cost. Add the mature token-based data/eval/serving stack and the fact that the pathologies are worked-around rather than fatal, and the switch simply hasn't paid off yet at scale.

## Exercise

Take a paragraph of English and the same paragraph in a non-Latin language (Chinese/Hindi/Arabic). (1) Count raw UTF-8 **bytes** for each — this is the sequence length a naive byte-level model sees. (2) Tokenize both with a real BPE tokenizer and count **tokens** — the tokenized `S`. Compute the byte/token ratio per language; notice it's much closer to 1 for the non-Latin text (BPE barely compresses it) than for English. (3) Reason through the compute: with `O(S²)` attention, what's the relative cost of the byte-level version vs. the tokenized version for each language? For a linear-`O(S)` model (MambaByte-style), how does that comparison change? You should come away seeing concretely why (a) tokenization's compute win is real, (b) it's *smallest* exactly where the pathologies are *worst* (non-Latin text), and (c) sub-quadratic mixing is what makes dropping the tokenizer plausible. For the deeper SSM machinery behind MambaByte, forward to Part 7.3.
