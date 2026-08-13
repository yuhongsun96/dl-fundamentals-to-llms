# The Subword Algorithms: BPE, WordPiece, Unigram

[01](01_granularity_tradeoffs.md) argued that *subword* is the only workable granularity: bound the vocab, kill OOV, share morphology. This file is the *how*. There are three algorithms that actually get used, plus one library that people constantly mistake for a fourth. You already know WordPiece from BERT; the goal here is to place it next to its cousins and see what criterion each one optimizes, because that criterion is the whole difference.

**Scope:** how a fixed subword vocabulary is *learned* from a corpus, and how a string is then segmented against it. No matrix math here, so no orientation convention needed.

## The shared setup

All three algorithms produce the same *kind* of artifact: a fixed vocabulary of subword units (plus, for BPE, an ordered merge list), such that any string can be segmented into vocab entries. They differ in **how they decide which pieces to keep**. Two build *bottom-up* (start small, grow), one builds *top-down* (start big, prune).

Think of a learned vocabulary as **learned compression** of the corpus: the tokenizer discovers which byte/character patterns recur often enough to deserve their own short code. This is the same lens as the bits-per-byte framing — a good tokenizer is a good compressor of the training text (see [1.3/03 bits-per-byte](../../part1_math_foundations/1.3_information_theory/03_bits_per_byte.md)).

## BPE — Byte-Pair Encoding (Sennrich et al., 2016)

Originally a 1994 data-compression algorithm, repurposed for NMT by Sennrich to solve the OOV problem in translation. The training procedure is almost embarrassingly simple:

1. Start with the vocabulary = all individual characters (or bytes — [03](03_byte_level_bpe.md)) in the corpus.
2. Count all adjacent token pairs across the corpus. Find the **most frequent pair**.
3. **Merge** it into a new single token, add that token to the vocab, and record the merge in an ordered list.
4. Repeat from step 2 until the vocab reaches the target size `V`.

The output is two things: the **vocabulary** and the **ordered merge list**. Encoding a new word replays the merges in the same order — deterministic, greedy, one segmentation per input.

Worked micro-example. Suppose the corpus makes `l`, `o`, `w`, `e`, `r`, `n`, ... the base characters, and the most frequent adjacent pair is `(e, r)`:

```
merge 1: e + r  → er        "lower" = l o w er
merge 2: o + w  → ow        "lower" = l ow er
merge 3: l + ow → low       "lower" = low er
...
```

After enough merges, common words collapse to one token while rare words stay fragmented into these learned pieces. The merge criterion is **raw frequency**: merge whatever pair co-occurs most. Nothing probabilistic, no model of the language — just "these two are seen together a lot, give them one code."

Used by: **GPT-2, GPT-3, GPT-4, RoBERTa** (all byte-level BPE — [03](03_byte_level_bpe.md)), and countless others. BPE is the workhorse.

## WordPiece — the one you know from BERT (Schuster & Nakajima 2012; BERT 2018)

WordPiece is BPE with a different merge criterion. Instead of merging the *most frequent* pair, it merges the pair that most increases the **likelihood of the training corpus** under a unigram language model over the current vocabulary. Concretely, it scores each candidate pair `(a, b)` by

```
score(a, b) = freq(ab) / (freq(a) · freq(b))
```

and merges the highest scorer. Read this as: don't just reward pairs that are *frequent*, reward pairs that are frequent *relative to how often their parts appear separately*. A pair like `(th, e)` might be very frequent, but if `th` and `e` are each independently super-common, merging them buys little; a pair whose parts almost only ever appear *together* is a better merge because it captures real structure. That ratio is the "likelihood gain" — it's what keeps WordPiece from wasting slots on high-frequency-but-low-information pairs.

The other WordPiece fingerprint you'll recognize: the **`##` continuation prefix**. WordPiece marks any subword that is *not* the start of a word:

```
"unhappiness" → ["un", "##happi", "##ness"]
"playing"     → ["play", "##ing"]
```

The `##` says "this piece attaches to the previous one, no space before it." This is how BERT's tokenizer signals word-internal boundaries, and it's the visual tell that you're looking at WordPiece output rather than BPE (which handles the space differently — see the `▁` note below and [03](03_byte_level_bpe.md)).

Used by: **BERT, DistilBERT, ELECTRA, and the BERT family** generally.

## Unigram LM (Kudo, 2018) — the top-down one

Unigram inverts the whole approach. Instead of starting from characters and merging up, it starts from a **large** candidate vocabulary (all frequent substrings) and **prunes down**:

1. Seed a big vocabulary of candidate pieces.
2. Fit a **unigram language model**: each piece has a probability `p(piece)`, and the probability of a segmentation is the product of its pieces' probabilities.
3. For each candidate piece, estimate how much removing it would *hurt* the total corpus likelihood (using EM to re-estimate the segmentations).
4. Remove the pieces that hurt least (the most redundant ones).
5. Repeat until the vocab reaches target size `V`.

Two consequences of keeping an explicit probabilistic model, both of which distinguish Unigram from BPE/WordPiece:

- **One string has *many* valid segmentations, each with a probability.** BPE/WordPiece give you one deterministic split. Unigram can enumerate alternatives — e.g. `"tokenization"` might be `["token", "ization"]` *or* `["token", "iz", "ation"]`, and the model has a probability for each.
- **This enables subword regularization / sampling** — during training you can *sample* different segmentations of the same word so the model learns robustness to how text gets chopped (Kudo's original motivation). BPE has a later analogue ("BPE-dropout"), but the probabilistic model makes it native to Unigram.

Used by: **T5, ALBERT, XLNet, mBART**, and — via SentencePiece's Unigram mode — a large fraction of multilingual models. **Llama** uses SentencePiece with a BPE-style model plus byte-fallback (see [03](03_byte_level_bpe.md)).

## SentencePiece is NOT a fourth algorithm

This trips people up constantly. **SentencePiece (Kudo & Richardson, 2018) is a library/framework, not an algorithm.** It *implements* BPE **or** Unigram — you pick — and its actual contributions are orthogonal to the merge criterion:

- **Raw-stream input, no pre-tokenization.** Classic BPE/WordPiece assume you've already split text into words on whitespace (a language-specific, brittle step — think of languages that don't use spaces, like Japanese or Chinese). SentencePiece treats the input as a **raw stream of Unicode**, including whitespace, and learns tokens over that directly. No language-specific pre-tokenizer needed → **language-agnostic**.
- **The `▁` meta-space.** To keep whitespace reversible, SentencePiece replaces spaces with a visible meta-symbol `▁` (U+2581) and treats it as a normal character that can be merged into tokens:

  ```
  "the cat"  →  ["▁the", "▁cat"]
  ```

  Because the space is *inside* the token, detokenization is **exactly reversible**: concatenate the pieces and turn `▁` back into a space. No guessing where spaces went (contrast WordPiece's `##`, which encodes the *absence* of a leading space and needs detokenization rules).

So when a model card says "SentencePiece," ask the follow-up: *SentencePiece running BPE, or running Unigram?* Those are the two real choices. SentencePiece is the plumbing.

## Comparison table

| Algorithm | Direction | Merge / prune criterion | `▁`/`##`? | Used by |
|---|---|---|---|---|
| **BPE** | Bottom-up (merge) | Most **frequent** adjacent pair | space-handling varies (byte-level: [03](03_byte_level_bpe.md)) | GPT-2/3/4, RoBERTa |
| **WordPiece** | Bottom-up (merge) | Pair with max **likelihood gain** (`freq(ab)/(freq(a)·freq(b))`) | `##` continuation prefix | BERT, DistilBERT, ELECTRA |
| **Unigram LM** | Top-down (prune) | Remove pieces that least hurt **unigram LM likelihood** | `▁` (via SentencePiece) | T5, ALBERT, XLNet, mBART |
| **SentencePiece** | — (library) | *Runs BPE or Unigram* | `▁` meta-space | Llama (BPE+byte-fallback), T5, many multilingual |

## Two ways to read the same word

To make the algorithmic differences concrete, here is roughly how each might handle `"unhappiness"` (exact output depends on the trained corpus):

```
BPE          "unhappiness" → ["un", "happ", "iness"]          (frequency-driven merges)
WordPiece    "unhappiness" → ["un", "##happi", "##ness"]      (## marks continuations)
Unigram      "unhappiness" → ["un", "happi", "ness"]  (one of several scored options)
```

The pieces differ, the philosophy differs, but the payoff is identical and is the point of all three: `un`, `happi`/`happ`, `ness`/`iness` are **reusable** across `unkind`, `happily`, `kindness` — the morphology sharing that word-level tokenization threw away, delivered without ever emitting `<UNK>`.

## Self-check

1. BPE and WordPiece are both bottom-up mergers. State the single difference between them in one sentence, and explain what it buys WordPiece.
2. Unigram can produce multiple segmentations of one word; BPE gives exactly one. Which property of Unigram makes multiple segmentations possible, and what training-time trick does it enable?
3. A colleague says "we switched from WordPiece to SentencePiece." Why is that sentence underspecified, and what's the actual question you should ask?

### Answers

1. **BPE merges the most *frequent* adjacent pair; WordPiece merges the pair with the highest *likelihood gain*** (`freq(ab)/(freq(a)·freq(b))`). The likelihood-gain criterion down-weights pairs whose parts are already individually common (merging them adds little information) and up-weights pairs whose parts almost always co-occur, so WordPiece spends vocab slots on genuinely bound units rather than on frequent-but-uninformative adjacencies.
2. Unigram keeps an explicit **unigram language model** — every piece has a probability, and any segmentation's probability is the product of its pieces'. Because multiple segmentations of a word are all assigned probabilities, the tokenizer can enumerate/score them rather than committing to one. This enables **subword regularization**: sampling different segmentations of the same text during training so the model becomes robust to how words get chopped (BPE's later analogue is BPE-dropout).
3. SentencePiece is a **library**, not an algorithm — it can run either BPE or Unigram, and its real features (raw-stream input, the `▁` meta-space, reversible detokenization, no whitespace pre-tokenization) are independent of the merge criterion. The actual question: *is the new SentencePiece tokenizer running BPE or Unigram?* — plus whether it's byte-level / has byte-fallback ([03](03_byte_level_bpe.md)).

## Exercise

Install `tokenizers` (Hugging Face) or `sentencepiece` and *train* two small tokenizers from scratch on the same modest corpus (a few MB of text is plenty): one BPE, one Unigram, both targeting the same `V` (say 8000). Then tokenize a held-out paragraph with each and diff the outputs. Look for: (a) words where the two disagree on the split, (b) whether Unigram's pieces look more "morpheme-like" than BPE's frequency-driven ones, (c) the average tokens-per-word for each. Finally, for the Unigram model, enable sampling (`nbest`/`alpha`) and tokenize the same word several times — watch the segmentation vary. That variance is subword regularization; there's no equivalent knob on the vanilla BPE model.
