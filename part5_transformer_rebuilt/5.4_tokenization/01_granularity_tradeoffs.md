# Tokenization Granularity: Character vs. Word vs. Subword

Tokenization is the map from a raw string to a sequence of integer ids that the embedding table can look up. It's the very first thing that happens to text — component #1 in [ARCHITECTURE.md](../../ARCHITECTURE.md), the `W_E ∈ R^(V × D)` lookup — and it fixes two numbers the rest of the model has to live with: the vocabulary size `V` and how many tokens a given document becomes (its effective sequence length `S`).

You already know the endpoints of this design space. You lived through word-level tokenization and its `<UNK>` problem; you used WordPiece with BERT. This file re-derives *why* the field settled on subword as the universal default, framed as a three-way tension so the modern choice stops looking arbitrary.

**Scope:** the granularity decision only — *what counts as one token*. The specific subword algorithms (BPE / WordPiece / Unigram) are [02](02_bpe_wordpiece_unigram.md); byte-level is [03](03_byte_level_bpe.md).

## The three levers, and why they fight

Every tokenizer trades off the same three quantities. You cannot win all three at once.

- **Vocabulary size `V`** — how many distinct tokens exist. Sets the embedding/unembedding cost (`V · D` params each) and the width of the final softmax.
- **Sequence length `S`** — how many tokens a document becomes. Attention is `O(S²)` in compute and memory ([ARCHITECTURE.md](../../ARCHITECTURE.md), Part 5.1), so `S` is not free.
- **Semantic granularity** — how much meaning one token carries. A whole word is a meaningful unit; a single character is almost none.

Plus the practical fourth axis you remember viscerally: **OOV (out-of-vocabulary) handling** — what happens when text contains something the vocab never saw.

## Character-level

One token per character (or, in the modern version, per byte — see [03](03_byte_level_bpe.md)).

| Lever | Value |
|---|---|
| `V` | Tiny — ~100 for ASCII, a few thousand for Unicode characters |
| `S` | Very long — one token per character, ~4–5× more tokens than subword for English |
| Granularity | Almost none per token — `t`, `h`, `e` carry no standalone meaning |
| OOV | **Impossible** — any string is a sequence of characters you already have |

The appeal is real: no OOV ever, trivially small vocab, no preprocessing decisions. The killer is `S`. Because attention is quadratic, making every token 4–5× shorter makes the attention cost ~16–25× worse for the same text, and a fixed context window holds 4–5× less actual content. On top of that, the model must *relearn* that `t-h-e` is a word — structure that word/subword tokenizers hand it for free. You spend most of your FLOPs modeling short-range spelling regularities instead of meaning.

## Word-level

One token per whitespace/punctuation-delimited word — the classic NLP setup (word2vec, early LSTMs).

| Lever | Value |
|---|---|
| `V` | Huge and *unbounded* — English has a long tail of rare words, names, typos, morphology; realistic vocabs hit 100K–1M+ and still miss things |
| `S` | Short — ~1 token per word, the fewest tokens of any scheme |
| Granularity | High — each token is a meaningful unit |
| OOV | **Brutal** — anything not in the vocab collapses to `<UNK>` |

This is the `<UNK>` world you remember. Two structural problems, not just annoyances:

1. **The vocab is unbounded but your table isn't.** You pick a cutoff (say top 50K words), and everything below it — rare words, proper nouns, misspellings, `covid`, `2026`, `tokenizer` — becomes a single `<UNK>` token. The model literally cannot represent or generate those words. For a *generative* LLM this is fatal: you can't have the model emit `<UNK>`.
2. **No morphology sharing.** `run`, `runs`, `running`, `runner` are four unrelated ids. The model can't exploit the shared stem; it has to learn each independently, wasting capacity and data. `unhappiness` and `happy` look as unrelated as `happy` and `banana`.

Word-level made sense for closed-vocabulary classification. It never made sense for open-ended generation, and that's the regime modern LLMs live in.

## Subword — the compromise that won

Keep frequent words whole; split rare words into meaningful, reusable pieces.

| Lever | Value |
|---|---|
| `V` | Fixed and moderate — you *choose* it (typically 32K–128K) |
| `S` | Moderate — ~1.3 tokens per word for English; ~3–5 bytes per token |
| Granularity | Adaptive — common words are one token, rare words a few morphemes |
| OOV | **None** — worst case, a word falls back to characters/bytes, all of which are in the vocab |

The core idea: **frequency should decide granularity.** A word common enough to earn its own slot (`the`, `running`, `tokenization` if the corpus is technical) stays a single token. A rare word (`antidisestablishmentarianism`) gets decomposed into subword units it shares with other words. This simultaneously:

- **Bounds `V`** — you set the target (e.g. 128K) and the algorithm fills it.
- **Kills OOV** — the vocab includes small enough pieces (down to single characters/bytes) that *any* string is representable. `<UNK>` is gone.
- **Shares morphology** — `run`, `runs`, `running` share the `run` piece; the model sees the relationship for free.
- **Keeps `S` reasonable** — far shorter than character-level because frequent chunks are merged into single tokens.

Concretely, a subword tokenizer might split:

```
"tokenization"   → ["token", "ization"]         (2 tokens: known stem + known suffix)
"unhappiness"    → ["un", "happ", "iness"]       (3 morpheme-ish pieces)
"the"            → ["the"]                        (1 token — frequent, kept whole)
"antidis..."     → ["anti", "dis", "establish", ...]   (rare → many pieces)
```

(Exact splits depend on the algorithm and training corpus — see [02](02_bpe_wordpiece_unigram.md).)

This is why **every modern LLM is subword.** It's the only point on the tradeoff surface that bounds the vocab, eliminates OOV, and keeps sequences short — all at once.

## Tying `V` back to the architecture

Why do modern vocabs land around **`V = 128000`** (the [ARCHITECTURE.md](../../ARCHITECTURE.md) reference config)? It's a direct read of the three-way tradeoff:

- **Bigger `V` → shorter `S`.** More/longer tokens means fewer tokens per document, so cheaper attention and more content per context window. This pushes `V` *up*.
- **Bigger `V` → more embedding params and a wider, slower final softmax.** The token embedding is `V · D`; the unembedding is another `V · D` (unless tied). At `V = 128K, D = 4096` that's ~524M params each. This pushes `V` *down* — though note from [ARCHITECTURE.md](../../ARCHITECTURE.md) that as models scale, embeddings *shrink* as a fraction of the total (~13% at 8B, ~1% at 405B), so a big model can "afford" a bigger vocab more easily than a small one.
- **Diminishing returns on `S`.** Past ~100K, extra vocab slots go to increasingly rare tokens that barely shorten real documents. You pay the full `V · D` cost for tokens that rarely fire.

The sweet spot balances these: large enough that common words and multi-token chunks are single tokens (short `S`, good multilingual coverage), small enough that the embedding matrix and softmax stay affordable. That's how `V ≈ 32K` (Llama-2) grew to `≈ 128K` (Llama-3, GPT-4o class) as models got bigger and more multilingual — the shrinking-embedding-fraction argument gave the budget to spend.

The `S`-side of this tradeoff connects directly to context length: tokenization decides what *one unit of context* is, and thus how much real text fits in a fixed window — see [5.3/05 context-length extension](../5.3_positional_information/05_context_length_extension.md).

## Self-check

1. Character-level tokenization has zero OOV and a tiny vocab — seemingly ideal. Why is it nonetheless the *most expensive* scheme for a given document, and where does the cost concentrate?
2. Word-level keeps sequences shortest of all. Give the two structural reasons (not just "`<UNK>` is annoying") it's disqualified for a generative LLM.
3. You're told a new model bumped `V` from 32K to 128K. Name one thing that gets cheaper and one thing that gets more expensive as a direct result.

### Answers

1. The cost is in sequence length `S`, and it concentrates in **attention**, which is `O(S²)`. Character-level produces ~4–5× more tokens than subword for English, so attention compute/memory rises ~16–25×, and a fixed context window holds ~4–5× less actual text. The tiny vocab saves a little on the embedding matrix but that's a rounding error next to the quadratic attention blow-up. Secondary cost: the model wastes capacity relearning spelling/word structure that subword tokenizers supply for free.
2. (1) **Unbounded vocab meets a fixed table** — you must truncate to the top-`k` words, and everything below the cutoff (rare words, names, numbers, typos) becomes `<UNK>`, which a generator cannot represent *or* emit. (2) **No morphology sharing** — inflected/derived forms (`run/runs/running`) get unrelated ids, so the model can't exploit shared stems and wastes data and capacity learning each in isolation.
3. **Cheaper:** sequence length `S` drops (fewer tokens per document → cheaper `O(S²)` attention, more real text per context window). **More expensive:** the embedding and unembedding matrices grow (`V · D` each, so ~524M → the same formula at 4× the rows), and the final softmax is over 4× more classes. Whether the net is a win depends on scale — big models absorb the extra `V · D` more easily (embeddings are a small fraction of their params).

## Exercise

Take a real subword tokenizer (e.g. `tiktoken` with the `cl100k_base` / GPT-4 vocab, or a Hugging Face `AutoTokenizer` for Llama-3) and tokenize a paragraph of ordinary English. Record the byte count, the token count, and bytes-per-token. Now do the same for (a) a snippet of Python code, (b) a paragraph of a non-Latin-script language (e.g. Japanese or Arabic), and (c) a long chemical or legal term. Compare bytes-per-token across all four. You should see English prose land around 4 bytes/token, code and rare technical terms fragment into more/shorter tokens, and non-Latin text fragment hard (foreshadowing the multilingual inequity in [04](04_tokenizer_pathologies.md)). Finally, estimate what the token count *would* be for the same text at pure character level, and compute the `S` blow-up factor — that ratio is exactly the attention-cost penalty character-level pays.
