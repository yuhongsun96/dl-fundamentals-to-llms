# Byte-Level BPE: The GPT-2-Onward Default

[02](02_bpe_wordpiece_unigram.md) described BPE as merging characters. Every production BPE tokenizer since GPT-2 does something subtly different and much more robust: it runs BPE over raw **bytes**, not Unicode characters. This one change — swapping the base alphabet from "all the characters in the corpus" to "the 256 possible byte values" — is what makes modern tokenizers immune to OOV in the strongest possible sense. It's worth its own file because the idea is small but its consequences are everywhere.

**Scope:** the byte-level base vocabulary and byte-fallback. Assumes BPE mechanics from [02](02_bpe_wordpiece_unigram.md).

## The problem with a character-level base

Recall character-level BPE: the base vocabulary is "every character seen in the training corpus." That's a subtle trap. What happens at inference when the input contains a character the tokenizer *never saw during training*? An emoji it wasn't trained on, a rare CJK glyph, a mathematical symbol, a character from a script absent from the corpus? There's no base token for it. You're back to an OOV — the `<UNK>` problem [01](01_granularity_tradeoffs.md) promised subword had killed, sneaking back in through the base alphabet.

Unicode has ~150,000 assigned code points and growing. You cannot put them all in the base vocab (that's most of your budget gone before a single merge), and any subset you *do* choose will miss something.

## The insight: bytes as the alphabet

Every string, in any language, is stored as a sequence of **UTF-8 bytes**, and there are exactly **256 possible byte values**. So:

> Run BPE over the raw UTF-8 byte stream. The base vocabulary is the 256 byte values, full stop.

Now *any* string — any language, emoji, code, mathematical notation, even arbitrary binary-ish junk — is representable, because every string is *by definition* a sequence of bytes, and all 256 bytes are in the vocab. **OOV becomes literally impossible.** There is no character the tokenizer can't fall back to expressing as its constituent bytes.

On top of these 256 base tokens, BPE learns merges exactly as before — frequent byte pairs merge into subword tokens, common words end up as single tokens. The base is just bytes instead of characters. This is **byte-level BPE**, introduced at scale by **GPT-2** and used by GPT-3, GPT-4, RoBERTa, and most of the GPT lineage.

A closely related mechanism is **byte-fallback**, used by Llama's SentencePiece tokenizer: the vocabulary is mostly learned subword pieces (character-level flavor), but any character the tokenizer can't otherwise represent is emitted as its raw UTF-8 bytes. Same OOV-elimination guarantee, reached from the other direction — a character-ish vocab with a byte safety net underneath.

## Why this matters

- **Robustness — no `<UNK>`, ever.** This is the headline. There is no input string that breaks the tokenizer. Whatever bytes come in, tokens come out, and the model can always represent and generate them.
- **Clean multilingual and code handling.** Chinese, Arabic, Hindi, Emoji, source code with unusual glyphs, mixed-script text — all just bytes. No per-language vocabulary curation, no separate handling. One tokenizer covers everything.
- **No preprocessing brittleness.** Character-level pipelines often need Unicode normalization, accent handling, script detection — brittle, locale-dependent steps that silently corrupt edge cases. Byte-level sidesteps most of it: the atoms are bytes, and normalization (if any) is a deliberate choice rather than a hidden dependency.

For anyone who fought `<UNK>` and encoding bugs in the pre-2020 pipeline, this is the quiet fix. Byte-level BPE is *why* modern models never choke on weird input the way old ones did.

## The costs (and where they lead)

Byte-level isn't free — it just moves the cost somewhere more tolerable.

- **Non-Latin characters cost multiple byte-tokens.** UTF-8 encodes ASCII in 1 byte, but Latin-with-accents in 2, most CJK in 3, and many emoji in 4. So before merges kick in, a single Chinese character is 3 bytes = potentially 3 tokens, versus 1 token for an English letter cluster. Merges *do* reclaim a lot of this for well-represented languages in the training corpus, but languages underrepresented in training stay fragmented. This is the root of **multilingual inequity** — non-English text costs 2–5× more tokens — which gets its own full treatment in [04](04_tokenizer_pathologies.md).
- **Bytes are opaque; merges must relearn character structure.** A character-level tokenizer at least knows `é` is one character. A byte-level one sees `é` as the two bytes `0xC3 0xA9` and has to *learn*, via merges, that those two bytes belong together. Most of the time the merges handle it — common characters get merged back — but it means some of the vocabulary budget is spent re-discovering character boundaries that were free at the character level. In exchange, you never hit an OOV. It's a good trade.

Contrast this with a **pure byte/character model** (no BPE at all, one token per byte — [05](05_tokenizer_free_future.md)): that has the same OOV-immunity but pays the full sequence-length blow-up from [01](01_granularity_tradeoffs.md) because it never merges. Byte-level *BPE* is the middle ground — byte base for robustness, learned merges to keep `S` short.

## Why `V` still ends up ~50K–128K

If the base vocab is only 256, why do byte-level tokenizers report `V ≈ 50257` (GPT-2), `≈ 100K` (GPT-4's `cl100k_base`), or `≈ 128000` (Llama-3, the [ARCHITECTURE.md](../../ARCHITECTURE.md) reference)?

Because `V` = **256 byte base tokens + all the learned merges + a handful of special tokens**. The 256 bytes guarantee coverage; the ~50K–128K merges are what make the tokenizer *efficient* (they collapse common byte sequences — words, morphemes, frequent code patterns — into single tokens so `S` stays short). The target `V` is chosen by exactly the [01](01_granularity_tradeoffs.md) three-way tradeoff: big enough that common content is one token per word-ish chunk (short sequences, decent multilingual coverage), small enough that the `V · D` embedding and the final softmax stay affordable. The byte base is a floor for *robustness*; the merges are the budget you spend on *compression*.

So the two numbers coexist without contradiction: **256 is the base alphabet, 128K is the vocabulary** — the base plus ~127,744 learned merges and special tokens.

## Self-check

1. Character-level BPE and byte-level BPE both eliminate OOV within the training corpus. What specific failure can still hit a *character-level* tokenizer at inference that a *byte-level* one is structurally immune to?
2. The base vocabulary is 256, but `V` ends up ~128K. Where do the other ~127.7K tokens come from, and what job do they do that the 256 bytes don't?
3. Byte-level BPE fragments a rare non-Latin character into several tokens while an English word is one token. Trace this to the UTF-8 encoding, and name the downstream fairness problem it creates.

### Answers

1. A **character it never saw in training** — a rare glyph, an emoji outside the training set, a symbol from an absent script. A character-level tokenizer has no base token for it (the base alphabet was "characters seen in the corpus"), so it's an OOV. A byte-level tokenizer's base alphabet is all 256 byte values, and *every* string is by definition a sequence of bytes, so there is no representable-in-UTF-8 string it can't tokenize. OOV is structurally impossible, not just rare.
2. The other ~127.7K are **learned BPE merges** (plus a few special tokens). The 256 bytes provide *coverage* — the guarantee that anything is representable. The merges provide *efficiency* — they collapse frequently co-occurring byte sequences (words, morphemes, common code/whitespace patterns) into single tokens, which is what keeps sequence length `S` short. Without merges you'd have coverage but a pure-byte-length sequence blow-up ([01](01_granularity_tradeoffs.md)).
3. UTF-8 uses 1 byte for ASCII but 2–4 bytes for non-Latin scripts and emoji (e.g. most CJK characters are 3 bytes). Before merges, that's up to 3–4 byte-tokens for a single non-Latin character versus 1 token for a cluster of English letters, and merges only reclaim this well for languages well-represented in the training corpus. Underrepresented languages stay fragmented, so their text costs 2–5× more tokens — **multilingual inequity** ([04](04_tokenizer_pathologies.md)): more expensive per request, effectively shorter usable context, and often worse quality.

## Exercise

Using `tiktoken` (GPT-4's `cl100k_base`) or a Llama-3 tokenizer, tokenize the same short sentence written in English, then translated into Chinese, Hindi, and Arabic (any translator is fine). Record tokens-per-sentence for each. You'll see the non-Latin versions cost noticeably more tokens for the *same meaning* — quantify the ratio versus English. Then feed the tokenizer something adversarial: an emoji sequence, a snippet of minified JavaScript, and a line of random bytes (e.g. read a few bytes from a binary file and decode as latin-1). Confirm that (a) nothing ever produces an `<UNK>` / error, and (b) the "junk" inputs fragment into many short (often single-byte) tokens — you're watching the byte base do its OOV-proofing job with no merges to lean on.
