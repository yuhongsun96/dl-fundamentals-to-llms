# Tokenizer Pathologies: The Bugs Hiding in the Vocabulary

Tokenization looks like boring plumbing, but it is a surprisingly deep source of real, shipped bugs — the kind that produce bizarre model behavior no amount of staring at the weights explains. This is the "oh, *these* are real" file. Every pathology below has bitten production systems, and each one traces back to a decision made in [01](01_granularity_tradeoffs.md)–[03](03_byte_level_bpe.md) that seemed innocent at the time.

**Scope:** the practical failure modes of a trained tokenizer and their fixes. Nothing here is math; all of it is operational reality.

## 1. Glitch tokens (SolidGoldMagikarp and friends)

**The symptom.** Certain rare tokens, when fed to a model, cause it to behave bizarrely — refusing to repeat the token, hallucinating unrelated words, spelling it wrong, becoming evasive or unhinged. The most famous is `SolidGoldMagikarp` (a Reddit username), one of a cluster of "glitch tokens" found in GPT-2/GPT-3's vocabulary in 2023.

**The mechanism** — and it's a clean one. The **tokenizer and the model are trained on different data, at different times.**

- The tokenizer is trained *first*, on some corpus, and it dutifully creates a token for anything frequent in *that* corpus. If the tokenizer corpus was scraped with a lot of Reddit, usernames like `SolidGoldMagikarp` and counting-subreddit artifacts appeared often enough to earn their own single token.
- The *model* is then trained on a *different* (often cleaned, deduplicated, or later) corpus where those strings barely appear — the Reddit artifacts got filtered out.

Result: the embedding row `W_E[i]` for that token gets **almost no gradient updates during model training**. It stays near its random initialization — essentially **untrained**. When you finally invoke that token at inference, you're pushing a near-random vector into the residual stream, and the model does something undefined and weird downstream. The token is a live wire connected to noise.

**Why it's a good bug to understand:** it makes vivid that the tokenizer is a *separate artifact* trained on *separate data*. Any mismatch between "what the tokenizer thinks is worth a token" and "what the model actually saw" leaves these dead embeddings lying around. Modern pipelines mitigate by training the tokenizer on the same/representative data and by pruning or checking rarely-used tokens, but the failure mode is intrinsic to the two-stage design.

## 2. Number tokenization (why LLMs are bad at arithmetic)

**The symptom.** Models make elementary arithmetic errors that have nothing to do with reasoning ability.

**The mechanism.** BPE groups digits by *frequency*, not by place value. So a number gets chopped inconsistently:

```
"327"   → ["327"]        (one token — this 3-digit string was frequent)
"328"   → ["3", "28"]    (split — this exact string was less frequent)
"1234"  → ["123", "4"]   (arbitrary grouping)
```

Now the model has to do addition where "the same digit" (`3` in the hundreds place) is packaged differently across numbers, and carries have to cross token boundaries that don't align with place value. It's being asked to add numbers written in an inconsistent, frequency-driven encoding. Of course it's error-prone — the representation actively fights the algorithm.

**The modern fix:** force **consistent digit grouping**. Options in use:
- **Single-digit tokenization** — every digit is its own token (`"327"` → `["3","2","7"]`). Llama does this. Uniform place-value alignment, at the cost of longer sequences for numbers.
- **Fixed 3-digit chunking** — group digits in consistent blocks of three (GPT-family and others have moved toward regularized digit handling). A compromise between sequence length and consistency.

Either way, the win is *consistency*: the same digit in the same place is always the same token, so the model can actually learn the carry algorithm. If you ever see a model that's mysteriously good or bad at arithmetic, check how its tokenizer handles digits first.

## 3. Whitespace and the leading-space trap

**The symptom.** Prompts that look identical to you produce different model behavior; a carefully crafted prompt breaks when you add or remove a trailing space.

**The mechanism.** In byte-level BPE (and SentencePiece's `▁`, [03](03_byte_level_bpe.md)), **the leading space is part of the token.** `" the"` (with leading space) and `"the"` (without) are *different token ids*:

```
" the"  → token 262   (the common one — words usually follow a space)
"the"   → token 1169  (rarer — appears mid-word or at string start)
```

The word `the` at the start of a document, or right after a newline, tokenizes differently from `the` in the middle of a sentence. This bites in several concrete ways:

- **Trailing space in a prompt.** If your prompt ends with `"The answer is "` (trailing space) versus `"The answer is"`, the model's *next* token is being predicted in a different context — and one of them may be off-distribution (models rarely see a dangling leading-space token to complete). Trailing spaces in prompts are a classic footgun.
- **Stop sequences and formatting.** String matching on generated text can silently fail because the token boundaries don't line up with your character-level expectations.
- **Few-shot template drift.** Inconsistent spacing between your examples changes their tokenization and can measurably move results.

Rule of thumb: treat leading/trailing whitespace in prompts as *load-bearing*, not cosmetic. When something behaves oddly, print the actual token ids.

## 4. Multilingual inequity

**The symptom.** Non-English text is more expensive, has effectively shorter context, and often lower quality — for the *same* content.

**The mechanism** (set up in [03](03_byte_level_bpe.md)). Byte-level BPE learns most of its merges on the training corpus, which is overwhelmingly English (and code). English text compresses to ~4 bytes/token. Underrepresented languages get few dedicated merges, so their text stays close to the raw UTF-8 byte length — and non-Latin scripts are 2–4 bytes/character in UTF-8 to begin with. Net effect:

| Text | Relative tokens for same meaning |
|---|---|
| English | 1× (baseline) |
| Latin-script European (Spanish, German) | ~1.1–1.5× |
| Non-Latin (Chinese, Hindi, Arabic, Thai) | ~2–5× |

The downstream consequences are not just aesthetic:

- **Cost.** APIs bill per token, so the same message costs 2–5× more in these languages.
- **Effective context.** A 128K-token window holds 2–5× *less* actual text in these languages — your long-context budget is silently smaller.
- **Quality.** More tokens per unit meaning means the model spreads its representation of a concept across more positions, and the language had less effective training signal per byte. Real, measured quality gaps follow.

This is a fairness issue baked into the vocabulary. It's a major motivation for the tokenizer-free work in [05](05_tokenizer_free_future.md), and for the vocabulary growth (`V` 32K → 128K, [01](01_granularity_tradeoffs.md)) that spends the extra slots partly on better multilingual coverage.

## 5. The tokenizer is frozen after pretraining

**The symptom.** You can't fix any of the above — or adapt the vocabulary to your domain — without essentially retraining the model.

**The mechanism.** The tokenizer defines the mapping from tokens to rows of `W_E` (and columns of `W_U`). The model spent its *entire* pretraining budget learning embeddings for *exactly those* token ids. Change the tokenizer and every id now means something different: the embedding table is meaningless, and you're back to pretraining. So in practice:

- **The tokenizer is chosen once, before pretraining, and never changed.** It's the most locked-in decision in the whole stack — you can swap the optimizer, the norm, even the attention variant during a run more easily than the tokenizer.
- **Domain adaptation is constrained.** If your domain (medical, legal, a low-resource language, a new programming language) tokenizes badly under the frozen vocab, you mostly have to live with it. You can *continue pretraining* on domain data, but the tokenization stays inefficient — the model just gets better at working within it.
- **Adding tokens is possible but limited.** You *can* append new rows to `W_E` for new special/chat tokens (`<|im_start|>`, `<|tool_call|>`, etc.) and train just those, but each new token starts from a fresh, untrained embedding (exactly the glitch-token situation, deliberately this time) and needs enough fine-tuning data to become useful. This is precisely how chat/instruct models get their **special tokens** for turn structure and tool calls — grafted on post-pretraining and trained during post-training. The mechanics of those chat templates and special tokens are Part 8.1.

The takeaway: the tokenizer is the one component you should think hardest about *before* you spend the compute, because it's the one you can't take back.

## Putting it together

These aren't exotic edge cases — they're the daily reality of working with tokenized models:

| Pathology | Root cause | Practical tell |
|---|---|---|
| Glitch tokens | Tokenizer/model trained on different data → untrained embeddings | Model melts down on a specific rare string |
| Number errors | Frequency-based digit grouping, inconsistent across numbers | Arithmetic fails in dumb ways; fixed by single/3-digit chunking |
| Whitespace sensitivity | Leading space is part of the token | Trailing-space prompts and stop-strings misbehave |
| Multilingual inequity | Merges learned on English-heavy corpus | Non-English costs 2–5× tokens, shorter context, worse quality |
| Frozen tokenizer | Vocab ↔ `W_E` rows fixed at pretraining | Can't fix any of the above without retraining; new tokens start untrained |

## Self-check

1. Explain the glitch-token mechanism in terms of gradients. Why does a token *present in the tokenizer* end up with an essentially random embedding, and why only *some* tokens?
2. Two numbers, `327` and `328`, tokenize as `["327"]` and `["3","28"]` respectively. Why does this inconsistency specifically wreck *addition*, and what does single-digit tokenization fix about it?
3. You append a new `<|tool_call|>` special token to a pretrained model's vocabulary and it behaves erratically when the token appears. Which two pathologies from this file are you simultaneously reproducing, and why?

### Answers

1. The embedding row for a token only receives gradient when that token appears in the *model's* training data. A glitch token was frequent enough in the *tokenizer's* corpus to earn a vocab slot, but nearly absent from the *model's* (different/cleaned/later) corpus — so its row `W_E[i]` gets almost no updates and stays near random init. It's only *some* tokens because only the ones with a tokenizer-vs-model frequency mismatch (Reddit usernames, scraping artifacts) are starved of gradient; the vast majority of tokens appear in both corpora and train normally. At inference, invoking the untrained row injects a near-random vector into the residual stream → undefined behavior.
2. Addition needs place value: the same digit in the same column should be the same symbol so the model can learn carries. When `327` is one atomic token but `328` is `["3","28"]`, "the hundreds digit 3" is packaged completely differently across the two numbers, and carries must cross token boundaries that don't align with place value. The model can't learn a consistent carry rule over an inconsistent encoding. **Single-digit tokenization** makes every digit its own token, so place value is uniform across all numbers and the carry algorithm is learnable — at the cost of longer sequences for numbers.
3. **Frozen-tokenizer token-addition** and the **glitch-token** mechanism — they're the same phenomenon, now self-inflicted. Appending `<|tool_call|>` adds a fresh row to `W_E` that starts *untrained* (random init). Until you fine-tune enough on data containing it, invoking it pushes a near-random vector into the stream, exactly like a glitch token. The fix is deliberate: train the new embedding (and the model's use of it) during post-training with sufficient data (Part 8.1).

## Exercise

Pick a tokenizer with an accessible vocabulary (`tiktoken` `cl100k_base`, or a HF tokenizer). (1) **Whitespace:** tokenize `"the"`, `" the"`, `"The"`, and `"\nthe"` and confirm they yield different ids; then tokenize a full sentence and note that interior words carry a leading-space token. (2) **Numbers:** tokenize `100`, `1000`, `12345`, `327`, `328`, and a phone number — record which stay whole and which fragment, and whether the grouping is consistent. (3) **Multilingual:** tokenize one sentence's meaning in English, Chinese, and Hindi; compute the token-count ratio to English. (4) **Glitch tokens:** look up the known GPT-2/GPT-3 glitch-token list, find their ids in the vocab, and (if you have model access) observe the model's response when asked to simply repeat one back. Each part should turn an abstract pathology above into something you've now seen fire with your own eyes.
