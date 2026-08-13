# Packing, Document Boundaries, and Loss Masking

Between "a corpus of documents" and "batches of `(B, S)` tensors" sits a serialization layer that most explanations skip entirely — and that harbors a disproportionate share of real training bugs. Three questions, each with a default answer and a sharp edge: how do variable-length documents fill fixed-length sequences (**packing**), what happens where two documents meet (**boundaries**), and which positions receive gradient (**loss masking**).

## Packing: documents → fixed-length sequences

GPUs want rectangles. Padding every document to `S` wastes compute on pad tokens (the mean web document is far shorter than 8192), so pretraining **packs**: concatenate documents in stream order, separated by a boundary token (`<eos>`/`<bos>`), and slice the stream into `S`-token windows:

```
[doc A .................. <eos> doc B ......... <eos> doc C (first half…]   ← seq 1
[…doc C second half) <eos> doc D ........................... <eos> doc E…]   ← seq 2
```

Near-100% token utilization, no padding — and two artifacts accepted in exchange:

- **Documents straddle windows** (doc C above): the second window sees C's continuation with no beginning. Harmless at bulk-pretraining scale, but **truncation is not free** for long documents — a 40k-token paper sliced into five 8k windows never exercises dependencies longer than 8k, which is one reason long-context ability requires deliberate long-sequence training ([6.2/04](../6.2_data/04_mixtures_and_midtraining.md)) rather than falling out of packing.
- **Unrelated documents share a window** — which raises the boundary question.

The refinement in modern reports: **best-fit packing** (bin-packing documents into windows to minimize splits) measurably reduces truncation artifacts versus naive concatenation — one of those pipeline details that turned out to matter enough to publish (it's in the Llama-3 report).

## Boundaries: does attention cross documents?

In a packed window, can tokens of doc B attend back into doc A? Two regimes:

- **Historical default: yes.** GPT-2/GPT-3-era packing let attention flow across `<eos>` freely. The model wastes some capacity learning that pre-boundary context is irrelevant (attention *can* learn to stop at `<eos>`, and largely does) — tolerated because masks were all-or-nothing and the waste is small at `S = 1–2k.`
- **Modern default: no — intra-document masking.** Build the attention mask so each token attends only within its own document (block-diagonal within the causal mask). Llama 3 does this, reporting it matters little at short `S` but meaningfully at long `S` — at `S = 128k` a packed window holds *dozens* of unrelated documents, and unrestricted cross-document attention becomes both a capacity tax and a long-context training artifact (the model learns spurious "attend 40k back into an unrelated document" patterns). FlashAttention-style kernels take the document lengths directly (varlen APIs), so this is now nearly free.

The companion decision: **position IDs**. With intra-document masking you typically also restart positions at each document start (RoPE angles reset, [5.3/04](../../part5_transformer_rebuilt/5.3_positional_information/04_rope.md)) so every document trains as if it began at position 0 — matching inference, where prompts do start at 0. Cross-attending regimes must instead use continuous positions. Mask choice and position choice come as a pair; mixing them (masked attention + continuous positions, or vice versa) is a classic silent bug.

## Loss masking: which positions get gradient

The per-position loss layout ([6.1/01](../6.1_pretraining_objectives/01_causal_lm.md)) means you choose, per token, whether it contributes. The regimes:

| Regime | Loss on | Rationale |
|---|---|---|
| Bulk pretraining | **everything** (incl. `<eos>`) | every token is signal; predicting `<eos>` is how the model learns when to *stop* |
| Padding (if any) | never | pads are tensor filler, not data — masked via `ignore_index` |
| Mid-training on Q&A-formatted text | often **completion-only** | you want the model to learn *answers given questions*, not to generate the questions |
| SFT (Part 8.1) | completion-only (prompt masked) | same logic, now standard |

Two edges worth naming precisely:

- **Predicting `<eos>` matters.** Mask loss on `<eos>` and the model never learns to end a document — a real deployed-model failure mode (runaway generations). Conversely, *cross-boundary* predictions (predicting doc B's first token from doc A's last) are pure noise under intra-document masking and are excluded by construction; without intra-doc masking, that one noisy prediction per boundary is simply absorbed.
- **The mechanics are one tensor.** All of this is implemented by setting target entries to `ignore_index` (PyTorch: `-100`) before `F.cross_entropy` — the same mechanism from the [capstone notebook](../../numpy_pytorch/05_capstone_transformer_block.ipynb), doing production duty. Loss masking (which targets count) and attention masking (who sees whom) are independent decisions implemented in different tensors; conflating them is the other classic bug.

## Why it matters in modern LLM work

- **This layer is where "my loss looks weird" bugs live:** off-by-one shifts, loss on pad tokens (deflated loss), missing `<eos>` loss (non-terminating models), position IDs mismatched to the attention mask.
- **It's the boundary Part 8 builds on:** SFT's prompt-masking and chat-template special tokens (Part 8.1) are exactly this file's machinery with a different masking policy — read a `labels[labels == pad] = -100` line in any fine-tuning repo and you're looking at the same tensor.
- **Long-context training** (Part 7.5) is largely *this file's decisions at large `S`*: intra-document masking, position resets, and packing policy dominate whether 128k-training teaches real long-range dependencies or artifacts.

## Self-check

1. Why does pretraining pack rather than pad, and what two artifacts does packing accept?
2. At `S = 2048`, cross-document attention was tolerable; at `S = 128k` it isn't. What changed, quantitatively and qualitatively?
3. Why must intra-document attention masking and per-document position resets be adopted (or rejected) *together*?
4. A colleague masks loss on all special tokens "since they're not real text." What breaks, and how does it present?
5. Distinguish attention masking from loss masking in one sentence each, including the tensor each lives in.

### Answers

1. Padding spends forward-pass FLOPs on tokens that are masked everywhere (pure waste — often the majority of positions, since mean document length ≪ `S`); packing achieves ~100% utilization. Accepted artifacts: (a) documents split across windows, capping learnable dependency length at `S` and fragmenting long documents; (b) unrelated documents cohabiting a window, raising the boundary question.
2. Quantitatively: windows go from holding ~1–3 documents to *dozens*, so cross-document positions dominate the attention matrix rather than being a sliver. Qualitatively: at 128k you are specifically trying to teach real long-range dependencies ([6.2/04](../6.2_data/04_mixtures_and_midtraining.md)) — and unrestricted packing teaches the opposite lesson, that distant context is usually irrelevant noise, poisoning exactly the capability the long-`S` stage exists to build.
3. Because each is calibrated to what the other implies. Masked attention + continuous positions: a document starting at stream position 90k trains entirely at RoPE angles ≥ 90k with no long-range context to justify them — angles inference will present at position 0. Cross-attention + reset positions: two tokens in different documents can share a position ID while attending to each other, making "relative offset" ill-defined. Consistency (mask+reset, or neither) keeps train-time geometry aligned with inference.
4. Masking loss on `<eos>` means the model is never trained to emit end-of-text; generations don't terminate (or terminate only via length caps). It presents *after* training, in sampling — the loss curve looks perfectly healthy, which is what makes it a classic silent bug: the pretraining gauges ([file 01](01_batches_steps_tokens.md)) have no column for "learned to stop."
5. Attention masking decides *who can see whom* — it lives in the attention-score mask (block-diagonal-causal for intra-doc) and shapes the forward computation. Loss masking decides *which predictions are graded* — it lives in the target tensor (`ignore_index` entries) and shapes only the gradient. Independent knobs, independent tensors, independently buggable.

## Exercise

Extend the [capstone notebook](../../numpy_pytorch/05_capstone_transformer_block.ipynb)'s pipeline to real packing. (a) Take ~20 variable-length "documents" (sentences), pack them into `S = 64` windows with `<eos>` separators, and train twice: with and without intra-document attention masking (build the block-diagonal-causal mask by hand from document lengths). (b) Instrument the boundary: log the loss specifically at first-tokens-after-`<eos>` in both regimes and explain the difference you see. (c) Deliberately introduce each classic bug — loss on `<eos>` masked off; position IDs continuous while attention is intra-doc masked — and write one sentence per bug on where it *would* surface in a real run (training gauge, eval, or deployment). (d) Confirm the packed run and a padded-batch run reach similar loss per *document*, then compare tokens-consumed-per-step: the utilization argument, measured.
