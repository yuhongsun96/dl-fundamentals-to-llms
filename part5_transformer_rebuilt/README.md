# Part 5 — The Transformer, Rebuilt From Scratch

You've seen this architecture. You read "Attention Is All You Need" when it came out and you fine-tuned BERT. But the mental model that survived from 2018 is *encoder, bidirectional, sinusoidal positions, LayerNorm-after, WordPiece* — and almost every one of those defaults has since flipped. This part rebuilds the Transformer from the inside so that when you read a 2024 model card and it says *decoder-only, GQA, RoPE, RMSNorm-pre, SwiGLU, byte-level BPE*, every word lands.

This is the centerpiece deep-dive. Parts 1–3 built the machinery (linear maps, softmax, backprop, residual streams, normalization); Part 4 disposed of the RNN lineage. Here we assemble the actual object the rest of the curriculum refines: Part 6 pretrains it, Part 7 makes it efficient, Part 8 post-trains it. [ARCHITECTURE.md](../ARCHITECTURE.md) is the one-page map of the finished machine — this part is the guided tour of each component on that map.

## Structure

- **5.1 Self-Attention Mechanics** — the one genuinely new primitive.
  - `01` Q/K/V projections — three learned views of the stream, and why three separate matrices.
  - `02` Scaled dot-product attention — the full operation and the `√d_k` scaling (the *why* was derived in [1.2/03](../part1_math_foundations/1.2_calculus_probability/03_softmax_logsumexp.md); here it's assembled).
  - `03` Attention as soft lookup / kernel smoothing — the mental models that make it click.
  - `04` Multi-head attention — the full treatment of *why more than one head* (promised in [1.2/03](../part1_math_foundations/1.2_calculus_probability/03_softmax_logsumexp.md)), what heads specialize in, the interpretability tie-in.
  - `05` Causal vs. bidirectional masking — the `-∞` trick, why causal masking makes training parallel, encoder vs. decoder.
- **5.2 The Full Block** — how attention and the FFN compose into the unit repeated `L` times.
  - `01` Assembling the block — token-mixing vs. channel-mixing; the residual + norm wrapping (leans on Part 3).
  - `02` The FFN — its role as per-position memory, why the `4×` width, and the pointer to gated variants.
- **5.3 Positional Information** — attention is permutation-equivariant, so position must be injected.
  - `01` Why position must be injected — the bag-of-words failure and the design space.
  - `02` Sinusoidal & learned absolute — the original two options and why neither extrapolates.
  - `03` Relative positions — T5 bias and ALiBi.
  - `04` RoPE — the modern default; the complex-plane rotation view and why *relative* falls out of the dot product.
  - `05` Context-length extension — position interpolation, NTK-aware, YaRN (forward to Part 7.5).
- **5.4 Tokenization** — the interface between text and the model, and a surprising source of bugs.
  - `01` Granularity tradeoffs — char vs. word vs. subword; the vocab-size / sequence-length tension.
  - `02` BPE, WordPiece, Unigram — the subword algorithms and SentencePiece the tool.
  - `03` Byte-level BPE — the GPT-2-onward default that can never hit an OOV.
  - `04` Tokenizer pathologies — glitch tokens, number splitting, whitespace, multilingual inequity.
  - `05` The tokenizer-free future — ByT5, MambaByte, BLT, and why we're not there yet.

## How to use

Read Part 3 first if the residual stream isn't yet second nature — 5.2 assumes it. The dependency chain into Part 1/2 is real: `√d_k` (5.1/02) uses the variance argument from [1.2/03](../part1_math_foundations/1.2_calculus_probability/03_softmax_logsumexp.md), and 5.2 defers all activation/SwiGLU detail to [2.1/02](../part2_neural_network_fundamentals/2.1_mlp_building_block/02_activations.md). Keep [ARCHITECTURE.md](../ARCHITECTURE.md) open alongside — every file here corresponds to a labeled component in its block diagram.

The one file to *do*, not just read, is 5.1: implement single-head attention, then multi-head, then add a causal mask, on toy tensors. The rest of the Transformer is machinery you already have from earlier parts.

## Target time

4–6 days. 5.1 (attention) and 5.3 (positions, especially RoPE) are the load-bearing subsections and deserve the most time; 5.2 is short because it leans on Parts 2–3; 5.4 (tokenization) is conceptually easy but full of practical gotchas worth internalizing.

## What's deliberately omitted

- **Efficient-attention mechanics** (FlashAttention, MQA/GQA, sliding-window, MLA) — those are Part 7. Here we build *exact, full* attention and note where Part 7 optimizes it.
- **Norm-layer formulas and pre-vs-post derivations** — [2.3/03](../part2_neural_network_fundamentals/2.3_init_normalization/03_norm_layers.md) and Part 3 own those; 5.2 only assembles them into the block.
- **The SwiGLU `2/3` derivation** — fully worked in [2.1/02](../part2_neural_network_fundamentals/2.1_mlp_building_block/02_activations.md); 5.2 references it.
- **Cross-attention and the full encoder-decoder** (T5/BART) — touched in 5.1/05 and Part 6.1; the reference architecture here is decoder-only.
