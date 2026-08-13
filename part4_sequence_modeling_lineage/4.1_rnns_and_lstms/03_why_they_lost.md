# Why RNNs and LSTMs Lost

Gating (file `02`) fixed the gradient, so RNNs/LSTMs were *trainable* and, for a decade, state-of-the-art in NLP. They still lost. This file is the autopsy — three causes — and the reframing that matters most: they didn't die so much as get **reincarnated** (Part 7.3), because the Transformer's win came with a bill that state-space models are now trying to pay off.

## Cause 1 — sequential compute (the hardware lottery)

The recurrence `h_t = f(h_{t-1}, x_t)` is a strict chain: step `t` waits for step `t−1`. So processing a length-`S` sequence is `O(S)` **sequential** operations — even in *training*, even on a GPU with thousands of idle cores. You cannot parallelize across the time axis.

The Transformer's decisive trick: with teacher forcing, every target position's input is already known, and self-attention has no position-to-position recurrence, so **all `S` positions go through each layer in one batched matmul**. Training parallelizes across the whole sequence. On modern accelerators — which are enormous parallel matmul machines — this is the difference between using 5% and 95% of the hardware. The Transformer didn't just have a better inductive bias; it *fit the hardware*, and that let it absorb far more data and parameters per wall-clock hour. (Generation is still sequential for both — one token at a time — but training throughput is where models are made.)

## Cause 2 — the fixed-state context ceiling

Everything an RNN knows about the past lives in the fixed-size state `h_t ∈ R^D` (file `01`). Two failures follow:

- **A compression bottleneck.** Arbitrarily long history must be squeezed into `D` numbers; distant information competes for bounded capacity and gets overwritten. Gating slows the leak (file `02`) but can't repeal the pigeonhole principle.
- **Residual long-range decay.** Even with the LSTM carousel, dependencies over hundreds-to-thousands of tokens degrade in practice.

Attention answers both by *refusing to compress*: it keeps **every** past token directly addressable and lets the current position attend to any of them with an `O(1)`-length path — perfect recall, no bottleneck (file `4.2/02` is where this idea arrives). The cost is that "keep everything" means `O(S²)` attention compute and an `O(S)` KV cache (Part 9.2) — which is exactly Cause 1's mirror: the Transformer traded RNN's cheap-but-forgetful inference for expensive-but-total-recall inference.

## Cause 3 — hard to scale

Deep RNN stacks are finicky to optimize (the time-unrolled depth compounds with layer depth), they showed no clean, predictable scaling behavior, and the sequential bottleneck made large-data training economically painful. The Transformer, by contrast, stacks cleanly with residuals + pre-norm (Part 3), trains stably at depth, and obeys smooth **scaling laws** (Part 6.3) — you can *predict* that more compute buys more capability. "Just add layers and data, and it reliably gets better" is a property RNNs never reliably had, and it's what made the scaling era possible.

## The reframing: reincarnation, not extinction

Line the two families up on what each is good and bad at:

| | RNN / LSTM | Transformer |
|---|---|---|
| Training over sequence | `O(S)` sequential — bad | `O(S)` parallel — great |
| Inference per token | `O(1)` compute, `O(1)` state — great | `O(S)` compute, `O(S)` KV cache — expensive |
| Long-range recall | bottlenecked by fixed state | perfect (direct access) |
| Scales predictably | no | yes |

The Transformer won by fixing the training/scaling columns — at the cost of the inference columns. So the obvious question, and the whole of Part 7.3: **can we get the RNN's `O(1)`-memory, `O(S)`-total inference back without losing parallel training and quality?** State-space models (S4 → **Mamba**) answer yes — they *are* recurrences (bounded state, cheap inference) rewritten so the training pass parallelizes (via a scan / convolutional form), with input-dependent gates that are the LSTM's `f_t`/`i_t` reborn (file `02`). Linear attention (Part 7.2) attacks it from the other side. Neither has dethroned attention, but the point stands: the RNN's core idea — a compact evolving state — was never wrong, only mismatched to 2017 hardware and training methods.

## Why it matters in modern LLM work

- **It explains the shape of Part 7.** Every "efficient sequence model" is a move in the tradeoff table above — trying to recover a column the Transformer gave up.
- **It's why "context length" is a headline metric.** RNNs had an implicit soft context limit (the state ceiling); Transformers made context an explicit, extendable, but quadratically-expensive resource — which is why long-context techniques (Part 7.5) are a whole subfield.
- **It calibrates the hardware lesson.** The best architecture is partly the one that maps to the accelerator. Attention beat recurrence in 2017 as much for parallelism as for modeling; keep that lens when judging any new architecture.

## Self-check

1. Both an RNN and a Transformer take `O(S)` work to process a sequence in training. Why is only one of them slow, and what property of the hardware makes the difference decisive?
2. Name the two distinct failures caused by the fixed-size hidden state, and the single thing attention does to fix both.
3. The Transformer "won by fixing some columns of the tradeoff table at the cost of others." Which columns did it improve, which did it worsen, and which Part-7 family tries to win the worsened ones back?
4. Why is "obeys clean scaling laws" a competitive advantage, not just a nice property?

### Answers

1. The RNN's recurrence forces `S` *sequential* steps (each waits for the last), while the Transformer does the `S` positions in *parallel* batched matmuls. GPUs/TPUs are massively parallel matmul engines, so the parallel version saturates the hardware and the sequential one starves it — orders of magnitude more effective throughput per wall-clock hour, which translates into more data and parameters.
2. (a) A compression bottleneck — unbounded history crammed into `D` numbers, so distant info is overwritten; (b) residual long-range decay even with gating. Attention fixes both by keeping every past token directly addressable (no compression) with an `O(1)`-length path to any position.
3. Improved: parallel training and predictable scaling. Worsened: per-token inference cost (`O(S)` compute) and memory (`O(S)` KV cache), where an RNN was `O(1)`/`O(1)`. State-space models / Mamba (Part 7.3) — recurrences with parallelizable training — try to recover the inference columns.
4. Because it turns capability into something you can *buy predictably*: you can forecast the loss/quality of a giant run from small ones and justify the spend. An architecture whose returns to scale are erratic can't support billion-dollar training decisions, so it loses the compute race regardless of its per-parameter quality.

## Exercise

You're choosing an architecture for two products. (a) A streaming, on-device assistant that must generate over a very long running conversation with tight, *constant* memory — which family's inference profile do you want, and why? (b) A model to be pretrained once on a trillion tokens across a big GPU cluster, then served — which family's training profile do you want, and why? (c) Given (a) and (b) pull in opposite directions, explain in two sentences why Part 7.3's Mamba-style models are interesting *specifically* as an attempt to have both, and which one property they had to engineer to make an RNN-like model trainable at scale.
