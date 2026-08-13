# Why Position Must Be Injected

Self-attention has a property that is easy to miss and impossible to ignore once you see it: **it does not know where tokens are.** Feed it "dog bites man" and "man bites dog" and — with position stripped out — it computes the *same* representation for "bites" in both. The mechanism that made attention fast (no recurrence, everything in parallel) is exactly the mechanism that threw away word order. Position has to be added back by hand. This file is about *why*; the next four are about *how*. (One caveat, developed in the NoPE aside below: for *causal decoders* the causal mask already breaks the symmetry, so explicit position is an efficiency win rather than a strict necessity. For encoders it is non-negotiable.)

**Convention:** row-vector (`Y = X W`), the repo default — activations are rows, so `X ∈ R^(B, S, D)`. Dimension names follow [NOTATION.md](../../NOTATION.md); numbers anchor to the 8B config in [ARCHITECTURE.md](../../ARCHITECTURE.md) (`D = 4096`, `H = 32`, `D_h = 128`, `S` up to 8192).

## The core problem: permutation equivariance

A function `f` on a sequence is **permutation-equivariant** if permuting the inputs permutes the outputs the same way, and nothing else changes. Formally, for any permutation `P` (a row-shuffle of the `S` positions):

```
f(P X) = P f(X)
```

Both sublayers of a Transformer block have this property when there is no positional signal:

- **The FFN is applied per position.** Row `i` of the output depends only on row `i` of the input — `FFN(X)[i] = FFN(X[i])`. Shuffle the rows and each output row rides along with its input row. Trivially equivariant.
- **Self-attention mixes across positions, but symmetrically.** The score between query `i` and key `j` is `q_i · k_j / √D_h` — a function of the *content* of tokens `i` and `j`, not of the indices `i` and `j`. There is no `i` or `j` anywhere in the arithmetic except as bookkeeping for which vectors to dot. So if you relabel the positions, the whole score matrix is relabeled identically and the outputs shuffle to match. (See [scaled dot-product attention](../5.1_self_attention/02_scaled_dot_product_attention.md) for the score computation and [multi-head attention](../5.1_self_attention/04_multi_head_attention.md) for why running `H` of these in parallel changes nothing about equivariance.)

Stacking equivariant layers gives an equivariant network. So a position-free Transformer treats its input as a **set of tokens, not a sequence.** It can tell you *which* words are present and how they relate content-wise, but not their order. For language — where "man bites dog" is news and "dog bites man" is not — that is fatal.

> Equivariance vs. invariance. *Invariant* would mean the output doesn't change at all under permutation (`f(P X) = f(X)`) — that's what a pooling layer over a set does. Attention is *equivariant*: the outputs do move, they just move in lockstep with the inputs. Neither is what we want for a sequence; we want the output for "bites" to genuinely differ depending on whether "dog" precedes or follows it.

## Contrast: RNNs never had this problem

In the RNN/LSTM world you came from (Part 4), position was **implicit in the recurrence.** The hidden state `h_t` was computed by folding tokens in one at a time, left to right: `h_t = g(h_{t-1}, x_t)`. Token 5's contribution reached the state through five applications of `g`; token 1's through one. Order *was* the computation — you literally could not present the tokens out of order without changing the arithmetic. Position came for free because the model was inherently sequential.

That sequentiality is also what made RNNs slow: step `t` needs step `t-1`, so the `S` steps cannot be parallelized. Attention's entire pitch was to drop the recurrence and compute all `S` positions at once — `O(1)` sequential depth instead of `O(S)`. **The trade is explicit: you buy parallelism by giving up the free positional ordering, so you owe position back as an added signal.** Everything in Part 5.3 is paying that debt.

## A worked example: two permutations, identical attention

Take three tokens with 2-dim embeddings (tiny, so the arithmetic is visible), no positional signal, single head, `√D_h` folded away:

```
a = [1, 0]     b = [0, 1]     c = [1, 1]
```

Suppose `W_Q = W_K = I` (identity) so `q_i = k_i = x_i`. The unnormalized score `x_i · x_j`:

```
        a       b       c
  a     1       0       1
  b     0       1       1
  c     1       1       2
```

Now feed the **reversed** sequence `[c, b, a]`. The score matrix is the *same numbers*, just with rows and columns relabeled — because each entry still only asks "how aligned are these two token vectors?" The attention output for token `a` is identical whether `a` sits at position 1 (before `b`, `c`) or at position 3 (after them). The model has no way to represent "`a` came first" versus "`a` came last." Reversing the sentence produces a reversed-but-otherwise-identical set of outputs — the network cannot distinguish a palindrome's two halves, cannot tell a subject from an object by position, cannot tell "not good, but great" from "not great, but good."

Inject a position-dependent signal and the diagonal symmetry breaks: `q_i · k_j` starts to depend on `i` and `j`, not just on token content, and reversing the sentence now changes the scores.

## Aside — the causal-mask loophole (NoPE)

The equivariance argument above is airtight for **bidirectional** attention. For a **causal decoder** it is weaker than it looks, and the gap is worth knowing.

Add a causal mask and equivariance **fails immediately**, for a reason unrelated to content: positions become *structurally* distinguishable because they see different numbers of tokens. Position 1 attends to 1 token, position 5 to 5, position 800 to 800. Shuffling the sequence does not shuffle *that* — the mask is pinned to indices. Concretely, a head attending roughly uniformly over its visible window has a softmax denominator of `t`, so its output scales like `1/t`: the model can **count its own position** and synthesize a positional signal from scratch.

This works in practice, and has a name — **NoPE** (no positional encoding). Haviv et al. (2022) trained decoder-only LMs with *zero* positional encoding and found them competitive, with probes recovering absolute position from the activations; Kazemnejad et al. (2023) showed NoPE matches or beats explicit schemes on length generalization for reasoning tasks and can in principle represent both absolute and relative encodings.

Two details on the mechanism. It is **not** that one attention op encodes order — inside a single op the visible prefix is pooled by a content-only softmax, which is order-blind. Order is recovered **through depth**: position `t`'s state is a function of `{x_1..x_t}` and position `t−1`'s of `{x_1..x_{t−1}}`, so a later layer reading both can difference them. And there is **no "first few tokens" exception** — prefix sets are nested and grow by exactly one element, so the chain of prefixes is equivalent to the ordered sequence. Position 1 is in fact the *most* positionally certain token: it attends only to itself.

**So why isn't NoPE mainstream?**

- **It costs capacity and sample efficiency.** NoPE must *learn* counting circuitry and spend residual-stream bandwidth representing position. RoPE hands the model exact relative offsets for zero parameters and zero learning — no reason to make it rediscover arithmetic.
- **Relative structure is what attention actually wants, and NoPE gets it indirectly.** RoPE puts the offset `i − j` straight into the dot product ([04](04_rope.md)); NoPE must first infer both absolute positions, then subtract.
- **It's a decoder-only privilege.** Encoders (BERT) have no mask, so equivariance genuinely holds and a positional encoding is non-negotiable. Half the architecture space can't use this loophole at all.
- **Track record.** RoPE is extremely well validated at frontier scale; NoPE evidence is concentrated at small-to-mid scale. It is gaining ground, though — Kimi K3 (2026) reports NoPE, and hybrid designs (NoPE in some layers, RoPE in others) are an active direction.

The precise claim, then: for causal decoders the mask already breaks the symmetry and makes position *learnable*, so explicit encodings are a strong **efficiency and inductive-bias** win rather than a logical necessity. For encoders, they remain strictly required.

## The design space (the map for the next four files)

Every positional scheme is a point in a small design space. Four axes, and knowing them turns the next files from a list of tricks into a coherent story:

| Axis | Option A | Option B |
|---|---|---|
| **What is encoded** | **Absolute** — token `i`'s slot index | **Relative** — the offset `i − j` between a query and a key |
| **Where it enters** | **Added to embeddings** at the input, then carried up the [residual stream](../5.1_self_attention/01_qkv_projections.md) | **Injected inside attention**, biasing or transforming `q`, `k` directly |
| **Fixed vs. learned** | **Fixed** function of position (no parameters) | **Learned** table or bias (trainable) |
| **Extrapolation** | Degrades / hard-fails past training length | Extrapolates — train short, run long |

The last axis is the one that came to dominate in the LLM era. You pretrain at some context length (say `S = 8192`) because attention is `O(S²)` and long sequences are expensive; then at inference you want to feed *more* — a long document, a big codebase, a multi-turn chat. **A scheme that only works up to its training length is a scheme with a hard wall.** The winning designs are the ones where a token seen at position 20,000 is handled gracefully even though training never went past 8,192.

Here is where each of the next files lands:

| File | Scheme | Encoded | Enters | Params | Extrapolation |
|---|---|---|---|---|---|
| [02](02_sinusoidal_and_learned_absolute.md) | Sinusoidal | absolute | added at input | fixed | mediocre |
| [02](02_sinusoidal_and_learned_absolute.md) | Learned absolute | absolute | added at input | learned | none (hard wall) |
| [03](03_relative_positions.md) | T5 bias | relative | in attention (logit bias) | learned | limited |
| [03](03_relative_positions.md) | ALiBi | relative | in attention (logit bias) | fixed | strong |
| [04](04_rope.md) | RoPE | relative (via absolute) | in attention (rotate q,k) | fixed | good, degrades late |
| [05](05_context_length_extension.md) | PI / NTK / YaRN | — extend a trained RoPE model — | | | the goal |

The arc is: absolute-and-additive (the 2017 answer) → relative-and-in-attention (the fix) → RoPE (the synthesis that won) → extending RoPE past its trained length (the current frontier).

## Self-check

1. The per-position FFN "mixes" nothing across positions, and attention *does* mix across positions. Yet both are permutation-equivariant. Explain why the mixing in attention doesn't break equivariance.
2. RNNs never needed an explicit positional encoding. What specifically about their computation supplied position, and what did the field give up to get rid of it?
3. You append a learned absolute position embedding to the input, then apply a causal, position-free attention stack. Is the resulting network still permutation-equivariant? Why or why not?

### Answers

1. Attention mixes, but the *rule* for mixing — the score `q_i · k_j` — depends only on the token vectors at `i` and `j`, never on the index values `i`, `j` themselves. Permuting the positions relabels every query and key identically, so the entire score matrix is relabeled identically and the mixed outputs come out permuted the same way as the inputs. Mixing across positions is fine; what matters is that the mixing weights are computed from content alone, so they carry no notion of *order*.
2. The recurrence `h_t = g(h_{t-1}, x_t)` folds tokens in strictly left to right, so token `t`'s influence passes through exactly `t` applications of `g`. Order is baked into the computation graph — you cannot reorder inputs without changing the result. The field gave that up to gain **parallelism**: attention computes all `S` positions simultaneously (`O(1)` sequential depth vs. the RNN's `O(S)`), which is what made large-scale training feasible — but it made the layers position-blind, so position must be added back explicitly.
3. No — the input embeddings are no longer interchangeable. Adding `PE(i)` to token `i` makes row `i` of the input carry an `i`-specific signal, so shuffling the tokens no longer shuffles identical rows; the position tags stay put while the tokens move. That is precisely the point of a positional encoding: it breaks the symmetry so the stack can tell orderings apart. (The causal mask *also* breaks equivariance, independently — it hard-codes that position `t` may only attend to `≤ t` — but even a bidirectional stack needs the positional signal.)

## Exercise

Implement a tiny position-free attention layer (single head, `W_Q = W_K = W_V = I`, softmax over scores) and run it on the embedding matrix for a 4-token sequence. Then run it on a random permutation of those 4 rows. Verify numerically that the output rows are the *same permutation* of the first output — i.e. `f(P X) = P f(X)` to floating-point precision. Now add a distinct constant vector to each row (a crude positional signal) before the layer and repeat: confirm the equivariance relation now *fails*, and inspect which output rows changed and by how much. Write one sentence explaining, in terms of the score matrix, exactly what the positional signal broke.
