# The Seed for Transformers

By 2016, attention (file `02`) was the best thing in sequence modeling — but it was an *add-on*: a module bolted onto an RNN encoder–decoder to relieve the bottleneck. "Attention Is All You Need" (Vaswani et al., 2017) made the radical move — **delete the recurrence, keep only the attention** — and it worked better. This file is the bridge: what the leap was, what attention had to grow to survive on its own, and the through-line that lands you in Part 5.

## The leap: attention as the model, not an add-on

The 2016 mental model was "RNN does the sequence processing; attention helps the decoder peek at the encoder." The 2017 insight inverted it: if attention already gives every position direct, weighted access to every other position (file `02`), then attention *is* the sequence-processing mechanism — the RNN is redundant. Remove it entirely. What's left is a stack of:

```
self-attention  →  add & norm  →  FFN  →  add & norm       (repeated L times)
```

no recurrence anywhere. This was genuinely counterintuitive: recurrence had been assumed *necessary* for sequence modeling for decades. The paper's real contribution wasn't inventing attention (file `02` did that) — it was proving attention **alone**, made parallel and stacked deep, beats recurrence on both quality and — decisively — training speed (the Cause-1 parallelism win, file `4.1/03`).

## What attention had to grow to stand alone

Bahdanau's cross-attention couldn't just be dropped in as-is; standing without an RNN exposed gaps, and each fix is a Part 5 topic:

| Upgrade | Why it was needed once the RNN was gone | Covered in |
|---|---|---|
| **Self-attention** | With no recurrence, positions must exchange information *some* way; attention pointed at the sequence itself does it. | Part 5.1 |
| **Scaled dot-product** (`/√d_k`) | Dot-product scores grow with dimension and saturate softmax; the scale keeps logits sane at init. | Part 5.1 |
| **Multi-head** | One attention distribution is roughly unimodal (file `02`); real sentences need to attend to several relations at once (syntax, coreference, position). | Part 5.1 |
| **Positional encoding** | The RNN encoded order *implicitly* via its step-by-step processing; a pure attention layer is **permutation-equivariant** (order-blind), so order must be injected explicitly. | Part 5.3 |
| **Residuals + pre/post-norm** | Stacking many attention+FFN layers deep needs the gradient-highway machinery (Part 3) to train at all. | Part 3, Part 5.2 |
| **Position-wise FFN** | Attention mixes *across* positions but does no per-token nonlinear transformation; the FFN supplies it. | Part 5.2 |

Read that table as the agenda for Part 5: the Transformer is Bahdanau attention, generalized to self-attention, made numerically stable (`√d_k`), made expressive (multi-head + FFN), given back its sense of order (positional encoding), and made stackable (residuals + norm).

## The through-line (and the pendulum)

The whole of Part 4 in one arc:

```
RNN                → seq2seq            → attention          → Transformer
(sequential state)   (fixed-vector        (direct weighted      (attention-only,
                      bottleneck)          access, alignment)    parallel, stackable)
```

Each stage fixes the previous one's fatal flaw: seq2seq gave RNNs a clean encode→decode shape but bottlenecked it; attention removed the bottleneck; the Transformer removed the recurrence that made attention slow and hard to scale. And the pendulum doesn't stop: having deleted recurrence for parallelism, the field pays for it with `O(S²)` compute and an `O(S)` KV cache (file `4.1/03`, Part 9.2) — so Part 7.3 brings **recurrence back** (state-space models / Mamba) to reclaim cheap inference, now with parallelizable training. RNNs weren't a dead end; they were one swing of a pendulum still in motion.

## Why it matters in modern LLM work

- **It's the on-ramp to Part 5.** Every Transformer component has a *reason for existing* that only makes sense against this history — the upgrade table above is that list of reasons. You'll build each one from scratch next.
- **"Delete the assumed-necessary component" is a repeatable pattern.** Recurrence was assumed necessary and wasn't. Later: convolutions (in vision), then — arguably — dense attention itself (MoE, SSMs). The lesson is to keep asking which "load-bearing" piece is actually load-bearing.
- **It fixes the correct framing for judging new architectures.** The Transformer won on a *combination* — quality, parallel training, clean scaling — not any single axis. When Part 7's alternatives claim to beat it, hold them to the same multi-axis bar.

## Self-check

1. Attention existed in 2015. In one sentence, what was actually new in the 2017 Transformer?
2. A pure self-attention layer is permutation-equivariant. What does that mean, why is it a problem the RNN never had, and what fixes it?
3. Give three upgrades attention needed to replace the RNN entirely, and the specific gap each one closes.
4. State the four-stage through-line of Part 4, and name the Part-7 development that represents the pendulum swinging back.

### Answers

1. Not attention itself, but the claim (and demonstration) that attention *alone* — recurrence deleted, self-attention stacked deep with FFN/residuals/positional encoding — outperforms recurrent models, crucially because it parallelizes over the sequence during training.
2. Permutation-equivariant = shuffle the input positions and the outputs shuffle the same way; the layer has no inherent notion of order. The RNN got order for free from processing tokens in sequence; a pure attention layer must be *told* the order, which is what positional encoding (Part 5.3) injects.
3. E.g.: **self-attention** (positions must exchange info with no recurrence to do it); **scaled dot-product `/√d_k`** (unscaled dot scores saturate softmax as dimension grows); **positional encoding** (attention is order-blind, so order must be added explicitly). (Also acceptable: multi-head for multiple simultaneous relations; residuals+norm for trainable depth; FFN for per-token nonlinearity.)
4. RNN (sequential state) → seq2seq (fixed-vector bottleneck) → attention (direct weighted access / learned alignment) → Transformer (attention-only, parallel, stackable). The pendulum swings back with state-space models / **Mamba** (Part 7.3), which reintroduce a recurrent bounded state to reclaim cheap `O(1)`-memory inference while keeping parallel training.

## Exercise

Write the one-sentence "reason for existing" for each Transformer component by pointing at the Part-4 failure it fixes: (a) self-attention, (b) positional encoding, (c) multi-head, (d) the residual connections around each sublayer, (e) the fact that training parallelizes over the sequence. Then, in two sentences: given all these fixes, what did the Transformer *give up* relative to an RNN, and why is that the thing Part 7.3 is trying to win back?
