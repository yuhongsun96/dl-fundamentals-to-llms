# Assembling the Decoder Block

You now have both primitives: self-attention (5.1) and the position-wise FFN (file `02`, and its activation guts in [2.1/02](../../part2_neural_network_fundamentals/2.1_mlp_building_block/02_activations.md)). This file is about the *arrangement* — how those two sublayers, wrapped in norm and residual, compose into the single unit that gets stacked `L` times to make the whole model. The arrangement is not arbitrary; it encodes a clean division of labor that, once you see it, makes the entire architecture read as one idea repeated.

**Convention:** row-vector (`Y = X W`), repo default — activations are rows, `(B, S, D) @ (D, d_out) → (B, S, d_out)`. Numbers anchor to the [ARCHITECTURE.md](../../ARCHITECTURE.md) 8B config: `D = 4096`, `d_ff = 14336`, `L = 32`, `H = 32`, `D_h = 128`.

## The two-sublayer structure

A decoder block is exactly two sublayers, each wrapped identically:

```
   x_in (B,S,D) ─────────────────────────┐  residual stream (never normalized)
        │                                 │
   ┌────▼─────┐                           │
   │ RMSNorm  │ γ₁                        │
   └────┬─────┘                           │
   ┌────▼──────────────┐                  │
   │  Self-Attention   │  ← token-mixing  │
   └────┬──────────────┘                  │
       (+)◄───────────────────────────────┘   x_mid = x_in + Attn(Norm(x_in))
        │
        ├─────────────────────────────────┐  residual stream
   ┌────▼─────┐                            │
   │ RMSNorm  │ γ₂                         │
   └────┬─────┘                            │
   ┌────▼──────────────┐                   │
   │  FFN (SwiGLU)     │  ← channel-mixing │
   └────┬──────────────┘                   │
       (+)◄────────────────────────────────┘   x_out = x_mid + FFN(Norm(x_mid))
        │
        ▼  to next block
```

Read top to bottom, that's: `(norm → attention → add)` then `(norm → FFN → add)`. Two sublayers, same wrapper on each. The full annotated version — with GQA, RoPE, QK-norm, and shapes — lives in [ARCHITECTURE.md](../../ARCHITECTURE.md); this stripped-down view is the one to hold in your head.

The wrapper is the **pre-norm residual** form from Part 3:

```
x ← x + Sublayer(Norm(x))
```

The stream `x` flows down the right edge untouched; each sublayer reads a *normalized copy*, computes, and adds its result back. This is not the post-norm `Norm(x + Sublayer(x))` you remember from the 2017 paper and BERT. Why pre-norm won — the `I + ∂f/∂h` gradient highway, why it trains deep stacks without learning-rate warmup babysitting — is derived in [2.3/04](../../part2_neural_network_fundamentals/2.3_init_normalization/04_pre_vs_post_norm.md) and is the whole story of [Part 3](../../part3_residual_connections_deep_networks/README.md). Do not re-derive it here; just know that the norm sits *inside* the residual branch, and the stream itself is never normalized in place.

## The central decomposition: mix across tokens, then mix across channels

Here is the one idea. Look at what each sublayer can and cannot touch:

| | Self-attention | FFN |
|---|---|---|
| Mixes information **across positions**? | **Yes** — each token's output is a weighted sum over *other* tokens | No — position `t` in, position `t` out, never looks sideways |
| Mixes information **across channels** (the `D` dim)? | Only linearly (via `W_O`); the nonlinearity is elementwise softmax over positions | **Yes** — a full nonlinear MLP applied within each token's `D`-vector |
| Weights are **input-dependent**? | Yes — attention weights `α` are computed from the data | No — the same `W_gate/W_up/W_down` for every token |
| Nickname | **token-mixing** / communication | **channel-mixing** / computation |

The block does the two in sequence: **every token first gathers context from the sequence (attention), then each token independently processes what it gathered (FFN).** Communication, then computation. Gather, then think. That alternation, repeated `L` times, *is* the Transformer.

### Why you need both

Neither sublayer can do the other's job, which is exactly why the block has two:

- **Attention alone can't do per-position nonlinear compute.** Strip out the FFN and attention is, per output position, a *convex combination* of value vectors — `A V` is a weighted average, and `W_O` is one more linear map. Stacking linear-in-values layers with data-dependent averaging weights is expressive about *routing* but weak about *transforming*: there is no per-position nonlinearity turning a gathered representation into something new. The FFN is where a token, having collected "the subject is plural, the verb is two tokens back," runs that through a nonlinear function to produce "so emit an agreement feature." (The key-value-memory view of exactly this computation is file `02`.)

- **FFN alone can't move information between positions.** The FFN is applied identically and independently to each of the `S` positions — position `t` never sees position `t-1`. A stack of pure FFNs is `S` parallel MLPs that never talk; the model could not resolve a pronoun, match a bracket, or copy a name, because those all require reading *another* position. Attention is the *only* component in the block that moves information horizontally across the sequence.

So the two are complementary by construction: attention is the sole cross-token operator, the FFN is the sole per-token nonlinear operator, and a working language model needs both. Everything else in the block (norm, residual) is plumbing that makes stacking them `L` deep trainable.

## Where position enters

Notice the block diagram adds *no* positional vector to the stream. Attention as defined in 5.1 is permutation-equivariant — shuffle the tokens and the outputs shuffle identically — so position has to be injected somewhere or the model is a bag of words. In the modern design it enters **inside attention**, as a rotation of the query and key vectors right before their dot product (RoPE, [5.3/04](../5.3_positional_information/04_rope.md)). It does not live in the residual stream and it does not touch the FFN. This is why the block-level picture can stay position-agnostic: the FFN genuinely does not care where a token is, and attention gets its position sense from RoPE at the last moment before scoring.

## Identical, stacked, weight-untied

The block is structurally **identical** at every depth: same two sublayers, same norm placement, same shapes. What differs across the `L = 32` copies is only the *weights* — each block has its own `W_Q, W_K, W_V, W_O, W_gate, W_up, W_down` and its two norm gains `γ₁, γ₂`. "Structurally uniform, weight-untied." (Contrast Universal Transformers / ALBERT, which *tie* weights across depth to save parameters — the mainstream LLM does not, because the extra parameters buy more than they cost at scale.)

The object that persists across all `L` blocks is the **residual stream**: one `(B, S, D)` tensor, born at the token embedding, modified only by the `+=` at each sublayer's exit, read one last time by the final norm and unembedding. Each block is a read-compute-write step against that shared stream — the interpretability "residual stream as a communication channel" framing (Part 3.1, and the [residual stream supplementary](../../part1_math_foundations/1.1_linear_algebra/supplementary/06_residual_stream.md)) is just this picture taken seriously. Blocks don't pass hidden states to each other through a bottleneck; they all read from and write to the same `D`-dimensional bus.

## Self-check

1. Classify each of the following as token-mixing or channel-mixing, and say whether its weights are input-dependent: (a) the attention `A V` product, (b) the FFN up-projection `x W_up`, (c) the output projection `W_O`.
2. Suppose you deleted every FFN and kept only the attention sublayers (with their residuals and norms). Give one concrete linguistic task the resulting model would struggle with, and name the mathematical reason.
3. The block adds no positional information to the residual stream, yet the model is not permutation-invariant. Reconcile this.

### Answers

1. (a) **Token-mixing, input-dependent** — `A` is computed from the data (`softmax(QKᵀ/√D_h)`), and it forms a weighted sum *across positions*. (b) **Channel-mixing, not input-dependent** — `W_up` is a fixed learned matrix applied within each token's `D`-vector, same for every position and every input. (c) Channel-mixing in the sense that it recombines the concatenated head outputs within a position, and **not input-dependent** (`W_O` is fixed). Note (c) is the *linear* part of attention's output; the input-dependence of the attention sublayer lives entirely in `A`, not in `W_O`.
2. Anything requiring a per-position *nonlinear transform* of gathered context — e.g. producing a subject-verb agreement feature, or any computation of the form "given what I gathered, emit a genuinely new feature." Mathematically, attention's output at a position is a data-weighted convex combination of value vectors followed by the linear `W_O`; without the FFN there is no per-position nonlinearity, so the block can *route and average* information but cannot nonlinearly *transform* it. (A pure-attention stack is far weaker than it looks precisely because of this.)
3. Position is injected *inside* attention via RoPE — the query and key vectors are rotated by an angle proportional to their absolute positions before the dot product, so the score `q·k` depends on the *relative* offset between the two tokens ([5.3/04](../5.3_positional_information/04_rope.md)). Position never exists as an additive vector in the residual stream (so the FFN and the stream stay position-agnostic), but it shapes the attention weights, which is enough to break permutation-equivariance. Shuffle the tokens and the RoPE angles change, so the attention pattern changes.

## Exercise

Take the 8B config and write out, in words, the full path of a single residual-stream vector through one block: what shape it is at each arrow in the diagram, which operations *change* its content versus merely read it, and at exactly which two points the `(B, S, D)` stream gets an additive update. Then argue in two sentences why swapping the order to `FFN → attention` within a block would still be a valid Transformer (it would) but why swapping the *global* order — all `L` FFNs first, then all `L` attentions — would not produce a usable language model. (Hint: with all FFNs first, the stream at the input to the first attention has never mixed across positions, so the FFNs computed on pure per-token embeddings with no context — the "think before you gather" failure mode.)
