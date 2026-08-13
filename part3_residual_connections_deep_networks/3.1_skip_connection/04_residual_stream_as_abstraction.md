# The Residual Stream as the Central Abstraction

The previous three files built the skip connection up from a problem (degradation), a mechanism (the gradient highway), and a structure (ensemble of paths). This file states the payoff that the rest of the curriculum leans on: the sequence of residual adds isn't just a training trick — it's a **shared communication channel** that every component reads from and writes to, and this single reframing organizes how you think about attention, interpretability, pruning, and steering.

There's a companion deep-dive with the full etymology, metaphors, and interpretability tie-ins: [residual stream supplementary](../../part1_math_foundations/1.1_linear_algebra/supplementary/06_residual_stream.md). This file is the compact, architecture-first version — read it first, go there for depth.

**Convention:** row-vector, per-token. The stream at layer `l` is `h_l ∈ R^D` for one position (or `(B, S, D)` batched). `D` is fixed across all layers — that fixedness is the whole point.

## The reframe: one bus, not a pipeline

Old mental model (CNN / pre-Transformer): a network is a **pipeline**. Layer 1 makes low features, layer 5 mid features, layer 50 predictions. Each layer's output is a *different kind of object* than its input.

Residual model: a network is `L` components all reading from and writing to **one persistent `D`-dimensional vector**. The token embedding writes the initial value; every attention head and every MLP does the same three-step move:

```
1. READ    a (normalized) copy of the current stream
2. COMPUTE something from it
3. ADD     the result back into the stream        (never overwrite)
```

Because step 3 is always `+=` and never `=`, the stream at any depth is a **running sum**:

```
h_L = embed(token) + Σ_l attn_l(·) + Σ_l ffn_l(·)
```

Every term is a `D`-vector living in the *same* space. Layer 50's output is not a categorically different thing from layer 5's output — they're both edits to the same shared vector. That uniform, shared, additive vector is the **residual stream**.

## Why the "never overwrite" matters: all-to-all communication

In a plain (overwriting) net, layer 50 sees only layer 49's output — everything earlier is accessible only as heavily-retransformed residue. In the residual stream, layer 50 reads a vector that *still literally contains* layer 5's additive contribution. So:

- Any later component can pick up what any earlier component wrote, as long as it reads the right **direction** in the `D`-dim space.
- Components communicate **indirectly, through the stream**, not by direct wiring. Think of it as a shared message bus: a head "sends a message" by writing a vector in some direction; a later head "receives" it by having a read (query/key/value) projection that's sensitive to that direction.

This is the architectural precondition for attention doing long-range routing (Part 5) and for **circuits** (Part 11.2): a circuit is a set of components that write-then-read compatible directions in the stream.

## Bandwidth, superposition, and the width knob

The stream is a *finite* resource: `D` real numbers per token, shared by all `2L` sublayers across all of depth. That framing pays off repeatedly:

- **Superposition.** Because `D` is large (thousands), far more than `D` features can coexist as *near-orthogonal directions* — [1.1/06 projections](../../part1_math_foundations/1.1_linear_algebra/06_projections_orthogonality.md) and [04 low-rank](../../part1_math_foundations/1.1_linear_algebra/04_outer_products_low_rank.md) give the geometry. Two features stored in near-orthogonal directions barely interfere, so the stream holds many features at once without a dedicated dimension each.
- **Width `D` is "bus bandwidth."** This is a cleaner reason width matters than "more parameters": a wider stream has more room for simultaneous features and less read/write crosstalk. It also reframes why `D_h = 128` per head stays fixed while you *add heads* with width ([ARCHITECTURE.md](../../ARCHITECTURE.md)) — you're widening the bus, not the individual reader:
  - Each attention head is a fixed-width **reader/writer** on the bus: it reads through `W_Q/W_K/W_V` (projecting the `D`-dim stream down to a `D_h`-dim view) and writes back through `W_O`. Its own working width is `D_h`; the bus it plugs into is `D`.
  - Scaling grows `D` but pins `D_h = 128`, so `H = D/D_h` grows instead (32 → 64 → 128 across 8B → 70B → 405B). A bigger model doesn't have *fatter* heads — it has *more* of them, each the same size, on a wider bus. (They always exactly tile it: `H·D_h = D`.)
  - Why keep `D_h` fixed rather than widen each head? Two reasons, one per side of the metaphor. **The reader has a sweet spot:** `D_h = 128` is where the `√D_h` logit scaling stays well-behaved (a much larger `D_h` raises `q·k` variance and feeds the attention-logit-growth instability) and where attention kernels are tuned. **The bus is what capacity wants:** by superposition, a wider `D` offers more near-orthogonal directions, so more features coexist with less interference — real added capacity, plus more independent read/write ports (heads) operating in parallel. So you buy capacity by widening the shared channel and adding readers, not by fattening each reader.
- **Skip-ability, restated.** A component that writes ≈0 costs the stream nothing; the additive structure makes useless contributions harmless. This is the [ensemble/pruning](03_ensemble_of_shallow_paths.md) story from the stream's side.

## Two consequences you'll meet again

**Interpretability — the logit lens (forward → Part 11).** Because the stream lives in one space across all depth, and the unembedding `W_U` maps that space to vocabulary logits, you can apply `W_U` to the stream *at an intermediate layer* and read off "what would the model predict if it stopped here?" This **logit lens** works *only* because the residual stream is a single persistent space that the output head can interpret at any depth — in a pipeline net, an intermediate activation is the wrong kind of object to unembed. Probes, steering vectors, and activation patching (Part 11.2) all rest on the same property: features are **directions in one shared space** with continuity across depth.

**Norm placement (→ 3.2).** If the stream is the thing everything reads and writes, you do *not* want to normalize it in place — that would erase the accumulated contributions and break the additive-sum semantics. Pre-norm normalizes only the *copy* a sublayer reads, leaving the stream itself untouched (the clean highway of [02](02_gradient_highway.md)). Post-norm normalizes the stream after each add, spoiling exactly this property. That's the residual-stream reason pre-norm won, developed in [3.2/01](../3.2_normalization_and_depth/01_normalization_placement_recap.md).

## The one-sentence version

> A Transformer is `2L` components reading from and writing to one big shared `D`-dimensional vector, additively — and that accumulating vector is the residual stream.

If you retain one sentence about Transformer architecture, this is it (it's also the closing line of the supplementary — deliberately repeated because it's the load-bearing idea).

## Why this matters for modern NLP/LLM work

- It's the vocabulary the entire modern literature speaks in — "writes to the residual stream," "reads a direction," "the stream at layer `l`" are standard phrases you'll now parse instantly.
- It makes attention (Part 5) legible before you see it: Q/K/V are just *read* projections off the stream, and `W_O` is the *write* back into it.
- It's the substrate for interpretability (Part 11), steering, and model editing — all of which are "manipulate a direction in the stream."
- It reframes width vs. depth ([3.2/02](../3.2_normalization_and_depth/02_scaling_the_residual_stream.md)) as "bus bandwidth vs. number of sequential edits," which is a more useful intuition than parameter counting.

## Self-check

1. In a 32-layer, `D = 4096` model, what is the dimensionality of the residual stream at layer 1? At layer 32? Why is the answer the same, and why is that the crux of the abstraction?
2. State the read → compute → add cycle, and say which step's choice of "overwrite vs. add" enables all-to-all communication. What breaks if you overwrite?
3. The logit lens applies the unembedding `W_U` to an *intermediate*-layer stream. Why does this produce something meaningful in a residual Transformer but not in a plain pipeline network?
4. "Width is bus bandwidth." Use superposition to explain why a wider stream helps beyond just "more parameters."

### Answers

1. `D = 4096` at every layer, including 1 and 32 — the stream's dimensionality is constant by construction. That constancy *is* the abstraction: every component reads and writes the *same* `R^4096` space, so the same direction can carry the same meaning at any depth, and components at different layers can communicate. A pipeline whose layer widths/meanings change can't offer this shared address space.
2. Read a (normalized) copy of the stream → compute → **add** the result back. The **add** (not overwrite) is what enables all-to-all communication: because contributions accumulate rather than replace, layer 50 still sees layer 5's writing. If you overwrote (`h_{l+1} = f(h_l)`), each layer would see only the previous layer's output, earlier information would survive only as retransformed residue, and the gradient highway ([02](02_gradient_highway.md)) would vanish too.
3. In a residual net the stream is one persistent space shared with the output head, so `W_U` (which maps that space → logits) is meaningful applied at *any* depth — you're asking "what does the stream currently point toward in vocab space?" In a pipeline net an intermediate activation is a different kind of object (mid-level features), not a point in the space `W_U` was trained to read, so unembedding it is a type error that yields noise.
4. Superposition says features are stored as near-orthogonal directions, and a `D`-dim space has room for many more than `D` of them with low interference. A wider stream (bigger `D`) has *more* near-orthogonal directions available, so more features coexist with less crosstalk and more read/write capacity per token — capacity of the shared channel, not just raw parameter count. That's why width improves models in a way the "bus bandwidth" metaphor captures better than "more weights."

## Exercise

Conceptual, no code. (a) Write the residual stream at the output as an explicit sum of the embedding plus each block's attention and FFN contributions for a 3-layer model. (b) Suppose one attention head at layer 2 writes a vector `v` in direction `d` (a "topic = cooking" feature). Describe, in terms of reads and directions, how a layer-3 head could *use* that information, and how the final logits could reflect it via `W_U`. (c) Now describe what a "cooking" steering vector would do if you added `α·d` to the stream at layer 1, and why the mechanism is the same as in (b).
