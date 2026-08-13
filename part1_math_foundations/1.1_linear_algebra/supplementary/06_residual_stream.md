# Supplementary: The residual stream — intuition and why it's called that

Companion to `../06_projections_orthogonality.md`'s "Residual stream as a vector space" section.

The primary file uses the residual stream as a workspace metaphor for understanding how Transformers move information. This sidecar gives the deeper intuition: where the name comes from, why the architecture works, and why this view is fundamentally different from how older deep-learning thinking treated layers.

Once you internalize this picture, a lot of seemingly disparate phenomena (residual connections, superposition, steering vectors, circuits analysis, why pre-norm helps) become natural consequences of one core idea.

## The mental picture: a shared whiteboard

Imagine the residual stream as a `d_model`-dim **whiteboard** that flows through the entire network. The token embedding writes the initial state on the whiteboard. Then every component (every attention head, every MLP) does the same thing:

1. **Read** what's currently on the whiteboard.
2. **Compute** something based on what it read.
3. **Add** its conclusions to the whiteboard.

Crucially, no one ever erases — each layer only *adds*. By the time you reach the final layer, the whiteboard has been edited many times, but its contents are still the cumulative sum of:

- the original token embedding
- + every attention head's contribution from every layer
- + every MLP's contribution from every layer

That cumulative-sum object is the residual stream.

## Why it's called "residual"

The name comes from **ResNets** (He et al., 2015). Before ResNets, a typical layer was:

```
h_{l+1} = f(h_l)     (output replaces input — a fresh representation)
```

ResNets changed this to:

```
h_{l+1} = h_l + f(h_l)
```

Now `f(h_l)` is the **residual** — the *difference* between the old and new representations. Instead of asking the layer "what should the next representation be?", you ask it "what should we *add to* what's already there?" That delta is the residual; learning the delta is empirically easier than learning the full new representation.

The original motivation was to enable training much deeper networks. The `+ h_l` path gives gradients an **identity bypass** — they can flow backward without being multiplied by the layer's Jacobian. So vanishing/exploding gradients across depth become much milder. Without the residual, you can't really train a 50-layer network; with it, you can train a 1000-layer one.

Transformers inherited this idea: every Transformer block is `h + f(h)` where `f` is attention or MLP. The accumulating sum of those `f`'s, threaded through the whole network, is the **residual stream**.

## Why "stream"

The "stream" metaphor came from Anthropic's interpretability work (Elhage et al. 2021, "A Mathematical Framework for Transformer Circuits"). The picture:

- A **river** of activations flowing forward through the network.
- Each layer is a **tributary** that adds its contribution.
- Information flows **downstream** through the model.
- You can **probe** at any point in the stream to see what features are present at that depth.

The metaphor emphasizes that the stream isn't just "whatever the last layer happened to output." It's a single coherent, evolving object that runs through the entire network end-to-end. Layer-5's output isn't categorically different from layer-50's output — they're both points on the same stream.

## What makes the residual-stream view powerful

In classical deep-learning intuition, layer outputs are *categorically* different objects: input is raw, layer 1 is "low features," layer 5 is "mid features," layer 50 is "predictions." Each layer transforms the data into a different *kind* of thing.

The residual-stream view rejects this. It says: **the same kind of object — a `d_model`-dim vector — flows through every layer.** What *changes* is the information stored in it, not its geometric type. This single uniform space becomes a workspace that all the model's components share.

That uniformity is what enables several powerful things:

### 1. All-to-all communication

Any later layer can read what any earlier layer wrote — they share the same address space. Attention at layer 50 can pull information from layer 5's MLP because both wrote/read in the same residual stream coordinates. Without residual connections, layer 50 would only see the heavily transformed output of layer 49, with all earlier information mediated through the chain.

### 2. Superposition

Because the stream is high-dimensional (`d_model` ≈ thousands), many features can coexist in it via near-orthogonal directions (see the orthogonality section in the primary file). The shared workspace has room for everyone — exponentially so. This is what lets a single residual stream encode many semantic, syntactic, and topical features simultaneously without crosstalk.

### 3. Skip-ability

If a layer's `f(h)` is useless, it can output ~zero and the stream just passes through. A bad layer doesn't ruin the network — the additive structure makes useless contributions harmless. The model can effectively "skip" layers that don't help by learning their `f` to be near-zero.

This is why deep Transformers can be trained from scratch and why pruned/distilled models work: many layers are doing relatively small additive contributions that can be removed or compressed without total destruction.

### 4. Tractable interpretability

You can ask "what direction in the stream encodes feature X?" because the stream is one persistent vector space across the whole model. The same direction that means "topic = cooking" at layer 5 can still mean "topic = cooking" at layer 50. Components add to that direction, read from it, transform it — but the direction itself has continuity across depth.

This is what lets **probes** work (find a direction that linearly correlates with a feature), **steering vectors** work (add a vector along a target direction to nudge generation), and **circuits analysis** work (track which heads write to / read from which directions).

## Three useful framings

All three say the same thing from different angles:

- **Workspace metaphor**: the stream is a shared whiteboard that all components read from and write to additively.
- **Message-bus metaphor**: components don't talk to each other directly; they talk *through* the stream by writing/reading at chosen directions.
- **Tally metaphor**: the stream is a running sum of all contributions, starting from the token embedding and accumulating to the final pre-unembed state.

The "workspace" view is most useful for thinking about parallelism (many components in the same layer all add concurrently to the same stream). The "message bus" view is most useful for circuits analysis (drawing arrows from writers to readers). The "tally" view is most useful for understanding gradient flow (the cumulative sum has a direct identity path through the whole network).

## Why this is so different from older intuitions

If you came up through CNNs or pre-Transformer NLP, you probably learned to think of each layer as a **transformation**: input goes in, transformed output comes out, the transformed output is *qualitatively different*. Each layer does its own job and produces a new kind of thing.

The residual stream view is fundamentally different: there's **one big shared vector**, and each layer adds its two cents to it. That's a different mental model — and it's much more accurate for modern Transformers.

Once you internalize this view, a lot of seemingly disparate phenomena become natural:

- **Why pre-norm Transformers train better than post-norm**: pre-norm preserves the stream's magnitude across depth, so the residual path stays clean.
- **Why attention can route long-range information**: because the stream itself carries everything from upstream, downstream attention can pull from any earlier layer's output.
- **Why steering vectors work**: you can directly inject a delta into the stream, and the rest of the network treats it as if some upstream component had written it.
- **Why superposition is plausible**: many features fit because the stream is high-dim and components communicate in near-orthogonal subspaces.
- **Why interpretability is tractable**: features are *directions* in a single persistent space, not "stuff that happens to come out of layer X."

## The one-sentence framing

> **A Transformer is `n` components reading from and writing to one big shared vector, additively.** That shared vector — accumulating residual updates as it flows through the network — is the residual stream.

If you only carry one sentence about Transformer architecture, this is the one.

## Self-check

1. In a 12-layer Transformer with `d_model = 768`, what's the dimensionality of the residual stream after layer 6? After layer 12?
2. What would happen if you replaced `h_{l+1} = h_l + f(h_l)` with `h_{l+1} = f(h_l)` (no residual connection)? Why is the gradient flow worse?
3. The token embedding is at the *start* of the residual stream — but the unembed (`lm_head`) is at the *end*, often sharing weights with the embedding via tying. Why does this make sense in the residual-stream view?
4. If you add a "steering vector" `s` to the stream at layer 5, predict roughly what happens for layers 5 through 12.

### Answers

1. `d_model = 768` at *every* layer. The residual stream's dimensionality is constant — that's the whole point. It's the same shared space, just with different information accumulated over time.

2. Without residual connections, `∂h_{l+1}/∂h_l = ∂f/∂h_l` — purely the layer's Jacobian. Across `L` layers, gradients get multiplied by a product of `L` Jacobians, which can vanish (norms < 1) or explode (norms > 1) exponentially. With residual connections, `∂h_{l+1}/∂h_l = I + ∂f/∂h_l`, so there's an identity term that keeps the gradient from disappearing. The product across layers includes a "1" path, preventing vanishing.

3. Both embed and unembed are projections between the same vector space (residual stream) and the vocabulary. The embed projects from vocab→stream (one-hot to vector); the unembed projects from stream→vocab (vector to logits). Tying their weights ties the "input direction for token *t*" to the "output direction for predicting token *t*" — a natural symmetry that saves parameters and aligns input/output token representations.

4. Layers 5+ will read from the modified stream, so any component whose read direction overlaps with `s` will compute differently. The effect propagates: layer 6 might write something different in response, which layer 7 reads, and so on. By layer 12, the unembed sees a stream that includes `s`'s contribution and produces logits accordingly. This is *exactly* how steering vectors work in practice — inject a delta, observe downstream behavior shift toward the target encoded by the direction of `s`.
