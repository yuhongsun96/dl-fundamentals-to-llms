# Scaling the Residual Stream, and Depth vs. Width

Two loose ends from [3.1](../3.1_skip_connection/02_gradient_highway.md) and [3.2/01](01_normalization_placement_recap.md), pulled into one file because they're the same question from two sides: *how deep can you actually go, and what has to be tuned to get there?* First the mechanical problem pre-norm creates — the residual stream's magnitude grows with depth — and its fixes. Then the strategic question — given that residuals make 1000 layers *trainable*, why do frontier LLMs stop at ~tens-to-~100?

**Convention:** per-token stream `h_l ∈ R^D`; `f_l` a sublayer whose output is added to the stream. `Var(·)` is per-coordinate variance at init.

## The variance-grows-with-depth problem

In pre-norm, each block *adds* to the stream and never rescales it ([3.2/01](01_normalization_placement_recap.md)). Treat successive contributions as roughly uncorrelated at init. Then variance adds:

```
h_{l+1} = h_l + f_l(Norm(h_l))
Var(h_{l+1}) ≈ Var(h_l) + Var(f_l)
```

If each sublayer output has variance ≈ `σ²` (its input was normed to unit scale, so this is natural), then after `L` blocks:

```
Var(h_L) ≈ Var(h_0) + L·σ²      →   stream magnitude grows like √L
```

For `L = 100` that's a ~10× larger stream at the end than at the start. Consequences:

- The **final logits** `h_L W_U` inherit that scale — without correction they'd grow with depth, saturating softmax and destabilizing the loss. (This is *the* reason pre-norm needs a **final norm**, [3.2/01](01_normalization_placement_recap.md).)
- Each sublayer's *relative* contribution shrinks: a fixed-variance `f_l` added to an ever-larger `h_l` moves the stream less and less as `l` grows. Deep layers have diminishing leverage on the stream unless something rescales.
- At init, before the model has learned anything, this pure-accumulation growth can already push activations out of the range mixed-precision ([2.4/05](../../part2_neural_network_fundamentals/2.4_optimization/05_mixed_precision.md)) handles well.

Residuals converted the *multiplicative* vanishing/exploding problem into this *additive* growth problem ([3.1/02](../3.1_skip_connection/02_gradient_highway.md) self-check 3) — much milder, but it still has to be managed. Here's the toolkit.

## Fix 1 — down-scale the writes at init (`1/√(2L)`)

If the problem is `Var(h_L) ≈ L·σ²`, make each write smaller so the *sum* stays O(1). Scale the **output projections that write into the stream** (`W_O` in attention, `W_down` in the FFN) by `1/√(2L)` at init:

```
each block adds variance σ²/(2L)  →  2L sublayers × σ²/(2L) = σ²    # constant in depth
```

The `2` is because there are `2L` writes (attention + FFN per block). This is a **pure init trick** — no architecture change, no extra parameters, no runtime cost — and it's what [2.3/02](../../part2_neural_network_fundamentals/2.3_init_normalization/02_xavier_kaiming_modern.md) and [ARCHITECTURE.md](../../ARCHITECTURE.md) (components 1, 8) refer to. GPT-2 introduced it; most modern models use some version. It keeps the stream O(1) *at step 0*; the final norm + training then handle the rest.

## Fix 2 — learn the per-layer contribution scale (LayerScale, ReZero)

Instead of a fixed init constant, make the write-scale a **learnable scalar** `α_l` per sublayer, initialized tiny so the network *starts* near the identity and learns how much each layer should contribute:

```
h_{l+1} = h_l + α_l · f_l(Norm(h_l))
```

Two named variants:

| Method | `α_l` form | Init | Notes |
|---|---|---|---|
| **ReZero** (Bachlechner 2020) | one scalar per sublayer | `α = 0` | Starts as *exact* identity — every block is a no-op at step 0, gradient flows perfectly. Layers "turn themselves on" as needed. |
| **LayerScale** (CaiT, Touvron 2021) | a *per-channel* diagonal `diag(λ) ∈ R^D` | `λ ≈ 1e-4`–`1e-6` | Per-dimension gate on the write; from deep vision Transformers, used in some deep LLMs/ViTs. |

Notice this is the **Highway/LSTM gate from [3.1/01](../3.1_skip_connection/01_degradation_problem.md), resurrected** — a learned coefficient on the transform branch — except initialized near 0 so identity is the *starting* behavior, not just a *reachable* one. ReZero with `α=0` is the purest statement of "let the network open each layer's valve only when it helps," which dovetails exactly with the [ensemble/effective-depth](../3.1_skip_connection/03_ensemble_of_shallow_paths.md) picture: most layers start as skips and only some become active.

**When you see which:** `1/√(2L)` init is the default in mainstream LLMs (simplest, zero cost). LayerScale/ReZero show up in *very* deep or unstable regimes (deep ViTs, some 100+-layer experiments) where a learnable, self-gating scale buys extra stability. They're the same idea at different points on the "fixed vs. learned" dial.

## The depth-vs-width question: why not 1000 layers?

Residuals + pre-norm + these scaling fixes mean depth is no longer *limited by trainability* — stochastic depth trained 1000-layer ResNets ([3.1/03](../3.1_skip_connection/03_ensemble_of_shallow_paths.md)), DeepNet trained 1000-layer Transformers. Yet look at real LLMs ([ARCHITECTURE.md](../../ARCHITECTURE.md)):

| Model class | `D` (width) | `L` (depth) | Ratio `D/L` |
|---|---|---|---|
| 8B | 4096 | 32 | 128 |
| 70B | 8192 | 80 | 102 |
| 405B | 16384 | 126 | 130 |

Depth grows *sublinearly* with scale; width grows faster; the ratio stays ~100–130. Nobody ships a 1000-layer LLM. Four reasons:

1. **Diminishing returns of depth (the ensemble argument).** From [3.1/03](../3.1_skip_connection/03_ensemble_of_shallow_paths.md): a residual net's *effective* depth is far below its nominal depth, because most gradient-carrying paths are short. Piling on layers mostly adds long paths that contribute little. Past a point, extra depth buys less per parameter than extra width. Empirically, for a fixed parameter budget there's a broad optimum: too shallow underfits, too deep wastes parameters on barely-active layers.

2. **Depth is sequential; width is parallel.** A forward pass must traverse `L` blocks *in order* — layer `l+1` needs layer `l`'s output. Depth is a **latency floor** you cannot parallelize away, on training or inference. Width, by contrast, is matmul size — trivially parallel across a GPU's cores and across devices (tensor parallelism, Part 12). Doubling width is cheap on modern hardware; doubling depth doubles the critical-path length. For a latency-sensitive product (a chatbot decoding token by token), depth is the expensive axis.

3. **Wider streams have more room (bandwidth).** The [residual-stream-as-bus](../3.1_skip_connection/04_residual_stream_as_abstraction.md) view: width `D` is how many features can coexist in superposition per token. Many capabilities want *more simultaneous features*, not *more sequential refinement steps* — so width often pays off more directly than depth. This is also why `D_h = 128` per head is held fixed and you *add heads with width* ([ARCHITECTURE.md](../../ARCHITECTURE.md)).

4. **Deep stacks stay finicky at scale.** Even trainable, very deep models are more prone to loss spikes and need more careful init/norm ([2.2/05](../../part2_neural_network_fundamentals/2.2_backpropagation/05_gradient_pathologies.md), [3.2/01](01_normalization_placement_recap.md)). The operational cost of `L=500` isn't worth a marginal quality gain when widening is safer.

The synthesis: **residuals removed depth as a *hard* limit, so the depth chosen is now an *economic* optimum** — set by diminishing returns, hardware parallelism, and stream-bandwidth needs, not by "can we train it." The scaling-laws literature (Kaplan, Chinchilla — Part 6.3) treats the aspect ratio `D/L` as a hyperparameter with a broad, shallow optimum around 100:1, which is exactly what the table shows.

## Why this matters for modern NLP/LLM work

- `1/√(2L)` init and the final norm are non-negotiable parts of any from-scratch training recipe; getting them wrong is a classic "why won't my deep model train / why do logits explode" bug.
- LayerScale/ReZero are the tools you reach for when a deep or multimodal stack is unstable and init tricks aren't enough — and recognizing them as "learned Highway gates" tells you what they're doing.
- The depth-vs-width intuition tells you *how to spend* a parameter budget and *why* inference latency scales with depth — directly relevant to serving (Part 9) and systems (Part 12).
- "Effective depth ≪ nominal depth" underpins depth-pruning, early-exit, and mixture-of-depths — all live compression techniques.

## Self-check

1. In a pre-norm model, why does the residual stream's variance grow with depth, and roughly how fast? What single architectural element is the direct consequence at the output?
2. The `1/√(2L)` output-projection scaling keeps the stream O(1). Where does the `2` come from, and is this an init trick or a permanent architectural change?
3. ReZero initializes its per-layer scalar `α = 0`. What does the network compute at step 0, and how does this relate to (a) the gradient highway and (b) the Highway-gate idea from [3.1/01](../3.1_skip_connection/01_degradation_problem.md)?
4. Residuals make 1000-layer models trainable, yet a 405B LLM uses only 126 layers. Give the four reasons depth is chosen far below what's trainable.
5. Why is depth a "latency floor" in a way width is not?

### Answers

1. Each block adds a roughly-uncorrelated contribution of variance ≈ `σ²` to a stream that's never rescaled in place, so variances sum: `Var(h_L) ≈ Var(h_0) + L·σ²`, i.e. magnitude grows like `√L`. The direct output consequence is the need for a **final norm** before the unembedding, to condition the accumulated stream before it becomes logits.
2. The `2` counts the `2L` writes into the stream (one from attention's `W_O`, one from the FFN's `W_down`, per block): scaling each write's variance by `1/(2L)` makes the total `2L · σ²/(2L) = σ²`, constant in depth. It's a pure **init trick** — a scaling applied to `W_O, W_down` at initialization, no architecture change, no runtime cost, no extra parameters.
3. At step 0 every block computes `h_{l+1} = h_l + 0·f_l(...) = h_l` — the network is the *exact identity* from input to output, so the gradient highway is perfectly clean (`∂h_{l+1}/∂h_l = I` exactly) and every layer gets undistorted gradient. (a) It's the strongest possible version of the identity-path guarantee. (b) It's the Highway/LSTM carry gate reborn as a learnable scale on the transform branch, but initialized *closed* (α=0, identity is the start state) rather than merely reachable — the network opens each layer's valve only as training finds it useful.
4. (i) Diminishing returns — effective depth ≪ nominal depth, so extra layers add mostly low-contribution long paths; (ii) depth is sequential (a non-parallelizable latency floor) while width is parallel matmul; (iii) wider streams give more superposition bandwidth, which many capabilities want more than more sequential steps; (iv) very deep stacks remain finicky/spike-prone at scale, so the operational cost isn't worth the marginal gain.
5. A forward pass must go through the `L` blocks *in order* — block `l+1` depends on block `l`'s output — so the critical path length is `L` and can't be parallelized; latency scales with depth. Width is the *size* of each matmul, which parallelizes across GPU cores and across devices (tensor parallelism), so adding width adds work that runs concurrently, not sequential steps.

## Exercise

(a) A pre-norm model has `L = 50`, and at init each sublayer output has per-coordinate variance `σ² = 1`. Estimate `Var(h_50)` (i) with no output scaling and (ii) with `1/√(2L)` scaling on the two write projections per block. (b) The stream feeds `W_U`; explain qualitatively what happens to the logit scale (and to softmax) in case (i) without a final norm. (c) You're given a fixed 7B parameter budget and can spend it as `(D=4096, L=32)` or `(D=2048, L=128)`. Using the four depth-vs-width reasons, argue which is likely better for a low-latency chat model and what the deep-and-narrow option might buy in return.
