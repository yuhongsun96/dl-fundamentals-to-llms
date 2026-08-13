# Residual Nets as an Ensemble of Shallow Paths

This is the view Part 2 never gave you, and it explains a cluster of otherwise-weird facts: why you can *delete* a layer from a trained ResNet or Transformer and barely dent it, why you can *reorder* layers, why **stochastic depth** and **LayerDrop** train fine, and why "effective depth" is much smaller than nominal depth. All of it falls out of one algebraic observation.

## The unraveled view (Veit, Wilber, Belongie, 2016)

Take a 3-block residual network and just substitute, without simplifying:

```
h_3 = h_2 + f_3(h_2)
    = [h_1 + f_2(h_1)] + f_3(h_1 + f_2(h_1))
    = h_0 + f_1(h_0) + f_2(·) + f_3(·) + f_2∘f_3 terms + ...
```

Because each block is `h + f(h)`, expanding the nesting produces a **sum over every subset of blocks**. For `L` blocks you get `2^L` terms, one for each way of choosing "use this block's `f`" or "skip it (take the identity)":

```
output = h_0                              (skip all — the 1 path of length 0)
       + Σ_i f_i                          (use exactly one block — L paths of length 1)
       + Σ_{i<j} f_j ∘ f_i                (use two — paths of length 2)
       + ...
       + f_L ∘ ⋯ ∘ f_1                    (use all — the 1 path of length L)
```

This is the *forward* twin of the backward `Π(I + J_l)` expansion from [02](02_gradient_highway.md) — same `2^L` subsets, read as data paths instead of gradient paths.

> **"Skip" doesn't mean switched off.** Every path always contributes — the network computes the whole sum in one forward pass. "Use this block's `f` or skip it" just labels *which* `f_l` appear in each of the `2^L` additive terms (exactly like `(a+b)(c+d) = ac + ad + bc + bd`, where "`ac` skips `b, d`" only names which factor each term picked). It's a decomposition of one deterministic computation, not a runtime routing choice. If that distinction feels slippery, see [`supplementary/03_paths_are_a_decomposition_not_routing.md`](supplementary/03_paths_are_a_decomposition_not_routing.md).

**The reframing:** a residual network of depth `L` is not one deep network. It's an implicit **ensemble of `2^L` networks of every depth from 0 to `L`**, all sharing weights, all summed. A "plain" deep net has exactly one path — the full-length one. Residuals add all the shorter ones for free.

```
       plain (1 path)                 residual (many paths, most SHORT)
   in → f → f → f → f → out       in ─┬─────────────────────────┬→ out   (length 0)
        (only the long path)          ├─ f₁ ────────────────────┤        (length 1)
                                      ├──── f₂ ─────────────────┤        (length 1)
                                      ├─ f₁ → f₂ ───────────────┤        (length 2)
                                      └─ f₁ → f₂ → f₃ → f₄ ─────┘        (length L)
```

## Most paths are short — "effective depth"

How long is a typical path? Each of the `L` blocks is independently either in or out, so path lengths follow a **Binomial(`L`, ½)** distribution: mean `L/2`, standard deviation `√L/2`, sharply concentrated. For `L = 54` blocks, essentially all paths have length between 20 and 35 — almost none use all 54.

Now combine that with the gradient view. Veit et al. measured how much gradient each path length actually delivers during training and found the long paths contribute **almost nothing** — gradient magnitude decays with path length, so only the *short* paths carry real training signal. The product: the paths that are both *common* (short) and *effective* (carry gradient) are a narrow band well below `L`.

> A very deep residual network trains as if it were a much shallower one. Its **effective depth** — the depth of the paths doing the work — is roughly `O(√L)`-ish, not `L`.

This dissolves the degradation paradox from [01](01_degradation_problem.md) from a third angle: you don't need to successfully optimize a genuinely `L`-deep function. You need to optimize an ensemble of mostly-shallow functions, which SGD handles fine — and depth just adds more shallow paths to average over.

## The lesion experiments — the memorable evidence

If a residual net were one rigid deep computation, deleting or reordering a layer would corrupt everything downstream (each layer expects the specific output of the one before). Veit et al. tested this on a trained ResNet:

- **Delete one residual block at test time** (replace `h + f(h)` with just `h`): accuracy drops by a *tiny* amount. Do the same to a plain VGG-style net and it's **catastrophic** — accuracy collapses to chance.
- **Delete several blocks:** performance degrades *gracefully and smoothly* as you remove more, rather than falling off a cliff.
- **Reorder blocks** at test time: the network is surprisingly robust to permuting layers it never saw permuted.

Why? Deleting block `i` just removes every path that passed through `f_i` — but every path *not* through `f_i` still connects input to output unchanged. The ensemble loses some members and keeps functioning. Layers behave less like sequential stages of a pipeline and more like **loosely-coupled, somewhat-redundant refinements** to a shared representation ([04](04_residual_stream_as_abstraction.md) develops this as "read/compute/add to the residual stream").

This is the rigorous version of the "skip-ability" intuition in the [residual stream supplementary](../../part1_math_foundations/1.1_linear_algebra/supplementary/06_residual_stream.md): a layer whose `f ≈ 0` (or a deleted layer) is harmless because the identity path carries the stream past it.

## What it licenses: stochastic depth, LayerDrop, and pruning

Once you believe "depth = an ensemble of shallow paths, robust to dropping members," several techniques become obvious rather than surprising:

- **Stochastic depth** (Huang et al., 2016): during training, *randomly drop entire residual blocks* per minibatch — set `h_{l+1} = h_l` with some probability, skipping `f_l` and its gradient. It's dropout at the granularity of whole layers, and it's coherent *only* because of the residual structure (drop a plain layer and the forward pass breaks). Benefits: faster training (fewer active layers per step), and a regularizer that forces paths to be individually useful. It trained the first 1000+-layer ResNets.
- **LayerDrop** (Fan et al., 2019, "Reducing Transformer Depth on Demand"): stochastic depth for Transformers. Train with random block-dropping, and at *inference* you can prune to any shallower depth without retraining — because you effectively trained the shallower sub-ensembles too. This is why some LLMs ship "you can lop off the top `k` layers" as a supported knob.
- **Depth pruning / layer merging** in LLM compression: works for the same reason — later layers in big models often make small, near-redundant contributions to the stream, so removing or merging them degrades gracefully. (Studies find some middle/late Transformer layers are near-identity and prunable with minimal loss.)

The unifying statement: **you can perturb the set of active layers because no single layer is load-bearing for the whole computation.** The ensemble is redundant by construction.

## Caveats — don't over-rotate on this

- **Attention couples the paths.** The clean `2^L`-subset expansion is exact for a purely additive `h + f(h)` stack. In a Transformer, attention *mixes across positions*, so paths aren't fully independent and some layers (e.g. ones hosting **induction heads**, Part 11.2) *are* disproportionately important. Lesioning those hurts more than the average layer. The ensemble view is a strong first-order picture, not a license to delete arbitrary layers blindly.
- **"Effective depth is small" ≠ "depth is useless."** The extra depth still helps — it enlarges the ensemble and enables the *occasional* genuinely-deep path (long-range, multi-step computation). [3.2/02](../3.2_normalization_and_depth/02_scaling_the_residual_stream.md) covers the real depth-vs-width tradeoff and why frontier models still pay for tens-to-~100 layers.

## Why this matters for modern NLP/LLM work

- It's the mechanistic basis for **layer pruning and depth-adaptive inference** in deployed LLMs — the "why" behind "you can drop the last few layers."
- It reframes interpretability: circuits (Part 11.2) are *paths through the ensemble*, and asking "which heads on which path" only makes sense because the model decomposes into paths.
- It explains empirical robustness you'll otherwise find baffling — early-exit models, layer-skipping/MoD (mixture-of-depths) routing, and why distillation into fewer layers works.
- It's the deepest reason "just add more layers" is safe in a residual architecture and dangerous in a plain one.

## Self-check

1. A residual net has `L` blocks. How many distinct input→output paths are there, and what's the distribution of their lengths? Where does the average path length sit relative to `L`?
2. Deleting one block from a trained ResNet barely hurts; deleting one layer from a plain VGG net is catastrophic. Explain both in terms of paths.
3. "Effective depth" is much smaller than `L`. Two facts combine to produce this — one about how *common* each path length is, one about how much *gradient* each path length carries. State both.
4. Why does stochastic depth (randomly dropping whole layers during training) work in a residual network but not in a plain one?
5. Give one reason the exact `2^L` ensemble picture is only approximate for a Transformer.

### Answers

1. `2^L` paths (each block independently in or out). Lengths follow Binomial(`L`, ½): mean `L/2`, std `√L/2`, tightly concentrated. So the *average* path is only half as deep as the network, and paths near the full length `L` are exponentially rare.
2. Deleting block `i` removes exactly the paths that route through `f_i` and leaves all paths that skip it intact — including the identity path — so input still connects to output and the ensemble degrades slightly. In a plain net there is only *one* path and every layer is on it; removing a layer severs the sole connection (and the next layer receives an input distribution it never trained on), so the computation collapses.
3. (a) Path lengths are Binomially concentrated around `L/2`, so most paths are moderate-length, and paths of length near `L` are vanishingly rare. (b) Gradient magnitude *decays with path length* — long paths deliver almost no training signal. The paths that are both common and gradient-carrying are the short-to-moderate ones, so the *effective* depth (paths actually doing the learning) is well below `L`.
4. Dropping a residual block means `h_{l+1} = h_l` — the identity path carries the activation through, so the forward pass is still well-defined and the rest of the ensemble is intact; you're just training a random shallower sub-ensemble each step. Dropping a plain layer breaks the chain: there's no identity path, so the forward pass has a hole and downstream layers get garbage. Residual structure is a prerequisite for coherently dropping layers.
5. Attention mixes information across positions, so the additive `h + f(h)` per-position decomposition into independent subset-paths isn't exact — paths share the attention-routed information and some layers (e.g. induction-head layers) are disproportionately important, unlike the uniform ensemble the pure-additive picture implies.

## Exercise

(a) For `L = 4` residual blocks, list all `2^4 = 16` paths by the subset of blocks they use, and tally how many paths have each length 0,1,2,3,4. Confirm the counts are the binomial coefficients `1,4,6,4,1`. (b) If you delete block `f_2`, how many of the 16 paths survive, and what's the length distribution of the survivors? (c) In one sentence, connect (b) to the observed "graceful degradation when you prune a layer."
