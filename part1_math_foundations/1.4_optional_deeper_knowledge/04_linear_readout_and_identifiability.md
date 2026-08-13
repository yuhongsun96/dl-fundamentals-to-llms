# Linear Read-out and Identifiability

Transformers add signals together constantly: a token embedding plus a positional encoding, then every attention head and FFN writing into the same residual stream. When can a learned linear layer pull an added signal back out — and when is it impossible?

**Convention:** row-vector. `x = e + p ∈ R^D` is an observed sum; we want to recover `p` (or `e`) with a learned linear map `W`.

## Linearity makes a sum filterable

```
x W = (e + p) W = e W + p W
```

One matrix acts on each addend independently. A `W` that sends `e` to zero while leaving `p` intact gives `x W = p W` — the content is filtered out and the position kept. Such a `W` is a **projection**: it keeps the component of `x` lying in one subspace and discards the rest.

Two subspaces matter: `T = span{token embeddings}` and `P = span{positional encodings}`. If they are **orthogonal**, an exact projection exists and recovery is perfect — verified at cosine **1.0000** with position in the first 64 coordinates and tokens strictly in the other 448.

Addition itself never destroys information. Whether that information is *accessible to a linear map* is a question about the geometry of `T` and `P`.

## Measuring separation: gain over the do-nothing baseline

A raw recovery score is not evidence of separation. Output `x = e + p` unchanged as your "estimate" of `p` and you already score

```
cos(e + p, p) = ‖p‖ / √(‖p‖² + ‖e‖²)
```

which is `0.707` at equal norms — purely because two independent vectors of comparable size sit at ~45° to their sum. **Separation is the gain above this baseline, never the raw number.**

Setup for all figures below: `D = 512`, vocab 50,257, embeddings at GPT-2-style `0.02` init times the paper's `√d_model` scaling, 40,000 samples, fit on half, evaluated held out.

| Position signal | fitted | do-nothing | **gain** |
|---|---|---|---|
| Orthogonal coordinate blocks (idealized) | **1.0000** | — | exact |
| Real sinusoidal `PE` | 0.9630 | 0.8423 | **+0.121** |
| Random vectors, same distribution and scale as tokens | 0.7805 | 0.7073 | **+0.073** |

The last row is the important one: when position vectors are statistically indistinguishable from token vectors and share the same full-rank subspace, the read-out achieves **almost nothing over passing the input through untouched**.

It cannot do better. When both signals are isotropic with the same covariance shape, the optimal linear estimator is a **pure scalar shrinkage** `W = c·I` — and scaling never changes a cosine. The best possible linear read-out is literally "output `x`, scaled."

> **Separability requires a structural difference between the two signals — high dimension alone is not enough.** Orthogonality is sufficient; something like it is close to necessary.

## What creates that structural difference

**Subspace dimension is the dominant lever.** Holding signal-to-noise fixed at `‖p‖/‖e‖ = 1.56` — so every row shares the same **0.842** baseline — and varying only how many dimensions the position signal occupies:

| Position in a random *k*-dim subspace | fitted | **gain** | `W` vs. pure shrinkage |
|---|---|---|---|
| `k = 4` | 0.9988 | **+0.156** | 0.996 — a genuine projector |
| `k = 16` | 0.9940 | +0.152 | 0.985 |
| `k = 64` | 0.9754 | +0.133 | 0.938 |
| `k = 256` | 0.9183 | +0.076 | 0.730 |
| `k = 512` (all of `R^D`) | 0.8812 | **+0.039** | 0.499 — collapsing toward `c·I` |

A **4× spread in real separation** across the sweep. The last column measures how far the fitted `W` sits from the nearest scalar multiple of the identity: at `k = 4` it is a true projector doing real work; at `k = 512` it has collapsed halfway toward pure shrinkage.

Why dimension matters: recovering a `k`-dimensional signal from `D`-dimensional observations means estimating `k` numbers from `D` measurements, so interference from the other addend averages down by roughly `√(D/k)`. Small `k` means redundancy working in your favor. This is why sinusoidal `PE`'s low effective rank is a genuine asset ([file 01](01_dimension_span_and_rank.md)).

**Relative magnitude is the second lever.** The original Transformer multiplies token embeddings by `√d_model` for exactly this reason: `‖PE(i)‖ = √(D/2) = 16` is fixed while a `0.02`-init embedding has norm `0.45`, so without rescaling position would dominate content ~35× and *content* would become the unrecoverable signal.

## Why adding a signal isn't "corrupting" it

`e` and `p` occupy the same `D` coordinates, so adding `p` overwrites the number the embedding put in each dimension. Measured on a real embedding table (`D = 576`, magnitudes matched at `‖e‖ = ‖p‖ = 3.18`):

| Measurement | Result |
|---|---|
| Median change to a coordinate | **137%** |
| Coordinates where `\|p_d\| > \|e_d\|` | **60.3%** |
| Correct token recovered by nearest-neighbour over 49,152 tokens | **100.0%** |
| Correct position recovered over 512 positions | **98.5%** |

All four hold simultaneously. Position overwrites most of the embedding coordinate-wise, and nothing is lost. The resolving fact:

> **High dimensions let you perturb everything while barely moving in a particular direction.**

**Nothing ever reads a coordinate.** Every downstream operation is a projection — `W_Q`, `W_K`, `W_V`, and every FFN key compute `x · w` across all `D` coordinates at once. So the question is never "did `p` change dimension 17" but "did `p` change `x · w` for the `w`s that carry meaning":

```
‖p‖ = 3.179       but      |p · w| ≈ 0.104  for a unit direction w
ratio ≈ 0.033     ≈  1/√D  (1/√576 = 0.042)
```

### Coherent vs. incoherent

Along the token's own read-out direction:

| Contribution to `x · e_token` | Size | Why |
|---|---|---|
| the embedding `e` | `‖e‖² ≈ 10.1` | **aligned** — adds coherently |
| the position `p` | `≈ ‖p‖‖e‖/√D ≈ 0.42` | **random** relative to it — adds incoherently |

A ~24× margin, exactly `√D`. Signal accumulates along its own direction; interference spreads across all of them. So the two signals don't occupy different *coordinates* — they point in **near-orthogonal directions**, for the ordinary reason that almost everything is near-orthogonal in high dimensions ([1.1/06](../1.1_linear_algebra/06_projections_orthogonality.md)).

### The learned weights can dodge

The `1/√D` figure is what a *random* read-out direction gets. But `W_Q`, `W_K`, `W_V` are **learned**, and the positional signal is low-rank — about 225 of 512 directions carry 99% of sinusoidal `PE`'s energy, and just 14 carry half ([file 01](01_dimension_span_and_rank.md)). A learned projection can orient its rows *away* from that subspace for pure content, or *into* it for position. That is the mechanism behind the sweep above: separation runs from **+0.156** at `k = 4` to **+0.039** at `k = 512`. Low-rank structure isn't only cheap to store — it's **easy to avoid**.

### The limit

This is a budget, not a guarantee. Each added signal costs roughly `1/√D` of interference along every direction, so one positional vector is nearly free — but ~100 sublayers writing across 32 layers is how the budget gets spent. That erosion, not the initial addition, is the real weakness of input-additive positional encoding, and why RoPE stops writing position into the stream at all ([5.3/04](../../part5_transformer_rebuilt/5.3_positional_information/04_rope.md)).

## Least squares as the ceiling

Given observations `X ∈ R^(N×D)` and targets `Y ∈ R^(N×k)`:

```
W* = argmin_W ‖X W − Y‖_F²      normal equations:  W* = (XᵀX)⁻¹ XᵀY
```

- **Least squares is the best linear estimator, so it upper-bounds what any linear layer can do.** If it can't recover the signal, no `W_Q`, `W_K`, or `W_V` can. That makes it the right tool for "is this information *linearly available*?"
- **Needs `N ≫ D`, evaluated held out.** With `N ≤ D` the system is underdetermined and fits anything exactly — including pure noise — so in-sample scores are determined by the shapes, not the signal.

## Identifiability: when recovery is impossible

- **Collinear signals.** If `p = c·e`, the two are the same direction and no linear map separates them.
- **Exact aliasing.** If two positions produce the identical encoding they are unidentifiable by construction. This is a learned table's failure past its last row, and why base 10000 keeps the slowest sinusoid from wrapping ([file 08](08_frequency_phase_and_periodicity.md)).
- **Gauge freedom.** When only a product is observed the factors are unidentifiable — attention behavior determines `W_Q W_Kᵀ`, never `W_Q` and `W_K` individually ([file 03](03_bilinear_and_quadratic_forms.md)).

*Not identifiable* is a proof; *hard to learn* is an engineering problem. Telling them apart saves a lot of wasted effort.

## Why this matters

- **Additive positional encoding** is defensible because position is low-dimensional and magnitude-matched — and because it stays *usable without being recovered*, via the four-term bilinear expansion ([file 03](03_bilinear_and_quadratic_forms.md)).
- **The residual stream** runs on this principle at scale: ~100 sublayers add into one vector, and each later layer's projections read out what they need ([1.1 supplementary](../1.1_linear_algebra/supplementary/06_residual_stream.md), [3.1/04](../../part3_residual_connections_deep_networks/3.1_skip_connection/04_residual_stream_as_abstraction.md)).
- **Linear probing** is exactly this experiment. A probe succeeding means the information is *linearly available*, not that the model uses it — and probe scores partly reflect the property's effective dimension, so they aren't comparable across properties.

## Self-check

1. Why does `x W = e W + p W` matter for "can addition destroy information"?
2. A read-out recovers a signal at cosine 0.78. What do you need before that number means anything?
3. Position in a 4-dim subspace separates at +0.156; spread over all 512 dims, at +0.039. Same signal-to-noise. Why?
4. Why does the original Transformer multiply embeddings by `√d_model`?
5. A linear probe reads "syntactic role" out of layer 12 at 95% accuracy. Name two things this does *not* establish.

### Answers

1. Because a linear map treats the addends **separately** — the sum is superposed, not scrambled. Whatever `W` does to `e` it does regardless of `p`. So "can I get `p` back" becomes "does a `W` exist that keeps `p` and kills `e`," a question about the geometry of two subspaces rather than about information loss.
2. The **do-nothing baseline** — `cos(e+p, p) = ‖p‖/√(‖p‖²+‖e‖²)`, which is 0.707 at equal norms with no separation occurring. At matched magnitudes a raw 0.78 is a gain of only ~0.07 and represents almost no separation; against a 0.5 baseline it would be substantial. Only the gain is meaningful.
3. **Subspace dimension.** Recovering 4 numbers from 512 measurements lets interference average down by `√(512/4)`; at `k = 512` there are as many unknowns as measurements and no redundancy left, so the optimal `W` collapses toward pure scalar shrinkage, which cannot change a cosine at all.
4. To stop **content** from becoming the unrecoverable signal. `‖PE(i)‖ = √(D/2) = 16` is fixed while a `0.02`-init embedding has norm ~0.45 — without rescaling, token identity is a tiny perturbation on a large positional vector, putting the information the model exists to process on the low-SNR side.
5. It does not establish that the **model uses** it — availability is not use, and the check is a causal intervention (ablate the direction, see if behavior changes). Nor that the model **computes** it there — the property may have been linearly available in the input embeddings, which is why probes need a layer-0 or random-weights baseline.

## Exercise

1. Build sinusoidal `PE` for `S = 512`, `D = 512` and a random embedding table scaled `0.02·√D`. Draw 40,000 `(token, position)` pairs, fit least squares on half to predict `p`, and report **both** the held-out cosine and the do-nothing baseline `cos(x, p)`. You should get ≈0.96 against ≈0.84.
2. Sweep the embedding scale over `[0.1×, 1×, 10×]` of the `√d_model` value and plot the *gain over baseline* for both `p` and `e`. Find the crossover where protecting position starts costing content.
3. Confine position to a `k`-dim subspace at fixed norm and sweep `k ∈ {4, 16, 64, 256, 512}`. Confirm the gain shrinks monotonically, and check the fitted `W` against the nearest scalar multiple of `I` to see it collapse toward shrinkage.
4. Assign the *same* vector to positions 10 and 20. Confirm no `W` can distinguish them, and connect this to choosing the sinusoidal base relative to sequence length.
