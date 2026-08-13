# Bilinear and Quadratic Forms

An attention score is a bilinear form. A loss surface's curvature is a quadratic form. Both phrases show up in explanations without definition, and both are simpler than they sound — but the *bilinearity* has a consequence that genuinely explains architectural choices, which is the payoff of this file.

**Convention:** row-vector (`x` is a row, `x W` maps it), matching [5.1](../../part5_transformer_rebuilt/5.1_self_attention/). `M ∈ R^(D×D)`.

## Definitions

A **bilinear form** takes *two* vectors and returns a scalar, linearly in each:

```
B(x, y) = x M yᵀ = Σᵢⱼ xᵢ Mᵢⱼ yⱼ
```

"Bilinear" = linear in `x` when `y` is held fixed, and linear in `y` when `x` is held fixed. The plain dot product is the special case `M = I`.

A **quadratic form** is a bilinear form fed the *same* vector twice:

```
Q(x) = x M xᵀ
```

That's it. `Q(x) = ‖x‖²` when `M = I`.

### "But `M` is a matrix — how can this be nonlinear?"

`M` is an ordinary matrix and never stops being a linear operator. What varies is **how many times you feed the input in**:

| Object | Takes | Returns | Under `x → cx` |
|---|---|---|---|
| Linear map `x ↦ xM` | one vector | a **vector** | `c` — linear |
| Bilinear form `B(x,y) = x M yᵀ` | **two** vectors | a scalar | `c` per slot separately; `c²` if both |
| Quadratic form `Q(x) = x M xᵀ` | one vector | a scalar | **`c²`** — not linear |

"Bilinear" means linear in each slot *one at a time*: fix `y`, and `M yᵀ` is a constant vector `v`, so `B(x, y) = x · v` — plainly linear in `x`. The nonlinearity appears only when the input occupies **both** slots, because then you're multiplying two things that each depend on it. Compare: `f(x) = 3x` is linear, `g(x) = x·f(x) = 3x²` is not — and the `3` never stopped being a constant. `M` is the `3`.

One consequence worth carrying: attention scores are bilinear in the residual stream, so they are **nonlinear in the input before softmax is applied at all**. `q` and `k` are each linear, but the score multiplies them.

### Symmetry, and what it means

`M` is **symmetric** if `M = Mᵀ`. For a bilinear form this means `B(x, y) = B(y, x)` — the two arguments are interchangeable. Asymmetric `M` gives a form where **order matters**: `B(x, y) ≠ B(y, x)`.

This is not a technicality in attention. `M = W_Q W_Kᵀ` is *deliberately asymmetric*, which is what lets "token A attends to token B" differ from "B attends to A" — a verb can look for its subject without the subject equally looking for the verb ([5.1/01](../../part5_transformer_rebuilt/5.1_self_attention/01_qkv_projections.md)).

Any `M` splits uniquely into symmetric and antisymmetric parts, `M = ½(M + Mᵀ) + ½(M − Mᵀ)`, and the quadratic form only ever sees the symmetric half — `x M xᵀ = x (½(M+Mᵀ)) xᵀ` for every `x`. So when discussing curvature you can always assume symmetry without loss.

## Attention as one bilinear form

The pre-softmax score between positions `i` and `j` is usually written as two projections and a dot product:

```
qᵢ = xᵢ W_Q ,   kⱼ = xⱼ W_K ,   score = qᵢ · kⱼ
```

But it collapses:

```
qᵢ · kⱼ = (xᵢ W_Q)(xⱼ W_K)ᵀ = xᵢ (W_Q W_Kᵀ) xⱼᵀ = xᵢ M xⱼᵀ
```

Verified numerically to float tolerance. **There is only one matrix in the QK circuit, `M = W_Q W_Kᵀ`.** Splitting it into `W_Q` and `W_K` is a rank constraint and a compute saving (`2 D D_h` parameters instead of `D²`), not extra expressive power.

### Gauge freedom

Because only the product matters, the factorization isn't unique. For *any* invertible `R ∈ R^(D_h × D_h)`:

```
W_Q → W_Q R ,   W_K → W_K R⁻ᵀ      leaves M unchanged
```

Verified: with `R` orthogonal, `(x W_Q R)(y W_K R)ᵀ = x M yᵀ` exactly. This is **gauge freedom** — a family of different parameter values describing the identical function. Practical consequences:

- **Individual `W_Q` and `W_K` are not meaningful objects.** Inspecting them, comparing them across models, or interpreting their singular vectors is comparing arbitrary gauge choices. `M` is what's real. (Same lesson as LoRA's `B` and `A`; see [file 01](01_dimension_span_and_rank.md) self-check #5.)
- It's why interpretability work talks about the **QK circuit** rather than the query and key matrices.
- The redundancy is `D_h²` parameters' worth, exactly matching the degrees-of-freedom count in [file 01](01_dimension_span_and_rank.md).

## The payoff: bilinearity distributes over sums

Here is the property that explains a design choice. Bilinearity means a form applied to a **sum** expands like multiplying out brackets:

```
(e + p) M (f + q)ᵀ  =  e M fᵀ  +  e M qᵀ  +  p M fᵀ  +  p M qᵀ
```

Verified exactly. Four terms, no approximation, no assumptions.

Now apply it to a Transformer input, where each position's vector is a token embedding **plus** a positional encoding (`x = e + p`):

| Term | What it computes |
|---|---|
| `eᵢ M eⱼᵀ` | **content–content** — semantic matching, position-blind |
| `eᵢ M qⱼᵀ` | **content–position** — "this kind of token cares about that region" |
| `pᵢ M eⱼᵀ` | **position–content** — "a token here cares about that kind of token" |
| `pᵢ M qⱼᵀ` | **position–position** — pure geometry: "attend `k` steps back" |

Four independent mechanisms fall out of nothing but adding the encodings and taking a bilinear score. Gradient descent grows whichever terms reduce loss, via a single `M`. **Content and position are never separated — they interact inside the form**, which is strictly more useful than either alone.

This is why "adding position to the token embedding" is a workable design rather than a signal-corrupting hack, and it's the decomposition Transformer-XL rewrites term-by-term to make position relative ([5.3/03](../../part5_transformer_rebuilt/5.3_positional_information/03_relative_positions.md)). The most striking instance: with the trivial choice `M = I`, the position–position term of sinusoidal `PE` is already a smooth, monotonically decaying function of `|i − j|` — a working locality bias present at initialization, before any learning ([5.3/02](../../part5_transformer_rebuilt/5.3_positional_information/02_sinusoidal_and_learned_absolute.md)).

## Quadratic forms as curvature

The other place these appear. A second-order Taylor expansion of a loss around a point `θ₀` is:

```
L(θ₀ + δ) ≈ L(θ₀) + g·δ + ½ δ H δᵀ
```

with `g` the gradient and `H` the **Hessian** of second derivatives. The `½ δ H δᵀ` term is a quadratic form, and it *is* the curvature — how the loss bends, as opposed to how it slopes.

The vocabulary attached to it:

- **Positive definite** (`x M xᵀ > 0` for all `x ≠ 0`): all eigenvalues positive, the surface curves up in every direction — a local minimum.
- **Indefinite** (mixed signs): a saddle. In high-dimensional loss landscapes, saddles vastly outnumber local minima, which is why "escaping saddles" is the more relevant concern.
- **Condition number** (`λ_max/λ_min`): the ratio of steepest to shallowest curvature. Large means a narrow valley — gradient descent zigzags, and this is precisely the pathology preconditioners fix.

This connects directly to optimizers: **Adam approximates the curvature with a diagonal matrix** (a per-parameter scale), which is why it can only stretch along the coordinate axes it's handed and not along the loss surface's natural axes. Full-matrix or block-wise preconditioners (Shampoo) approximate more of `H` at higher cost ([2.4/01](../../part2_neural_network_fundamentals/2.4_optimization/01_sgd_to_adamw.md)).

## Why this matters

- **Attention is one bilinear form**, and its asymmetry is a feature. Reading `W_Q`/`W_K` separately is a category error.
- **The four-term expansion** justifies additive positional encoding and is the entry point to relative-position schemes.
- **Curvature vocabulary** — positive definite, indefinite, condition number, preconditioner — is the language of every optimizer discussion past plain SGD.
- **Gauge freedom** recurs wherever a product of matrices is learned: QK, OV, LoRA's `BA`, tied embeddings.

## Self-check

1. `M = W_Q W_Kᵀ` with `W_Q, W_K ∈ R^(D×D_h)`. What is the maximum possible rank of `M`, and why is that a deliberate design choice rather than a limitation to fix?
2. Why is it a mistake to compare the singular vectors of `W_Q` between two trained models?
3. Attention scores use an *asymmetric* `M`. Give a concrete linguistic reason symmetry would be harmful.
4. Expand `(e + p) M (f + q)ᵀ`. Which single term would a head implementing "attend to the token 3 positions back, regardless of content" rely on?
5. If a quadratic form only ever sees the symmetric part of `M`, why can attention's asymmetry matter at all?

### Answers

1. `rank(M) ≤ D_h`, since the product of a `D × D_h` and a `D_h × D` matrix can't exceed the inner dimension. It's deliberate: it costs `2 D D_h` parameters instead of `D²`, and — more importantly — it forces each head to compare tokens within a `D_h`-dimensional *subspace* rather than the full stream, which is what makes different heads specialize on different kinds of similarity. Multi-head attention is a bet that many narrow comparisons beat one wide one.
2. Because of gauge freedom: `W_Q → W_Q R` and `W_K → W_K R⁻ᵀ` leave the function identical, so the singular vectors of `W_Q` alone are an arbitrary choice within a `D_h²`-dimensional family. Two models computing the *same* attention could have unrelated `W_Q`. Only gauge-invariant objects — `M = W_Q W_Kᵀ`, or the attention patterns themselves — are comparable.
3. Symmetry would force "how much does the verb attend to its subject" to equal "how much does the subject attend to the verb." But the useful relations are directional: a pronoun needs to look back at its antecedent much more than the antecedent needs to look forward at the pronoun; an adjective's head noun matters more to the adjective than vice versa. Symmetric attention would make every retrieval mutual, collapsing all directed syntactic relations into undirected similarity.
4. `eᵢ M eⱼᵀ + eᵢ M qⱼᵀ + pᵢ M eⱼᵀ + pᵢ M qⱼᵀ`. A purely positional head relies on the **position–position** term `pᵢ M qⱼᵀ`, since that's the only one independent of both tokens' content — and with sinusoidal `PE` it depends only on the offset `i − j`, exactly the quantity such a head needs.
5. Because attention forms are **bilinear, not quadratic** — the two arguments are different vectors (`xᵢ` and `xⱼ`), not the same one. The "only the symmetric part survives" fact applies specifically to `x M xᵀ`, where you feed in one vector. With two distinct inputs, the antisymmetric part contributes `½(xᵢ M xⱼᵀ − xⱼ M xᵢᵀ)`, which is generally nonzero and is precisely the directional information. Curvature is quadratic (hence symmetric); attention is bilinear (hence can be directional).

## Exercise

1. Build random `W_Q, W_K ∈ R^(64×16)` and verify `(x W_Q)(y W_K)ᵀ = x (W_Q W_Kᵀ) yᵀ` to float tolerance.
2. Generate a random invertible `R ∈ R^(16×16)` and confirm that `W_Q R` and `W_K R⁻ᵀ` give an identical score matrix. Then check how different `W_Q` and `W_Q R` look — compare their singular values and the cosine between their first columns. This is what gauge freedom costs you interpretively.
3. Split a random `M` into symmetric and antisymmetric parts. Confirm `x M xᵀ` equals `x M_sym xᵀ` for random `x`, but `x M yᵀ ≠ x M_sym yᵀ` for `x ≠ y`. Then compute, for a random pair, the size of the asymmetric contribution relative to the total score — that fraction is how much directionality this `M` carries.
4. With `e, p, f, q` random, verify the four-term expansion sums exactly to `(e+p) M (f+q)ᵀ`. Then zero out `M`'s effect on the position components (project them away) and confirm only the content–content term survives.
