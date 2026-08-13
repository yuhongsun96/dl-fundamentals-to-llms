# Dimension, Span, and Rank

The word "dimension" gets used for at least three different things in ML writing, and confusing them is why sentences like *"position is a 1-D curve, but its span is 62-dimensional, inside a 512-dimensional space"* read as gibberish. That sentence is precise and all three numbers are different. This file separates them.

**Convention:** vectors are elements of `R^D`. Matrices act as `y = W x` (column convention) where a distinction matters; nothing here depends on it.

## The vocabulary, in order

Each term builds on the previous.

**Vector space** — `R^D`, the set of all `D`-tuples of real numbers. Nothing more exotic than "vectors with `D` slots."

**Linear combination** — given vectors `v₁, …, v_k`, any `a₁v₁ + a₂v₂ + … + a_kv_k` with real coefficients. *Adding and scaling, no other operations.*

**Span** — the **set of all** linear combinations of a given collection:

```
span{v₁, …, v_k} = { a₁v₁ + … + a_kv_k  :  aᵢ ∈ R }
```

"The span of `{v}`" for a single nonzero vector is the infinite **line** through the origin in direction `v` — every scalar multiple. "The span of `{u, v}`" for two non-parallel vectors is the **plane** containing both.

**Linear subspace** — any set that is closed under addition and scaling, i.e. anything that *is* the span of something. Concretely, in `R³`: the origin alone (dim 0), lines through the origin (dim 1), planes through the origin (dim 2), and all of `R³` (dim 3). **Crucially, a subspace must contain the origin and must be flat.** A line that misses the origin is not a subspace; neither is a circle.

**Basis** — a minimal spanning set: enough vectors to span the subspace, with none redundant (they're linearly independent). A subspace has many possible bases but they all have the same size.

**Dimension of a subspace** — the size of any basis. This is the number that "1-D subspace" refers to: **the number of independent directions you need**, equivalently the number of coefficients required to name a point in it.

So the phrase from the intro decodes as:

> "A **1-D linear subspace**" = a single line through the origin. One number (`a`) picks out any point on it.

## The three dimensions that get conflated

| Name | Question it answers | Example: a helix in `R³` |
|---|---|---|
| **Ambient dimension** | How many coordinates does each vector have? | **3** — points are `(x,y,z)` |
| **Linear dimension** (dim of the span) | How many independent *directions* do the points collectively occupy? | **3** — the helix genuinely uses all of `R³`; no plane contains it |
| **Intrinsic dimension** | How many *parameters* determine a point? | **1** — one number `t` gives `(cos t, sin t, 0.1t)` |

Verified: 500 points sampled from that helix have a mean-centered span of rank **3**, from **1** parameter, in an ambient space of **3**.

That gap between intrinsic 1 and linear 3 is the whole subtlety. A curve can wander through many dimensions while still being *one-dimensional as an object*. And the two numbers answer genuinely different questions:

- **Linear dimension** answers *"how much room does this occupy?"* — the right question for capacity, interference, and whether a projection can isolate it.
- **Intrinsic dimension** answers *"how much is there to learn?"* — the right question for generalization, because a smooth 1-parameter family means knowledge transfers between nearby parameter values.

### The term that covers curved sets: manifold

A **manifold** is, informally, a set that *locally* looks like `R^k` even if globally it curves. The helix is a 1-D manifold; the surface of a sphere is a 2-D manifold in `R³`. "Intrinsic dimension" is really "dimension of the manifold." A subspace is the special case of a manifold that happens to be flat and pass through the origin.

The **manifold hypothesis** is the version of this that matters for ML: real high-dimensional data (images, text embeddings) is believed to lie on or near a manifold of far lower intrinsic dimension than the ambient space. A `4096`-dim embedding of English sentences does not fill `R^4096`; it clusters on something much thinner.

## Rank

For a matrix, **rank** is the dimension of its span — but a matrix has two natural spans:

- **Column space** = span of the columns; **row space** = span of the rows.
- These always have the **same** dimension, and that shared number is the **rank**.

```
rank(A) ≤ min(#rows, #cols)
```

`rank(A) = min(rows, cols)` is **full rank**; less than that is **rank-deficient** or **low-rank**. Verified: 100 random vectors in `R^512` span a subspace of dimension exactly **100** — random vectors are as independent as they're allowed to be, so rank hits its ceiling `min(100, 512)`.

Rank is what makes LoRA work: a full update `ΔW ∈ R^(D×D)` has `D²` parameters, but a rank-`r` update `ΔW = BA` with `B ∈ R^(D×r)`, `A ∈ R^(r×D)` has only `2Dr`. See [1.1/04](../1.1_linear_algebra/04_outer_products_low_rank.md).

### Effective rank

Real matrices are almost never *exactly* low-rank; they're low-rank **plus small noise**, which makes the exact rank misleadingly high. **Effective rank** is the informal fix: the number of directions that carry non-negligible weight, read off the singular values.

Singular values `[10, 5, 1, 0.01]` have exact rank 4 but effective rank ~3 — the last direction contributes essentially nothing ([1.1/05](../1.1_linear_algebra/05_eigen_svd.md) self-check #1). The usual quantitative version is *"how many directions hold 90% of the energy"*, which is [file 02](02_energy_and_spectra.md).

Note that effective rank is a **judgment call, not a definition** — it depends on the threshold you pick. Always state the threshold: "62 dimensions for 90% of the energy" is a claim; "the effective rank is 62" is that claim with the interesting part hidden.

## Degrees of freedom

The same counting idea under a different name: **how many numbers can be chosen independently.** A `D × D` matrix has `D²` degrees of freedom. Constrain it to be symmetric and you have `D(D+1)/2`. Constrain it to be a rotation in `R^D` and you have `D(D−1)/2`. Constrain it to rank `r` and you have `r(2D − r)`.

This is the right lens for "how expressive is this parameterization" questions — and for spotting when two parameterizations that look different describe the same set of functions (see gauge freedom in [file 03](03_bilinear_and_quadratic_forms.md)).

## Why this matters

- **Positional encodings.** Sinusoidal `PE` traces a curve with **intrinsic dimension 1** whose **linear span** is a few hundred dimensions inside an ambient `D = 512`. The intrinsic-1 part is why the model generalizes across positions; the span is what determines how much residual stream it consumes ([5.3/02](../../part5_transformer_rebuilt/5.3_positional_information/02_sinusoidal_and_learned_absolute.md)).
- **LoRA and adapters.** The bet is that useful fine-tuning updates have low *rank* — few independent directions — even though they're stored as full matrices.
- **Superposition.** The residual stream packs far more features than `D` by using *nearly* orthogonal directions rather than a basis. The counting argument for how many fit is in [1.1/06](../1.1_linear_algebra/06_projections_orthogonality.md).
- **Embedding spectra.** "Is this matrix compressible?" is a question about effective rank — and the answer for real embedding matrices is a surprising *no* ([1.1 supplementary](../1.1_linear_algebra/supplementary/05_embedding_spectrum.ipynb)).
- **Attention head width.** Each head's `W_Q`, `W_K` map `D → D_h`, deliberately projecting into a low-dimensional subspace; `W_O` writes a low-rank update back to the stream ([5.1/04](../../part5_transformer_rebuilt/5.1_self_attention/04_multi_head_attention.md)).

## Self-check

1. Is the set of vectors in `R³` with `x + y + z = 1` a linear subspace? What about `x + y + z = 0`?
2. A curve in `R^512` is described by a single parameter `t`. What is its intrinsic dimension? What can you say about the dimension of its linear span?
3. Why is "the effective rank is 62" an incomplete statement?
4. You have 1000 vectors in `R^768`. What is the largest possible dimension of their span? If you measured it as exactly 768, what would that tell you?
5. A rank-`r` update to a `D × D` matrix has `r(2D − r)` degrees of freedom, but you store it as `B ∈ R^(D×r)` and `A ∈ R^(r×D)`, which is `2Dr` numbers. Why the discrepancy?

### Answers

1. `x + y + z = 1` is **not** a subspace — it doesn't contain the origin (`0+0+0 = 1` is false), and it isn't closed under scaling. It's an *affine* plane: a subspace shifted off the origin. `x + y + z = 0` **is** a subspace: it contains the origin, and sums and scalar multiples of solutions are solutions. It's a 2-D subspace (a plane) in `R³`.
2. Intrinsic dimension **1** — one parameter names a point. Its linear span can be anywhere from **1** (if the curve is a straight line through the origin) up to **512** (if the curve wanders enough to be linearly dense), and in general has nothing to do with the intrinsic dimension. You cannot infer one from the other in either direction.
3. Because effective rank isn't defined without a threshold. "62 dimensions hold 90% of the energy" is a measurement; the same matrix might need 109 for 99% and 225 for 99.9%. Quoting a single number hides both the threshold and the shape of the tail — and the tail is usually where the interesting behavior is.
4. At most **768** — the span is a subspace of `R^768`, so `min(1000, 768) = 768` is the ceiling. Measuring exactly 768 tells you the vectors are *not* confined to any lower-dimensional subspace, which usually means "no exactly-low-rank structure." It tells you nothing about whether they're *effectively* low-rank; a spectrum with 700 tiny singular values is full rank and still highly compressible in practice.
5. The `2Dr` storage is an **over-parameterization** of the `r(2D − r)`-dimensional set of rank-`r` matrices. You can insert any invertible `r × r` matrix `R` between them — `(BR)(R⁻¹A) = BA` — giving the same product from different factors. That's `r²` redundant parameters, and indeed `2Dr − r² = r(2D − r)`. This redundancy is exactly the "gauge freedom" idea in [file 03](03_bilinear_and_quadratic_forms.md), and it's why LoRA's `B` and `A` are not individually meaningful — only their product is.

## Exercise

Build the sinusoidal `PE` matrix for `S = 4096`, `D = 512` (formula in [5.3/02](../../part5_transformer_rebuilt/5.3_positional_information/02_sinusoidal_and_learned_absolute.md)). Then answer, with code:

1. What is its **intrinsic** dimension? (No code needed — count the parameters in the formula.)
2. What is `torch.linalg.matrix_rank(PE)`? Is it full rank? (You should find **236**, not 512 — so the span doesn't even fill the ambient space, despite 4096 position vectors being available to span it.)
3. How many singular directions hold 50%, 90%, and 99% of the energy? (You should find **14, 173, and 225**. Note how close the 99% figure is to the numerical rank in step 2 — the spectrum genuinely dies rather than trailing off.)
4. Reconcile 1–3 in one sentence: how can a 1-dimensional object have a 236-dimensional span, and which of these numbers would you quote to argue that "position doesn't take up much room"?

Then do the same for a **learned** absolute position table initialized randomly, and explain why its intrinsic dimension is a fundamentally different kind of number.
