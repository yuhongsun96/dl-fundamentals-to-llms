# Projections and Orthogonality

## Orthogonal vectors

`<u, v> = 0` means `u` and `v` are perpendicular. In high dimensions, random vectors are *almost* orthogonal — this is why embedding spaces can pack many nearly-independent concepts into a fixed `d_model`.

Specifically: two random unit vectors in `R^d` have expected cosine similarity 0 and std `~1/√d`. For `d=4096`, that's std ≈ 0.016 — extremely tight concentration around orthogonal. This is the **Johnson-Lindenstrauss** / **concentration of measure** phenomenon, and it's *the* reason neural nets can represent so many features in one vector ("superposition" in interpretability).

### Why this is true

Two complementary intuitions:

**Dimensional accounting.** Pick a fixed unit vector `v`. The space "parallel to `v`" is **1-dimensional** — just the line through `v` and `-v`. The space "orthogonal to `v`" is **`(d-1)`-dimensional** — the entire hyperplane perpendicular to it.

So in dimension `d`:
- Fraction of directions that are "parallel-ish" to `v`: `1/d`
- Fraction of directions that are "orthogonal-ish" to `v`: `(d-1)/d → 1`

In 2D, half the plane is "perpendicular-ish" to a given direction. In 4096D, 99.98% is. **There's just way more room to be perpendicular than to be parallel.** A random vector has overwhelmingly more places to go that don't align with `v`.

**Cancellation in the dot product.** Each component of a random unit vector has typical magnitude `±1/√d` (the components have to sum-of-square to 1, so each one carries `1/d` of the "energy" on average). The dot product `<u, v> = Σ u_i v_i` is then a sum of `d` independent terms, each typically `±1/d`. With random signs these terms mostly cancel — by CLT, the sum has standard deviation `√d · (1/d) = 1/√d`. Bigger `d` → tighter concentration around 0.

These are aspects of the broader **concentration of measure** phenomenon: random quantities in high dimensions concentrate sharply around their means. Distances, norms, and inner products all become "almost deterministic" as `d` grows.

### Exponential capacity for nearly-orthogonal directions

Strict orthogonality is dimension-limited:

- In 2D you can fit at most 2 mutually orthogonal unit vectors.
- In 3D, 3.
- In `d` dimensions, exactly `d`.

But *nearly* orthogonal is qualitatively different. By the **Johnson-Lindenstrauss lemma** (and related random-projection arguments), you can fit roughly `e^(O(d))` vectors that are all pairwise nearly orthogonal (within angle `≈ 90° ± ε`) in `d`-dim space. **Exponential** in `d`.

| Dimension | Mutually orthogonal (max) | Nearly orthogonal (≈ 90° ± ε) |
|---|---|---|
| 2D | 2 | ~4 before significant overlap |
| 100D | 100 | billions |
| 4096D | 4096 | more than the number of atoms in the observable universe |

This is the basis of **superposition** in interpretability: a model can pack many features into a fixed `d`-dim residual stream because there's exponentially much "room" for nearly-orthogonal directions, even though the dimension is only `d`. A `d = 4096` LM can in principle encode many more than 4096 distinguishable features as long as they're sparsely active.

### One-line takeaway

> In high dimensions, perpendicular is the default; alignment is the exception. The 2D/3D intuition that "perpendicular is special, generic is messy" is actually backwards — *generic* random directions are perpendicular; *aligned* directions are rare.

### The consequence that does the most work

Near-orthogonality has a restatement that resolves a surprising number of "how is that not destructive?" questions:

> **High dimensions let you perturb *every coordinate* while barely moving in any *particular direction*.**

These sound contradictory and aren't. A vector `p` of norm `‖p‖` spread across `d` dimensions has typical coordinate size `‖p‖/√d` — but its projection onto any *given* direction `w` is also only about `‖p‖/√d`, because `p` and `w` are nearly perpendicular. So adding `p` to a vector `e` can change all `d` coordinates substantially while leaving `e · w` almost untouched.

The asymmetry that makes this useful is **coherent vs. incoherent** contribution. Along `e`'s own read-out direction:

```
e contributes  ‖e‖²          — aligned, adds coherently
p contributes  ≈ ‖p‖‖e‖/√d   — random relative to it, adds incoherently
```

That's a factor of `√d` in favor of the signal. Measured on a real embedding table with `d = 576` and magnitudes matched (`‖e‖ = ‖p‖ = 3.18`): adding `p` changes the **median coordinate by 137%**, and position exceeds content in **60%** of coordinates — yet the correct token is still recovered by nearest-neighbour search over a 49,152-token vocabulary **100%** of the time. Coordinate-level damage and direction-level integrity are simply different measurements.

This is why the residual stream can be a shared bus at all: every sublayer writes into all `d` coordinates, but each write is nearly orthogonal to the directions the other sublayers read. Nothing is "reserved" — everything is merely pointed elsewhere. Worked out with measurements in [1.4/04](../1.4_optional_deeper_knowledge/04_linear_readout_and_identifiability.md).

## Projection onto a vector

Project `x` onto the direction of `u`:
```
proj_u(x) = (<x, u> / <u, u>) · u
```

If `u` is a unit vector: `proj_u(x) = <x, u> · u`. The scalar `<x, u>` is the coordinate of `x` in the `u` direction.

### Reading the formula

The dot product `<x, u>` alone is **not** the projection — it's an unnormalized alignment scalar. The full formula has three pieces, each doing a different job:

```
proj_u(x) = ( <x, u>     /     <u, u> )     ·     u
              ──┬─────         ──┬─────         ─┬─
                alignment        ‖u‖² norm       direction
                scalar           correction
```

- **`<x, u>`** — raw alignment in "dot product units" (depends on both magnitudes).
- **`<u, u> = ‖u‖²`** — corrects for `u`'s length, so the result doesn't depend on how long `u` happens to be drawn. Without this, scaling `u` by 2 would scale the projection by 2 — geometrically wrong, since projecting onto a *line* shouldn't care which length-vector specifies the line.
- **`u`** — picks the output direction.

`<u, u>` and `‖u‖²` are literally the same: `<u, u> = Σ u_i · u_i = Σ u_i² = ‖u‖²`. The `<u, u>` notation generalizes cleanly to abstract inner-product spaces; treat them as interchangeable.

### Two different "projection" quantities

The word "projection" overloads two related but distinct things:

| Quantity | Formula | Type | Geometric meaning |
|---|---|---|---|
| **Scalar projection** (component) | `<x, u> / ‖u‖` | scalar | Signed *length* of `x`'s shadow on `u`'s line |
| **Vector projection** | `(<x, u> / <u, u>) · u` | vector | The shadow itself, as a vector along `u` |

The dot product `<x, u>` alone is neither — it equals the scalar projection times `‖u‖` (i.e., unnormalized). All three encode the same alignment information, just in different units.

When `u` is a **unit vector** (`‖u‖ = 1`), both simplify dramatically:
- Scalar projection = `<x, u>`
- Vector projection = `<x, u> · u`

This is why "the dot product is the projection" feels right when working with unit vectors — the normalization is invisible. When `u` isn't unit-length, you need the `<u, u>` correction.

### Symmetry vs. directionality

A subtle point that trips people up:

- **Dot product is symmetric**: `<x, u> = <u, x>`. The scalar doesn't care about order.
- **Vector projection is NOT symmetric**: `proj_u(x) ≠ proj_x(u)`. The two are different vectors — one along `u`, one along `x`.

So when you say "project `x` onto `u`," the order matters for the *vector* result, but the *underlying alignment scalar* is order-invariant. The formula encodes this by taking the symmetric scalar `<x, u>` and then deciding direction via the multiplication by `u` (not `x`).

### Geometric picture

Drop a perpendicular from the tip of `x` onto the line through `u`. The foot of that perpendicular — measured from the origin — is `proj_u(x)`. The signed length of the foot-line is `comp_u(x)` (the scalar projection). The dot product `<x, u>` is the same length, just unnormalized for `u`'s magnitude.

```
       x
       •
      ╱│
     ╱ │
    ╱  │ ← perpendicular drop
   ╱   │
  •────•──────────►  u-line
 origin  ↑
        proj_u(x)  (the "foot" of the perpendicular)
        ⊢──comp──⊣  (length of the foot)
```

### Quick reference

- `<x, u>` — alignment scalar (unnormalized).
- `<x, u> / ‖u‖` — scalar projection: signed length of `x`'s shadow on `u`.
- `(<x, u> / ‖u‖²) · u` — vector projection: the shadow as a vector.

All three carry the same directional alignment, expressed in different units. The vector projection is the one most often meant by "the projection of `x` onto `u`."

## Orthogonal decomposition

Any vector `x` can be **uniquely** split into two perpendicular components — one along a chosen direction `u`, one orthogonal to it:

```
x = proj_u(x) + x_perp     where x_perp ⊥ u
```

Concretely:
```
x_perp = x - proj_u(x)
```

Take `x`, subtract its shadow on `u`'s line, and what's left over is — by construction — perpendicular to `u`.

### Why subtracting the projection produces something perpendicular

Quick algebraic check that `x_perp = x - proj_u(x)` really is orthogonal to `u`:

```
<x_perp, u> = <x - proj_u(x), u>
            = <x, u> - <(<x, u>/<u, u>) · u, u>
            = <x, u> - (<x, u>/<u, u>) · <u, u>
            = <x, u> - <x, u>
            = 0   ✓
```

Subtracting the projection removes exactly the right amount of `u`-content. The formula for `proj_u(x)` is *defined* to be the unique scalar multiple of `u` whose removal leaves a residual with zero `u`-alignment.

### The Pythagorean view

The decomposition is length-preserving in a specific sense — squared norms add:
```
‖x‖² = ‖proj_u(x)‖² + ‖x_perp‖²
```

This is just Pythagoras on the right triangle formed by `x` (hypotenuse), `proj_u(x)` (leg along `u`), and `x_perp` (leg perpendicular to `u`). It works *because* the two legs are orthogonal — otherwise you'd pick up a cross-term `2 · <proj_u(x), x_perp>`, which is exactly zero here.

The ratio `‖proj_u(x)‖² / ‖x‖²` is the **fraction of `x`'s squared length captured by `u`**. For unit `u`, this equals `cos²(θ)` where `θ` is the angle between `x` and `u`. This is what "fraction of variance explained by direction `u`" means in PCA-style analyses.

### Uniqueness

The decomposition is unique: given `x` and `u`, there's *exactly one* way to write `x = a + b` with `a` along `u` and `b ⊥ u`. The pair `(proj_u(x), x_perp)` is determined by the orthogonality constraint — no other split into "along-`u`" and "perpendicular-to-`u`" pieces exists.

This uniqueness is what makes the decomposition useful as an analytical tool: there's a single, well-defined "`u`-component" of any vector.

### Generalization to subspaces

The same idea extends when `u` is replaced by a *subspace* `U` (spanned by multiple vectors):
```
x = proj_U(x) + x_perp     where x_perp ⊥ U
```

Now `proj_U(x)` is the closest point to `x` lying *in* `U` (the "shadow on the plane" instead of "on a line"), and `x_perp` is what's left over — orthogonal to *every* vector in `U`. The Pythagorean relation still holds.

This is the foundation of **least-squares regression**: fitting a linear model means projecting the target vector onto the column space of the design matrix. The residual is `x_perp` — what the model can't explain. Minimizing `‖x_perp‖²` is the OLS objective.

### Numerical example

Let `x = [3, 4]` and `u = [1, 0]` (the x-axis).
- `<x, u> = 3·1 + 4·0 = 3`
- `<u, u> = 1·1 + 0·0 = 1`
- `proj_u(x) = (3 / 1) · [1, 0] = [3, 0]`
- `x_perp = x - proj_u(x) = [3, 4] - [3, 0] = [0, 4]`

Verify:
- `<x_perp, u> = 0·1 + 4·0 = 0` ✓ (perpendicular)
- `proj_u(x) + x_perp = [3, 0] + [0, 4] = [3, 4] = x` ✓ (reconstructs `x`)
- `‖x‖² = 3² + 4² = 25`,  `‖proj_u(x)‖² = 9`,  `‖x_perp‖² = 16`,  sum = 25 ✓ (Pythagoras)

### Geometric picture

```
          x
          •
         ╱│
        ╱ │
       ╱  │
      ╱   │  x_perp  (perpendicular drop)
     ╱    │
    ╱     │
   •──────•──────────►  u-line
  origin  ↑
        proj_u(x)  (foot of the perpendicular)
```

`x_perp` is the perpendicular drop from `x` to the line; `proj_u(x)` is its foot on the line. Together they reconstruct `x`.

### Why this matters in DL

The decomposition is the workhorse of any operation that asks "what's `x`'s component along — or perpendicular to — some direction `d`?"

- **Ablation in interpretability**: to test whether direction `d` is causally responsible for some behavior, replace `x` with `x_perp` (subtract its `d`-component) and observe whether behavior changes. If yes, `d` mattered.
- **Steering vectors / activation engineering**: adding a direction `d` to the residual stream increases `x`'s `d`-component, pushing generation along the axis encoded by `d` (style, sentiment, topic, etc.).
- **Gradient surgery in multi-task learning**: when two tasks' gradients conflict (negative dot product), project one gradient onto the orthogonal complement of the other before the optimizer step. The decomposition removes the conflicting component, letting both tasks make progress.
- **Sparse autoencoders**: the SAE reconstruction is a projection onto the span of currently-active features; the SAE's reconstruction loss is essentially `‖x_perp‖²` — the part of `x` not explainable by the sparse code.
- **Least-squares / linear regression**: the optimal predictor `ŷ` is the projection of `y` onto the design matrix's column space; the regression residual is `y - ŷ = x_perp`.

Whenever you hear "remove the X direction," "ablate this feature," "the residual after Y," or "project out the Z component" — under the hood it's orthogonal decomposition.

## Orthogonal matrices

A square matrix `Q ∈ R^(n × n)` is **orthogonal** if:
```
Q^T Q = I    (equivalently:  Q^(-1) = Q^T)
```

This single condition is extremely restrictive — it's what makes orthogonal matrices the "rigid motion" matrices of linear algebra.

### What `Q^T Q = I` actually says

Compute `Q^T Q` entry-wise. The `(i, j)` entry is the inner product of the `i`-th and `j`-th columns of `Q`:
```
(Q^T Q)_{ij} = <q_i, q_j>
```

For this to equal the identity:
- Diagonal entries `<q_i, q_i> = ‖q_i‖² = 1` → each column is **unit length**.
- Off-diagonal entries `<q_i, q_j> = 0` for `i ≠ j` → distinct columns are **mutually perpendicular**.

Together: the columns of `Q` form an **orthonormal basis** for `R^n`. The matrix-form condition `Q^T Q = I` is just the compact way of saying "columns are orthonormal."

### The three key properties

| Property | Statement | Meaning |
|---|---|---|
| **Length-preserving** | `‖Qx‖ = ‖x‖` | Doesn't stretch or shrink any vector |
| **Inner-product-preserving** | `<Qx, Qy> = <x, y>` | Doesn't change angles between vectors |
| **Inverse = transpose** | `Q^(-1) = Q^T` | Inverse is "free" — no computation needed |

All three follow directly from `Q^T Q = I`:

```
‖Qx‖² = (Qx)^T (Qx) = x^T Q^T Q x = x^T I x = ‖x‖²

<Qx, Qy> = (Qx)^T (Qy) = x^T Q^T Q y = x^T I y = <x, y>

Q^T Q = I  ⟺  Q^T = Q^(-1)
```

Because they preserve **both** lengths and angles, orthogonal matrices preserve all geometric structure — they're the **rigid motions** of `R^n`.

### Geometric meaning: rotations and reflections

A rigid motion of Euclidean space (one that fixes the origin) is exactly an orthogonal transformation. Two flavors, distinguished by determinant:

- **`det(Q) = +1`**: pure **rotation**. Preserves orientation (right-handed coordinate frames stay right-handed). The set of these matrices forms the *special orthogonal group* `SO(n)`.
- **`det(Q) = -1`**: rotation **with a reflection** (mirror flip). Flips orientation (right-handed → left-handed).

Both are "rigid" — preserve lengths and angles. They differ only in whether they preserve handedness.

The determinant of any orthogonal matrix is always exactly `±1`: from `det(Q^T) · det(Q) = det(Q^T Q) = det(I) = 1` and `det(Q^T) = det(Q)`, you get `det(Q)² = 1`.

### Square implies both directions

For a *square* orthogonal matrix, `Q^T Q = I` automatically implies `Q Q^T = I` — so both columns *and* rows are orthonormal. (If `Q^T Q = I`, then `Q^T` is the unique inverse of `Q`, so multiplying on the other side gives the same identity.)

For a **rectangular** matrix with orthonormal columns (`Q ∈ R^(m × n)` with `m > n`), only `Q^T Q = I_n` holds — `Q Q^T` is *not* the identity. These are called **semi-orthogonal** or **column-orthonormal**. They preserve lengths but aren't invertible. Used in: SVD's `U` matrix for tall input matrices; truncated PCA bases; orthogonal projections onto subspaces.

### Special examples

- **Identity `I`**: trivially orthogonal; `det = +1`.
- **2D rotation by θ**: `R_θ = [[cos θ, -sin θ], [sin θ, cos θ]]`; `det = +1`.
- **Reflection across x-axis**: `[[1, 0], [0, -1]]`; `det = -1`.
- **Permutation matrices** (each column is a unit basis vector with the rest zero): orthogonal — they just rearrange coordinates. So *swapping or shuffling axes* counts as a rigid motion.
- **Householder reflection**: `H = I - 2 v v^T` for unit vector `v` — reflects across the hyperplane perpendicular to `v`. Orthogonal, `det = -1`. Building block of QR decomposition.
- **DFT matrix** (after normalization): the matrix performing the discrete Fourier transform is unitary in the complex case and orthogonal up to scaling in the real-valued analogues.

### Why this matters in DL

- **RoPE (Rotary Position Embedding)** is a block-diagonal orthogonal matrix that rotates each pair of dimensions of `Q` and `K` by a position-dependent angle. Because it's orthogonal, it preserves the norms of `Q` and `K` *exactly* — it can't blow them up or shrink them. Position information is encoded purely as a phase shift that affects the dot product `<q, k>`, not as a magnitude change. The orthogonality is what makes RoPE composable across many positions without numerical drift.

- **Orthogonal initialization** (used in some RNNs and recent Transformer variants like nGPT): initialize weight matrices as random orthogonal matrices. Because `Q` preserves norms, gradients flowing through `Q` neither explode nor vanish — the Jacobian's spectral norm is exactly 1. This stabilizes training, especially for deep recurrent models with long credit-assignment paths.

- **SVD's `U` and `V` are orthogonal.** Every linear map factors through two orthogonal transformations and a diagonal scaling. Those two orthogonals are the "rotate into the right coordinate system" rotations from file 05's three-step picture.

- **Group-equivariant networks**: build models whose layers commute with a group of orthogonal transformations (rotations, reflections). Spherical CNNs use `SO(3)` equivariance; molecular property predictors use `O(3)`. The orthogonal-matrix constraint enables the equivariance machinery.

- **Random projections for proofs (Johnson-Lindenstrauss)**: random orthogonal matrices preserve pairwise distances on average. Used in dimensionality reduction analyses and concentration-of-measure arguments.

- **Spectral normalization / weight-orthogonalization**: some training-stability work explicitly constrains or regularizes weights to be (near-)orthogonal. The motivation: condition number = 1, gradients flow cleanly, training is stable even at large scale or in low-precision arithmetic.

- **Spherical / unit-norm embeddings** (CLIP, some retrieval models): the natural transformations on a unit sphere are orthogonal matrices — they keep you on the sphere. The math of orthogonal transformations is the underlying language for any "embedding lives on a sphere" architecture.

### One-line summary

> Orthogonal matrices are the **rigid motions** of `R^n`: they preserve all lengths, angles, and inner products. The defining condition `Q^T Q = I` says "columns are orthonormal," and from it you get length preservation, angle preservation, and a free inverse (`Q^(-1) = Q^T`) all at once.

## The residual stream as a vector space

Modern interpretability views the Transformer's hidden state (residual stream) as a `d_model`-dimensional vector space that **all the model's components share**. Every attention head and MLP reads from it, transforms what it reads, and writes back. Orthogonality is what lets many components share this one workspace without stepping on each other.

> **Deeper intuition**: where the name comes from (ResNets), why "stream," what makes the workspace metaphor so different from older deep-learning thinking, and why this single insight clarifies superposition, gradient flow, steering vectors, and circuits analysis at once: see [`supplementary/06_residual_stream.md`](supplementary/06_residual_stream.md). If you only carry one mental model from Part 1 into the Transformer-internals work later, make it this one.

### What the residual stream is

Each Transformer block updates a running sum:
```
h_{l+1} = h_l + attention_l(LN(h_l)) + mlp_l(LN(h_l))
```

Because the residual is *added* (not replaced), every layer's output is a *delta* on top of what's already there. The stream `h` is the shared communication channel that runs through the entire forward pass — it starts at the input embedding and accumulates contributions all the way to the unembed.

### Reading and writing

Each component interacts with the stream via two kinds of linear maps:

- **Read**: extract a subspace via an input projection. Attention's `W_Q`, `W_K`, `W_V` are reads — they ask "which directions of the stream define my queries / keys / values?" The MLP's first weight matrix is also a read.
- **Write**: compute a result and add it back via an output projection. Attention's `W_O` and the MLP's output matrix are writes.

A component's role can be summarized as: *read from a subspace, compute, write to a subspace.* The whole model is a network of read/compute/write loops sharing one high-dim stream.

### Why orthogonality lets components coexist

If component A writes along direction `d_A` and component B reads from a different direction `d_B` that's nearly orthogonal to `d_A`, then `<d_A, d_B> ≈ 0` and B effectively doesn't see A's contribution. With `d_model = 4096` (or larger) and random subspaces near-orthogonal in high dim — see the orthogonality section above — **many components can read/write in "their own" subspaces with minimal cross-talk**. The workspace is large enough that every component can have its own corner.

This is the **superposition** mechanism in action: many features pack into one residual stream because their read/write directions are nearly orthogonal. The geometric capacity argument from earlier in this file is what makes the workspace metaphor mathematically viable.

### What this view enables in interpretability

Treating the residual stream as a vector space lets you ask sharp, falsifiable questions:

- **Which direction encodes feature X?** Find a unit vector `d` such that `<h, d>` correlates with feature X being present. Foundation of linear probes, sparse autoencoders, and steering vectors.
- **How does information flow?** Track which heads write to which directions and which later components read them. This is "**circuits**" analysis — a directed graph of write/read pairs.
- **Is a component causally responsible?** Project out its writing direction (replace `h` with `h_perp`) and see if downstream behavior changes.
- **Can the model be steered?** Add a vector along a target direction; the model treats it as if that direction had been written by some upstream component.

Projections and orthogonality aren't abstract — they're the literal arithmetic of how Transformers move information between components.

## Self-check

1. Given `u = [1, 1, 0]`, compute `proj_u([3, 1, 2])` by hand.
2. Why does RoPE preserve the `<q, k>` dot product when `q` and `k` are at the same position, but modify it when they're at different positions?
3. Two random unit vectors in R^12288 (GPT-3 d_model). What's the std of their dot product? Why does this enable "superposition" of features?

### Answers

1. `proj_u(x) = (<x, u> / <u, u>) · u`. Compute: `<x, u> = 3·1 + 1·1 + 2·0 = 4`. `<u, u> = 1 + 1 + 0 = 2`. Scalar coefficient: `4/2 = 2`. Result: `2 · [1, 1, 0] = [2, 2, 0]`.
2. RoPE applies a rotation `R_p` (orthogonal matrix) to a vector at position `p`. Same position: `<R_p q, R_p k> = q^T R_p^T R_p k = q^T I k = <q, k>` since `R^T R = I` for orthogonal matrices. Different positions: `<R_p q, R_q k> = q^T R_p^T R_q k = q^T R_{q-p} k` — depends only on the **position difference** `q - p`. So RoPE encodes relative position purely as a phase rotation in the dot product, with the magic property of being relative without any explicit "relative position embedding" lookup.
3. **Std ≈ `1/√12288 ≈ 0.009`** (for unit vectors in dim `d`, the dot product has std `1/√d`). Random vectors are nearly orthogonal — typical cos sim ~0.01. **Superposition**: this near-orthogonality means many features can share the same residual stream without much cross-talk. Each feature lives along its own (nearly orthogonal) direction; a layer can write to one direction without significantly perturbing others. So a `d`-dim space can encode many more than `d` distinguishable features as long as they're sparsely active. Anthropic's interpretability work formalizes this as the "superposition hypothesis."

## Exercise

In PyTorch, generate 10000 random unit vectors in R^128 and R^4096. Plot the histogram of pairwise cosine similarities. Verify the std matches the `1/√d` prediction. Understand *viscerally* that high-dim space is mostly orthogonal.
