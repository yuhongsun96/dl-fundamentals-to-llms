# Eigenvalues, Eigenvectors, and SVD

## Eigenstuff in one paragraph

**The big idea**: an eigenbasis is the coordinate system in which a matrix's action reduces to independent per-axis stretching. In the standard basis a matrix looks like opaque coordinate-mixing; in the eigenbasis the same transformation is just `n` separate scalings, one per axis, by its eigenvalue. Eigenvalues and eigenvectors are the matrix's "native viewing angle" — the perspective from which it stops looking complicated.

For a square matrix `A`, an eigenvector `v` satisfies `A v = λ v`. The matrix stretches `v` by `λ` without rotating it. A diagonalizable matrix can be written `A = V Λ V^(-1)` — its action is "rotate into eigenbasis, scale each axis, rotate back."

### A bit more detail

`A = V Λ V^(-1)` factors the action of `A` on any input `x` into three geometric steps:

1. **`V^(-1) x`** — change basis. Express `x` in eigenvector coordinates: find the scalars `c_i` such that `x = Σ c_i v_i`. The output of this step is the column vector `(c_1, ..., c_n)`.
2. **`Λ`** — scale each coordinate by its eigenvalue: `(c_1, ..., c_n) → (λ_1 c_1, ..., λ_n c_n)`. Pure axis-aligned stretching.
3. **`V`** — change back to the standard basis: rebuild the output vector as `Σ λ_i c_i v_i`.

So multiplying by `A` decomposes any input along the eigenvectors, scales each component independently, and sums back up. Once you're in the right coordinate system, the matrix is just a diagonal scaling — no twisting.

When `A` is symmetric, `V` is orthogonal, so step 1 is a true rotation (no shear) and `V^(-1) = V^T`. The picture is then literally: rotate axes to align with eigenvectors, stretch independently along each, rotate back. For non-symmetric matrices the eigenvectors aren't orthogonal — the "basis change" is a more general affine transform — but the three-step factorization still holds.

### Worked example

`A = [[2, 1], [1, 2]]` is symmetric.

- Eigenvalues: solve `det(A - λI) = (2-λ)² - 1 = 0` → `λ_1 = 3`, `λ_2 = 1`.
- Eigenvectors: `v_1 = (1, 1)/√2` (for `λ = 3`), `v_2 = (1, -1)/√2` (for `λ = 1`). Orthogonal because `A` is symmetric.

So `A` stretches the `(1, 1)` direction by 3×, and leaves the `(1, -1)` direction unchanged. Let's apply `A` to `x = (1, 0)` via the three-step recipe:

1. **Decompose**: `(1, 0) = ½ · (1, 1) + ½ · (1, -1)`. Eigen-coordinates `(c_1, c_2) = (½, ½)` (modulo a `√2` from the unit-length normalization — keeping it loose for clarity).
2. **Scale**: `(c_1, c_2) → (λ_1 c_1, λ_2 c_2) = (3·½, 1·½) = (1.5, 0.5)`.
3. **Recombine**: `1.5 · (1, 1) + 0.5 · (1, -1) = (2, 1)`.

Verify directly: `A x = [[2,1],[1,2]] · (1, 0)^T = (2, 1)^T`. ✓

The big idea: matrix multiplication looks like opaque coordinate-mixing in the standard basis, but in the eigenbasis it's just `n` independent stretches. That's why eigenvalues capture so much about a matrix's behavior — they're the "true" scaling factors once you find the right axes.

**Why you should care, minimally:**
- The spectrum of the Hessian describes local loss curvature — eigenvalues = curvature in each direction.
- "Sharp" minima have large max eigenvalues; "flat" minima have small ones.
- In RNNs, eigenvalues of the recurrent weight matrix determine whether gradients explode (|λ| > 1) or vanish (|λ| < 1) over long sequences — this is *why* LSTMs exist.

You don't need to compute eigenvalues by hand. You need the *vocabulary* to read papers.

## The determinant — signed volume scaling

A square matrix has one more single-number summary worth knowing alongside its eigenvalues: the **determinant**. It captures, in one scalar, how the linear transformation stretches (or collapses) `n`-dimensional volume.

### The geometric picture

Take the unit square in 2D and apply a matrix `A` to it. The square's image is a **parallelogram**. The determinant is the (signed) **area** of that parallelogram.

```
  unit square         applied A=[[2,0],[0,3]]      applied A=[[1,2],[2,4]]
  ┌──┐                ┌────────┐                   ───────────
  │  │       →        │        │       →           (collapsed to a line)
  │  │                │        │
  └──┘                │        │
                      │        │
                      └────────┘
  area = 1            area = 6                     area = 0
                      |det| = 6                    |det| = 0
```

In 3D, same picture but with the unit cube → parallelepiped, and `|det|` is its **volume**. In `n` dimensions, the unit hypercube maps to an `n`-dim parallelotope whose `n`-volume is `|det(A)|`.

### What the sign tells you

Determinant is *signed*: it can be positive, negative, or zero.

- `det > 0` — the transformation **preserves orientation**. Right-handed coordinate frames stay right-handed; in 2D, counterclockwise loops stay counterclockwise.
- `det < 0` — orientation is **flipped**. Reflections (mirror flips) have negative determinant. The classic 2D swap matrix `[[0, 1], [1, 0]]` has `det = -1`: same area, but the plane gets reflected across the diagonal.
- `det = 0` — the transformation **collapses** space into a lower-dim subspace. Volume is destroyed; some directions get squashed to nothing.

### Why `det = 0` means "non-invertible"

If a matrix flattens `n`-dim space onto an `(n-1)`-dim subspace (or lower), multiple input points get mapped to the same output. There's no way to recover which input you started from — information is lost. So the matrix can't be inverted: it has no `A^{-1}`.

Equivalent ways of saying the same thing:
- `det(A) = 0`
- `A` is **singular** (the technical word for "non-invertible")
- The columns of `A` are linearly dependent
- `A` has rank `< n`
- At least one eigenvalue of `A` is `0`
- The map collapses some direction in space to zero

These are all the same statement viewed from different angles. SVD makes the geometry precise: a singular value of zero is exactly a "collapsed direction."

### Connection to eigenvalues

Since `A` scales eigenvector `v_i` by `λ_i`, the *total* volume scaling is just the product of those scalings. Concretely:

```
det(A) = λ_1 · λ_2 · ... · λ_n
```

The determinant is the product of all eigenvalues. If any single eigenvalue is `0`, the product is `0`, and `A` has flattened space along that eigendirection. This is why "all eigenvalues nonzero" and "determinant nonzero" and "matrix invertible" are the same condition.

### Why only square matrices have determinants

It only makes sense to ask "how does the transformation scale `n`-dim volume?" when the input and output spaces have the same dimension `n`. A rectangular matrix `R^n → R^m` (with `m ≠ n`) is mapping between spaces where "volume" means different things — there's no single scaling number that captures it. (For rectangular matrices, the right generalization is the singular value spectrum — see SVD next.)

### A few examples to anchor the picture

- **Pure scaling** `[[2, 0], [0, 3]]`: stretches x-axis by 2, y-axis by 3. Unit square → 2×3 rectangle. `det = 6`. Areas multiplied by 6.
- **Pure rotation** (e.g., 30° rotation): rigid motion, `det = 1`. Areas preserved, orientation preserved.
- **Reflection** `[[0, 1], [1, 0]]`: swaps axes — equivalently a flip across the line `y = x`. `det = -1`. Areas preserved, orientation flipped.
- **Shear** `[[1, 0.5], [0, 1]]`: tilts vertical edges, leaves horizontal edges alone. Parallelogram has the same base and height as the square, so `det = 1`. Areas preserved despite obvious deformation.
- **Singular** `[[1, 2], [2, 4]]`: rows (or columns) are proportional. The whole plane collapses onto the line through `(1, 2)`. `det = 0`. Information lost.

### Where it shows up in DL

- **Normalizing flows** (generative models): the change-of-variables formula for probability densities involves `|det(Jacobian)|`. Designing invertible neural network layers with cheap-to-compute Jacobian determinants is the central engineering problem.
- **Theoretical analyses** of weight matrix conditioning, stability, and certain interpretability arguments.
- **Rarely** in routine training code — you won't compute determinants in a typical loss function. But the underlying concept (does this matrix collapse space?) shows up via rank, condition number, and singular value analyses.

The one-line takeaway: **determinant = signed `n`-volume scaling factor of a square matrix's linear transformation, equal to the product of its eigenvalues**.

## SVD — the one decomposition that matters

**What SVD really says**: for any real matrix — square or rectangular, full-rank or not — there exists a pair of orthonormal bases (one for the input space, one for the output space) in which the transformation reduces to pure 1-to-1 axis-to-axis scaling. Each input axis maps to exactly one output axis (scaled by some `σ_i ≥ 0`), or maps to zero. **No input axis is ever split across multiple outputs; no output axis is ever assembled from multiple inputs.** The apparent coordinate-mixing of `A x` in the standard basis is purely an artifact of using the wrong coordinate system — SVD finds the right pair.

**The big idea**: SVD finds an orthonormal basis for the **input** space *and* an orthonormal basis for the **output** space such that the linear map becomes pure independent stretching between matched axes. Said differently: SVD finds the most natural input coordinates and the most natural output coordinates such that the map is as simple as possible — just per-axis scaling between the two.

The three pieces of `A = U Σ V^T` each play a clean role:
- **`V`** — the important modes of variation in the **input** space (an orthonormal basis of `R^n`).
- **`U`** — the corresponding modes in the **output** space (an orthonormal basis of `R^m`).
- **`Σ`** — the **coupling strength** between matched modes. `σ_i` says how much of input mode `v_i` gets transmitted to output mode `u_i`.

The eigenstuff above does the same trick for square matrices using a *single* basis. SVD generalizes by allowing two different bases, which lets it work for any matrix and decouple the input-side and output-side rotations.

Every real matrix `A ∈ R^(m×n)` has the decomposition:
```
A = U Σ V^T
```
where:
- `U ∈ R^(m×m)` orthogonal — columns `u_1, ..., u_m` are the **left singular vectors**, an orthonormal basis for the output space `R^m`.
- `V ∈ R^(n×n)` orthogonal — columns `v_1, ..., v_n` are the **right singular vectors**, an orthonormal basis for the input space `R^n`.
- `Σ ∈ R^(m×n)` diagonal, non-negative, sorted descending — entries `σ_1 ≥ σ_2 ≥ ... ≥ 0` are the **singular values**.

### What is a singular vector?

A **singular vector** is the SVD analogue of an eigenvector — a special direction along which the matrix acts as pure scaling — adapted for the case where input and output spaces can differ.

Where an eigenvector `v` satisfies `A v = λ v` (output is a scaled copy of the *same* input vector, in the *same* space), a **singular vector pair** consists of:
- a direction `v` in the **input** space (`R^n`),
- a direction `u` in the **output** space (`R^m`),
- tied together by a single non-negative gain `σ ≥ 0`,

such that:
```
A v = σ u         (input direction maps cleanly to one output direction, scaled by σ)
A^T u = σ v       (the reverse map recovers v from u, with the same gain)
```

Because the input and output live in *different* spaces (whenever `m ≠ n`), one vector isn't enough to describe a "clean direction" — you need a pair: the input-side direction and the output-side direction it lands on. The two halves of the pair get separate names:

- **Right singular vector** `v` — the input-space half.
- **Left singular vector** `u` — the output-space half.
- **Singular value** `σ` — the shared gain.

When `A` is square and symmetric PSD, the input and output spaces coincide, the pair collapses to a single direction, and singular vectors equal eigenvectors (`u_i = v_i`, `σ_i = λ_i`). For everything else they're genuinely different objects, and you need both halves of the pair.

### Why "left" and "right"?

The names track where each matrix sits in the product `A = U Σ V^T`:

- `U` is on the **left** of `Σ` → its columns `u_i` are the **left** singular vectors.
- `V^T` is on the **right** of `Σ` → the columns of `V` (i.e., the rows of `V^T`) are the **right** singular vectors `v_i`.

These two sets live in **different spaces** and play complementary roles:

- **Right singular vectors `v_i` ∈ R^n** — orthonormal directions in the **input** space. They're the "cleanly behaved" axes for `A`: each `v_i` gets mapped by `A` to a single output direction, scaled by exactly `σ_i`.
- **Left singular vectors `u_i` ∈ R^m** — orthonormal directions in the **output** space. The image of input direction `v_i` lies along the matched `u_i`.

The matched pairs are linked by:
```
A v_i = σ_i u_i
```
Reading: "input direction `v_i` becomes output direction `u_i`, scaled by `σ_i`." This is the core SVD identity — it pairs up an input direction, an output direction, and a gain factor into a single triple `(v_i, u_i, σ_i)`.

For a tall matrix (`m > n`): there are `n` such triples, and the extra `m - n` columns of `U` span the orthogonal complement of `A`'s image (i.e., output directions that `A` can never reach). For a wide matrix (`m < n`): there are `m` triples, and the extra `n - m` columns of `V` span `A`'s kernel (input directions that `A` squashes to zero). The math is symmetric; only the bookkeeping shifts.

### Where they come from: eigenvectors of `A^T A` and `A A^T`

A useful handle for computation and intuition:

- The **right** singular vectors are the eigenvectors of the `n × n` matrix `A^T A` (always symmetric PSD).
- The **left** singular vectors are the eigenvectors of the `m × m` matrix `A A^T` (always symmetric PSD).
- The eigenvalues of both are `σ_i²` — the squared singular values.

This is why `σ_i ≥ 0`: they're square roots of eigenvalues of a PSD matrix. It's also the standard recipe for computing SVD: form `A^T A` (or `A A^T`, whichever is smaller), eigendecompose it, take square roots of the eigenvalues to get singular values, and the eigenvectors give you `V` (or `U`).

### The three-step picture

SVD factors `A` into the same kind of recipe as the eigendecomposition, but with **two different orthogonal bases** because `A` can map between different spaces (`R^n → R^m`):

1. **`V^T x`** — rotate the input into the right-singular-vector basis (`v_1, ..., v_n`). Pure rotation/reflection in `R^n`.
2. **`Σ`** — scale each axis by its singular value `σ_i`, and convert from an `n`-dim vector to an `m`-dim vector (extra dims padded with zeros, or trimmed away).
3. **`U`** — rotate into the left-singular-vector basis in `R^m`.

The whole point: `A` takes the unit vector `v_i` to `σ_i · u_i`. So `σ_i` is the **gain factor** connecting input direction `v_i` to output direction `u_i`. Where eigendecomposition asks "which directions does `A` only stretch?", SVD asks "which input direction does `A` map to which output direction, and by how much?" — a richer question that always has an answer.

### Rank, kernel, and "how much lives in each direction"

The number of nonzero `σ_i` equals the **rank** of `A`. Input directions with `σ_i = 0` get squashed to zero (they're in the **kernel** / null space of `A`). The largest singular values capture the dominant structure of `A`; tiny ones contribute small corrections.

Equivalent expansion as a sum of rank-1 outer products:
```
A = Σ σ_i u_i v_i^T
```

This sum-of-rank-1 form is the foundation of the next section: truncating it at the top `k` terms gives the **best rank-`k` approximation** of `A`.

### Eigendecomposition vs SVD, in one table

|  | Eigen `V Λ V^(-1)` | SVD `U Σ V^T` |
|---|---|---|
| Works on | square diagonalizable matrices only | **any** real matrix (square or rectangular) |
| Bases | one (eigenvectors) | two (left and right singular vectors) |
| Scaling | eigenvalues `λ_i` (can be negative or complex) | singular values `σ_i ≥ 0` (always real, non-negative) |
| Always exists | no (defective matrices have none) | yes |

For **symmetric positive semi-definite** matrices the two coincide: `σ_i = λ_i`, and the columns of `U` and `V` are both the orthogonal eigenvectors. So SVD is the natural generalization of eigendecomposition that drops the square-and-diagonalizable requirement and replaces signed scaling with magnitudes plus a separate output-side rotation.

### Worked example

Take this 2×3 matrix (maps `R^3 → R^2`):
```
A = ┌ 1  0  1 ┐
    └ 0  1  0 ┘
```

**Step 1: form `A^T A`** (a 3×3 matrix in the input space):
```
A^T A = ┌ 1  0  1 ┐
        │ 0  1  0 │
        └ 1  0  1 ┘
```

**Step 2: eigendecompose `A^T A`.** Its eigenvalues are `λ = 2, 1, 0`, so the singular values are `σ_i = √λ_i`:
```
σ_1 = √2,   σ_2 = 1,   σ_3 = 0
```

The eigenvectors of `A^T A` (orthonormalized) are the right singular vectors:
```
v_1 = (1, 0, 1) / √2     ← stretched the most (σ_1 = √2)
v_2 = (0, 1, 0)          ← stretched by 1 (σ_2 = 1)
v_3 = (1, 0, -1) / √2    ← in the kernel (σ_3 = 0)
```

**Step 3: get the matched output directions** via `u_i = A v_i / σ_i`:
- `A v_1 = (1·(1/√2) + 0 + 1·(1/√2), 0 + 0 + 0) = (√2, 0)`. Divide by `σ_1 = √2`: `u_1 = (1, 0)`.
- `A v_2 = (0, 1)`. Divide by `σ_2 = 1`: `u_2 = (0, 1)`.
- `A v_3 = (1·(1/√2) + 0 + 1·(-1/√2), 0) = (0, 0)`. Kernel direction — squashed to zero, no paired `u`.

**Putting it all together:**
```
U = ┌ 1  0 ┐         Σ = ┌ √2   0   0 ┐
    └ 0  1 ┘             └ 0    1   0 ┘

      ┌ 1/√2   0   1/√2 ┐
V^T = │  0     1    0   │
      └ 1/√2   0  -1/√2 ┘
```

**Verify** `U Σ V^T = A`:
```
Σ V^T = ┌ √2 · 1/√2   0    √2 · 1/√2 ┐ = ┌ 1  0  1 ┐
        └    0        1       0       ┘   └ 0  1  0 ┘

U · (Σ V^T) = I · ┌ 1  0  1 ┐ = ┌ 1  0  1 ┐  = A   ✓
                  └ 0  1  0 ┘   └ 0  1  0 ┘
```

**What the SVD tells you about this `A`:**
- The diagonal direction `v_1 = (1, 0, 1)/√2` is the "principal" input — anything along it gets amplified by `√2` and emitted as `u_1 = (1, 0)`.
- The middle axis `v_2 = (0, 1, 0)` passes through unchanged to `u_2 = (0, 1)` — gain factor exactly 1.
- The anti-diagonal `v_3 = (1, 0, -1)/√2` is in the **kernel**: any input along this direction gets squashed to zero. This is the geometric expression of "row 1 minus row 3 = 0" being baked into `A`.

Three input directions, two of them produce nonzero output → `rank(A) = 2`. Two (`v_1, v_2`) get matched with output directions `u_1, u_2` and gains `σ_1, σ_2`; one (`v_3`) is in the kernel with `σ_3 = 0`. The full 3D input space cleanly decomposes into "the part `A` cares about" and "the part `A` ignores."

## SVD as a recipe for low-rank approximation

**Eckart-Young theorem**: the best rank-`k` approximation of `A` (in Frobenius or spectral norm) is:
```
A_k = U[:, :k] Σ[:k, :k] V[:, :k]^T
```

"Best" needs a way to measure error — *how far* is the approximation from the original? Two natural choices for matrix error `E = A - A_k`:

- **Frobenius norm** `‖E‖_F = √(Σ_{ij} E_{ij}²) = √(Σ σ_i(E)²)`: total squared deviation across all entries. Minimizes **total error**.
- **Spectral norm** `‖E‖_2 = σ_max(E)`: largest singular value of the error. Minimizes **worst-case error** along any single direction.

These typically pull in different directions (low total vs. low worst-case). The remarkable part of Eckart-Young: the same SVD truncation minimizes **both simultaneously**:

- `‖A - A_k‖_F² = σ_{k+1}² + σ_{k+2}² + ...` (sum of squared *discarded* singular values)
- `‖A - A_k‖_2 = σ_{k+1}` (the largest discarded singular value)

So you don't have to pick which kind of error you care about — top-`k` truncation is optimal for both. (For other matrix norms, the optimal rank-`k` approximation can differ.) See `03_inner_products_norms.md` for a fuller treatment of these norms and their cousins.


Just truncate to the top `k` singular components. This is **optimal** — no other rank-`k` matrix is closer.

This is the mathematical foundation of:
- LoRA (approximating weight deltas with low rank)
- PCA (SVD of centered data)
- Matrix factorization in general
- Weight compression / pruning
- Some embedding analyses (looking at the singular value spectrum of the embedding matrix)

## Condition number

```
κ(A) = σ_max / σ_min
```

The ratio of the biggest singular value to the smallest. It measures **how distorted the transformation is** — equivalently, how much `A` cares about some directions vs. others.

### Geometric picture

A matrix maps the unit sphere to an ellipsoid. `σ_max` is the longest ellipsoid axis; `σ_min` is the shortest. The condition number is the ratio of those two axes:

- `κ = 1`: ellipsoid is a sphere → `A` is essentially a rotation (equal stretch in every direction).
- `κ = 10`: one direction stretched 10× more than another. Moderate.
- `κ = 1000`: severely anisotropic — most directions barely move while one is hugely amplified.
- `κ = ∞`: some `σ_i = 0` → `A` collapses a direction to zero (singular).

### What it predicts

**Error amplification**. When you "undo" `A` (e.g., solve `Ax = b`), a small input perturbation `δb` can blow up into an output perturbation as large as `κ · δb`. Rule of thumb: you lose about `log₁₀(κ)` digits of precision. `κ = 10⁶` in fp32 (~7 digits) leaves you with one usable digit.

### Why it matters in DL

- **Optimization speed.** Large κ means gradients flow well in some directions and poorly in others; one learning rate can't balance both. Ill-conditioned losses train slowly under vanilla SGD.
- **Why Adam helps.** Adam, RMSprop, etc. estimate per-parameter gradient scale and divide it out — they effectively precondition the loss to look like `κ = 1`. This is why adaptive optimizers train on problems where SGD stalls.
- **Numerical stability at scale.** In bf16/fp16, large κ amplifies precision errors. Loss spikes in large-model training often correlate with runaway κ on some weight matrix.
- **Attention instability.** If Q or K projections develop large κ, attention scores become scale-variable across rows — some heads saturate while others stay soft. This is part of why QK-norm was introduced for very large models (see `supplementary/03_scaling_and_saturation.md`).

### One-line summary

> κ summarizes "how anisotropic is this matrix?" — the ratio of biggest stretch to smallest stretch. Small κ = well-behaved; large κ = numerically dangerous and optimization-hostile.

## PCA in one breath

Stack your data in `X ∈ R^(n×d)`, center it. SVD: `X = U Σ V^T`. The columns of `V` are principal components — orthogonal directions of maximum variance. Projecting onto the top `k` gives the best `k`-dim linear embedding.

You see this in analyses of learned embeddings ("what does the top PC of BERT's CLS token encode?").

## What not to waste time on

- Memorizing eigen-decomposition algorithms.
- Jordan normal form.
- Anything involving hand-computing eigenvectors of >3×3 matrices.

You need *reading fluency*, not *computational fluency*.

## Self-check

1. If a matrix has singular values `[10, 5, 1, 0.01]`, what's its effective rank? What does the condition number tell you?
2. Why does PCA work on the covariance matrix `X^T X` — what's the relationship between its eigendecomposition and the SVD of `X`?
3. In the RNN exploding-gradient analysis, why does the spectral radius (max |eigenvalue|) of the recurrent matrix matter more than its norm?

### Answers

1. **Effective rank ~3**: the first three singular values are O(1+); the fourth is two orders of magnitude smaller, contributing negligibly. The matrix is "essentially rank-3 with a tiny rank-4 perturbation." **Condition number** `κ = σ_max/σ_min = 10/0.01 = 1000`. Moderately ill-conditioned: solving linear systems with this matrix loses roughly `log10(1000) = 3` digits of precision. Not catastrophic, but small `σ` directions are unreliable.
2. SVD: `X = U Σ V^T`. So `X^T X = V Σ U^T U Σ V^T = V Σ² V^T` (using `U^T U = I` since `U` is orthogonal). This is the **eigendecomposition of `X^T X`**: eigenvectors are the columns of `V` (right singular vectors of `X`), eigenvalues are `σ_i²` (squared singular values). PCA's "top-k principal directions" = "top-k eigenvectors of the covariance" = "top-k right singular vectors of `X`." SVD and PCA are the same computation viewed two ways.
3. RNN backprop multiplies the recurrent matrix `W` once per timestep. After `T` steps the gradient is scaled by `W^T` (not `‖W‖^T`). For a diagonalizable `W = V Λ V^{-1}`, `W^T = V Λ^T V^{-1}` — the diagonal of `Λ^T` has entries `λ_i^T`, and asymptotic growth is dominated by the largest `|λ_i|` = spectral radius `ρ(W)`. If `ρ(W) > 1`, gradients explode exponentially; if `< 1`, they vanish. The operator norm `‖W‖₂ = σ_max(W)` can be much bigger than `ρ(W)` for non-normal matrices, but it doesn't compound the same way across many multiplications. For long-horizon dynamics, spectral radius rules.

## Exercise

Take an embedding matrix from a small pretrained model (e.g. `distilbert`'s token embeddings, ~30K × 768). Compute its singular value spectrum. Plot `σ_i` on a log scale. Observe a sharp drop from `σ_1` to `σ_2`, then a long, slowly-decaying tail. The shape — anisotropic at the top, fat-tailed in the middle — is the structural fingerprint of training, but (perhaps surprisingly) it does **not** make the matrix cleanly truncatable via SVD. "Anisotropic" and "low-rank" are different properties; the notebook below pulls them apart.

## Practice

Runnable walk-through — load distilbert, compute singular values, plot the spectrum, compare against a random Gaussian baseline, and quantify what the spectrum does and doesn't imply about compression: [`supplementary/05_embedding_spectrum.ipynb`](supplementary/05_embedding_spectrum.ipynb). Key counterintuitive result: σ_1 alone holds ~56% of the Frobenius energy, but rank-`k` truncation to 1% relative Frobenius error needs `k = 765` of `768` — the long tail refuses to be dropped. The notebook also discusses why Frobenius error is only a proxy for embedding *quality* (the geometric/inner-product structure that downstream code actually consumes is not what Frobenius measures).
