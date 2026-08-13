# Outer Products and Low-Rank Structure

## The outer product

For `u ∈ R^m`, `v ∈ R^n`:
```
u ⊗ v = u v^T ∈ R^(m×n)
```

Every entry is `(u v^T)[i, j] = u_i v_j`. The result is a **rank-1 matrix**: all its rows are scalar multiples of `v^T`, all its columns are scalar multiples of `u`.

### Why it's called "outer"

The name mirrors the inner product by tracking the matmul shape:
```
u v^T :  (m, 1) @ (1, n) → (m, n)        # matrix
```
The shared dim is `1` and sits on the **outside**; `m` and `n` are the "outer" dims and both survive into the output. Nothing collapses. Inner collapses a shared inner dim to a scalar; outer has no real inner dim to collapse, so both outer dims expand into a 2D pattern.

Same two vectors, opposite transpose placement:
- Transpose on the **left** (`u^T v`) → inner product → scalar (`d` is inside, summed).
- Transpose on the **right** (`u v^T`) → outer product → matrix (`1` is inside, nothing summed).

### Geometric picture

An outer product `u v^T` is a linear map with a specific, very constrained shape:

```
(u v^T) x = u (v^T x) = <v, x> · u
```

Two steps applied to any input `x`:
1. **Listen**: measure how much `x` aligns with `v` — that's the scalar `<v, x>`.
2. **Speak**: output that scalar, scaled along the direction `u`.

So `u v^T` is a "**listen on `v`, speak on `u`**" operator. It takes the infinite information in `x`, collapses it to one number (alignment with `v`), and re-expresses that number as a multiple of `u`. The output always lies along the line through `u`. Its image is 1D. That's what "rank 1" means geometrically: 1D range.

Equivalently, view the matrix itself as a multiplication table:

```
u = (2, 1)         v = (3, 4)

u v^T =  ┌ 2·3   2·4 ┐  =  ┌ 6  8 ┐
         └ 1·3   1·4 ┘     └ 3  4 ┘
```

Every row is `v^T` scaled by some entry of `u`. Every column is `u` scaled by some entry of `v`. The whole matrix is "all pairs of products of entries" — the tensor product of the two vectors written out as a 2D grid.

### Trace, briefly

The **trace** of a square matrix `A ∈ R^(n×n)` is the sum of its diagonal entries:
```
trace(A) = Σ_i A_{ii}
```
A single scalar summarizing the diagonal. Two useful properties:
- **Cyclic**: `trace(AB) = trace(BA)` (operands commute *inside* a trace, even though `AB ≠ BA` in general).
- **Linear**: `trace(A + B) = trace(A) + trace(B)`.

You'll see trace show up in covariance computations (`trace(Σ)` = total variance), Frobenius inner products (`<A, B>_F = trace(A^T B)`), and the duality below.

### The inner/outer duality

Same two vectors, two products related by a trace:

```
<u, v> = trace(u v^T)
```

The diagonal entries of `u v^T` are `u_i v_i`, so summing them gives `Σ u_i v_i = <u, v>`. So `<u, v>` is "what you get when you collapse the outer product to its trace." Expand vs. collapse, linked by the trace operator.

One-line versions:
- **Inner** = the number saying how much two directions agree.
- **Outer** = the matrix representing "project onto one direction, re-emit along another."

Neural nets use both constantly: inner products for similarity (attention scores, cosine sim), outer products for weight updates (`∂L/∂W = δ x^T`) and low-rank structure (LoRA, SVD).

## Rank, intuitively

The rank of a matrix is:
- The number of linearly independent rows (or columns — they're equal).
- The minimum number of rank-1 matrices you need to sum to reconstruct it.
- The number of nonzero singular values (see next file).

A full-rank `n×n` matrix has rank `n` and "uses" all `n²` degrees of freedom. A rank-`r` matrix really only has `~(m+n)r` degrees of freedom — it's secretly compact.

## Why this matters: LoRA

Fine-tuning updates a big weight matrix `W ∈ R^(d×d)` with some delta `ΔW`. The LoRA hypothesis: **`ΔW` is low-rank**, even though `W` isn't. So parameterize:

```
ΔW = B @ A       where B ∈ R^(d×r), A ∈ R^(r×d), r « d
```

You store `2dr` parameters instead of `d²`. For `d=4096, r=16`: 131K params instead of 16.7M — **128× fewer**.

At inference you can either:
- Apply as `(W + BA) @ x` (merged — same speed as base model)
- Apply as `W @ x + B @ (A @ x)` (un-merged — lets you swap adapters)

The un-merged form is exactly the associativity trick from the matmul file: you never form the `d×d` intermediate.

## Other places low-rank shows up

- **Embedding matrices**: often factorized (ALBERT did this explicitly).
- **Attention Q/K/V projections**: multi-head decomposes one big `d×d` matrix into `H` smaller `d × d/H` blocks — a structured low-rank decomposition.
- **Multi-head Latent Attention (MLA)** in DeepSeek: compresses KV into a low-rank latent space to shrink the KV cache.
- **Embedding compression** for edge deployment.

## The key insight

Large neural nets have lots of parameters, but the *functions they implement* and the *updates they need during fine-tuning* often live in a much smaller subspace. Low-rank parameterizations exploit this. Every efficiency technique in modern DL is some version of "find and exploit the low-dimensional structure."

## Self-check

1. What's the rank of `u v^T + u w^T` where `u, v, w` are distinct non-zero vectors?
2. If `W ∈ R^(d×d)` and `ΔW = BA` with `A ∈ R^(r×d)`, `B ∈ R^(d×r)`, what's the max possible rank of `ΔW`?
3. Why does LoRA typically initialize `A` with a Gaussian and `B` with zeros? (Hint: what is `ΔW` at step 0, and why does that matter?)

### Answers

1. **Rank 1.** Factor: `u v^T + u w^T = u (v + w)^T`. Still an outer product (assuming `v + w ≠ 0`), so rank-1. The two outer products share the same left vector `u`, so they live in the same 1D column space.
2. **Max rank `r`.** General fact: `rank(BA) ≤ min(rank(B), rank(A))`. Both `B` and `A` have at most rank `r` (limited by the shared dim), so `rank(BA) ≤ r`. This is the whole point of LoRA — restrict updates to a rank-`r` subspace.
3. At step 0: `ΔW = B · A = 0 · A = 0`. So **the model behaves exactly like the base model at init**, and fine-tuning starts from a smooth, known-good state. Without this (e.g., both `A` and `B` random), `ΔW` would be small but nonzero random noise — perturbing pretrained behavior before any optimization step has happened. The zero-init of `B` is a "safe start" guarantee. Once training begins, gradients flow into both `A` and `B`, and `B` becomes nonzero.

## Exercise

Take a random `100×100` matrix. Do SVD. Reconstruct using only the top `k` singular values for `k ∈ {1, 5, 20, 100}`. Compute Frobenius error vs. `k`. This is the foundation of understanding what "low-rank approximation" means in practice.

## Practice

Runnable walk-through of this exercise — plus a contrast between random Gaussian matrices (flat singular-value spectrum, slow error decay) and naturally near-low-rank matrices (sharp 'knee' in the spectrum, fast error decay), with the LoRA tie-in: [`supplementary/04_low_rank_approximation.ipynb`](supplementary/04_low_rank_approximation.ipynb).
