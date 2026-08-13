# Generators and One-Parameter Families

This file closes the argument the previous two set up: **why RoPE rotates in 2-D pairs, and why that isn't a design choice but a theorem.** The tools are the matrix exponential, skew-symmetric generators, and the idea of a one-parameter family — the last piece of vocabulary needed to read rotation-based positional encodings properly.

**Prerequisites:** [file 05](05_complex_eigenvalues_and_rotation_planes.md) (complex eigenvalues as rotation planes) and [file 06](06_groups_and_the_rotation_group.md) (`SO(n)`, abelian).

## A one-parameter family

A **one-parameter family** (or **one-parameter subgroup**) of transformations is a map `t ↦ R(t)` from a single real number into a group, satisfying

```
R(0) = I           starting at "do nothing"
R(s + t) = R(s)R(t)    advancing by s then t == advancing by s+t
```

The second line is the whole content: it's a **homomorphism from `(ℝ, +)` into the group** — addition of parameters becomes composition of transformations.

Why this is exactly what a positional encoding needs: **position is a single number.** You want "the transformation for position `i`" such that moving from `i` to `j` depends only on `j − i`. That's precisely `R(i)⁻¹R(j) = R(j − i)`, which follows immediately from the homomorphism property. Verified on RoPE's matrices: `R(5)·R(8) == R(13)` and `R(5)ᵀ·R(8) == R(3)`.

One immediate consequence, worth noticing because it constrains everything: **every one-parameter family is abelian.** `R(s)R(t) = R(s+t) = R(t+s) = R(t)R(s)`, since real numbers commute under addition. So even inside a non-abelian group like `SO(3)`, any one-parameter family you build lives in a commutative corner of it. The extra richness of `SO(3)` is *unreachable* when your input is one scalar — which is the first hint that 3-D buys nothing here.

## The matrix exponential and the generator

Every one-parameter family of matrices has the form

```
R(t) = exp(tA)          where   exp(M) = I + M + M²/2! + M³/3! + …
```

`A` is called the **generator** — the infinitesimal version of the transformation. Concretely `A = R′(0)`: the velocity at which the family leaves the identity. Knowing `A` determines the entire family, because `R(t) = exp(tA)` solves `dR/dt = A·R` with `R(0) = I`.

The intuition: `A` says "here is the instantaneous motion," and the exponential integrates that motion for `t` units of time. Small rotations compose into a big one.

## Which generators produce rotations? Skew-symmetric ones

A matrix is **skew-symmetric** (or *antisymmetric*) if

```
Aᵀ = −A
```

which forces zeros on the diagonal and makes the lower triangle the negated mirror of the upper.

**Claim: `exp(tA)` is a rotation for every `t` exactly when `A` is skew-symmetric.** The derivation is two lines — differentiate the orthogonality condition `R(t)ᵀR(t) = I` at `t = 0`:

```
d/dt [ RᵀR ] = R′ᵀR + RᵀR′  = 0
at t = 0 (R = I, R′ = A):      Aᵀ + A = 0      ⟹   Aᵀ = −A
```

So "stays orthogonal" and "generator is skew-symmetric" are the same condition. Verified:

| `n` | `Aᵀ = −A` | `RᵀR = I` | `det R` |
|---|---|---|---|
| 3 | True | True | 1.0000000000 |
| 5 | True | True | 1.0000000000 |
| 8 | True | True | 1.0000000000 |
| 4, **non-skew** `A` | — | **False** | — |

Note the determinant is always `+1`, never `−1`: the exponential moves continuously from `I`, and it can never jump to the reflection half of `O(n)` ([file 06](06_groups_and_the_rotation_group.md)). One-parameter families generate **rotations only**.

The set of skew-symmetric matrices is called the **Lie algebra** `so(n)` — the "infinitesimal version" of the group `SO(n)`. Its dimension is the count of free entries in the strict upper triangle, `n(n−1)/2`, which matches `dim SO(n)` exactly, as it must.

## The decomposition theorem — the punchline

Here's the fact that answers the RoPE question. A real skew-symmetric matrix has a **canonical form**: in some orthonormal basis it is block-diagonal, built from

```
2×2 blocks   [ 0   −θ_k ]        plus   zeros on the remaining diagonal entries
             [ θ_k   0  ]
```

You can read this off the eigenvalues, which for a skew-symmetric matrix are **purely imaginary and come in `±iθ` conjugate pairs** ([file 05](05_complex_eigenvalues_and_rotation_planes.md)) — one pair per rotating plane, plus a zero for each inert direction:

| `n` | imaginary parts of eigenvalues | structure |
|---|---|---|
| 4 | `±0.408i, ±2.949i` | **2 rotating planes** |
| 5 | `±2.558i, ±4.841i`, **0** | 2 planes **+ 1 inert axis** |
| 6 | `±0.086i, ±1.492i, ±4.020i` | **3 rotating planes** |
| 7 | `±0.946i, ±2.281i, ±5.413i`, **0** | 3 planes **+ 1 inert axis** |

Exponentiating a block-diagonal matrix exponentiates each block independently, so:

> **Every one-parameter family of rotations, in every dimension, is a set of independent 2-D rotations at independent angular speeds — plus, in odd dimensions, at least one direction that never moves.**

This is not a simplification anyone chose. It's what one-parameter rotation families *are*.

## Therefore: RoPE

RoPE is exactly this construction, written in its canonical basis. Its generator is block skew-symmetric with `θ_k` the frequency ladder, and — verified — the RoPE matrix **is** the matrix exponential of it:

```
pos    1:  exp(1·A) == RoPE(1)     True
pos    7:  exp(7·A) == RoPE(7)     True
pos  300:  exp(300·A) == RoPE(300) True
A skew-symmetric: True     →  RoPE is a one-parameter subgroup of SO(D_h)
```

Now every "why not 3-D?" question answers itself:

- **Why not 3×3 blocks?** A one-parameter family in `SO(3)` is rotation about a fixed axis — verified eigenvalues `{1, −0.744 ± 0.668i}`, i.e. **one 2-D rotating plane plus one direction with eigenvalue exactly 1 that never moves**. You'd spend a third of the head dimension on a coordinate that does nothing.
- **Why not 4×4 blocks?** A one-parameter family in `SO(4)` rotates in two independent planes at two speeds — which *is* two RoPE pairs with two frequencies. Identical, relabelled.
- **Why not 1-D?** No rotation exists in one dimension; the only norm-preserving maps are `±1`. Two is the minimum for a rotation to exist.
- **Why must `D_h` be even?** Odd would guarantee an inert axis carrying no positional signal.

**Two is the minimum that can rotate and the maximum that's irreducible.** The only genuine design freedom left is choosing the `θ_k` — the frequency ladder of [file 08](08_frequency_phase_and_periodicity.md).

### One loose end: why coordinate pairs?

The theorem promises *some* orthonormal basis in which the blocks are 2×2, not that it's the coordinate axes. RoPE simply declares the basis to be adjacent coordinate pairs. That's free: `W_Q` and `W_K` are **learned** and sit immediately before the rotation, so they can absorb any fixed orthonormal change of basis. Fixing the basis costs no expressiveness — the same gauge-freedom argument as [file 03](03_bilinear_and_quadratic_forms.md).

## Why this matters

- **RoPE and its descendants** — you can now read "RoPE applies a one-parameter subgroup of `SO(D_h)`" as a plain sentence, and see why context-extension methods only ever adjust the `θ_k` ([5.3/05](../../part5_transformer_rebuilt/5.3_positional_information/05_context_length_extension.md)).
- **Sinusoids and rotations are the same object.** A 2-D rotation block *is* a `(sin, cos)` pair advancing in phase — the bridge to [file 08](08_frequency_phase_and_periodicity.md).
- **Parameterizing constrained weights.** If you need an orthogonal weight matrix, don't constrain `Q` directly — parameterize a skew-symmetric `A` and use `exp(A)`, which is unconstrained and automatically lands in `SO(n)`. This is the standard trick for orthogonal RNNs and normalizing flows.
- **The general pattern** — a hard constraint (orthogonality) becomes an unconstrained parameterization (skew-symmetry) through the exponential — recurs across ML: softmax for the simplex, `exp` for positivity, Cholesky for positive-definiteness.

## Self-check

1. What two conditions make `t ↦ R(t)` a one-parameter family, and which one gives RoPE its relative-position property?
2. Why is every one-parameter family abelian, even inside a non-abelian group?
3. Derive in one line why the generator of a rotation family must be skew-symmetric.
4. A one-parameter family in `SO(7)` — how many rotating planes, and how many inert directions?
5. Why can `exp(tA)` never produce a reflection?
6. You need a weight matrix constrained to be orthogonal throughout training. How would you parameterize it, and why is that easier than projecting back onto `O(n)` after each step?

### Answers

1. `R(0) = I` and `R(s+t) = R(s)R(t)`. The **second** gives the relative-position property: it implies `R(i)⁻¹R(j) = R(j−i)`, so `(R_i q)·(R_j k)` depends only on the offset. The first is a normalization that just fixes where the family starts.
2. Because composition inherits commutativity from addition of the parameters: `R(s)R(t) = R(s+t) = R(t+s) = R(t)R(s)`, and real numbers commute. The group may be non-abelian overall, but the one-dimensional path you trace through it with a single scalar always lands in an abelian subgroup. This is why no amount of dimensionality gives RoPE access to non-commutative structure.
3. Differentiate the orthogonality constraint `R(t)ᵀR(t) = I` and evaluate at `t = 0`, where `R = I` and `R′ = A`: `Aᵀ·I + I·A = 0`, hence `Aᵀ = −A`.
4. **3 rotating planes and 1 inert direction.** Skew-symmetric eigenvalues come in `±iθ` pairs; 7 is odd so one eigenvalue must be 0, leaving 6 to form 3 pairs. Verified in the table above. Every odd dimension wastes at least one direction, which is why head dimensions are even.
5. Because reflections have `det = −1` while `exp(tA)` has `det = +1` for all `t` — and the family is **continuous**, starting at `det(exp(0)) = det(I) = 1`. Determinant is a continuous function that can only take values `±1` on orthogonal matrices, so it cannot jump between them without passing through a non-orthogonal matrix. `O(n)` is disconnected, and the exponential can only reach the component containing the identity.
6. Parameterize an **unconstrained** matrix `M`, form `A = M − Mᵀ` (automatically skew-symmetric), and take `Q = exp(A)`, which is automatically in `SO(n)`. Gradients flow through `matrix_exp` normally, so it's just another differentiable layer and the constraint can never be violated — even mid-step. Projecting after each update instead means you're always slightly off the manifold, the projection interacts badly with momentum and adaptive optimizers (which assume a Euclidean parameter space), and you pay a decomposition every step. The tradeoff is that `matrix_exp` is itself expensive for large `n`.

## Exercise

1. Build a random skew-symmetric `A` for `n = 6`. Confirm `Aᵀ = −A`, that its eigenvalues are purely imaginary in `±iθ` pairs, and that `exp(A)` is orthogonal with `det = 1`.
2. Verify the homomorphism property: `exp(sA)·exp(tA) == exp((s+t)A)` for several `s, t`. Then check it **fails** for two different generators — `exp(A)·exp(B) ≠ exp(A+B)` when `AB ≠ BA` — and connect that to why `SO(3)` is non-abelian.
3. Construct RoPE's generator explicitly (block skew-symmetric with the frequency ladder on the off-diagonals) and confirm `torch.matrix_exp(p*A)` equals the standard RoPE rotation matrix at `p = 1, 7, 300`.
4. For `n = 5`, compute `exp(tA)` and find its eigenvector with eigenvalue 1 — the inert axis. Verify that `exp(tA)` leaves that vector *exactly* unchanged for every `t`, then explain in one sentence what including it in a positional encoding would accomplish.
