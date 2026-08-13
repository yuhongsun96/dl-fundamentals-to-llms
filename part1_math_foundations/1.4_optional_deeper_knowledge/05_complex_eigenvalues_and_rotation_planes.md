# Complex Eigenvalues and Rotation Planes

The first surprise in [1.1/05](../1.1_linear_algebra/05_eigen_svd.md) is that a perfectly ordinary **real** matrix can have **complex** eigenvalues. That isn't a curiosity — it's the algebra telling you the matrix *rotates*, and it's the reason rotations come in two-dimensional pairs everywhere in this repo. This file is the geometric reading of complex eigenvalues.

**Prerequisite:** eigenvalues/eigenvectors as covered in [1.1/05](../1.1_linear_algebra/05_eigen_svd.md). Nothing here needs complex analysis — only that `i² = −1` and that `e^(iθ) = cos θ + i sin θ`.

## The problem a rotation poses

An eigenvector is a direction the matrix merely **scales**: `Av = λv`, same direction out as in. Now picture a 2-D rotation by 40°. Is there *any* direction it leaves pointing the same way?

No. Every real direction gets turned. So a rotation matrix has **no real eigenvectors at all** — and since the eigenvalue equation still has solutions over the complex numbers, those solutions must be complex.

Compute them for `R(θ)`:

```
R(0.7)  eigenvalues:  0.7648 + 0.6442i   and   0.7648 − 0.6442i
e^(0.7i) = cos(0.7) + i·sin(0.7) = 0.7648 + 0.6442i        ← identical
|λ| = 1.0000000000
```

So the eigenvalues of a rotation by `θ` are exactly `e^(±iθ)`. **The angle you rotate by is sitting in the phase of the eigenvalue**, and `|λ| = 1` says the rotation doesn't scale anything.

## Reading `λ = r·e^(iθ)`

Any complex number splits into a **magnitude** `r` and a **phase** `θ`, and both have direct meaning: applying the matrix once multiplies lengths by `r` and turns by `θ`. Verified on `M = 1.15 · R(0.4)`, starting from `(1,0)`:

| after | norm | `r^step` | angle | `step·θ` |
|---|---|---|---|---|
| 1 application | 1.150000 | 1.150000 | 0.400000 | 0.400000 |
| 2 applications | 1.322500 | 1.322500 | 0.800000 | 0.800000 |
| 3 applications | 1.520875 | 1.520875 | 1.200000 | 1.200000 |

Exactly `r` per step in magnitude, exactly `θ` per step in angle. This is why the eigenvalue's **modulus** governs growth and decay while its **argument** governs oscillation:

- `|λ| > 1` → spirals outward (explodes under repeated application)
- `|λ| = 1` → pure rotation, forever
- `|λ| < 1` → spirals inward (decays)

That's precisely the RNN story: repeated multiplication by `W_h` drives the hidden state like `|λ|^t`, which is why the **spectral radius** (largest `|λ|`) decides between vanishing and exploding gradients ([4.1/02](../../part4_sequence_modeling_lineage/4.1_rnns_and_lstms/02_vanishing_gradient_and_gating.md)). The complex phase of those eigenvalues is the oscillation you see riding on top.

## Why they always come in conjugate pairs

A real matrix's characteristic polynomial has **real coefficients**. If a polynomial with real coefficients has a complex root `λ = a + bi`, then `λ̄ = a − bi` is automatically a root too — take the complex conjugate of the whole equation and nothing on the left changes. So complex eigenvalues of a real matrix **always** arrive in matched `λ, λ̄` pairs.

Verified on random real matrices:

| `n` | imaginary parts of the eigenvalues | real eigenvalues |
|---|---|---|
| 3 | `−0.9436, 0, +0.9436` | **1** |
| 4 | `−1.2407, 0, 0, +1.2407` | 2 |
| 5 | `−1.1375, 0, 0, 0, +1.1375` | **3** |

Two consequences that matter later:

1. **Each conjugate pair corresponds to one real 2-D plane.** The pair `λ, λ̄` has complex eigenvectors `v, v̄`; their real and imaginary parts span a genuine **real two-dimensional subspace**, and the matrix maps that plane into itself, acting on it as "scale by `r`, rotate by `θ`." Complex eigenvalues aren't describing anything imaginary — they're the bookkeeping for a **rotation plane**.
2. **Odd dimensions must have at least one real eigenvalue.** Complex ones pair up, so an odd count can't be all-complex — there's always a leftover real one, i.e. a direction that is only scaled, never rotated. That's the "inert axis" that makes odd-dimensional rotation blocks wasteful, developed in [file 07](07_generators_and_one_parameter_families.md).

## The real canonical form

Putting it together: any real matrix can be brought (by a real change of basis) to a block-diagonal form of

- **1×1 blocks** for each real eigenvalue — directions that are purely scaled, and
- **2×2 blocks** `r·[[cos θ, −sin θ], [sin θ, cos θ]]` for each conjugate pair — planes that are scaled *and* rotated.

**Two-by-two is the smallest block that can rotate, and it's all you ever need.** A 3×3 "rotation block" doesn't exist as an irreducible thing: it's always a 2-D rotating plane plus a fixed axis. That single fact is the answer to "why does RoPE use pairs," and [file 07](07_generators_and_one_parameter_families.md) closes the argument.

## Why this matters

- **RoPE's pair structure** is this canonical form, not a design choice ([5.3/04](../../part5_transformer_rebuilt/5.3_positional_information/04_rope.md), [file 07](07_generators_and_one_parameter_families.md)).
- **RNN dynamics**: `|λ|` sets exponential growth/decay, `arg λ` sets oscillation frequency ([4.1](../../part4_sequence_modeling_lineage/4.1_rnns_and_lstms/)).
- **Sinusoids and rotations are the same object** viewed two ways — the bridge to [file 08](08_frequency_phase_and_periodicity.md).
- **Why SVD is often preferred to eigendecomposition** in ML: singular values are always real and non-negative, so you never deal with any of this. The price is that the SVD tells you about stretching, not about dynamics under repeated application ([1.1/05](../1.1_linear_algebra/05_eigen_svd.md)).

## Self-check

1. Why does a 2-D rotation matrix have no real eigenvectors, in one sentence about geometry rather than algebra?
2. A real 7×7 matrix has how many real eigenvalues, at minimum? Why?
3. `λ = 0.9 e^(0.3i)`. Describe what repeated application of the matrix does to a vector in the corresponding plane.
4. Why do complex eigenvalues of a *real* matrix always come in conjugate pairs — and would this still hold for a matrix with complex entries?
5. Both a 2×2 rotation and the identity matrix have `|λ| = 1` for all eigenvalues. What distinguishes them spectrally?

### Answers

1. Because an eigenvector is a direction the matrix leaves pointing the same way (up to scaling), and a rotation by a non-zero angle turns *every* direction. There is no line through the origin that maps to itself, so no real eigenvector can exist. (The exceptions are θ = 0, the identity, and θ = π, which negates every vector — both leave lines fixed.)
2. **At least one.** Complex eigenvalues come in conjugate pairs, so they always occupy an even count; with 7 eigenvalues total you can have at most 6 complex ones, leaving ≥1 real. In general any odd-dimensional real matrix has at least one real eigenvalue — and for an odd-dimensional *rotation* it must equal 1, giving a fixed axis.
3. It **spirals inward while turning**: each application shrinks the vector's length by a factor of 0.9 and rotates it by 0.3 radians within that plane. After `t` steps the length is `0.9^t` (decaying to zero) and the total angle is `0.3t`. Under repeated application this direction's contribution vanishes — the classic vanishing-gradient behavior, with oscillation on the way down.
4. Because the characteristic polynomial `det(A − λI)` has **real coefficients** when `A` is real. Conjugating the whole equation `p(λ) = 0` leaves the coefficients unchanged and sends `λ → λ̄`, so `p(λ̄) = 0` too. For a matrix with genuinely complex entries the coefficients are complex, conjugation changes them, and the argument collapses — complex-entry matrices have no pairing requirement.
5. Their **phases**. The identity has `λ = 1` twice — zero phase, so nothing rotates and every direction is an eigenvector. A rotation by `θ ≠ 0` has `λ = e^{±iθ}` with non-zero phase and *no* real eigenvectors. Modulus alone tells you about growth; you need the argument to know whether anything is turning. This is exactly why spectral radius is insufficient to describe dynamics — it discards the phase.

## Exercise

1. Build `R(θ)` for `θ ∈ {0, 0.1, π/2, π}` and compute eigenvalues for each. Confirm they're `e^{±iθ}`, and identify the two angles where real eigenvectors reappear. Explain geometrically why those two are special.
2. Take `M = r·R(θ)` with `r = 0.9` and `θ = 0.3`. Apply it 50 times to `(1,0)`, recording norm and angle at each step. Plot the trajectory — you should see a logarithmic spiral. Then set `r = 1.05` and watch it invert.
3. Generate random real matrices for `n = 2…8`, and for each count real vs. complex eigenvalues. Verify complex ones always come in pairs and that odd `n` always leaves at least one real.
4. Take a random real 4×4 matrix with two conjugate pairs. For one pair, extract the complex eigenvector `v`, form the real vectors `Re(v)` and `Im(v)`, and verify that the plane they span is **invariant** — that `A` maps vectors in that plane back into it. This is the 2×2 block of the canonical form, found by hand.
