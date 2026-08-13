# Rotations in n Dimensions

File [06](06_projections_orthogonality.md) established *what* a rotation is algebraically: an orthogonal matrix with `det = +1`. This file is about how to **picture** one when `n` is large. The short version, and the only thing you need to carry:

> **A rotation happens *in a plane*, not around an axis.**

Almost all confusion about high-dimensional rotation comes from the 3D habit of naming a rotation by the line that stays still. That habit is a coincidence of three dimensions and it does not generalize. The plane that *moves* is the real object.

> **Convention note.** This file uses the **column-vector** convention (`y = R x`, `R` acts on the left), matching [08](08_jacobians_chain_rule.md), because rotation matrices read more naturally applied from the left. Most other files use row-vector (`Y = X W`).

## The atom: a simple rotation

Pick any **2D plane** `P` sitting inside `R^n`. A *simple rotation* does exactly two things:

1. Spins vectors **within** `P` by an angle `θ` — the ordinary 2D picture.
2. Leaves everything **orthogonal to `P`** completely fixed, pointwise.

That's it. Note the payoff for intuition: **you never have to picture more than two dimensions at once.** A rotation in `R^4096` is still just the familiar 2D spin, happening inside one plane, with 4094 dimensions sitting there untouched.

### Where the `2 × 2` block comes from

The `sin`/`cos` grid isn't a formula to memorize — a matrix's **columns are just where the basis vectors land**, so build it by asking where each axis goes. Read the answer off the **unit circle**, where the point at angle `θ` is *by definition* `(cos θ, sin θ)`:

- `e₁ = (1, 0)` sits at angle `0`. Rotate by `θ` → it lands at angle `θ`, i.e. **`(cos θ, sin θ)`**.
- `e₂ = (0, 1)` sits at angle `90°`. Rotate by `θ` → it lands at angle `90° + θ`, i.e. **`(−sin θ, cos θ)`** (tipping left, hence the minus).

Write those two destinations side by side as columns and you have `R(θ)`. Sine and cosine appear for one reason: they *are* the coordinates of a point on the unit circle, so they're the only way to say "a direction at angle `θ`." Everything else follows from linearity — `v = x e₁ + y e₂`, so `Rv = x(Re₁) + y(Re₂)`: move the two axes and every vector rides along with them.

In coordinates, if `P` is the plane spanned by basis vectors `e_i` and `e_j`, the matrix is the identity everywhere except a `2 × 2` block at rows/columns `i, j`:

```
G(i, j, θ) =  ⎡ 1                    ⎤
              ⎢    cos θ   -sin θ    ⎥  ← row i
              ⎢       1              ⎥
              ⎢    sin θ    cos θ    ⎥  ← row j
              ⎣                    1 ⎦
                       ↑        ↑
                    col i    col j
```

This is called a **Givens rotation**. It's the building block used in QR decomposition, and it's the concrete form of "spin one plane, fix the rest."

## Why "axis" is a 3D-only nickname

Ask what *doesn't* move. For a 2D plane inside `R^n`, the fixed orthogonal complement has dimension `n − 2`:

| `n` | Rotation acts in | Fixed complement | Nameable by an "axis"? |
|---|---|---|---|
| 2 | the whole space | `0`-dim — just the origin | No. There's no axis, only a fixed *point* |
| 3 | a plane | **`1`-dim: a line** | **Yes** — "the axis." This is the coincidence |
| 4 | a plane | `2`-dim: another plane | No — no single line to point at |
| `n` | a plane | `(n−2)`-dim | No |

The complement is a single line exactly when `n − 2 = 1`, i.e. **only when `n = 3`**. So in 3D you *can* label a rotation by what stays still, and everyone learns to. In every other dimension there is no such line, and the label is meaningless. The moving plane is what generalizes.

## The canonical form: every rotation is a set of independent plane spins

The general statement (the real-orthogonal analogue of eigendecomposition):

> **Any rotation of `R^n` decomposes into simultaneous simple rotations in a set of mutually orthogonal 2D planes, each with its own angle.**

You get at most `⌊n/2⌋` such planes — pair up the dimensions, and if `n` is odd there's one leftover dimension that stays fixed. In the basis adapted to those planes, the matrix is **block-diagonal with `2 × 2` rotation blocks**:

```
R = ⎡ R(θ₁)                    ⎤                    ⎡ cos θ   -sin θ ⎤
    ⎢         R(θ₂)            ⎥      R(θ) =        ⎣ sin θ    cos θ ⎦
    ⎢               ⋱          ⎥
    ⎣                     [1]  ⎦   ← only if n is odd
```

So a high-dimensional rotation is not one exotic operation. It is **a stack of independent 2D spins in perpendicular planes**, each minding its own business.

**The eigenvalue view** ([05](05_eigen_svd.md)). A rotation matrix has no interesting *real* eigenvalues — a real eigenvector would be a direction mapped to a multiple of itself, and rotations preserve length, so the only possibilities are `±1` (fixed or flipped directions). All the actual rotating shows up as **complex-conjugate eigenvalue pairs `e^{±iθ_d}`**, one pair per plane, each sitting on the unit circle. Reading off the eigenvalues of an orthogonal matrix therefore reads off its rotation angles directly.

## Rotations don't care about your basis

The block-diagonal form is easy to read backwards, so state the theorem carefully. It does **not** say "rotations happen in coordinate planes." It says:

> For any rotation `R`, **there exists an orthonormal basis** in which `R` is block-diagonal.

The causality runs the other way from what the tidy matrix suggests. The rotation comes first and **determines its own planes**; you then *choose a basis aligned to them*. A generic rotation of `R^4096` spins in planes that align with **no** pair of standard basis vectors — they're 2-planes tilted at arbitrary angles through the space.

An arbitrary-plane rotation in coordinates is just the coordinate-aligned one seen through a rotated lens. For a plane `P = span(u, v)` with `u, v` orthonormal but pointing anywhere, extend to an orthonormal basis and stack it as the columns of an orthogonal `Q = [u\ v\ w_3 \cdots w_n]`:

```
R = Q · G(1, 2, θ) · Q^T
      ↑        ↑        ↑
   change   easy     change
   back     spin     coordinates so P
                     becomes the first two axes
```

Read right to left: rotate the world so `P` is axis-aligned, do the simple spin, rotate back. The resulting `R` is a dense matrix with no visible structure, yet it is the *same geometric operation* as the tidy Givens block. Conjugation by an orthogonal matrix is exactly "the same rotation, different coordinates."

| Basis-independent (the rotation itself) | Basis-dependent (bookkeeping) |
|---|---|
| The set of rotation **planes** | Which coordinates they happen to align with |
| The **angles** `θ_d` | The individual matrix entries |
| The eigenvalues `e^{±iθ_d}` | Whether the matrix looks block-diagonal or dense |

The eigenvalue row is the proof that the angles are real invariants: eigenvalues are unchanged by similarity, and `R = Q G Q^T` is a similarity, so `R` and `G` have *identical* eigenvalues. The angles belong to the rotation, not to your choice of axes.

**So why use Givens rotations at all?** Two reasons, neither of which is "rotations are secretly coordinate-aligned":

1. **They generate everything.** Any rotation factors into a product of at most `n(n−1)/2` Givens rotations — one per coordinate pair, exactly the parameter count below. This is the algorithm behind **QR decomposition**: zero out one off-diagonal entry at a time by spinning one coordinate plane.
2. **They're cheap.** A dense rotation costs `O(n²)` to apply; a Givens rotation costs `O(1)`. If you can arrange for your planes to be coordinate-aligned, the operation is nearly free.

> **A matrix is a *description* of a rotation in chosen coordinates, not the rotation itself.** The same rotation has infinitely many matrix representations, all conjugate, all sharing the same angles. Block-diagonal is simply what you get when you pick the basis the rotation already prefers.

## Degrees of freedom

One angle per **pair** of axes, so the number of independent parameters is

```
dim SO(n)  =  C(n, 2)  =  n(n − 1) / 2
```

| `n` | Free parameters |
|---|---|
| 2 | 1 (a single angle) |
| 3 | 3 (roll, pitch, yaw) |
| 128 (`D_h`) | 8128 |
| `n` | `n(n−1)/2` — grows quadratically |

Notice 3D is the odd one out *again*: `3` parameters happens to equal the size of a 3-vector, which is why "axis + angle" (a direction plus a magnitude) fits so neatly there and nowhere else. Two coincidences of `n = 3`, one cause.

Also worth separating two different counts that are easy to conflate: a rotation involves **at most `⌊n/2⌋` planes**, but the *space of rotations* has **`n(n−1)/2` dimensions** — because you must also specify *which* planes, not just the angles.

## When you can't picture it, use the algebra

The definition that needs no visualization, from [06](06_projections_orthogonality.md):

```
R^T R = I        (preserves all lengths and inner products)
det(R) = +1      (orientation-preserving — no reflection)
```

Together: `R ∈ SO(n)`, a **rigid, orientation-preserving reshuffling of space**. Nothing stretches, nothing flips. Three consequences do most of the work:

| Property | Statement |
|---|---|
| **Norm-preserving** | `‖Rx‖ = ‖x‖` — nothing blows up or decays |
| **Angles add** | `R(θ₁) R(θ₂) = R(θ₁ + θ₂)` within a plane — rotations compose |
| **Free inverse** | `R(θ)^{-1} = R(θ)^T = R(−θ)` |

### One condition, not two

"Preserves lengths *and* angles" is redundant — **length preservation alone forces angle preservation**, by the polarization identity:

```
⟨x, y⟩ = ½ ( ‖x + y‖² − ‖x‖² − ‖y‖² )
```

Inner products are recoverable from norms, so any length-preserving linear map automatically preserves inner products and therefore angles. You only ever need to check one thing.

Stronger still: **distance preservation forces linearity.** If `f: R^n → R^n` satisfies `‖f(x) − f(y)‖ = ‖x − y‖` and `f(0) = 0`, then `f` *must* be linear and orthogonal. There are no exotic curved isometries lurking — the matrix picture is the whole story.

### What the two conditions exclude

Rotations are *nearly* characterized as "the rigid maps," but two other families are equally rigid, and seeing what each condition rules out is what makes the definition precise:

| Preserve | What you get | Group |
|---|---|---|
| Angles only | Uniform scaling composed with a rotation/reflection: `c·Q` | similarity / conformal |
| **Lengths** (⟹ angles), origin fixed | Rotations **and reflections** | `O(n)` |
| Lengths + **orientation** (`det = +1`) | **Rotations only** | `SO(n)` |
| Lengths, origin free | Rotations, reflections, **and translations** | `E(n)` (Euclidean isometries) |

- **Reflections** preserve every length and angle perfectly — just as rigid as a rotation. They are excluded *only* by `det = +1`. This is why the determinant clause is not a technicality.
- **Translations** also preserve every distance, but are excluded by **linearity**. The reason is worth internalizing: a translation preserves the *distance function* but **not the inner product**, because the inner product is defined relative to the origin:

```
⟨x + t, y + t⟩ = ⟨x, y⟩ + ⟨x, t⟩ + ⟨t, y⟩ + ‖t‖²  ≠  ⟨x, y⟩
```

Norms change, and angles measured at the origin change. So "a translation is just a change of description" is true in an **affine** space (where only differences between points are meaningful) and false in a **vector** space (where the origin is structure). Linear algebra is the second. Concretely, `T(x) = x + t` fails linearity at once since `T(0) = t ≠ 0`, and no `n × n` matrix can express it — which is exactly why graphics uses homogeneous coordinates, embedding `R^n` in `R^{n+1}` at height 1 so translation becomes a genuine linear shear.

In DL this is precisely the **weight/bias split**: `Wx` is linear, `Wx + b` is affine — **the bias term is the translation** ([2.1/03](../../part2_neural_network_fundamentals/2.1_mlp_building_block/03_bias_terms.md)). And it is not cosmetic for embeddings: cosine similarity, norms, and dot products are all measured from the origin, so translating an embedding space changes every similarity score.

### Why this makes rotations special

Pulling it together — three reasons beyond "they're convenient":

1. **They are the symmetry group of Euclidean structure itself.** The inner product defines every metric fact (lengths, angles, orthogonality, volume). Rotations are exactly the maps leaving all of it untouched — they change your *description* of space without changing any *fact* about it.
2. **They are continuously reachable from the identity.** `SO(n)` is connected; `O(n)` has two components. You can physically turn an object into any rotated version of itself through intermediate states; you cannot get to its mirror image that way. Rotations are the rigid motions you can actually *perform*.
3. **They are one of the two atoms of all linear algebra.** The SVD `A = U Σ V^T` ([05](05_eigen_svd.md)) says *every* linear map is **rotate → scale → rotate**. Rotations aren't a special case among matrices; with diagonal scaling they are the ingredients everything else is built from.

### Do rotations commute?

Usually **no**, but there's a clean rule:

- **Rotations in the same plane commute** — angles just add.
- **Rotations in *completely orthogonal* planes commute** — they act on complementary invariant subspaces, so their block-diagonal matrices touch disjoint coordinates.
- **Rotations in planes that share a direction do *not* commute.** This is the familiar 3D fact that yaw-then-pitch ≠ pitch-then-yaw — any two distinct planes in `R^3` necessarily intersect in a line, so 3D rotations essentially never commute.

Non-commutativity is therefore about **plane overlap**, not about dimension.

## The complex-number shortcut

A 2D rotation is multiplication by `e^{iθ}` — because `e^{iθ} = cos θ + i sin θ` *is* the unit-circle point at angle `θ`, so multiplying by it adds `θ` to a point's angle while leaving its magnitude alone. Derived two ways (series and "velocity perpendicular to position") in [supplementary/07](supplementary/07_eulers_formula.md).

Identify `R^n` with `C^{n/2}` by pairing coordinates into complex numbers, and an `n`-dimensional rotation becomes:

```
multiply each complex coordinate z_d  by its own unit-modulus  e^{iθ_d}
```

Angles add because exponents add: `e^{iθ₁} · e^{iθ₂} = e^{i(θ₁+θ₂)}`. This is the most compact formalism for the canonical form, and it's why rotation-heavy code is often written with complex arithmetic.

## Why this matters in DL

Kept brief — the details live elsewhere:

- **RoPE** pairs up a head vector's `D_h` coordinates into `D_h/2` planes and spins each by a position-dependent angle. That is *literally* the canonical form above, with the angles chosen as a frequency ladder. Full treatment in [Part 5.3/04](../../part5_transformer_rebuilt/5.3_positional_information/04_rope.md); the orthogonality consequences are in [06](06_projections_orthogonality.md).
- **Orthogonal initialization.** A rotation's Jacobian has spectral norm exactly 1, so gradients flowing through it neither explode nor vanish — the motivation for orthogonal init in deep recurrent and some Transformer variants.
- **Norm-preservation as a design tool.** Any time you want to transform a representation without changing its scale — unit-norm embeddings, phase-only encodings — a rotation is the transformation that does it.

## One-line summary

> A rotation is a set of independent 2D spins in mutually orthogonal planes, and nothing else changes. Forget axes: to picture a rotation in any dimension, picture *one plane* spinning with everything perpendicular to it holding still — then remember there can be up to `⌊n/2⌋` such planes going at once.

## Self-check

1. Why doesn't "the axis of rotation" generalize beyond 3D, and what object replaces it?
2. How many independent parameters does a rotation of `R^n` have, and why? Reconcile that count with the fact that a rotation involves at most `⌊n/2⌋` planes.
3. A rotation matrix is real, but its "rotation angles" are recovered from complex eigenvalues. Why can't a genuine rotation (other than the identity) have a full set of real eigenvalues?
4. A translation preserves every distance between points, so it is just as "rigid" as a rotation. Why is it nonetheless not a rotation — and what breaks if you translate an embedding space?
5. The canonical form writes a rotation as a block-diagonal matrix. Does that mean rotations act in coordinate planes? If not, what is the block-diagonal form actually telling you?

### Answers

1. A rotation acts *in* a 2D plane and fixes that plane's orthogonal complement, which has dimension `n − 2`. That complement is a single line only when `n = 3`. In 2D it's just the origin; in 4D and above it's a subspace of dimension ≥ 2, so there is no unique line to call "the axis." What generalizes is the **plane of rotation** (plus its angle) — the thing that moves, rather than the thing that stays still.
2. `n(n−1)/2 = C(n, 2)` — one angle per pair of coordinate axes. The reconciliation: `⌊n/2⌋` counts how many planes are simultaneously *active* in a given rotation, while `n(n−1)/2` is the dimension of the *space of all rotations*, which must also account for **which** planes were chosen. Specifying the planes costs the remaining parameters. (For `n = 3`: at most 1 active plane with 1 angle, but 3 total parameters — 2 to orient the plane, 1 for the angle.)
3. A real eigenvector `v` with real eigenvalue `λ` satisfies `Rv = λv`, and rotations preserve norms, so `‖Rv‖ = |λ|·‖v‖ = ‖v‖` forces `|λ| = 1`, i.e. `λ = ±1`. A real eigenvalue therefore means a direction that is either **fixed** (`+1`) or **reversed** (`−1`) — neither of which is rotating. Genuine rotation in a plane sends no direction in that plane to a multiple of itself, so those components must appear as **complex-conjugate pairs `e^{±iθ}`**. (And `det = +1` constrains how many `−1`s can appear.)
4. Because it preserves the **distance function** but not the **inner product** — and linear algebra is built on the latter. `⟨x+t, y+t⟩ ≠ ⟨x, y⟩`, norms change, and angles measured at the origin change, because the inner product is defined *relative to the origin* and a translation moves it. Formally, `T(x) = x + t` isn't linear at all (`T(0) = t ≠ 0`), so no `n × n` matrix represents it; translations live in the affine group `E(n)`, not `SO(n)`. Translating an embedding space breaks every quantity measured from the origin: norms, dot products, and cosine similarities all change, so retrieval and similarity scores change. (In a purely affine setting, where only *differences* between points are meaningful, a translation genuinely would be just a change of description — but embeddings are not that setting.)
5. No — the theorem says only that *there exists* an orthonormal basis in which the matrix is block-diagonal. The rotation determines its own planes, and generically they align with no standard basis vectors; you *choose* a basis to match them. An arbitrary-plane rotation is `R = Q G Q^T` — the same operation conjugated into different coordinates, generally a dense matrix. What's basis-independent is the set of planes and the angles (equivalently the eigenvalues `e^{±iθ}`, invariant under similarity); what's basis-dependent is whether the matrix *looks* tidy.

## Exercise

Build a Givens rotation `G(i, j, θ)` in `R^5` by hand: start from `I`, then set the four entries of the `2 × 2` block. Verify numerically that `G^T G = I` and `det(G) = 1`, and confirm that any vector lying entirely outside the `(i, j)` plane is returned unchanged.

Then compose two Givens rotations in **disjoint** coordinate pairs — say `(1,2)` and `(3,4)` — and check that they commute (`G₁G₂ = G₂G₁`). Repeat with two rotations in **overlapping** pairs — `(1,2)` and `(2,3)` — and confirm they do *not*. Finally, take the composed disjoint rotation, compute its eigenvalues, and verify you recover `e^{±iθ₁}`, `e^{±iθ₂}`, and a single `1` for the leftover fifth dimension — the canonical form read straight off the spectrum.
