# Groups, and the Rotation Group

`SO(2)`, `SO(3)`, "abelian," "the rotation group," "a one-parameter subgroup" — this vocabulary shows up whenever rotations are discussed seriously, and it's almost never defined in ML writing. It's a small amount of machinery with a large payoff: it's the precise language for **transformations you can compose and undo**, which is what a neural network's weight matrices, RoPE's rotations, and gauge freedom all are.

## What a group is

A **group** is a set `G` of things you can combine, with one operation (write it as multiplication), satisfying four rules:

| Rule | Statement | In words |
|---|---|---|
| **Closure** | `a, b ∈ G  ⟹  ab ∈ G` | combining two members gives a member |
| **Associativity** | `(ab)c = a(bc)` | grouping doesn't matter |
| **Identity** | `∃ e` with `ea = ae = a` | there's a "do nothing" element |
| **Inverses** | `∀a ∃a⁻¹` with `aa⁻¹ = e` | everything can be undone |

The useful mental model is not "a set with an operation" but:

> A group is a collection of **transformations** that is closed under composition and under undoing.

Familiar examples: the integers under addition (identity 0, inverse `−n`); invertible `n × n` matrices under multiplication (identity `I`, inverse `A⁻¹`); the symmetries of a square under composition.

Non-examples, and why they fail: the integers under *multiplication* (no inverse for 2 — `1/2` isn't an integer); **all** `n × n` matrices under multiplication (singular matrices have no inverse); the odd integers under addition (not closed — odd + odd = even).

### Abelian

A group is **abelian** (or *commutative*) if `ab = ba` for every pair. Most familiar number systems are; most transformation groups are not.

Rotations make the distinction vivid — verified:

```
SO(2):  R(0.7)·R(1.3) == R(1.3)·R(0.7)          True
SO(3):  Rx(0.7)·Ry(1.3) == Ry(1.3)·Rx(0.7)      False   (they differ by 0.62)
```

Rotations in a plane commute because they all turn about the same (implicit) axis and the angles simply add. In 3-D, rotating about `x` then `y` genuinely differs from `y` then `x` — a fact you can confirm with a book on a table. **This is the single most important structural difference between 2-D and 3-D rotations**, and [file 07](07_generators_and_one_parameter_families.md) shows why RoPE only ever needs the abelian case.

### Subgroup

A **subgroup** is a subset that is itself a group under the same operation — closed, containing the identity, closed under inverses. Rotations form a subgroup of the rigid motions; rotations about a *single fixed axis* form a subgroup of all 3-D rotations (and that one **is** abelian, even though the full group isn't). That last example is the seed of the "one-parameter subgroup" idea.

## The matrix groups you'll actually meet

| Group | Definition | Name | Meaning |
|---|---|---|---|
| `GL(n)` | invertible `n × n` real matrices | **general linear** | all reversible linear maps |
| `O(n)` | `QᵀQ = I` | **orthogonal** | rigid motions fixing the origin: rotations *and* reflections |
| `SO(n)` | `QᵀQ = I` **and** `det Q = +1` | **special orthogonal** | rotations only |

**"Special" always means "determinant 1."** That's the whole content of the S — it's the conventional way to say "orientation-preserving," which for `O(n)` means excluding the reflections.

Recall from [1.1/06](../1.1_linear_algebra/06_projections_orthogonality.md) that `det Q = ±1` is forced for orthogonal `Q`, so `O(n)` splits cleanly into two halves and `SO(n)` is the half containing the identity.

### `O(n)` and `SO(n)` really are groups

Worth checking once, because it's what licenses all the algebra:

- **Closure:** `(Q₁Q₂)ᵀ(Q₁Q₂) = Q₂ᵀQ₁ᵀQ₁Q₂ = Q₂ᵀQ₂ = I`, and determinants multiply, so `det = 1 · 1 = 1`.
- **Identity:** `I` is orthogonal with `det = 1`.
- **Inverses:** `Q⁻¹ = Qᵀ`, which is itself orthogonal with determinant 1.

Verified in `SO(5)`: `det Q₁ = 1.000000`, `det Q₂ = 1.000000`, `det(Q₁Q₂) = 1.000000`, the product is orthogonal, and `inv(Q₁) == Q₁ᵀ`.

The practical upshot of "inverse = transpose" is that **undoing a rotation is free** — no solve, no inversion. That's what makes the RoPE cancellation `R_iᵀ R_j = R_{j−i}` a transpose rather than a matrix inverse.

### How big is each group?

The **dimension** of a group is its degrees of freedom — how many independent numbers specify one element:

```
dim SO(n) = n(n − 1) / 2        one per pair of axes, i.e. per plane you could rotate in
```

| `n` | `dim SO(n)` | reading |
|---|---|---|
| 2 | **1** | a single angle — that's all a planar rotation is |
| 3 | 3 | three (roll, pitch, yaw) |
| 4 | 6 | six |
| 64 | 2016 | a head-dimension-sized rotation has 2016 free parameters |

Note that `dim SO(2) = 1` is exactly why 2-D rotations are so well-behaved: the group is **one-dimensional and abelian**, so "rotate by `θ`" is a complete description and angles just add.

## Where groups show up in this repo

- **RoPE** applies a one-parameter family inside `SO(D_h)` — [file 07](07_generators_and_one_parameter_families.md), [5.3/04](../../part5_transformer_rebuilt/5.3_positional_information/04_rope.md).
- **Gauge freedom** is a group acting on parameters: `W_Q → W_Q R`, `W_K → W_K R⁻ᵀ` for `R ∈ GL(D_h)` leaves the function unchanged. The set of transformations that don't change behavior forms a group, which is exactly why `W_Q` alone is meaningless while `W_Q W_Kᵀ` is real ([file 03](03_bilinear_and_quadratic_forms.md)).
- **Orthogonal initialization** samples from `O(n)` so that forward and backward passes preserve norms exactly ([2.3](../../part2_neural_network_fundamentals/2.3_init_normalization/)).
- **Equivariance** — the design principle that a model's output should transform predictably when its input is transformed by a group element. Central to geometric deep learning; peripheral on this repo's NLP track, but the vocabulary is the same.
- **Permutation symmetry** — attention is permutation-equivariant over positions until you break that symmetry with positional information, which is exactly the framing of [5.3/01](../../part5_transformer_rebuilt/5.3_positional_information/01_why_position_must_be_injected.md). "The model is invariant under a group action" is the precise version of "the model can't tell the order."

## Self-check

1. Are the invertible matrices a group under multiplication? Are *all* matrices? Are the orthogonal matrices?
2. What does the "S" in `SO(n)` add over `O(n)`, and why does that correspond to excluding reflections?
3. `SO(3)` is not abelian, but rotations about a single fixed axis in 3-D do commute. Reconcile these.
4. `dim SO(2) = 1` but `dim SO(3) = 3`. What is the "1" and what are the "3"?
5. Why is "inverse = transpose" more than a computational convenience for RoPE?

### Answers

1. **Invertible matrices: yes** — that's `GL(n)`; closed under multiplication (product of invertibles is invertible), associative, `I` is the identity, and inverses exist by definition. **All matrices: no** — singular matrices have no inverse, so the inverse axiom fails (closure and associativity are fine, which is why all matrices form a *monoid*, not a group). **Orthogonal matrices: yes** — verified above; closure, identity, and inverses all hold, and `Q⁻¹ = Qᵀ` is itself orthogonal.
2. "S" = **special** = **determinant exactly +1**. Every orthogonal matrix has `det = ±1`, and the sign records whether orientation (handedness) is preserved. Reflections flip handedness and so have `det = −1`; restricting to `det = +1` keeps precisely the orientation-preserving rigid motions, which are the rotations. `O(n)` is two disconnected pieces; `SO(n)` is the piece containing `I`.
3. Non-abelian is a statement about the group **as a whole** — there *exist* pairs that don't commute (rotations about different axes). It doesn't mean *no* pair commutes. Rotations about one fixed axis form an **abelian subgroup** isomorphic to `SO(2)`, since they're really just planar rotations in the plane perpendicular to that axis, and their angles add. A non-abelian group can contain abelian subgroups, and that's exactly what a one-parameter subgroup always is.
4. The **1** is the single angle `θ` — a planar rotation is fully specified by one number. The **3** counts independent planes of rotation in 3-D (`xy`, `xz`, `yz`), equivalently roll/pitch/yaw, equivalently `3·2/2 = 3`. Note that in 3-D people often say "an axis plus an angle," which is 2 + 1 = 3 numbers — same count, different parameterization.
5. Because the RoPE identity is `(R_i q)·(R_j k) = qᵀ R_iᵀ R_j k = qᵀ R_{j−i} k`, and that middle step requires `R_iᵀ` to *be* `R_i⁻¹`. Orthogonality is what makes the transpose an inverse, so it's the property that makes the absolute positions cancel and leave a pure offset. Take away orthogonality and the transpose doesn't undo the rotation, so there's no relative-position identity at all — it's load-bearing, not an optimization.

## Exercise

1. Verify the group axioms for `SO(3)` numerically: generate two random rotation matrices (QR-decompose a random matrix, then fix the sign of the determinant), confirm the product is orthogonal with `det = 1`, and confirm `inv(Q) == Qᵀ` to float tolerance.
2. Demonstrate non-commutativity concretely: build `Rx(π/2)` and `Ry(π/2)`, apply both orders to the vector `(0,0,1)`, and confirm you land somewhere different. Then check that `Rx(a)` and `Rx(b)` *do* commute for any `a, b`.
3. Confirm `dim SO(n) = n(n−1)/2` empirically by counting the free parameters of a skew-symmetric `n × n` matrix (its strictly-upper triangle) for `n = 2…6`. [File 07](07_generators_and_one_parameter_families.md) explains why the skew-symmetric matrices are the right thing to count.
4. Show `O(n)` is disconnected in a concrete sense: generate 1000 random orthogonal matrices and histogram their determinants. You should see exactly two values with nothing in between — and explain why a continuous path from one to the other is impossible.
