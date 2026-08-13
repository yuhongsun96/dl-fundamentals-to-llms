# Energy, and What Spectra Actually Tell You

"σ₁ alone holds 56% of the Frobenius energy." "62 dimensions capture 90% of the energy." **Energy** is used constantly and defined almost never. It's a simple idea with one non-obvious justification and one important limitation.

**Convention:** `A ∈ R^(m×n)` with singular values `σ₁ ≥ σ₂ ≥ … ≥ σ_min(m,n) ≥ 0`. SVD background: [1.1/05](../1.1_linear_algebra/05_eigen_svd.md).

## Energy = squared magnitude

That's the whole definition. The energy of a vector is `‖v‖² = Σᵢ vᵢ²`. The energy of a matrix is the sum of its squared entries, which is the **squared Frobenius norm**:

```
‖A‖_F² = Σᵢⱼ Aᵢⱼ²
```

The name is borrowed from physics (kinetic energy `∝ v²`, signal power `∝ amplitude²`) and carries no extra content in ML. When someone says "energy," read "sum of squares."

## Why squares rather than absolute values

Not arbitrary. Squares are the unique choice that makes energy **additive over orthogonal components** — the Pythagorean theorem:

```
if  u ⊥ v   then   ‖u + v‖² = ‖u‖² + ‖v‖²
```

This fails for `Σ|vᵢ|` or for `max|vᵢ|`. Additivity is what licenses every "X% of the energy" statement: it lets you *decompose* a total into per-direction contributions that sum to it, so percentages are meaningful.

Two consequences worth having:

**Energy is rotation-invariant.** For any orthogonal `Q` (a rotation/reflection), `‖QA‖_F = ‖A‖_F`. Verified: a random `64 × 32` matrix has `‖A‖_F² = 2058.5659`, and after multiplying by a random orthogonal `Q`, `‖QA‖_F² = 2058.5659` — identical. So energy is a property of the *object*, not of the coordinate system you happen to write it in. (This is the linear-algebra version of Parseval's theorem.)

**Energy and variance are the same measurement.** For a mean-zero vector, `‖v‖²/D` is exactly the variance of its entries. So every "variance preservation" argument in initialization ([2.3/01](../../part2_neural_network_fundamentals/2.3_init_normalization/01_init_variance_preservation.md)) is an energy-conservation argument in different words.

## The spectral decomposition of energy

Here's the fact that makes spectra useful. The SVD writes `A` as a sum of rank-1 pieces, `A = Σᵢ σᵢ uᵢ vᵢᵀ`, and because the `uᵢ` are mutually orthogonal (likewise `vᵢ`), the energies add:

```
‖A‖_F² = Σᵢ σᵢ²
```

Verified on a random `64 × 32` matrix: `Σ Aᵢⱼ² = 2058.5659` and `Σ σᵢ² = 2058.5659`. Exact.

So **each singular value's squared magnitude is that direction's share of the total energy**, and you can rank directions by importance and take partial sums:

```
energy fraction of top k  =  (σ₁² + … + σ_k²) / (σ₁² + … + σ_r²)
```

This is the **cumulative energy** curve. Reading it:

| Shape of the curve | What it means |
|---|---|
| Rises steeply, plateaus early | Genuinely low-rank — a few directions explain the matrix |
| Rises almost linearly to 1 | Flat spectrum, no preferred directions (a random Gaussian matrix looks like this) |
| Steep jump then a long slow climb | One dominant direction plus a fat tail — the common shape for trained matrices |

Verified for a random Gaussian `64 × 32`: top-1 holds **8.9%** of the energy, top-8 holds **50.7%** — spread out, as expected with no structure to find.

And by **Eckart-Young**, truncating to the top `k` singular values is the *optimal* rank-`k` approximation in Frobenius norm, with error exactly the energy you discarded:

```
‖A − A_k‖_F = √(Σ_{i>k} σᵢ²)
```

This is why energy fractions and approximation error are the same conversation.

## The limitation: energy fraction ≠ importance

This is the part that gets skipped, and it matters.

Energy weights every matrix entry equally and measures average-case behavior under uniformly random inputs. Specifically, for a random unit input `x`, `E_x[‖(A − Â)x‖²] = ‖A − Â‖_F² / n`. **Real usage looks nothing like uniformly random inputs.**

The concrete case from this repo: distilbert's embedding matrix has `σ₁` alone holding ~56% of its Frobenius energy — which sounds like "one direction does most of the work, so it must be compressible." It isn't. Reaching 1% relative Frobenius error requires keeping **765 of 768** directions, i.e. essentially **no compression at all** ([1.1 supplementary](../1.1_linear_algebra/supplementary/05_embedding_spectrum.ipynb)). Two reasons the energy story misleads here:

1. **The dominant direction is often an artifact, not a feature.** That `σ₁` is largely a token-frequency effect (the "rogue dimension"), not semantic structure — high energy, low usefulness.
2. **The tail carries the distinctions you care about.** Embeddings are consumed via *inner products* — what matters is the relative geometry between specific token pairs, not the average reconstruction error. Small singular directions can carry the differences that separate tokens, so discarding them costs far more than their energy share suggests.

**The rule to carry:** energy fraction is a statement about *reconstruction in Frobenius norm*, and nothing more. Before believing "X% of the energy means we can drop the rest," ask what the matrix is actually used for and whether Frobenius error is a proxy for that. Usually it isn't.

## Why this matters

- **Reading spectrum plots** in papers — the y-axis is nearly always energy fraction or log `σ`.
- **Justifying low-rank methods** (LoRA, SVD compression, KV-cache compression like MLA). Energy arguments are how the case is made — and the embedding-matrix result above is why the case sometimes fails.
- **Effective rank** ([file 01](01_dimension_span_and_rank.md)) is usually *defined* via an energy threshold.
- **Variance-preservation** arguments in initialization and normalization are energy bookkeeping.
- **PCA** is exactly this: the top-`k` energy directions of a mean-centered data matrix.

## Self-check

1. Why can't you make meaningful "% of the total" statements using `Σ|vᵢ|` the way you can with `Σvᵢ²`?
2. A matrix has singular values `[10, 1, 1, 1]`. What fraction of the energy is in the top direction? Would you call this matrix low-rank?
3. You rotate a matrix by an orthogonal `Q`. Which of these change: its entries, its Frobenius norm, its singular values, its rank?
4. `σ₁` holds 56% of a matrix's energy, yet rank-truncation to 1% error requires keeping 99.6% of the directions. Explain how both can be true.
5. An "energy fraction" argument concludes a weight matrix is 90% compressible. What one question should you ask before believing it?

### Answers

1. Because `Σ|vᵢ|` isn't additive over orthogonal components — `‖u+v‖₁ ≠ ‖u‖₁ + ‖v‖₁` even when `u ⊥ v`. Percentages of a total only make sense when the parts genuinely sum to the whole, and squares are what guarantee that (Pythagoras). With L1 you'd get "shares" that don't add to 100%.
2. Energy is `100 + 1 + 1 + 1 = 103`, so `σ₁²/103 = 97.1%`. And yet **it is not low-rank in any useful sense**: all four singular values are nonzero, and the three small ones are equal — there's no gap to truncate at, and dropping them costs a fixed 2.9% of energy with no natural stopping point. This is the trap in question 4 in miniature: high top-1 energy fraction is compatible with a spectrum you can't truncate.
3. **Entries change.** Frobenius norm, singular values, and rank are **all unchanged** — `QA` has the same SVD up to replacing `U` with `QU`, so `Σ` (hence every `σᵢ`, hence `‖·‖_F` and rank) is identical. This is why these three are considered intrinsic properties of the linear map, while entries are an accident of basis.
4. Because the energy is concentrated but the *remainder* is spread thin across hundreds of directions rather than sitting in a few. After removing `σ₁`'s 56%, the leftover 44% is divided nearly evenly among ~700 directions, so no small subset of them can be dropped without losing more than 1%. Concentration at the top says nothing about whether the tail is truncatable — those are independent facts about the shape of the spectrum.
5. **"Is Frobenius reconstruction error a proxy for how this matrix is actually used?"** For a matrix consumed via inner products against a specific, highly non-uniform distribution of inputs (like an embedding table), it usually is not. The follow-up: measure the downstream metric you care about after truncation, rather than trusting the energy curve.

## Exercise

Take a real weight matrix — an embedding table, or an FFN `W_up` from any small open model — and:

1. Verify `‖A‖_F² = Σσᵢ²` numerically. Then multiply `A` by a random orthogonal matrix and confirm the singular values are unchanged.
2. Plot cumulative energy vs. `k`. Report `k` for 50%, 90%, 99%.
3. Now do the honest check the energy curve can't do for you: truncate to the `k` that gives 90% energy, and measure something *usage-relevant* — e.g. for an embedding table, how much the cosine similarities between 100 random token pairs change. Compare that degradation to the 10% figure the energy curve advertised.
4. Write one sentence on whether you'd ship the truncation.
