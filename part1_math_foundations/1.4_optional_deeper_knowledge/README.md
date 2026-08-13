# 1.4 Optional Deeper Knowledge

**This subsection is a vocabulary layer, not new theory.** Everything here is the mathematical language that gets used *casually* in explanations of modern architectures — "the position signal occupies a low-dimensional subspace," "σ₁ holds 56% of the energy," "the score is a bilinear form," "RoPE is a one-parameter subgroup of `SO(D_h)`," "to first order the variance is preserved." If you've hit an explanation where the ML content was clear but a phrase like *intrinsic dimension*, *skew-symmetric generator*, or *abelian* slid past unexamined, this is the section that pins those down.

It is **optional and non-linear**. Nothing in Parts 2–12 requires reading it front to back. Treat it as a reference: find the group below whose vocabulary is currently costing you, read those files, come back later for the rest.

## The four groups

### A — Space and size *(what lives where, and how much room it takes)*

| File | Answers |
|---|---|
| [01_dimension_span_and_rank.md](01_dimension_span_and_rank.md) | What is a subspace, a span, a basis? What does "1-D subspace" mean, and how is it different from a 1-D *curve*? Ambient vs. linear vs. intrinsic dimension. Rank and effective rank. |
| [02_energy_and_spectra.md](02_energy_and_spectra.md) | What is "energy"? Why squares? Why `‖A‖_F² = Σσᵢ²`, and what does "90% of the energy in 62 dimensions" *not* claim? |

### B — Forms and read-out *(how information gets combined and extracted)*

| File | Answers |
|---|---|
| [03_bilinear_and_quadratic_forms.md](03_bilinear_and_quadratic_forms.md) | What is `x M yᵀ`? If `M` is a matrix, how can the form be nonlinear? Why does an attention score split into four terms when the input is a sum? Gauge freedom; curvature. |
| [04_linear_readout_and_identifiability.md](04_linear_readout_and_identifiability.md) | If you *add* two signals, when can a learned linear layer pull them apart? Why doesn't adding a positional encoding corrupt the token? When is recovery provably impossible? |

### C — Symmetry and rotation *(the language of RoPE)*

Read these three in order — each depends on the one before.

| File | Answers |
|---|---|
| [05_complex_eigenvalues_and_rotation_planes.md](05_complex_eigenvalues_and_rotation_planes.md) | Why does a *real* matrix have complex eigenvalues? What do conjugate pairs mean geometrically? Why `λ = r·e^(iθ)` splits into growth and oscillation. Why rotation planes are 2-D. |
| [06_groups_and_the_rotation_group.md](06_groups_and_the_rotation_group.md) | What is a group? What does **abelian** mean and why do 2-D rotations commute while 3-D ones don't? What are `O(n)` and `SO(n)`, and what does the "S" add? |
| [07_generators_and_one_parameter_families.md](07_generators_and_one_parameter_families.md) | What is a **one-parameter family**? What is a **skew-symmetric generator** and the matrix exponential? The decomposition theorem — and therefore **why RoPE uses 2-D pairs and not 3-D or larger**. |

### D — Signals and estimates *(reading quantitative claims)*

| File | Answers |
|---|---|
| [08_frequency_phase_and_periodicity.md](08_frequency_phase_and_periodicity.md) | Frequency vs. wavelength vs. phase. Why geometric spacing. Aliasing and the Nyquist limit. Why a set of frequencies "never repeats." |
| [09_approximations_and_orders_of_magnitude.md](09_approximations_and_orders_of_magnitude.md) | What "to first order" means. Why `sin x ≈ x` and when it stops being true. How to read `O(1/√D)`, "scales like," and log-log plots. |

## Where each group is load-bearing

- **A** is the language of the SVD, LoRA's low-rank updates, embedding spectra, and the claim that a positional signal is "small" relative to `d_model`.
- **B** is the language of attention itself — `W_Q W_Kᵀ` as one bilinear form — plus second-order optimization and why additive positional encoding works at all.
- **C** is the language of RoPE, orthogonal initialization, and any "equivariance" discussion. Group C also feeds directly into D: a 2-D rotation block **is** a `(sin, cos)` pair advancing in phase, so 07 and 08 are two descriptions of one object.
- **D** is the reading skill behind sinusoidal and rotary encodings, context-length extension, variance-preservation arguments, and scaling laws.

## Prerequisites and overlap

These files assume [1.1 Linear Algebra](../1.1_linear_algebra/) and cross-reference it rather than repeating it — 1.1 already covers inner products and norms ([03](../1.1_linear_algebra/03_inner_products_norms.md)), low-rank structure ([04](../1.1_linear_algebra/04_outer_products_low_rank.md)), eigenvalues and the SVD ([05](../1.1_linear_algebra/05_eigen_svd.md)), and projections, orthogonal matrices, near-orthogonality and Johnson-Lindenstrauss ([06](../1.1_linear_algebra/06_projections_orthogonality.md)). Where a concept lives there, these files point to it and add only the part that gets used loosely elsewhere. Information-theoretic vocabulary lives in [1.3](../1.3_information_theory/).

## Target time

A day if you read all nine. Much less if you use it as intended — one group at the moment you need it. Group C is the most self-contained and the most worth reading end-to-end, since the three files build a single argument.
