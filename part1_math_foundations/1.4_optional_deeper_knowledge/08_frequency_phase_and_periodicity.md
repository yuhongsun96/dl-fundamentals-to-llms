# Frequency, Phase, and Periodicity

Sinusoidal positional encodings, RoPE, and every context-extension method are described in signal-processing vocabulary — frequency, wavelength, phase, aliasing, base. None of it is hard, but it's unfamiliar to a practitioner whose background is embeddings and gradients rather than DSP. This file is the glossary plus the three facts that actually do work.

## The basic quantities

For `f(i) = sin(ω i + φ)`:

| Symbol | Name | Meaning |
|---|---|---|
| `ω` | **angular frequency** | radians advanced per step of `i`. Bigger = faster oscillation. |
| `2π/ω` | **wavelength** (or period) | how many steps of `i` for one full cycle. Bigger = slower. |
| `φ` | **phase** | where in the cycle you start; shifts the wave left/right without changing its shape. |
| amplitude | — | the multiplier out front; always 1 in positional encodings. |

The only conversion to keep straight is that **frequency and wavelength are reciprocal**. In the sinusoidal `PE` formula the denominator `10000^(2d/D)` *is* the wavelength scale, so a **larger** denominator means a **lower** frequency and a **longer** wavelength — slower oscillation. That inversion is the single most common source of confusion in reading these formulas.

Concretely, at `D = 512`: `ω` runs from `1` (wavelength ≈ 6.3 tokens) down to `10⁻⁴` (wavelength ≈ 62,832 tokens).

## Fact 1: `(sin, cos)` at one frequency is a rotating point

A pair `(sin θ, cos θ)` is a point on the unit circle at angle `θ`. Advancing position by `k` advances the angle by `ωk`, and the angle-addition identities say that advance is a **rotation matrix**:

```
[ sin(ω(i+k)) ]   [ cos(ωk)   sin(ωk) ] [ sin(ωi) ]
[ cos(ω(i+k)) ] = [ −sin(ωk)  cos(ωk) ] [ cos(ωi) ]
```

The matrix depends only on the **offset** `k`, never on the absolute position `i`. Three things follow:

- **Rotations preserve length**, so `‖PE(i)‖` is the same for every position.
- **Rotations preserve inner products**, so `PE(i)·PE(j)` depends only on `i − j`.
- A rotation needs **two** coordinates. This is why frequencies come in `(sin, cos)` pairs rather than single sines — the pair is the minimum unit in which "advance by `k`" is a linear operation.

RoPE is precisely the move of applying this rotation directly to `q` and `k` instead of adding a vector at the input ([5.3/04](../../part5_transformer_rebuilt/5.3_positional_information/04_rope.md)). A `(sin, cos)` pair advancing in phase *is* a 2-D rotation block — the same object described two ways. Why rotations decompose into 2-D pairs in every dimension, and never need to be larger, is [file 07](07_generators_and_one_parameter_families.md).

## Fact 2: sampling at integers creates aliasing, and Nyquist is the limit

Positional encodings are only ever evaluated at **integer** `i`. That has a consequence with no continuous analogue: **frequencies that differ by `2π` are indistinguishable.**

```
sin((ω + 2π)i) = sin(ω i)      for every integer i
```

Verified exactly. So `ω` and `ω + 2π` are the same function as far as the model can tell — and worse, `ω = 2.0` and `ω = 2.0 − 2π = −4.283` are also identical when sampled at integers (verified). This is **aliasing**: a fast oscillation, sampled coarsely, is impersonated by a slow one.

The consequence is a ceiling. Only `ω ∈ [0, π]` gives distinguishable frequencies; anything faster folds back down. That ceiling is the **Nyquist limit**, `ω_max = π ≈ 3.1416`.

Now check the sinusoidal design against it: the fastest frequency is `ω₀ = 1`, comfortably **below** `π`. So the ladder is bounded at *both* ends deliberately:

| End of the ladder | Constraint | Why |
|---|---|---|
| **Fast** (`ω₀ = 1`) | must stay below Nyquist `π` | above it, frequencies alias onto slower ones and carry no new information |
| **Slow** (`ω_min = 10⁻⁴`) | wavelength (≈62,832) must exceed sequence length | otherwise the coarse "where am I" signal wraps around and repeats |

Both ends are aliasing constraints — the same failure at two scales. And the slow-end constraint is exactly what breaks when you extend context beyond the trained length, which is why context-extension methods adjust the base ([5.3/05](../../part5_transformer_rebuilt/5.3_positional_information/05_context_length_extension.md)).

## Fact 3: incommensurate frequencies never repeat

A single clock repeats every wavelength. Many clocks together repeat only when they *simultaneously* return to their starting angles — which requires their wavelengths to have a common multiple.

Two frequencies are **commensurate** if their ratio is rational (a common period exists) and **incommensurate** if irrational (no common period ever). Geometrically, the `D/2` pairs trace out a trajectory on a **torus** — the product of `D/2` circles — and with incommensurate frequencies that trajectory is **quasi-periodic**: it comes arbitrarily close to repeating but never exactly does.

In the sinusoidal ladder consecutive frequencies have ratio `10000^(2/D)`, which is irrational for essentially any `D`. **That non-repetition is the guarantee that every position gets a unique encoding**, and it's why the scheme works for arbitrarily long sequences in principle.

The caveat that makes "in principle" load-bearing: uniqueness is not the same as *distinguishability under noise*, and it says nothing about whether the model has ever learned to read the phase combinations that occur past its training length. Sinusoidal `PE` is well-defined at position 100,000 and still performs badly there.

## Why geometric spacing

The frequencies are geometrically spaced (each `10000^(2/D)` times the last) rather than linearly. Reasons:

- **Coverage.** A geometric ladder spans four orders of magnitude in `D/2` steps. Linear spacing would cluster nearly all frequencies at similar speeds — highly redundant dimensions and no long-range resolution.
- **Scale-free structure.** Relative resolution is constant: each step resolves distances about `1.3×` longer than the last, so no length scale is privileged. Language has structure at every scale (adjacent words, clauses, paragraphs), so a scale-free encoding is the natural match.
- **It's the same reason we use log scales** for anything spanning orders of magnitude — see [file 09](09_approximations_and_orders_of_magnitude.md).

## Why this matters

- **Sinusoidal `PE`** ([5.3/02](../../part5_transformer_rebuilt/5.3_positional_information/02_sinusoidal_and_learned_absolute.md)) and its supplementary deep dive are written entirely in this vocabulary.
- **RoPE** ([5.3/04](../../part5_transformer_rebuilt/5.3_positional_information/04_rope.md)) is Fact 1 relocated into the attention dot product.
- **Context extension** ([5.3/05](../../part5_transformer_rebuilt/5.3_positional_information/05_context_length_extension.md)) — position interpolation, NTK-aware scaling, YaRN — are all reallocations of this frequency ladder, and they're unreadable without knowing what "base," "wavelength," and "high-frequency dims" mean.
- **Rotations preserve norms and inner products**, which is why they're the safe way to inject information into a stream you don't want to perturb.

## Self-check

1. In `10000^(2d/D)`, does a larger value mean faster or slower oscillation? Which dimensions have large values?
2. Why can't a positional encoding use a frequency of `ω = 4`?
3. Why do the frequencies come in `(sin, cos)` pairs rather than `D` independent sines?
4. The slowest wavelength is ≈62,832 tokens and you want to train at `S = 100,000`. What breaks, and at roughly what position does the problem first appear?
5. What makes every position's encoding unique, given that each individual dimension repeats?

### Answers

1. **Slower.** `10000^(2d/D)` is the wavelength scale and appears in the *denominator* of the angle (`i / 10000^(2d/D)`), so large values mean small angular frequency, long wavelength, slow oscillation. Large values occur at **large `d`** — the top of the ladder — reaching ~10000 at `d = D/2 − 1`.
2. Because `4 > π`, above the Nyquist limit for integer sampling. `sin(4i)` at integer `i` is indistinguishable from `sin((4 − 2π)i) = sin(−2.283i)` — it *aliases* onto a slower frequency, so it adds no information beyond what a slower dimension already carries, while looking erratic. Anything above `π` is wasted capacity.
3. Because the useful operation is "advance position by `k`," which is a **rotation**, and a rotation requires a two-dimensional plane. The pair makes `PE(i+k)` a fixed linear function of `PE(i)` depending only on `k` — which in turn gives constant norm and offset-only inner products. `D` independent sines would have none of these properties, and a lone sine is also ambiguous (`sin θ = sin(π − θ)`), so it couldn't even distinguish `+k` from `−k`.
4. The slowest clock completes more than one revolution inside the trained range, so the **coarse positional signal wraps**: positions about 62,832 apart get nearly identical encodings across the whole ladder, making them unidentifiable. The problem first appears around `i ≈ 62,832`, where the slowest pair returns to its starting angle — but degradation begins earlier, since the slow dims stop being monotone as they approach the top of their cycle. The fix is to raise the base (lengthen all wavelengths), which is the core of NTK/YaRN scaling.
5. **Incommensurate frequencies.** Each dimension individually repeats every `2π/ω_d` steps, but with irrational frequency ratios there is no position at which *all* the clocks simultaneously return to their starting angles. The joint trajectory on the torus is quasi-periodic — it never exactly repeats — so the full vector is unique even though every component is periodic.

## Exercise

1. Confirm the aliasing claim: check `sin((ω + 2π)i) = sin(ωi)` for integer `i` and a few `ω`. Then plot `sin(4i)` and `sin((4 − 2π)i)` at integer `i` and observe they're the same points.
2. For the `D = 512` ladder, print `ω_d` and wavelength for `d = 0, 8, 16, 24, 255`. Verify `ω_0 < π` and note the margin.
3. Take two frequencies with a **rational** ratio (say `ω = 1` and `ω = 0.5`) and confirm the joint `(sin, cos)` trajectory exactly repeats. Then use the real ratio `10000^(2/512)` and show it doesn't repeat over 10,000 positions (measure the minimum distance between `PE(i)` and `PE(0)` for `i > 0` and confirm it never reaches 0).
4. Build `PE` with a base of `100` instead of `10000` and find the position where the slowest dimension completes a full cycle. Confirm that two positions separated by that distance have near-identical encodings — you've reproduced the wraparound failure mode on purpose.
