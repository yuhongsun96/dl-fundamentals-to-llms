# Supplementary: Euler's formula, and why `e^{iθ}` is a rotation

Companion to `../07_rotations.md`, whose "complex-number shortcut" section leans on `e^{iθ}` without justifying it. It's also the machinery behind RoPE's complex formulation ([Part 5.3/04](../../../part5_transformer_rebuilt/5.3_positional_information/04_rope.md)).

Read this if "a rotation *is* multiplication by `e^{iθ}`" feels like an assertion rather than a fact. Otherwise the one-line version in the primary file is enough.

## What it says

```
e^{iθ} = cos θ + i sin θ
```

Read the right side as a **point**: `(cos θ, sin θ)`. That is exactly the point on the unit circle at angle `θ`. So the formula says:

> **`e^{iθ}` is the point you reach by walking `θ` radians counterclockwise around the unit circle, starting from `1`.**

The reason this is startling: the left side is an *exponential* — the language of growth and compounding — and the right side is *trigonometry* — the language of circles and angles. Euler's formula says they're the same thing once the exponent is imaginary. Growth in an imaginary direction is rotation.

## Why it's true, version 1: the series (the proof)

Expand `e^x = 1 + x + x²/2! + x³/3! + ...` at `x = iθ`, using the fact that powers of `i` cycle with period 4 (`i⁰=1, i¹=i, i²=−1, i³=−i`):

```
e^{iθ} = 1 + iθ + (iθ)²/2! + (iθ)³/3! + (iθ)⁴/4! + (iθ)⁵/5! + ...
       = 1 + iθ −  θ²/2!  −  iθ³/3!  +   θ⁴/4!  +  iθ⁵/5!  − ...
```

Now separate the terms with an `i` from those without:

```
real part:       1 − θ²/2! + θ⁴/4! − ...   =  cos θ
imaginary part:      θ − θ³/3! + θ⁵/5! − ...   =  sin θ
```

Those are precisely the Taylor series for cosine and sine. So `e^{iθ} = cos θ + i sin θ`. The formula isn't a definition or a coincidence — **the cosine and sine series were hiding inside the exponential series all along**, split apart by the alternating powers of `i`.

## Why it's true, version 2: the motion (the intuition)

The series proves it but doesn't explain it. This does.

Two facts:

1. **`e^x` is the function that is its own derivative.** So `\frac{d}{dθ} e^{iθ} = i · e^{iθ}` — the derivative of the position is `i` times the position.
2. **Multiplying by `i` rotates by 90°.** Check it: `i·1 = i` sends `(1,0) → (0,1)`; `i·i = −1` sends `(0,1) → (−1,0)`. Four multiplications return you to the start, so each one is a quarter turn.

Put them together and read `θ` as time. Then:

> **Velocity = position, turned 90°.**

An object whose velocity is always perpendicular to its position vector is moving in a **circle** — perpendicular motion is exactly the motion that doesn't change your distance from the origin. (Formally: `\frac{d}{dθ}|f|^2 = 2\,\mathrm{Re}[\overline{f} \cdot i f] = 2\,\mathrm{Re}[i|f|^2] = 0`, since `i|f|²` is purely imaginary. The radius is constant.)

Starting at `f(0) = 1`, the radius stays `1`, and the speed is `|i·f| = |f| = 1` — constant. On a unit circle, distance travelled equals angle swept, so after "time" `θ` you are at angle `θ`. That point is `(cos θ, sin θ)`.

**Why base `e` specifically?** Because `e` is the base whose derivative factor is exactly 1. For any other base, `\frac{d}{dθ}b^{iθ} = i\ln(b)\,b^{iθ}` — you'd circle at speed `ln(b)` instead of `1`, so `θ` would no longer measure radians. `e` is the base that makes "one unit of exponent = one radian."

## Consequences you actually use

Everything below follows from the one formula, and each is why the complex notation beats the matrix notation.

**Magnitude is always 1.** `|e^{iθ}| = √(cos²θ + sin²θ) = 1`. Multiplying by `e^{iθ}` can never change a vector's length — the complex-number statement of "rotations are norm-preserving."

**Angles add, because exponents add.**

```
e^{iα} · e^{iβ} = e^{i(α+β)}
```

This single line replaces the rotation-composition theorem `R(α)R(β) = R(α+β)`. It's just the exponent rule.

**Conjugation is the inverse rotation.** `\overline{e^{iθ}} = e^{−iθ} = 1/e^{iθ}`. Flipping the sign of the imaginary part flips the angle — replacing the matrix fact `R(θ)^T = R(−θ)`.

**Polar form separates size from angle.** Any complex number is `z = r e^{iθ}` — magnitude `r`, angle `θ`. Multiplication becomes trivial:

```
(r₁ e^{iθ₁}) · (r₂ e^{iθ₂}) = r₁r₂ · e^{i(θ₁+θ₂)}      multiply magnitudes, add angles
```

**Repeated rotation is free.** `(e^{iθ})^n = e^{inθ}` — spinning `n` times is one multiplication in the exponent.

### The trig identities fall out

A nice check that the formula carries real content. Expand both sides of `e^{i(α+β)} = e^{iα}e^{iβ}`:

```
LHS:  cos(α+β) + i sin(α+β)
RHS:  (cos α + i sin α)(cos β + i sin β)
    = (cos α cos β − sin α sin β)  +  i (sin α cos β + cos α sin β)
```

Match real parts and imaginary parts and you have **both** angle-addition formulas at once. The trig identities memorized in school are just the statement that rotations compose by adding angles.

### Special values

| `θ` | `e^{iθ}` | As a point | Meaning |
|---|---|---|---|
| `0` | `1` | `(1, 0)` | no rotation |
| `π/2` | `i` | `(0, 1)` | quarter turn |
| `π` | `−1` | `(−1, 0)` | half turn |
| `3π/2` | `−i` | `(0, −1)` | three-quarter turn |
| `2π` | `1` | `(1, 0)` | full turn, back to start |

The `θ = π` row rearranges to `e^{iπ} + 1 = 0`, Euler's *identity* — famous for tying together `e`, `i`, `π`, `1`, and `0`, but it's just "a half turn takes you from `+1` to `−1`."

## Where this shows up

- **[`../07_rotations.md`](../07_rotations.md)** — an `n`-dimensional rotation, viewed as `C^{n/2}`, is "multiply each complex coordinate by its own `e^{iθ_d}`." The canonical block-diagonal form in one line.
- **[RoPE](../../../part5_transformer_rebuilt/5.3_positional_information/04_rope.md)** — queries and keys are cast to complex, multiplied by a precomputed `e^{i·pos·θ_d}` table, and cast back. The relative-position property is then just `e^{imθ}e^{−inθ} = e^{i(m−n)θ}`.
- **[Sinusoidal position encodings](../../../part5_transformer_rebuilt/5.3_positional_information/02_sinusoidal_and_learned_absolute.md)** — the `sin`/`cos` frequency ladder is the same object with the real and imaginary parts written out separately.
- **Fourier transforms / the DFT matrix** — built entirely from `e^{−2πikn/N}`; the "frequency" view of any signal is a sum of rotations at different rates.

## One-line summary

> `e^{iθ}` is the unit-circle point at angle `θ`. Multiplying by it rotates, because the exponential's defining property (derivative = itself) combined with `i`'s defining property (multiplying by it turns 90°) forces motion that is always perpendicular to the radius — i.e. a circle. Everything else — angles adding, conjugates inverting, magnitudes preserved — is the exponent rule doing the bookkeeping a rotation matrix would otherwise need theorems for.
