# Supplementary: "Skip vs take" is a decomposition, not routing

Companion to [`../03_ensemble_of_shallow_paths.md`](../03_ensemble_of_shallow_paths.md). Clears up the single most common confusion about the ensemble view.

**The confusion:** the unraveled network is a *sum* over `2^L` paths — so every path is always contributing. Then why does the ensemble view talk about "taking or skipping" paths, as if some are switched off? Nothing is being switched off, so what does "skip" mean?

**The answer in one line:** "skip vs take" is not a runtime decision — it's a *label on the terms of a sum that always computes all of them at once*. Every path is always taken; the vocabulary just names which sublayers `f_l` appear in each summand.

## It's distributivity, not routing

Each residual block is a **sum of two operators**: `h + f(h) = (I + f)(h)`. Composing `L` blocks and expanding is just distributing a product of sums — and a product of sums expands into every combination of picking one term from each factor. The clean analogy:

```
(a + b)(c + d) = ac + ad + bc + bd
```

All four products are in the result simultaneously. Saying "the `ac` term skips `b` and `d`" does **not** mean `b, d` were turned off — it means *that summand happened to pick `a` from the first factor and `c` from the second*. Same object, four contributing terms, each labeled by which choice it made at each factor.

A residual net is exactly this with `L` factors of `(I + f_l)`:

```
(I + f_1)(I + f_2) ··· (I + f_L)   →   a sum of 2^L terms
```

Each term picks either `I` or `f_l` at every block. **A "path" is one of those summands.** "Path skips block 3" means *this summand took the `I` at block 3, so `f_3` doesn't appear in it* — precisely like "the `ac` term." Every summand is present in the total at once. There is no skipping *event*; there's a single deterministic forward pass that, expanded algebraically, equals a sum over `2^L` terms, each named by which blocks it routes through vs. bypasses via the identity.

## Then why bother with the path fiction?

Because the decomposition predicts real, measurable behavior the "one deep function" view can't:

- **Lesioning is exact in path terms.** Delete block `i` at test time → you remove exactly the summands containing `f_i`, and every summand that took `I` at block `i` survives untouched. That's *why* accuracy barely drops (main doc, lesion section) — evidence the sum really does behave like an ensemble with no single load-bearing member.
- **Short paths dominate learning.** Long summands (many `f`'s composed) carry almost no gradient, so the short ones drive training — a statement about *which terms in the sum matter*, testable by measuring gradient magnitude per path length. This is the "effective depth ≪ `L`" result.

Neither claim makes sense for "one monolithic deep computation," but both fall out cleanly once you read the single sum as a sum over its terms. The path picture earns its keep by predicting what happens when you *delete* members and *which* members do the learning.

## The honest caveat

Because the `f_l` are **nonlinear**, the terms are not truly independent functions — `f_3` is applied to `h_2`, which already contains contributions from `f_1, f_2`. So "a sum of `2^L` independent sub-networks" is exact only in a linearized reading; strictly it's an approximation, and attention couples the paths further (main doc, caveats section). As a *first-order* model it's both algebraically grounded (distributing `∏(I + f_l)`) and empirically confirmed (lesions, gradient-by-length).

## The takeaway

Every path *is* always taken. "Skip vs take" simply labels which `f_l` appear in each of the `2^L` additive terms that the network's one forward pass expands into. The ensemble view is reading that single sum as a sum over its terms — not proposing that any path is switched off at runtime. (Techniques like stochastic depth / LayerDrop are the *separate* case where you genuinely *do* drop terms — during training, on purpose — which is coherent precisely because the identity path keeps the forward pass well-defined when an `f_l` is removed.)
