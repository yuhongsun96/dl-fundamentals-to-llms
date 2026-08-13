# Entropy Rate, Optimal Compression, and Scaling Limits

**The fact this doc unpacks**: English has an entropy rate of ~1–1.5 bits per character (Shannon's original estimate, confirmed by modern LMs), and frontier models are approaching that floor. That single observation is doing more work in the current "are we hitting a wall?" conversation than most people realize, and it's also the source of more confusion than most realize.

This doc explains what "approaching the entropy floor" actually means in precise information-theoretic terms, why it implies a real diminishing-returns story on one specific axis of scaling, and why the broader conclusion *isn't* "scaling is dead" — it's "the original scaling axis is bending against a hard floor, but new axes have opened up that are nowhere near theirs." The loss function powering frontier capability has also shifted in step with this, in a way that's worth understanding explicitly.

## Background — the four concepts this doc rides on

Brief refreshers, in case any are rusty.

**Entropy** `H(p) = −Σ p_i log p_i` of a distribution `p`. Read as the *average surprise per draw from `p`*. The building block is `−log p_i` (the "surprise" of outcome `i` — zero for certain events, large for rare ones), and you average it weighted by how often each outcome actually happens (`p_i`). Zero when one outcome is certain; maximal when all outcomes are equally likely. In bits, it's literally the minimum average number of yes/no questions needed to identify a draw from `p` (Shannon's source coding theorem).

**Cross-entropy** `H(p, q) = −Σ p_i log q_i` of a true distribution `p` and a predicted distribution `q`. Read as the average surprise you'll feel *if you assume `q` but reality is `p`*. Surprise is computed from your beliefs (`−log q_i`), but averaged over reality's frequencies (`p_i`). Always ≥ `H(p)`, with equality iff `q = p`. **This is the LM training loss**: for each token position, `q` is the model's softmax output, `p` is the one-hot true next token, and the loss is `−log q(true_token)` — the model's surprise at the right answer. Minimizing it = training the model to stop being surprised by real data.

**Perplexity** `PPL = exp(H(p, q))`. The human-readable form of cross-entropy. Interpretable as the *effective number of equally-likely next-token choices* the model is torn between. PPL = 10 means the model is as uncertain as if it were choosing uniformly among 10 options. CE is what backprop differentiates; PPL is what humans look at. Since `exp` is monotonic, `argmin(CE) = argmin(PPL)` — same optimization, different units.

**Entropy rate** of a stochastic process (like a language): the per-symbol entropy in the limit of long sequences, conditioning each new symbol on the full preceding context. Formally `H = lim_{n→∞} (1/n) H(X_1, ..., X_n)` for stationary sources. Intuitively: "how much new uncertainty does each additional character/token bring, on average, once you've seen everything before it?" For English, this is ~1–1.5 bits/char. For code or formal proofs, it's substantially lower (more structure → next-token more predictable from context).

One more identity to keep in mind: **cross-entropy on real data = entropy rate + KL divergence from the model to reality.** So the lower bound on any model's CE on a given data distribution is the entropy rate of that distribution. Pushing CE down to the entropy rate means the model has matched the true distribution as well as anyone ever could.

## Part 1: What "approaching optimal compression" actually means

There's a deep equivalence sitting underneath: **a probability model and a lossless compressor are the same object.** Specifically:

- A predictor that assigns probability `q(x)` to each next token can be plugged into **arithmetic coding** to produce a compressor whose average code length per token is exactly `−log q(x)` — the cross-entropy of the model on the data.
- Conversely, any compressor implicitly defines a probability model.

So an LM with `CE = 1.5 bits/char` on English isn't *just* good at predicting — it's *literally a 1.5-bit-per-character compressor of English*. You could use it with arithmetic coding to compress a novel at ~1.5 bits/char. People have demonstrated this experimentally; it works.

Now stack the two key facts:

- Shannon proved **no compressor can do better than the entropy rate of the source.** English's entropy rate is ~1–1.5 bits/char — a property *of the language*, not of any predictor. It's the irreducible uncertainty left over after you've understood everything understandable about English.
- GPT-4-class models hit roughly that number on held-out clean text.

Therefore: **frontier LMs are approaching the theoretical limit of how well any model — present, future, or hypothetical — can predict English.** The remaining gap isn't ignorance; it's that English genuinely has that much intrinsic randomness in it. When two synonyms fit a context equally well, when an author picks a comma over a semicolon, when a sentence could legitimately continue or end — that's real variability the model can't be punished for missing.

This is a much stronger claim than "the model is good." It's "the model has converged to near-optimal," with the optimality measured against an unconditional lower bound from information theory.

## Part 2: Why this matters for scaling and data

Scaling laws (Kaplan, Hoffmann/Chinchilla) say: as you scale compute, parameters, and data together, cross-entropy on held-out text drops as a predictable power law. For a long time "just scale" worked because we were far from the floor and the curve was nearly linear in log-compute.

But CE has an asymptote — the **entropy rate of the data distribution**. You can't push below it. So the scaling curve flattens as it approaches:

```
Loss
 │
 │  large headroom regime (early scaling: each 10× compute → big quality jump)
 │\
 │ \
 │  \
 │   \____
 │        \________  ← we live around here on clean text
 │                 ──────────  ← Shannon floor for English
 │ ────────────────────────────  ← (entropy rate of English)
 └────────────────────────────────────── Compute / Data / Params
```

Two consequences:

**(a) Pretraining scaling has diminishing returns on saturated distributions.** Going from 2× the floor to 1.5× is a much bigger lift than 10× → 5×. Each subsequent halving of (CE − entropy_rate) costs more compute than the last. We aren't out of headroom *everywhere*, but we're out of headroom on "predict the next character in clean English prose." That's why frontier evals have shifted from "perplexity on Wikipedia" (saturated, uninformative) to reasoning, code, and multi-step problem-solving — places models are still far from any plausible floor.

**(b) "Running out of data" has a precise meaning.** If the data distribution has entropy rate `H` and current models hit near-`H` on available data, more data *of the same distribution* doesn't help much — you've extracted everything extractable. The remaining headroom lives in **other distributions**: code, math, multimodal, reasoning chains, each with their own entropy rates that we're nowhere near. This is one of the deepest arguments for why **synthetic data and post-training are now central** — they push performance in regimes that aren't yet against the floor.

## The rough strategic picture

- **Until ~2023**: lots of headroom on web-text pretraining. "Just scale" dominates. CE on held-out text is the right north-star metric.
- **Now**: pretraining bumps into the floor on the distributions it was trained for. CE/perplexity improvements on standard corpora are noisy and uninformative. The action moves to (i) data distributions still far from their floors and (ii) objectives other than next-token prediction (RLHF, RL on verifiable rewards, test-time reasoning).
- **Long term**: every text-like distribution eventually saturates. Real gains then come from regimes not subject to a Shannon-style floor at all — test-time compute (think longer to find better answers), tool use (offload computation to the world), new modalities.

## Part 3: But wait — does the "wall" apply to math, code, and reasoning?

The previous parts describe what's happening on **general web text**. A natural follow-up: are math, code, and reasoning subject to the same floor, or is the "scaling has stalled" narrative overgeneralizing from one specific distribution?

**Short answer: the wall is only on the pretraining-on-web-text axis. Other axes have lots of room.**

### Structured language has a lower text-entropy floor

Code, math, and formal proofs are *more compressible* than prose:

- **Code** follows strict grammars. After `def foo(`, the next token is far more constrained than after "The cat sat on the". Indentation, naming conventions, and library idioms collapse the next-token distribution.
- **Math** has formal syntax with precise semantics. A proof step is constrained by what was just established.
- **Formal proof languages** (Lean, Coq) are nearly deterministic given enough context.

Empirically, trained LMs achieve per-token CE on code roughly 2× lower than on prose, and even lower on formal proofs. As text distributions, these have lower Shannon floors *and* current models are not particularly close to those floors on long-context corpora yet. That's already headroom you can scale into.

### The more important shift: capability ≠ text modeling

For reasoning tasks, **the relevant metric isn't text-perplexity at all**. It's: *did the model produce a correct solution?* Pass@1 on math benchmarks, solve rate on coding problems, accuracy on GPQA. These are not log-likelihoods and **are not subject to Shannon's bound**.

The floor for "fraction of problems you can solve" is 100%, not some entropy number. As long as the problem is solvable in principle, there's no information-theoretic ceiling — only compute, algorithm quality, and search efficiency. The optimization changes shape entirely:

- Pretraining LM on prose → fighting against an entropy rate.
- Optimizing a model to solve reasoning problems → fighting against compute and algorithms, *not* against an entropy floor.

### Four scaling axes, only one saturating

When people say "scaling has hit a wall," they almost always mean *one specific axis*: pretraining-on-web-text. At least four distinct axes exist, and they aren't all saturated:

| Axis | What you scale | Status |
|---|---|---|
| **Pretraining on web text** | params + general web data + compute | Diminishing — approaching English's entropy rate |
| **Pretraining on structured corpora** | params + more code/math/reasoning data | Still alive — lower entropy floor + smaller corpora to date |
| **Post-training / RL on verifiable rewards** | RL compute, reward signal density, verifier quality | Very alive — currently the main driver of frontier gains |
| **Test-time compute** | tokens spent thinking per problem (CoT, reflection, search) | New axis — no near-term ceiling observed |

The "wall" people see is *only on the first row*. The other three are where the field has pivoted, and they're producing the most dramatic capability gains right now (o1/o3-style reasoning models, R1-style RL-distilled chains-of-thought, agentic coding systems).

### Why reasoning specifically has lots of headroom

- **The data is sparse and gettable.** Most of the internet is conversational English and narrative writing. Structured reasoning, verified proofs, correct derivations — these make up a tiny fraction of pretraining corpora. Even within the entropy-rate framing, the reasoning-flavored slice of the data distribution is underexploited and can be synthesized further.
- **Verification is cheap.** A math problem either is solved or isn't. A program either passes tests or doesn't. This asymmetry enables **RL with verifiable rewards** at scale — a luxury you don't have for "is this paragraph well-written?". Whole training procedures work for reasoning that don't work for prose.
- **Test-time compute changes the rules.** A standard LM gets one forward pass per token. A reasoning model can spend arbitrary tokens "thinking" before committing to an answer. Empirically, accuracy on hard problems scales smoothly with thinking length, with no observed ceiling. This is genuinely orthogonal to pretraining — and isn't entropy-bounded, it's bounded by how cleverly you can search/refine.
- **Compositionality and tool use.** Once a model can decompose hard problems, call external tools (code interpreters, formal verifiers, search), and synthesize results, the relevant entropy isn't of the *whole* problem-solving process — it's of small local sub-steps, each of which is much more compressible/learnable.

### The updated scaling-laws picture

The original Kaplan/Chinchilla laws are about pretraining loss vs. compute, and that curve is bending against the entropy rate of web text. But other scaling laws are emerging that haven't bent:

- **RL-scaling laws** (2024–2025 work): predictable improvement in pass-rate as you scale RL compute, no signs of immediate saturation.
- **Inference-compute scaling laws** (o1 paper, Snell et al.): solve rate on hard benchmarks scales as a power law in test-time compute, curve still straight.
- **Reasoning-data scaling**: targeted high-quality reasoning corpora (verified proofs, curated solutions, synthetic chains) keep producing capability gains beyond what their raw token counts predict.

So instead of "scaling has hit a wall," the more accurate statement is: **the original power law on pretraining-loss is bending, but new power laws on other axes are emerging that haven't bent yet.** Much less dramatic than the headline version.

### Caveat — does math/code eventually hit its own floor?

In principle, yes. Every distribution has some entropy rate; once approached, text-modeling perplexity on that distribution saturates too. But:

1. We're far from that point empirically — current pass rates on hard math benchmarks (Putnam, IMO-level, FrontierMath) are well below 100%.
2. Even at 100% on existing benchmarks, harder problems can be generated. We're not running out of math; we're running out of *easy* math.
3. The relevant ceiling for capability gain isn't text entropy at all — it's "what problems can be verified," and that frontier keeps moving as models get better.

So the "we'll eventually hit a floor on reasoning too" objection is true in principle but plausibly years away, not quarters.

## Part 4: Has the loss function itself shifted?

If the floor isn't hit on reasoning, can we just keep training with CE on more deterministic domains forever? Yes — but only up to a point, and the *capability-generating* loss has dramatically shifted away from CE. The structure of modern training has changed in a way that's worth understanding explicitly.

### Why CE alone leaves capability on the table — even on low-entropy domains

CE is a **per-token local loss**. Reasoning is a **global property of an entire chain**. CE penalizes each token against the next observed token; it doesn't know whether the chain of tokens, taken as a whole, *arrives at the right answer*. You can have a model that's perfectly calibrated per-token but still concludes wrong, because token-level calibration doesn't compose into chain-level correctness.

Three concrete failure modes when relying on CE alone in a low-entropy domain like math or code:

- **Coverage vs. density.** CE makes the model good at predicting proofs *that exist in the corpus*. The proof for a new theorem isn't in the corpus, and CE never directly rewards "found a novel valid proof" — only "predicted the next token of a known one."
- **Search vs. imitation.** Solving a hard problem requires searching over candidate steps, evaluating them, backtracking. Pure imitation (CE) learns to *replay* a good proof, not to *discover* one.
- **Credit assignment over long chains.** A reasoning trace has many tokens but only the final answer determines correctness. CE weights every token equally regardless of whether it contributes to the conclusion. End-of-trajectory rewards (RL) assign credit correctly to the steps that mattered.

So CE on low-entropy domains is *necessary but not sufficient* — there's entropy headroom, but token-level CE can't fully exploit it for chain-level capability.

### The modern training stack (frontier model recipe, late 2025 / early 2026)

```
1. Pretraining          → CE on next-token, huge web/code/math corpus.     ★ still CE
2. SFT                  → CE on next-token, curated instruction data.       ★ still CE
3. Preference tuning    → DPO/IPO/SimPO/KTO. CE-flavored but response-      ◐ CE-shaped,
                          level (preferred vs. rejected pair).                response unit
4. RL w/ verifiable     → PPO/GRPO. Sample response → verify with checker   ✗ NOT CE.
   rewards (RLVR)         (tests, answer match, proof checker) → policy
                          gradient: −E[reward · ∇ log π(response)] + β·KL.
```

Steps 1–2 are token-level CE. Step 3 is logistic regression on preference pairs — technically CE on a binary label, but the unit being graded is a full response, not a token. Step 4 is **not CE at all** — it's policy gradient with reward from an *external verifier* (Python interpreter, unit tests, answer checker), shaped by a KL penalty to a reference model.

The deepest shift: the reward signal in step 4 comes from *outside the model entirely*. Whether the response was "correct" is decided by something other than predicting the next token. This is qualitatively different from anything in steps 1–3.

### Timeline of where compute and capability gains live

- **2020–2022**: ~99% of useful training is CE pretraining. RLHF is a thin polish layer. Scaling laws are king; bigger pretrained model = better everything.
- **2023–2024**: Pretraining still dominates by compute, but RLHF/DPO becomes important enough that "post-training recipe" matters as much as the base model. ChatGPT, Claude, Gemini all differentiated as much by post-training as by pretraining.
- **2025–2026**: RL with verifiable rewards claims a substantial fraction of frontier compute. **The capability differentiator between top models is increasingly the RL stage, not the CE-pretrained base.** R1 (early 2025) showed you could train a strong reasoning model with R1-Zero by doing essentially *nothing but* RL on a pretrained base — no SFT, no preference data — and chain-of-thought capability emerges from RL alone. Strong signal that CE pretraining is now the *substrate*, not the *driver*, of reasoning ability.

### Where the frontiers are right now

- **RL with verifiable rewards on math, code, formal proofs.** Driving most of the visible capability gains on benchmarks like AIME, FrontierMath, SWE-bench, Codeforces.
- **Test-time compute scaling.** o1/o3-style models and their open-source counterparts (R1, QwQ) demonstrate that "thinking longer" is a real scaling axis with its own power law and no observed ceiling.
- **Process reward models (PRMs) and step-level supervision.** Denser reward signals than just final-answer correctness — train a verifier on partial solutions, use it to shape RL.
- **Self-play / search-augmented training.** AlphaZero-style setups for proofs and math (AlphaProof, formal-language work). MCTS rollouts + policy distillation. Loss looks more like AlphaZero's (CE to visit counts + value error) than like standard LM training.
- **Agentic / tool-using RL.** Reward signal comes from real-world task completion — did the agent fix the bug, complete the booking, find the answer. Same RL machinery, different verifiers.

### The cleanest mental model

**CE builds the substrate; RL with verifiable rewards builds the capability.** CE has not been replaced, but it has been demoted from "the loss" to "the *first* loss in a stack." Pretraining gives you a model that knows what valid tokens look like; RL on top teaches it to *use* those tokens to actually accomplish things. Neither alone is enough; the stack is the point.

This is also why "scaling has hit a wall" is the wrong framing. **Pretraining-CE-on-web-text** is hitting a wall (entropy rate of English). **RL-on-verifiable-rewards** has no such wall and is currently the steepest part of the capability curve. The two regimes use different loss functions, sit at different points in the training stack, and have different scaling characteristics. Conflating them is the central confusion in most "are we done?" arguments.

## Why this framing is uniquely useful

Most "are we hitting a wall?" arguments are vibes. Entropy rate gives you a **calculable, principled lower bound on loss**. If a researcher reports "perplexity improved from 2.5 to 2.3 on WikiText," entropy tells you whether that's a big deal (closing the gap to the floor) or noise (last fractional bit on a saturated distribution).

It's also philosophically tight: **compression = prediction = understanding**. There's a strand of thought (Solomonoff, Hutter, Sutskever's "compression is intelligence") arguing that lossless compression of a corpus *requires* understanding what's in it — you can't compress what you can't predict. So as models approach the entropy rate of human-generated text, the claim is they must have internalized a meaningful model of human cognition. Whether you buy the strong version or not, the weak version is unavoidable: we now have predictive systems for natural language that are near-optimal in a precise, provable sense. That's a milestone you can put a number on, and the number tells you both why scaling worked and why it's getting harder.

## TL;DR

1. **Approaching the entropy rate** = the model has become a near-optimal lossless compressor of the data. By Shannon's theorem, no model could do meaningfully better. The residual loss isn't model failure — it's intrinsic randomness in the data.
2. **Scaling limits** become visible exactly when CE nears the entropy rate. The compute-to-CE curve flattens against the asymptote. This is *why* pretraining-on-web-text shows diminishing returns now, *why* "more data of the same kind" doesn't help, and *why* the field has pivoted toward new distributions (code, math, reasoning) and new objectives (RLHF, test-time compute) — places with headroom against their own floors.
3. **The "wall" is one axis, not all of them.** Math/code have lower text-entropy than prose (more headroom even within the entropy framing). More fundamentally, reasoning capability isn't measured by text-entropy at all — it's measured by solve rate, which has no Shannon-style ceiling. Three other scaling axes (structured-data pretraining, RL on verifiable rewards, test-time compute) are wide open and currently driving the bulk of frontier progress. The right statement is: *the original power-law on pretraining-loss is bending against English's entropy rate; new power-laws on other axes have not bent yet.*
4. **The loss itself has shifted, in a stacking way.** CE is still the foundation (pretraining + SFT), but capability-generating training has moved to preference losses (DPO/IPO/SimPO — CE-flavored at response level) and especially **RL with verifiable rewards** (PPO/GRPO with external checkers — not CE at all). CE is now the *first* loss in a stack, not *the* loss. As of late 2025, frontier reasoning models differentiate primarily on the RL stage, and R1-Zero showed you can produce a strong reasoning model with essentially nothing but RL on a CE-pretrained base. **CE builds the substrate; RL with verifiable rewards builds the capability.**
