# Routing and Load Balancing

The router is a `d_model × E` matrix followed by a `top-k` — the smallest component in an MoE model and the source of nearly all its difficulty. This file is about why, because the problem is structural rather than incidental: **routing is a discrete decision inside a differentiable model, and it has a degenerate equilibrium that training will find unless you actively prevent it.**

## The core pathology: routing collapse

Nothing in the loss rewards using all the experts. Consider what gradient descent does with a slight early advantage:

1. Expert 3 is randomly a little better at initialization, so the router sends it more tokens.
2. More tokens means more gradient updates, so expert 3 improves faster.
3. Being better, it attracts more tokens.

This is a **rich-get-richer feedback loop**, and its fixed point is a handful of experts receiving nearly all traffic while the rest stay near their initialization — dead weight that costs memory and contributes nothing. You've paid for 256 experts and trained 12. Note the loop needs no bug to start: random initialization asymmetry is sufficient, and the dynamics are self-reinforcing from there.

Two secondary problems ride along:

- **The router is not differentiable through its choice.** `top-k` is discrete; gradients flow through the *gate values* (the softmax weights multiplying each expert's output) but not through *which* experts were picked. The router learns only via the experts it already chose — an exploration problem, and another reason collapse is sticky.
- **Load imbalance is a systems catastrophe, not just a quality one.** Experts live on different devices ([file 04](04_the_systems_reality.md)); an overloaded expert's device becomes the straggler that every other device waits for. Imbalance converts directly into idle hardware.

## Fix 1: the auxiliary load-balancing loss

The standard answer since Shazeer 2017. Add a term that penalizes uneven routing, typically (Switch Transformer's form):

```
L_aux = α · E · Σ_i  f_i · P_i
```

where `f_i` is the **fraction of tokens actually routed** to expert `i` and `P_i` is the **mean router probability** for expert `i` over the batch. The construction is worth understanding rather than memorizing:

- Using both `f` (a hard count, no gradient) and `P` (soft, differentiable) is deliberate: `f_i` provides the *magnitude* of the imbalance while `P_i` provides the *gradient path* — so the loss pushes the router's probabilities down for over-subscribed experts.
- The dot product `Σ f_i P_i` is minimized when the distribution is uniform (`f_i = P_i = 1/E`), and grows when mass concentrates. It's a soft proxy for "load is concentrated."
- `α` is small (~0.001–0.01). It's a **tax, not an objective**: too large and you force uniform routing, destroying the specialization that is the entire point.

A companion, **router z-loss**, penalizes large router logits (`log²` of the partition function) to keep the softmax numerically stable — MoE routers are prone to logit blowup in low precision, and this is a cheap guard.

## Fix 2: capacity factors and dropping

Even with the aux loss, batches are uneven. Implementations set a per-expert **capacity**:

```
capacity = capacity_factor · (tokens_per_batch · k / E)
```

`capacity_factor = 1.25` means "each expert may accept 25% more than its fair share." Tokens beyond that are **dropped** — they skip the expert entirely and pass through on the residual stream only (the residual makes this survivable rather than fatal, [3.1](../../part3_residual_connections_deep_networks/3.1_skip_connection/01_degradation_problem.md)). This is a memory/quality dial: fixed capacity means fixed buffer sizes, which is what makes the all-to-all communication in [file 04](04_the_systems_reality.md) implementable with static shapes.

Dropping tokens is obviously unpleasant — a token silently receives less computation than its neighbors — which motivates the next idea.

## Fix 3: change who chooses

**Token-choice** routing (the default) has each token pick its top-`k` experts. The imbalance is intrinsic: nothing coordinates the tokens.

**Expert-choice** routing (Zhou et al., 2022) inverts it — each *expert* picks its top-`c` tokens from the batch. Load balance becomes **exact by construction** (every expert takes precisely `c` tokens; no aux loss needed, no dropping from overflow), at the cost that a given token may be selected by zero experts or by many. It trades "every token gets `k` experts" for "every expert gets `c` tokens." Elegant, and used less than you'd expect — mostly because token-choice's guarantee is the one that's easier to reason about for quality, and because expert-choice leaks information across the batch (which experts a token gets depends on the *other* tokens present, awkward for inference determinism).

## Fix 4: auxiliary-loss-free balancing (the 2024 turn)

The most interesting recent development, and DeepSeek-V3's choice (`topk_method: "noaux_tc"` in its config). The observation: the aux loss balances load by **corrupting the gradient of the language-modeling objective** — you're adding a term that isn't about predicting tokens, and it degrades quality in proportion to its strength. Can you balance load without touching the loss at all?

Yes: maintain a **per-expert bias** added to the router scores *for selection purposes only*:

```
score_i  = sigmoid(x · W_router)_i        # the actual gate value used for combination
select on: score_i + b_i                  # b_i shifts WHO is chosen, not HOW MUCH they contribute
after each step:  b_i ← b_i − γ  if expert i was overloaded
                  b_i ← b_i + γ  if underloaded
```

The bias is updated by a **control loop, not by gradients**, and it's excluded from the gating weight that scales the expert's output. So load is regulated without any extra term in the loss and without distorting the gate magnitudes the model learned. DeepSeek reports better quality than aux-loss balancing at equal balance.

Two details from V3's config worth noting: it uses **sigmoid** rather than softmax for scoring (so expert scores are independent rather than competing), and **node-limited routing** (`n_group: 8`, `topk_group: 4`) — each token's experts are restricted to at most 4 of 8 device groups, which caps cross-node communication ([file 04](04_the_systems_reality.md)). That last one is a pure systems constraint expressed in the routing algorithm, in the same spirit as `n_kv = 8` matching tensor parallelism ([7.1/02](../7.1_attention_variants/02_mqa_and_gqa.md)).

## Why it matters in modern LLM work

- **Routing is where MoE training actually fails**, and "which balancing scheme" is the most consequential MoE design choice after the expert count.
- **The aux-loss-free turn is a transferable lesson**: when a regularizer fights your objective, look for a mechanism that achieves the same constraint *outside* the loss — a control loop, a constraint on selection, a reparameterization.
- **Config literacy:** `topk_method`, `scoring_func`, `n_group`/`topk_group`, `norm_topk_prob`, `router_aux_loss_coef` are all this file.

## Self-check

1. Describe routing collapse as a feedback loop, and explain why random init is enough to start it.
2. In `L_aux = α·E·Σ f_i·P_i`, why include both `f_i` and `P_i` rather than just one?
3. What does `capacity_factor = 1.25` mean operationally, what happens to overflow tokens, and why is the residual connection relevant?
4. Contrast token-choice and expert-choice routing on load balance and on per-token guarantees. Name expert-choice's inference-time awkwardness.
5. What exactly is the complaint against the auxiliary loss, and how does a bias-based scheme avoid it?
6. Why does DeepSeek-V3 restrict each token to 4 of 8 expert groups?

### Answers

1. A slightly-favored expert receives more tokens → more gradient updates → improves faster → becomes more attractive to the router → receives still more tokens. It's self-reinforcing with no counterbalancing term in the objective. Random initialization suffices because the loop only needs an infinitesimal initial asymmetry; the dynamics amplify it, so collapse is the *default* outcome rather than a failure mode requiring a cause.
2. `f_i` (the hard fraction of tokens routed) measures the real imbalance but has no gradient — it comes from a discrete `top-k`. `P_i` (the mean router probability) is differentiable and provides the path back to `W_router`. Multiplying them makes the *penalty size* reflect actual load while the *gradient* flows through the router's probabilities, so over-subscribed experts get their probabilities pushed down. Either alone is insufficient: `f` alone can't train, `P` alone doesn't see the realized routing.
3. Each expert may accept at most 1.25× its fair share (`tokens·k/E`) in a batch; tokens beyond that are **dropped** — they skip the FFN and continue via the residual path only. The residual matters because it means a dropped token still passes through the block with its representation intact ([3.1](../../part3_residual_connections_deep_networks/3.1_skip_connection/01_degradation_problem.md)) — degraded, not destroyed. Fixed capacity also gives static buffer shapes, which is what makes the all-to-all in [file 04](04_the_systems_reality.md) practical.
4. **Token-choice:** each token picks `k` experts — guarantees every token gets `k` experts' worth of compute, but load balance is uncoordinated and needs an aux loss plus capacity limits. **Expert-choice:** each expert picks `c` tokens — load is *exactly* balanced by construction with no aux loss or overflow drops, but a token may be picked by zero or many experts, so per-token compute is not guaranteed. Awkwardness: which experts a token receives depends on the *other tokens in the batch*, so outputs aren't independent per sequence — bad for inference determinism and batching semantics.
5. The aux loss adds a term unrelated to next-token prediction, so it **corrupts the gradient of the actual objective** and trades quality for balance in proportion to `α`. A bias-based scheme moves the balancing signal out of the loss entirely: a per-expert bias adjusts *which* experts are selected via a control loop (increment if underloaded, decrement if overloaded) while being excluded from the gate value that scales outputs. Load is regulated, the LM gradient is untouched, and the learned gate magnitudes aren't distorted.
6. To bound **cross-node communication**. Experts are distributed across device groups, and a token whose 8 experts are spread over all 8 groups requires all-to-all traffic with every group. Limiting each token to 4 groups halves the worst-case fan-out, making the all-to-all cheaper and more predictable — a systems constraint implemented as a routing rule.

## Exercise

Reproduce collapse, then fix it three ways. (a) Build a toy MoE (8 experts, top-1, small `d_model`) with **no** balancing and train on any dataset; log per-expert token counts each step and plot them. You should watch collapse happen — a few experts taking nearly everything within a few hundred steps. (b) Add `L_aux` and sweep `α ∈ {0, 1e-4, 1e-2, 1e-1}`; plot both the balance (entropy of the load distribution) and the task loss, and identify the point where the tax starts costing real quality. (c) Implement the bias-based scheme: a per-expert bias updated by `±γ` on over/under-load, applied to selection only, and confirm you get comparable balance with a *better* task loss than the equivalent-balance `α`. (d) Implement expert-choice and verify load is exactly uniform by construction — then count how many tokens received zero experts, which is the cost you just accepted.
