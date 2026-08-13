# The Sparse FFN — Buying Capacity With Memory

Every other technique in Part 7 reduces a cost. MoE does something different: it **decouples two quantities that were previously locked together** — how many parameters a model has, and how many it uses per token. Once separated, you can buy capacity with memory instead of compute, and that is the single most consequential architectural change of 2023–25.

**Config anchor:** DeepSeek-V3, from its published config — `d_model = 7168`, `L = 61`, 256 routed experts + 1 shared, top-8 routing, `moe_intermediate_size = 2048`, first 3 layers dense.

## The idea

Replace the block's single FFN with `E` parallel FFNs ("experts") plus a small **router** that picks `k` of them per token:

```
dense block:   y = x + FFN(x)                       # every token, every parameter
MoE block:     scores = softmax(x · W_router)        # W_router: d_model → E
               pick top-k experts by score
               y = x + Σ_{i ∈ top-k}  gate_i · FFN_i(x)
```

Two properties do all the work:

- **Per-token FLOPs depend on `k`, not `E`.** Adding experts adds parameters and *no* compute per token.
- **Routing is per token, per layer.** Not per sequence — token 5 and token 6 may use disjoint experts, and the choice is remade at every layer. So a 61-layer, top-8-of-256 model has an astronomically large space of expert paths; capacity comes from *combinations*, not just from count.

## Why the FFN, specifically

Three reasons, and they compose:

1. **That's where the parameters are.** The FFN is ~75–80% of a transformer block ([5.2/02](../../part5_transformer_rebuilt/5.2_the_full_block/02_the_ffn.md)), so sparsifying anything else moves little.
2. **The FFN is already a lookup table.** Its key-value-memory reading ([5.2/02](../../part5_transformer_rebuilt/5.2_the_full_block/02_the_ffn.md)) says the FFN is `d_ff` independent "if pattern, then write" memories, of which only a few fire per token (measured there: <9% of units meaningfully active, one unit carrying 81% of the output norm). **MoE takes that observed sparsity and makes it architectural** — if only a few memories fire, don't compute the rest. This is the intellectual justification, and it's why MoE feels natural rather than bolted-on.
3. **The FFN is position-wise.** Each token is processed independently, so routing tokens to different experts is trivially parallel and changes no sequence semantics. Attention mixes across positions and can't be split this way without changing what's computed.

## The accounting

This is the file's core, and it's worth doing exactly once with real numbers. DeepSeek-V3, verified:

```
one expert    = 3 · d_model · d_moe = 3 · 7168 · 2048   = 44.0M params   (SwiGLU: 3 matrices)
MoE layers    = 61 − 3 dense        = 58
all experts   = 58 · 257 · 44.0M                        = 656.5B
active/token  = 58 · (8 + 1) · 44.0M                    =  23.0B
```

Adding attention, embeddings, and the dense layers gives the published **671B total / 37B active** — a **~18× sparsity ratio**. Read the implication carefully:

> A dense model with 37B active parameters *has* 37B parameters. DeepSeek-V3 has 37B active and 671B total. **Same FLOPs per token; 18× the parameters.**

So the trade is stated in one line: **capacity is bought with memory bandwidth and capacity, not with compute.** Mixtral 8×7B is the smaller canonical example — 46.7B total, ~12.9B active (top-2 of 8, attention shared, which is why it isn't 8×7 = 56B).

## Where the win actually lands

Given [7.2/01](../7.2_efficient_attention/01_the_memory_bandwidth_wall.md)'s roofline, be precise about which resource MoE saves:

| Regime | Effect of MoE |
|---|---|
| **Training FLOPs** | Big win — quality of a much larger model at a fraction of the compute |
| **Prefill (compute-bound)** | Win — FLOPs scale with active params |
| **Decode (memory-bound)** | Win *if* you can read only the active experts' weights; the bytes read per token fall with `k/E` |
| **Total memory** | **Loss** — you must hold all 671B parameters resident |
| **Small-batch serving** | Loss — see [file 04](04_the_systems_reality.md); different tokens want different experts, so a small batch touches many experts anyway |

That last row is the one that catches people. MoE's decode advantage assumes batching lets you amortize each expert's weight read across many tokens routed to it. At batch 1, a single token's 8 experts must each be read in full — you get the FLOP saving but a much worse *bytes-per-useful-FLOP* ratio than a dense model of the same active size. **MoE is a high-throughput, high-batch architecture**, which is exactly the regime frontier labs serve and exactly the wrong regime for a hobbyist with one GPU. That asymmetry explains a lot of the "MoE models feel disappointing locally" experience.

## The historical arc, briefly

- **Sparsely-Gated MoE** (Shazeer et al., 2017) — the original, on LSTMs, with top-k gating and a load-balancing loss. All the core ideas were present.
- **Switch Transformer** (Fedus et al., 2021) — simplified to **top-1** routing, showed it works at scale, made MoE a Transformer technique.
- **Mixtral 8×7B** (2023) — the model that made MoE mainstream in open weights; top-2 of 8, and *good*.
- **DeepSeek-MoE → V3** (2024) — **fine-grained** experts (many small ones), a **shared** always-on expert, and eventually **auxiliary-loss-free** balancing ([file 02](02_routing_and_load_balancing.md), [file 03](03_fine_grained_and_shared_experts.md)).
- **2024–25** — MoE becomes the default at the frontier for large models; the open question moves from "does it work" to "how fine-grained, and how do you route."

## Why it matters in modern LLM work

- **"Total vs active parameters" is now the basic literacy** for reading a model card. `671B/37B` tells you the memory footprint *and* the serving cost, which are different numbers for the first time.
- **It changes the scaling-law question.** Chinchilla optimized `(N, D)` with `N` unambiguous ([6.3/02](../../part6_pretraining_paradigms/6.3_scaling_laws/02_chinchilla.md)); with MoE there are two `N`s, and the sparsity ratio becomes a third axis to allocate over ([file 04](04_the_systems_reality.md)).
- **It's the direct continuation of the FFN-as-memory story** — 5.2/02 observed the sparsity, 7.4 exploits it.

## Self-check

1. Why does adding experts increase parameters but not per-token FLOPs?
2. Give the three reasons the FFN is the component that gets sparsified.
3. Compute active and total expert parameters for a hypothetical `d_model = 4096`, `d_moe = 1024`, 64 experts, top-4, 32 MoE layers (SwiGLU). What's the sparsity ratio?
4. MoE has the same per-token FLOPs as a dense model of its active size. Why is it nonetheless *slower per token* at batch 1?
5. Routing is per token per layer. Why does that matter more than the raw expert count?
6. What did Switch Transformer simplify, and why was that significant?

### Answers

1. Because only the `k` selected experts run. The router's cost is negligible (`d_model × E` for a dot product), and the un-selected experts' weights are never multiplied by anything — they're storage, not computation. So compute scales with `k` while parameters scale with `E`.
2. (a) **Parameter share** — the FFN is ~75–80% of a block, so it's the only place with enough parameters to matter; (b) **observed sparsity** — the FFN is already a key-value memory where <9% of units fire per token, so MoE just formalizes existing behavior; (c) **position-wise structure** — each token's FFN is independent, so routing different tokens to different experts is embarrassingly parallel and changes no sequence semantics, unlike attention which must mix across positions.
3. One expert `= 3 · 4096 · 1024 = 12.6M`. Total `= 32 · 64 · 12.6M ≈ 25.8B`; active `= 32 · 4 · 12.6M ≈ 1.61B`. Sparsity ratio `= 64/4 = 16×` (matching `25.8/1.61`).
4. Because decode is memory-bound ([7.2/01](../7.2_efficient_attention/01_the_memory_bandwidth_wall.md)) and at batch 1 there's nothing to amortize: the token's `k` experts must be read from HBM in full to compute a single token's worth of FLOPs, so bytes-read-per-useful-FLOP is poor. A dense model of the same active size reads the same bytes but with better locality and no routing overhead. MoE's decode win requires large batches so each expert's weight read serves many tokens.
5. Because capacity comes from *combinations of experts along the depth*, not from the pool size alone. With top-8 of 256 chosen independently at each of 58 layers, the number of distinct expert paths a token can take is astronomically larger than 256 — so a modest expert count supports enormous effective specialization, and fine-grained routing ([file 03](03_fine_grained_and_shared_experts.md)) exploits exactly this combinatorial argument.
6. It reduced routing to **top-1** (one expert per token) from top-2 or more. Significant because it showed the simplest possible routing works at scale — halving expert compute and communication versus top-2 — and because it demonstrated MoE in a Transformer at then-unprecedented parameter counts, converting MoE from an LSTM-era curiosity into a mainstream Transformer technique.

## Exercise

Build the accounting tool and then the layer. (a) Write a function computing total params, active params, and sparsity ratio from `(d_model, d_moe, E, k, L, n_dense, n_shared, SwiGLU?)`; validate it reproduces DeepSeek-V3's ~671B/37B and Mixtral's ~46.7B/12.9B (you'll need to add attention and embedding params — do that too, and note what fraction of each model is *not* experts). (b) Implement a real MoE layer in the [capstone notebook](../../numpy_pytorch/05_capstone_transformer_block.ipynb): router linear → `topk` → gather → run experts → weighted combine. Verify that with `E = 1, k = 1` it exactly reproduces the dense FFN. (c) Instrument it: for a batch of real text, log which experts each token picks at each layer and count how many *distinct expert paths* the batch traverses — the combinatorial argument from self-check 5, measured. (d) Time a forward pass at batch 1 versus batch 64 and compare against a dense model of equal active size; you should reproduce the batch-1 disadvantage from self-check 4.
