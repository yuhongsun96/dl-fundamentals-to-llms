# Fine-Grained and Shared Experts

Two design moves that separate the 2024–25 MoE generation from the Switch/Mixtral generation, both from the DeepSeek-MoE line, and both with a clean combinatorial or economic argument behind them. The trend in one line: **many small experts, plus a small always-on core.**

The shift, concretely:

| | Mixtral 8×7B (2023) | DeepSeek-V3 (2024) |
|---|---|---|
| Routed experts | 8 | **256** |
| Active per token | 2 | **8** |
| Expert inner width | 14336 (full FFN size) | **2048** (⅐ of the dense `d_ff`) |
| Shared/always-on | none | **1** |

## Fine-grained experts: the combinatorial argument

Suppose you fix the compute budget — the total width of FFN actually run per token. You can spend it as a few wide experts or many narrow ones. Splitting each expert into `m` smaller ones and raising `k` by the same factor `m` keeps active parameters identical while multiplying the number of *combinations* available:

```
8 experts, top-2:      C(8,2)   = 28 possible expert sets
64 experts, top-16:    C(64,16) ≈ 4.9e14 possible expert sets
```

Same FLOPs per token, ~13 orders of magnitude more distinct ways to compose a computation. That is the DeepSeek-MoE argument, and it rests on the [5.2/02](../../part5_transformer_rebuilt/5.2_the_full_block/02_the_ffn.md) reading of the FFN: if an expert is a bundle of key-value memories, then **coarse experts force unrelated memories to share a routing fate.** A token needing "Python syntax" and a token needing "French grammar" may both be routed to the same wide expert simply because that expert is the closest match on average — and then that expert must serve both, diluting its specialization. Narrow experts let each token assemble a *bespoke combination* of narrow specialists instead of picking the least-bad generalist.

Empirically, DeepSeek-MoE reported matching a much larger conventional MoE at equal active parameters, and the fine-grained design is now standard (Qwen's MoE variants, Kimi K2 at 384 experts).

The limits, which are real:

- **Routing overhead grows** — the router is `d_model × E`, so `E = 256` costs 8× the router compute of `E = 32` (still small, but no longer free), and `top-k` over 256 with `k = 8` is more work than top-2 over 8.
- **Matmul efficiency falls.** Each expert's matmul becomes small and skinny; GPUs are inefficient on small matrices, so grouped/batched GEMM kernels are needed to recover throughput. This is another "the kernel is part of the architecture" case ([7.2/02](../7.2_efficient_attention/02_flashattention.md)).
- **Communication grows** with `k` — each token's activations must reach `k` experts, so top-8 moves 4× the data of top-2 ([file 04](04_the_systems_reality.md)), which is exactly why DeepSeek adds node-limited routing ([file 02](02_routing_and_load_balancing.md)).

## Shared experts: don't make specialists relearn the basics

The second move. Alongside the routed experts, designate one (or a few) expert that **every token always visits**, unrouted:

```
y = x + FFN_shared(x) + Σ_{i ∈ top-k} gate_i · FFN_i(x)
```

The argument is about **redundancy**. Some knowledge is needed by essentially every token — basic syntax, common-word statistics, the "the follows a noun phrase" regularities. Without a shared expert, every routed expert must learn its own copy of these basics, because each must function on whatever tokens it receives. That's `E` redundant copies of the same common knowledge, consuming capacity that could hold specialized knowledge.

A shared expert factors that out: **common knowledge lives once, in a component guaranteed to run; routed experts are freed to be genuinely specialized.** It's straightforward parameter economics, and it also stabilizes training — every token now has a guaranteed compute path regardless of routing decisions, which softens the impact of load imbalance and dropped tokens ([file 02](02_routing_and_load_balancing.md)).

Note the connection to hybrids ([7.3/03](../7.3_alternatives_to_attention/03_hybrids_and_recall.md)): both designs answer "a capability needed by *all* inputs shouldn't be provided by a mechanism that only *some* inputs reach." Shared expert is to MoE what the always-present global attention layer is to a hybrid stack.

## Two more details you'll meet in configs

**Dense first layers.** DeepSeek-V3 sets `first_k_dense_replace = 3` — the first three blocks use ordinary dense FFNs, not MoE. Early layers do broadly-applicable low-level work (detokenization, local syntax) where there's little to specialize *on*, and routing on barely-contextualized representations is unreliable — the router's input at layer 1 is close to the raw token embedding, so its decisions are near-lexical. Dense early layers are now common.

**Gate normalization.** `norm_topk_prob: true` renormalizes the selected experts' gates to sum to 1. Without it, the FFN branch's output magnitude varies with how much probability mass happened to land in the top-`k`, injecting noise into the residual stream's scale ([3.2/02](../../part3_residual_connections_deep_networks/3.2_normalization_and_depth/02_scaling_the_residual_stream.md)). Small detail, real stability effect.

## Why it matters in modern LLM work

- **These are the parameters you read off a modern MoE config** — `n_routed_experts`, `num_experts_per_tok`, `n_shared_experts`, `moe_intermediate_size`, `first_k_dense_replace` — and now each has an argument attached rather than being a magic number.
- **The combinatorial argument is the key intuition** for why the field went from 8 experts to 256+: capacity comes from *compositions*, and compositions grow combinatorially in `E` at fixed compute.
- **The "factor out the common case" pattern** recurs: shared experts, global attention layers in hybrids, the shared attention block in Zamba. When a mechanism is needed by everything, don't make it routed.

## Self-check

1. State the fine-grained argument quantitatively: what's held fixed, and what grows?
2. Why does the FFN-as-key-value-memory reading make coarse experts look wasteful?
3. Give the three costs of fine-grained experts, one of which is a kernel problem.
4. Explain the redundancy argument for shared experts, and the second benefit they provide.
5. Why do the first few layers stay dense?
6. What breaks without `norm_topk_prob`, and which earlier file owns that concern?

### Answers

1. **Active parameters (hence per-token FLOPs) are held fixed** by splitting each expert into `m` narrower ones and multiplying `k` by `m`. What grows is the number of distinct expert *combinations* — from `C(8,2) = 28` to `C(64,16) ≈ 4.9e14`, about 13 orders of magnitude, at identical compute. Capacity is bought from combinatorics rather than from parameters.
2. Because an expert is a bundle of independent "if pattern, then write" memories ([5.2/02](../../part5_transformer_rebuilt/5.2_the_full_block/02_the_ffn.md)), and a coarse expert forces unrelated memories to share one routing decision. A token is routed to the expert that's best *on average* over that whole bundle, so it receives many irrelevant memories and misses relevant ones held in other experts. Fine granularity lets a token assemble the specific memories it needs rather than accepting a pre-bundled compromise.
3. (a) **Router overhead** — `d_model × E` grows linearly in `E`, and `top-k` over a larger pool costs more; (b) **matmul inefficiency** — narrow experts mean small skinny GEMMs, which GPUs execute poorly, requiring grouped-GEMM kernels to recover throughput; (c) **communication** — larger `k` means each token's activations travel to more experts, multiplying all-to-all traffic ([file 04](04_the_systems_reality.md)).
4. Without a shared expert, every routed expert must independently learn the knowledge that all tokens need (basic syntax, frequent-word statistics), because each must handle whatever it receives — so that knowledge is duplicated `E` times, consuming capacity that could be specialized. A shared expert stores it **once** in an always-executed component. Second benefit: it guarantees every token a compute path regardless of routing, which stabilizes training and softens dropped-token and load-imbalance damage.
5. Because early representations are barely contextualized — near the raw token embedding — so the router's decisions would be essentially lexical and unreliable, and there's little semantic structure to specialize *on*. Early layers do universal low-level work (detokenization, local syntax) that all tokens need, which is precisely the case where routing adds risk without adding specialization.
6. The FFN branch's output magnitude becomes dependent on how much softmax mass happened to fall in the top-`k` — varying per token — so the residual stream receives writes of inconsistent scale, adding noise and a stability risk. Renormalizing the selected gates to sum to 1 fixes it. The concern is residual-stream scaling, owned by [3.2/02](../../part3_residual_connections_deep_networks/3.2_normalization_and_depth/02_scaling_the_residual_stream.md).

## Exercise

Test both moves at fixed compute. (a) Using the MoE layer from [file 01](01_the_sparse_ffn.md)'s exercise, build three variants with **identical active parameters**: 8 experts/top-2 at width `w`, 32 experts/top-8 at width `w/4`, and 64 experts/top-16 at width `w/8`. Train on a task with clear latent structure (e.g. text mixing code, prose, and arithmetic) and compare loss. (b) Instrument specialization: for each variant, compute the per-expert distribution over your three domains and measure its entropy — fine-grained experts should be measurably *more* specialized (lower entropy per expert). (c) Add a shared expert to the best variant at matched total compute (shrink `k` by one to pay for it) and compare; also compare each routed expert's specialization entropy with and without the shared expert, which is the redundancy argument measured directly. (d) Finally, time all variants — you should see the fine-grained ones lose wall-clock to small-GEMM inefficiency, quantifying the cost that grouped-GEMM kernels exist to recover.
