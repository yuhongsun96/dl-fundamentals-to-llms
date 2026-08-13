# The Systems Reality of MoE

The files above described MoE as a modeling technique. This one is about why it's a *distributed systems* technique wearing a modeling costume — and about the conditions under which it stops paying. This matters because MoE's benefits are frequently overstated by people quoting FLOP counts, and the gap between "same FLOPs as a 37B model" and "behaves like a 37B model" is where all the engineering lives.

## Expert parallelism and the all-to-all

An MoE layer's experts don't fit on one device: DeepSeek-V3's 671B parameters need hundreds of GiB. So experts are **sharded across devices** — expert parallelism, a fourth axis alongside data/tensor/pipeline parallelism (Part 12). And since routing is per token, each device's tokens need experts living on *other* devices. Every MoE layer therefore performs:

```
1. route          — compute top-k per token (local)
2. all-to-all     — send each token's activations to the devices owning its experts
3. expert compute — each device runs its own experts on whatever arrived
4. all-to-all     — send the outputs back to the tokens' original devices
```

**Two all-to-all collectives per MoE layer, per forward pass** (and two more in the backward). All-to-all is the least forgiving collective: every device talks to every other, so its cost is set by the *slowest link* in the topology — typically the inter-node network, which is an order of magnitude slower than intra-node NVLink. Consequences:

- **Communication can dominate.** At 58 MoE layers, that's ~116 all-to-alls per forward pass. If each token's activations are `d_model × 2` bytes and `k = 8`, the traffic per token per layer is substantial and scales with `k`.
- **This is why node-limited routing exists.** DeepSeek-V3's `n_group: 8, topk_group: 4` ([7.4/02](02_routing_and_load_balancing.md)) caps each token at 4 of 8 device groups, bounding the fan-out. A routing rule that exists purely to shape network traffic.
- **Load imbalance becomes a straggler problem.** All-to-all is a barrier: every device waits for the slowest. An expert receiving 3× its share makes its device 3× slower and *all* devices idle for the difference. This is the systems half of [7.4/02](02_routing_and_load_balancing.md)'s balancing story, and it's usually the more urgent half — quality degradation from imbalance is gradual, but a straggler wastes the entire cluster synchronously.
- **Fixed capacity is what makes it implementable.** Static buffer shapes let the collective be planned; that's the real reason capacity factors and token dropping exist ([7.4/02](02_routing_and_load_balancing.md)) rather than a quality-motivated design.

DeepSeek's DualPipe (overlapping computation with these communications) and the general obsession with all-to-all overlap in MoE training frameworks follow directly.

## Memory: the cost MoE doesn't hide

The trade from [file 01](01_the_sparse_ffn.md), stated as a bill:

| | Dense 37B | DeepSeek-V3 (37B active) |
|---|---|---|
| Weights in bf16 | ~74 GiB | **~1.3 TiB** |
| GPUs to hold weights (80 GiB) | 1 | **~17** |
| FLOPs per token | same | same |

You cannot serve an MoE model on hardware sized for its active parameter count. The 18× sparsity ratio is 18× the memory, and memory is what you buy GPUs for. This is why MoE is a *frontier-lab and serious-inference-provider* architecture: it converts a compute problem into a capital-expenditure problem, which is a good trade at scale and a terrible one for a single node.

Corollary worth stating: **MoE and quantization are complements, not alternatives.** MoE's cost is bytes-resident; quantization attacks exactly that (Part 9.2), which is why heavily-quantized MoE models are disproportionately common in local-inference communities.

## The batch-size condition

The most under-appreciated point, and it follows from [7.2/01](../7.2_efficient_attention/01_the_memory_bandwidth_wall.md)'s roofline. MoE's decode advantage requires that each expert's weight read be **amortized across many tokens routed to it**. Whether that happens depends on batch size versus expert count:

```
tokens per expert per step ≈ (batch × k) / E
```

For DeepSeek-V3 (`E = 257`, `k = 9`): batch 8 gives `≈ 0.3` tokens per expert — most experts are read from HBM to serve *less than one token*. You've paid the full weight-read cost of a 671B model to compute one token's worth of FLOPs. At batch 256 it's `≈ 9` tokens per expert, and the economics work.

So:

| Batch | MoE decode behavior |
|---|---|
| 1–8 | **Worse than dense-37B** — reads far more bytes for the same FLOPs |
| ~64 | Break-even-ish, depends on `E` and topology |
| 256+ | **Big win** — the regime it was designed for |

This resolves the common confusion where an MoE model with "37B active parameters" runs slower than a dense 37B on a workstation. The FLOP count was never the binding constraint ([7.2/01](../7.2_efficient_attention/01_the_memory_bandwidth_wall.md)); bytes-per-token were, and at batch 1 MoE's bytes-per-token are terrible.

## When MoE doesn't pay

A checklist, since "use MoE" is not universal advice:

- **Small models.** Under a few billion parameters there isn't enough knowledge to partition, and the routing/communication overhead isn't amortized.
- **Low-batch or latency-critical serving** — the condition above.
- **Memory-constrained deployment** — edge, on-device, single-GPU.
- **Fine-tuning.** MoE models are harder to fine-tune: sparse gradients per expert mean each expert sees a fraction of your (already small) dataset, routing can shift and destabilize, and full-model memory is required even for LoRA-style methods.
- **When you can't fix the balance.** An imbalanced MoE is strictly worse than dense — you pay full memory for partial capacity.

The clean summary: **MoE optimizes for the frontier-lab objective function** — best quality per training FLOP, served at high throughput on large clusters. That's not everyone's objective.

## Why it matters in modern LLM work

- **It explains model availability.** Why frontier open models are increasingly MoE and why they're awkward to run locally is one fact, not two.
- **It adds an axis to the scaling-law question.** Chinchilla allocated `(N, D)` ([6.3/02](../../part6_pretraining_paradigms/6.3_scaling_laws/02_chinchilla.md)); with MoE you also allocate the **sparsity ratio**, and the optimum depends on your serving batch size — i.e. the *architecture* is now downstream of the deployment plan, in the same spirit as inference-aware overtraining ([6.3/03](../../part6_pretraining_paradigms/6.3_scaling_laws/03_beyond_chinchilla.md)).
- **All-to-all sensitivity is why MoE training is hard to reproduce** on modest clusters even when you have the weights and the recipe.

## Self-check

1. List the four steps of an MoE layer's forward pass and count the collectives per forward for a 58-MoE-layer model.
2. Why is load imbalance a *worse* systems problem than a quality problem?
3. DeepSeek-V3 has 37B active parameters. How many 80 GiB GPUs are needed just to hold its weights in bf16, and why doesn't the active count help?
4. Compute tokens-per-expert-per-step for `E = 257, k = 9` at batch 8 and batch 512. Explain what each implies for decode.
5. Someone reports that a 37B-active MoE is slower than a dense 37B on their single GPU. Explain, using intensity.
6. Name three deployment situations where dense beats MoE.

### Answers

1. Route (local) → all-to-all dispatch → expert compute → all-to-all combine. That's **2 all-to-alls per MoE layer per forward**, so ~**116 per forward pass** at 58 MoE layers (plus roughly as many again in the backward pass during training).
2. Because all-to-all is a **synchronization barrier**: every device must finish before the collective completes, so one overloaded expert's device becomes a straggler and *all* other devices idle for the difference — wasting the whole cluster, synchronously, every layer. Quality damage from imbalance is gradual and partially recoverable; wasted cluster-time is immediate and total. Hence balancing is enforced for throughput reasons at least as much as for quality.
3. All 671B parameters must be resident: `671e9 × 2 bytes ≈ 1.34 TiB`, so **~17 GPUs** at 80 GiB. The active count doesn't help because *which* 37B are active changes per token and per layer — any expert may be needed by the next token, so nothing can be evicted. Sparsity is in the *computation*, not in the *storage*.
4. Batch 8: `8 × 9 / 257 ≈ 0.3` tokens per expert — most experts are read from HBM for less than one token of work, so you pay a 671B model's memory traffic for a 37B model's FLOPs: strictly worse than dense. Batch 512: `512 × 9 / 257 ≈ 18` tokens per expert — each weight read serves ~18 tokens, so the read is amortized and MoE's FLOP advantage translates into real throughput.
5. Decode is memory-bound with intensity ≈ tokens-per-weight-byte-read ([7.2/01](../7.2_efficient_attention/01_the_memory_bandwidth_wall.md)). At batch 1, the dense 37B reads ~74 GiB to produce one token; the MoE reads the weights of its 9 selected experts *per layer* — a large fraction of a much bigger model — to produce that same one token, with worse locality and routing overhead on top. Same FLOPs, more bytes, memory-bound regime ⇒ slower. The advantage only appears when batching amortizes those reads.
6. (a) **Small models** — insufficient knowledge to partition, overhead not amortized; (b) **low-batch/latency-critical or single-device serving** — the batch condition fails; (c) **memory-constrained/edge deployment** — must hold all parameters. (Also acceptable: fine-tuning workflows, and any setting where you can't guarantee load balance.)

## Exercise

Price an MoE deployment. (a) Write a function taking `(E, k, d_model, d_moe, L, n_dense, batch, GPU_HBM, GPU_bandwidth)` and returning: total weight memory, GPUs required, tokens-per-expert-per-step, and estimated bytes-read-per-token at decode. (b) Plot bytes-per-token for DeepSeek-V3 versus a dense 37B across `batch ∈ [1, 1024]` and find the crossover batch where MoE becomes cheaper — that single number is the deployment decision. (c) Add an all-to-all cost model (traffic `= batch × k × d_model × 2` bytes per layer, divided by an assumed inter-node bandwidth) and identify the batch size where communication starts dominating expert compute. (d) Redo (b) with int4 weights for both and note how quantization moves the crossover — the complementarity claim, quantified. (e) One paragraph: given your own hardware, would you ever choose an MoE, and at what batch size does the answer change?
