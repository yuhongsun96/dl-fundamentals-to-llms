# Part 7 — Modern LLM Architecture Refinements

Part 5 built the textbook Transformer; Part 6 pretrained it. Everything in this part is a modification made **after** someone tried to serve that model and found the economics unacceptable. That's the organizing insight: the 2017 architecture is compute-optimal for *training* and badly wrong for *inference*, and almost every refinement here — GQA, MLA, sliding windows, FlashAttention, MoE, SSMs — is a trade that gives up something cheap to buy back something expensive at serving time.

Where Part 6 stayed conceptual, this part goes **quantitative**. The reason is that these designs are only legible through their arithmetic: GQA is unmotivated until you compute that a 70B model's KV cache at 128K context is **40 GiB per sequence**, and MoE is unmotivated until you see that DeepSeek-V3 spends 37B parameters of compute to access 671B parameters of memory. Nearly every claim below is checkable, and the exercises are mostly "verify this number."

Two structural threads run through all five subsections:

1. **Memory bandwidth, not FLOPs, is the binding constraint at inference.** Decoding one token with a 70B dense model has an arithmetic intensity of **1 FLOP/byte** against an H100's machine balance of ~296 — off by 300×. Once you internalize that, GQA/MLA (shrink the bytes), FlashAttention (stop moving bytes), MoE (read fewer weight bytes per token), and SSMs (replace a growing cache with a fixed state) are all the same move.
2. **Sparsity and compression are everywhere, and are usually *not* approximations.** FlashAttention is exact. GQA and MLA are architectural (trained in, not applied after). Distinguishing "changes the math" from "changes the memory schedule" is the main reading skill this part teaches.

## Structure

- **7.1 Attention Variants** — the KV cache is the problem; these are the four answers.
  - `01` The KV-cache problem — what's cached, the bytes-per-token formula, why it beats the weights at long context, and the memory-bound decode.
  - `02` MQA and GQA — sharing K/V across query heads; why `n_kv = 8` became the default.
  - `03` Multi-head Latent Attention — DeepSeek's low-rank KV compression, and the RoPE incompatibility that forces *decoupled* rotary keys.
  - `04` Local and sparse patterns — sliding windows, local/global layer interleaving, attention sinks, and the 2025 return of learned sparsity.
- **7.2 Efficient Attention** — same math, better memory schedule.
  - `01` The memory-bandwidth wall — arithmetic intensity, the roofline, prefill vs. decode as two different machines.
  - `02` FlashAttention — the online-softmax trick derived, tiling, recomputation, and why it's *exact*; v1 → v2 → v3.
  - `03` KV-cache memory management — fragmentation, PagedAttention, prefix sharing (serving policy deferred to Part 9).
- **7.3 Alternatives to Attention** — Part 4's recurrence, rehabilitated.
  - `01` Linear attention — drop the softmax, reassociate the matmul, and a fixed-size state falls out; why it lost the first time.
  - `02` State-space models — S4 → Mamba's selectivity → Mamba-2's duality with linear attention.
  - `03` Hybrids and the recall tradeoff — why pure SSMs fail at in-context retrieval, and what ratio actually ships.
- **7.4 Mixture of Experts** — buy capacity with memory instead of compute.
  - `01` The sparse FFN — why the FFN is the thing you sparsify, and the param-vs-FLOP accounting.
  - `02` Routing and load balancing — top-k, token- vs expert-choice, capacity factors, auxiliary losses, and the aux-loss-free turn.
  - `03` Fine-grained and shared experts — why 256 small experts beat 8 big ones, and what the shared expert is for.
  - `04` The systems reality — expert parallelism, all-to-all, and the conditions under which MoE stops paying.
- **7.5 Long Context** — what actually breaks, and what actually measures it.
  - `01` Where quadratic bites — the crossover arithmetic (attention passes the FFN at `S ≈ 30K`), and why memory beats FLOPs as the real wall.
  - `02` Training at long context — sequence/context parallelism, ring attention, and the staged recipe (RoPE scaling itself is [5.3/05](../part5_transformer_rebuilt/5.3_positional_information/05_context_length_extension.md)).
  - `03` Evaluating long context — needle-in-a-haystack's inadequacy, RULER, and effective vs. advertised context.

## How to use

Read **7.1/01 first and don't skip it** — it's the file the other sixteen depend on, and the arithmetic in it is what makes every subsequent design look inevitable rather than arbitrary. After that: 7.2 is systems, 7.3 is theory, 7.4 is the biggest live design space, 7.5 is practice. 7.4 and 7.3 can be read in either order.

The file to *do* rather than read is [7.2/02](7.2_efficient_attention/02_flashattention.md) — implement online softmax in 20 lines of NumPy and confirm it matches standard attention exactly. It's the single best demystification in this part.

## Target time

5–7 days (shares "weeks 4–5" with Part 6). 7.1 and 7.4 are the load-bearing subsections; 7.3 can be skimmed unless SSMs are directly relevant to you.

## What's deliberately omitted

- **Serving-system design** — continuous batching, scheduling, speculative decoding, quantization: Part 9. 7.2/03 covers only the memory *layout* idea (paging), not the scheduler that exploits it.
- **RoPE scaling mechanics** (PI, NTK-aware, YaRN) — fully worked in [5.3/05](../part5_transformer_rebuilt/5.3_positional_information/05_context_length_extension.md). 7.5 assumes them and covers the *systems* and *evaluation* sides instead.
- **Distributed-training machinery** — data/tensor/pipeline parallelism and ZeRO/FSDP are Part 12; 7.5/02 covers only the sequence-dimension parallelism unique to long context, and 7.4/04 only the expert-parallel all-to-all unique to MoE.
- **SSM theory depth** — HiPPO, the continuous-time derivation, and the S4 kernel's mathematics. 7.3/02 takes the selective-scan and duality results and spends its pages on *why they matter*; the derivations are in the papers.
- **Norm/stability tricks at the attention interface** (QK-norm, logit soft-capping) — owned by [3.2/02](../part3_residual_connections_deep_networks/3.2_normalization_and_depth/02_scaling_the_residual_stream.md).
