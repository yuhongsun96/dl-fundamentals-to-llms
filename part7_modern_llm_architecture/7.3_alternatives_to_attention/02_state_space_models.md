# State-Space Models — S4, Mamba, and the Duality

The other route to a fixed-state sequence model, arrived at from signal processing rather than from attention — and after four years of looking like a separate field, revealed to be the same thing as [linear attention](01_linear_attention.md). This file covers the lineage's three load-bearing ideas and skips the continuous-time mathematics, which is beautiful and not necessary for reading model cards.

## The starting point: a linear recurrence

A state-space model is the control-theory workhorse, discretized:

```
hₜ = A hₜ₋₁ + B xₜ          # state update:  h ∈ R^N per channel
yₜ = C hₜ                    # readout
```

Compare to a vanilla RNN, `hₜ = tanh(W hₜ₋₁ + U xₜ)`. The difference is one omission: **no nonlinearity in the recurrence.** That single absence is what makes the family work, for exactly the reason [file 01](01_linear_attention.md) gave: a linear recurrence can be *unrolled and computed in parallel* (as a convolution, or via an associative scan), while a nonlinear one is stuck being sequential. Removing the softmax and removing the `tanh` are the same move — both restore associativity, and associativity is what buys parallel training.

**S4** (Gu et al., 2021) made this competitive by constraining `A` to a special structure (the HiPPO initialization) so the recurrence provably retains long-range information, and by computing the whole sequence as a **long convolution** via FFT — `O(S log S)` training, no sequential dependency. It won long-range benchmarks decisively. It also had a problem.

## Mamba's contribution: selectivity

S4's `A`, `B`, `C` are **fixed** — the same for every token. That makes it a *linear time-invariant* system, which is exactly why the convolutional form works, and also why it can't do the one thing language needs most: **decide what to remember based on content.** A time-invariant system compresses history by a schedule fixed at initialization; it cannot look at a token and choose to overwrite its state.

**Mamba** (Gu & Dao, 2023) makes `B`, `C`, and the discretization step `Δ` **functions of the input** — "selective" SSMs:

```
Δₜ, Bₜ, Cₜ = linear projections of xₜ        # data-dependent!
hₜ = Ā(Δₜ) hₜ₋₁ + B̄(Δₜ, Bₜ) xₜ
```

`Δₜ` acts as a learned, per-token gate: large `Δ` means "reset and absorb this token," small `Δ` means "ignore this input, keep the state." That is a **forget/input gate** ([4.1/02](../../part4_sequence_modeling_lineage/4.1_rnns_and_lstms/02_vanishing_gradient_and_gating.md)) reinvented via discretization step size — and it's the same fix gated linear attention applies ([file 01](01_linear_attention.md)).

The cost: input-dependent `A`, `B`, `C` break time-invariance, so the FFT convolution no longer applies. Mamba's second contribution is therefore a **systems** one — a *hardware-aware parallel scan* that keeps the state in SRAM and never materializes it to HBM, plus recomputation in the backward pass. That is FlashAttention's playbook ([7.2/02](../7.2_efficient_attention/02_flashattention.md)) applied to a scan: spend FLOPs, avoid HBM. Without it, selective SSMs would be too slow to matter — which is a recurring lesson in this part, that an architectural idea and its kernel are one artifact.

## Mamba-2 and the duality

**Mamba-2** (Dao & Gu, 2024) is the conceptually important one. It shows that structured SSMs and linear attention are **two views of one class of operators** — "structured state-space duality" (SSD). Concretely, a selective SSM's action over a sequence can be written as a *structured mask* applied to a matrix product; choose the structure one way and you have linear attention with decay, another and you have an SSM.

Three consequences:

- **Two literatures merged.** RetNet, GLA, DeltaNet, S4, Mamba, and lightning attention stop being separate inventions and become a design space with shared vocabulary: *what is the state, and what is the decay/gating structure?*
- **Better algorithms.** The dual view exposes a block-decomposition that uses dense matmuls (tensor-core-friendly) instead of a pure elementwise scan — Mamba-2 is substantially faster to train than Mamba-1 mostly for this reason.
- **The tradeoff becomes legible.** Mamba-2 enlarges the state relative to Mamba-1, and that is directly a *recall-capacity* choice ([file 01](01_linear_attention.md)'s pigeonhole limit, [file 03](03_hybrids_and_recall.md)'s measurements). The knob is now explicit rather than implicit.

## What SSMs actually buy, in Part 7's currency

| | Softmax attention | Selective SSM |
|---|---|---|
| Per-token inference compute | `O(S)` (attend to all) | **`O(1)`** |
| Inference memory | KV cache, `O(S)` — 320 KiB/token at 70B | **fixed state, `O(1)`** |
| Training | parallel matmuls | parallel scan (or SSD matmuls) |
| Exact retrieval of arbitrary past token | **yes** | no — superposed state |
| Throughput on long sequences | falls with `S` | ~flat |

The first two rows are the entire reason anyone cares: an SSM's inference cost **does not grow with context**. No cache, no paging ([7.2/03](../7.2_efficient_attention/03_kv_cache_memory_management.md)), no 40 GiB per sequence. The fourth row is why they haven't won ([file 03](03_hybrids_and_recall.md)).

## Why it matters in modern LLM work

- **It closes Part 4's loop explicitly** — the recurrence returns, having fixed both of its historical defects: parallel training (via linearity) and selective forgetting (via input-dependent gates).
- **The duality is the right mental filing system.** When a new "linear-time architecture" appears, ask the two SSD questions — what's the state size, what's the gating structure — and you'll usually place it in five minutes.
- **The kernel-is-part-of-the-architecture lesson** repeats: Mamba's scan, FlashAttention's tiling, MLA's absorbed matrices. An architecture without a good kernel is a paper, not a model.

## Self-check

1. What single omission distinguishes an SSM's recurrence from a vanilla RNN's, and what does it buy?
2. Why can S4 use an FFT convolution, and why can't Mamba?
3. What is `Δₜ` doing intuitively, and which classical mechanism is it?
4. Why was a custom kernel necessary for Mamba rather than a nice-to-have?
5. State the SSD duality claim and one practical benefit that followed from it.
6. An SSM's inference memory is `O(1)`. Name the capability that buys, and the capability it costs.

### Answers

1. **No nonlinearity inside the recurrence** — the state update is linear in `hₜ₋₁`. That restores associativity, so the recurrence can be unrolled into a parallel form (convolution or associative scan) instead of being computed sequentially. It's the same structural move as deleting softmax in [file 01](01_linear_attention.md): give up a nonlinearity, gain parallel training.
2. S4's `A`, `B`, `C` are fixed for all tokens, making the system **linear time-invariant** — so its action over the whole sequence is a convolution with a single fixed kernel, computable by FFT in `O(S log S)`. Mamba makes `Δ, B, C` **input-dependent**, so the effective kernel differs at every position; there's no single kernel to convolve with, and it must use a scan instead.
3. `Δₜ` is the per-token discretization step, and functionally it's a **gate**: large `Δₜ` makes the state absorb the current token strongly (and forget more of the past), small `Δₜ` makes the token nearly ignored and the state persist. It is the LSTM's forget/input gate ([4.1/02](../../part4_sequence_modeling_lineage/4.1_rnns_and_lstms/02_vanishing_gradient_and_gating.md)) reached through a different door.
4. Because dropping the FFT for a scan removes the fast path, and a naive scan materializes the (large) hidden state to HBM at every step — making the model memory-bound and slower than the attention it was trying to beat ([7.2/01](../7.2_efficient_attention/01_the_memory_bandwidth_wall.md)). The hardware-aware scan keeps state in SRAM and recomputes in the backward pass, which is what makes the architecture *usable*; without it the idea doesn't ship.
5. **Structured state-space duality:** selective SSMs and linear attention (with decay) are two parameterizations of one operator class — an SSM's sequence-level action is a structured-mask matrix product. Practical benefits: it unified the two research literatures into a single design space, and it exposed a block decomposition built from dense matmuls, making Mamba-2 substantially faster to train than Mamba-1 by using tensor cores instead of a purely elementwise scan.
6. It buys **context-length-independent inference** — constant per-token compute and constant memory, so no KV cache, no paging, and throughput that doesn't decay as the conversation grows. It costs **exact retrieval of an arbitrary past token**: the fixed state superposes all history, so recall of specific distant details degrades once capacity saturates ([file 03](03_hybrids_and_recall.md)).

## Exercise

Build the smallest honest SSM. (a) Implement the linear recurrence `hₜ = a·hₜ₋₁ + b·xₜ`, `yₜ = c·hₜ` for scalar state, and verify your sequential loop matches a parallel implementation via `torch.cumsum` on the log-space unrolling (`hₜ = Σᵢ a^(t−i) b xᵢ`) — this is the associativity that buys parallelism, in one line. (b) Make it *selective*: let `a` depend on the input via a sigmoid gate, then show that the closed-form cumsum trick **no longer applies** and you need a scan — reproducing exactly why Mamba abandoned the FFT. (c) Train two tiny models on a copy task (emit a marked 8-token substring from earlier in the sequence): a fixed-`a` SSM and a selective one. The selective model should win, and you should be able to plot its learned gate spiking at the marked tokens. (d) Then extend the substring to 64 tokens and watch *both* fail against a small attention baseline — the setup for [file 03](03_hybrids_and_recall.md).
