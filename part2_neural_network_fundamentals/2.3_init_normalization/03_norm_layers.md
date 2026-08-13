# Normalization: BatchNorm → LayerNorm → RMSNorm

Normalization layers solve the same problem init solves: keep activations at a controlled scale across depth. Init does it at step 0; norm layers do it at every step throughout training. They're the standing-wave version of a one-shot scaling.

NLP's normalization history is short: BatchNorm was tried, failed, LayerNorm dominated through BERT/GPT-3, and RMSNorm has now replaced it in most LLMs. Understanding what each does and why the transition happened is essential for reading any post-2020 paper.

## Why normalize at all

The variance-preservation argument from file `01` only holds at init. During training, weight matrices evolve, activations drift, and layer-to-layer variance can grow or shrink arbitrarily. Without ongoing normalization:
- A layer can produce activations 10× larger than the next layer expects.
- LR-sized updates that worked at step 0 become tiny or huge for layers whose activations have drifted.
- Pre-softmax inputs (output head, attention) drift into saturated regimes and gradients vanish.

A norm layer reinjects scale control at each layer. Think of it as a "speed governor" that prevents activations from drifting too far from the unit-scale regime the rest of the network is designed around.

## BatchNorm (Ioffe & Szegedy, 2015)

The original. Normalize across the **batch** dimension for each feature:
```
For each feature d:
    μ_d = mean over batch of x_{b, d}
    σ_d = std over batch of x_{b, d}
    x_norm_{b, d} = (x_{b, d} - μ_d) / σ_d
    y_{b, d} = γ_d · x_norm_{b, d} + β_d
```

`γ` and `β` are learnable per-feature scale and shift. They let the layer un-normalize if normalization would hurt — but usually do not, because the optimizer benefits from the normalized regime.

**During training**: use batch statistics.
**During inference**: use **running** mean/std (an EMA accumulated during training). The discrepancy between training and eval behavior is BatchNorm's first big footgun.

### Why BN works for vision

In conv nets (CNNs), each feature dimension corresponds to a channel. Many images, many spatial locations all contribute to one channel's statistics — large effective batch size for normalization, low noise. BN gave a 2–3× training-speed boost when introduced; it dominated vision through 2018+.

### Why BN failed for NLP

Several reasons, in roughly decreasing importance:

1. **Batch-size sensitivity.** With small batches (which NLP often uses for long sequences), the per-batch statistics are noisy estimates of the true distribution. Tiny batches → bad normalization → training instability.
2. **Sequence-length variability.** NLP batches have variable sequence lengths (padding tokens). Including padding tokens in the statistics pollutes them; excluding them complicates the implementation. Both are bad.
3. **Train/eval mismatch.** During inference (especially for autoregressive generation), batch size can be 1. Running statistics from training may not match a single-sample distribution, causing degraded performance.
4. **No good story for sequence dimension.** Should you normalize across sequence positions? Across batch? Both? The "right" answer wasn't obvious for sequence data.

BN works fine on toy NLP tasks. At scale, all four problems compound. By 2018 it was clear NLP needed something else.

## LayerNorm (Ba, Kiros, Hinton 2016)

Normalize across the **feature** dimension for each sample, **independently per sample, per position**:
```
For each (b, s) position:
    μ = mean over D of x_{b, s, :}
    σ = std over D of x_{b, s, :}
    x_norm_{b, s, :} = (x_{b, s, :} - μ) / σ
    y_{b, s, :} = γ · x_norm_{b, s, :} + β
```

`γ` and `β` are now per-feature `(D,)` vectors, shared across batch and sequence positions.

**Crucial differences from BN**:
- Statistics computed *per token*, not over the batch. No batch-size dependence. No train/eval mismatch.
- Same operation at train and inference.
- Works fine with batch size 1 (which matters for inference and for some fine-tuning setups).

This is why Transformers adopted LN. The 2017 "Attention is All You Need" paper used it, and every Transformer through ~2022 followed suit.

### What LayerNorm actually does

The transformation `(x - μ)/σ` does two things:
1. Centers the vector at 0 (subtract mean).
2. Scales it to unit RMS (divide by std).

Geometrically: project `x` onto the hyperplane `Σ x_i = 0`, then radially scale to the unit sphere of that hyperplane. The output lives on a `(D-1)`-dimensional sphere of radius `√D`.

The learnable `γ, β` then rescale and reshift each coordinate. After all this, the per-token vector is at a controlled scale, but the network has flexibility to recover any constant rescale or shift it wants.

## RMSNorm (Zhang & Sennrich, 2019)

The minor-variant-that-took-over. Skip the mean subtraction:
```
For each (b, s) position:
    rms = √(mean over D of x_{b, s, :}²)
    y_{b, s, :} = γ · x_{b, s, :} / rms
```

That's it. Divide by the RMS (root-mean-square), multiply by a learnable scale `γ`. No mean centering, no shift `β`.

### Why RMSNorm replaced LN

Three reasons:

1. **Cheaper compute.** No mean subtraction → one fewer pass over the data. Roughly 7–10% faster than LN at inference time, which is non-trivial when LayerNorm runs once or twice per layer for hundreds of layers.
2. **Empirically as good or better.** The original RMSNorm paper showed near-identical quality with the simpler formula. The Llama paper repeated this and adopted it. No one has found a strong empirical reason to prefer LN.
3. **The mean subtraction isn't doing much.** In a deep pre-norm Transformer, by the time you reach layer 10+, the activation distribution is centered around zero anyway (because that's what learned features tend to do). Centering it again is a no-op most of the time.

The omission of `β` (the learnable shift) similarly removes redundancy: a `+ β` in a pre-norm Transformer is immediately followed by a Linear layer, which has its own (potentially) `+ b`. The shift is absorbed downstream.

### What modern LLMs use

| Model | Norm |
|---|---|
| BERT, GPT-2, GPT-3, T5, OPT | LayerNorm |
| Llama 1, 2, 3 | RMSNorm |
| Mistral, Mixtral | RMSNorm |
| DeepSeek-V2, V3 | RMSNorm |
| PaLM, Gemini (likely) | RMSNorm |
| Qwen 1, 2, 2.5 | RMSNorm |
| Gemma | RMSNorm |

Pre-2021: LayerNorm. Post-2021: RMSNorm. The transition was driven by Llama 1's release. No modern from-scratch LLM uses LayerNorm anymore.

## How norm interacts with the rest of the architecture

- **Init**: norm layers expect inputs roughly unit-scaled. Init schemes (file `02`) provide this at step 0; norms maintain it through training.
- **Pre-norm vs. post-norm** (file `04`): *where* you put the norm matters enormously. The interior computation is the same; the residual flow it enables is what makes pre-norm so much more trainable at depth.
- **Mixed precision** (file 2.4/05): norms are usually kept in fp32 even when the rest of the model runs in bf16, because the variance computation needs the precision. Casts are typically inserted automatically.
- **QK-norm**: a recent variant — apply RMSNorm to Q and K before the attention matmul. Stabilizes attention scores at very large widths. Used in some recent LLMs (Reka, some Qwen variants).

## Computation cost

For an input of shape `(B, S, D)`:
- **LN**: one mean (`O(D)` per token), one variance (`O(D)`), one elementwise divide and rescale (`O(D)`). Total: `O(B · S · D)`.
- **RMSNorm**: one RMS (`O(D)` per token), one rescale (`O(D)`). Total: `O(B · S · D)` but with fewer ops.

In a Transformer block, norm cost is small relative to attention's `O(B · S² · D)` or FFN's `O(B · S · D²)` for typical shapes. But it's run twice per block (once before attention, once before FFN) and the operations are bandwidth-bound — so the cost isn't negligible at inference time for long sequences.

## Self-check

1. Why does BatchNorm fail with batch size 1, and why is LayerNorm immune?
2. RMSNorm doesn't center its input (no mean subtraction). Doesn't this mean activations could drift in mean over time? Why doesn't this break training?
3. The learnable `γ` in any norm is initialized to 1. Why? What goes wrong if you init it to 0 or to a small random value?

### Answers

1. BN computes statistics across the batch. With batch size 1, the "batch" is a single sample — `μ` is just that sample's per-feature value, and `σ = 0` (no variance with one sample). The normalization becomes `(x - x)/0 = NaN`. Some implementations special-case this with a small epsilon to avoid the NaN, but the resulting "normalization" carries no useful information (you're dividing by epsilon, getting massive scaled values). LayerNorm computes statistics across the feature dimension of *each sample independently*. Batch size doesn't enter — the statistics are well-defined with batch size 1 (you have `D` features to compute mean/std over). This is why LN works for autoregressive generation, single-sample inference, and small-batch fine-tuning, where BN doesn't.
2. The mean of the input *can* drift, but it doesn't matter much because the subsequent Linear layer can absorb any constant shift. If the post-RMSNorm activations have mean `m ≠ 0` going into a Linear `y = Wx + b` (or `y = Wx` for bias-free LLMs), the Wx output has a deterministic mean `Wm` that the downstream layers learn to handle. In deep pre-norm Transformers, the activations naturally center around 0 anyway, so RMSNorm's missing centering is a no-op most of the time. Empirically: the absence of mean subtraction has no measurable effect on final loss in the regimes tested. RMSNorm wins on simplicity and compute.
3. `γ = 1` makes the norm at step 0 an **identity operation** (up to the centering/dividing): the network starts in the configuration "no learned per-feature rescaling, raw normalized signal passes through." This is a sensible neutral starting point — the model can then learn `γ ≠ 1` if needed. Init `γ = 0`: the entire norm layer outputs 0, no signal passes through, no gradient flows to the next layer's input (or rather, only the bias `β` matters). The model can recover by learning `γ ≠ 0` but takes longer. Init `γ = 0.01` random: similar — most signal blocked at init, slow start. The standard recipe (`γ = 1`, `β = 0`) starts with identity-ish behavior and lets the network learn deviations.

## Exercise

In a notebook, implement RMSNorm from scratch (3 lines). Compare its output to `torch.nn.RMSNorm` (PyTorch ≥ 2.4) on random `(B, S, D) = (2, 8, 64)` tensors. Should match to ~1e-6.

Then take a small Transformer and replace its LayerNorm modules with your RMSNorm. Train both versions on a tiny LM task. The loss curves should be nearly identical, but the RMSNorm version should be slightly faster per step (~5–10% in PyTorch eager mode, more under `torch.compile`).
