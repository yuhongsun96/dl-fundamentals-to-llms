# Part 2 — Neural Network Fundamentals

The mechanics that make a stack of linear layers actually train. You've seen all of this before, but in the BERT era many of the modern choices (RMSNorm, AdamW, pre-norm, bf16, no-dropout) had not yet settled. This part restores the mental model and points out where the consensus has moved.

## Structure

- **2.1 The MLP as a Universal Building Block** — linear + nonlinearity, modern activations, the death of bias terms.
- **2.2 Backpropagation, Deeply** — forward / backward as graph traversal, autograd, checkpointing, gradient pathologies.
- **2.3 Initialization & Normalization** — variance preservation, Xavier/Kaiming, LayerNorm/RMSNorm, pre- vs. post-norm.
- **2.4 Optimization** — SGD → AdamW, schedules, clipping, sharpness, mixed precision.
- **2.5 Regularization** — dropout (and its disappearance), weight decay, label smoothing, data augmentation in NLP.

Each subsection folder contains primary study files at the top level, numbered in reading order (`01_...md`, `02_...md`, ...). No supplementaries in this part — the primary files stand alone.

## How to use

Same as Part 1: ~5–10 min per file. Don't move on from a file until the self-check answers feel trivial. The arc here is: build the smallest trainable unit (MLP) → understand how it learns (backprop) → understand why it sometimes doesn't (init, norm, schedules) → understand the modern tricks that made deep stacks routine.

## Target time

3–4 days. The optimization and norm files are dense — that's where most of the post-2018 evolution lives.

## What's deliberately omitted

- **Vision-specific normalization** (GroupNorm, InstanceNorm, BatchNorm details beyond "why NLP doesn't use it").
- **CNN-specific init** (orthogonal init for conv stacks, etc.).
- **Anything specific to a particular framework's autograd internals** beyond the mental model — PyTorch idioms are illustrative, not the point.
