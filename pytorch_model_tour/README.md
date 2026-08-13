# PyTorch Model Tour — small end-to-end models, increasing complexity

A short, runnable tour of the core model families, each **trained and evaluated end-to-end** on a tiny dataset so you see real results (loss/accuracy curves, generated text) in seconds. The point is breadth and fluency: how each architecture is built in PyTorch, what task each is *for*, and how they connect — using models and data small enough to train live on a laptop.

This is a **side-track**, a companion to [../numpy_pytorch_schedule.md](../numpy_pytorch_schedule.md). Do the [numpy_pytorch/](../numpy_pytorch) sessions first (tensors, autograd, `nn.Module`, the training loop) — this tour assumes them and reuses the same loop with progressively richer models.

## The arc (do them in order)

Each notebook adds exactly one idea over the previous, and every one ends with a plot or sample so you *see* it work:

1. **[01_the_workflow_linear_classifier.ipynb](01_the_workflow_linear_classifier.ipynb)** — the reusable workflow (data → train/val split → train loop → eval → plot) with the simplest model, a linear classifier, on 2-D blobs.
2. **[02_mlp_nonlinearity_and_overfitting.ipynb](02_mlp_nonlinearity_and_overfitting.ipynb)** — why nonlinearity matters (linear *fails* on concentric circles, an MLP solves it), then **overfitting** (watch val loss turn up) and the regularizers that tame it.
3. **[03_embeddings_text_classification.ipynb](03_embeddings_text_classification.ipynb)** — from continuous features to **tokens**: `nn.Embedding` + mean-pooling (a bag-of-words classifier) on a token-sequence task.
4. **[04_rnn_lstm_sequences.ipynb](04_rnn_lstm_sequences.ipynb)** — **order matters**: a task bag-of-words can't do (~chance), solved by an **LSTM**; plus char-level text generation.
5. **[05_self_attention_and_position.ipynb](05_self_attention_and_position.ipynb)** — **self-attention** on the same order task, and the key lesson that attention is order-blind until you add **positional embeddings**.

The through-line mirrors the curriculum: linear → nonlinear (MLP) → embeddings → recurrence (Part 4) → attention + position (Part 5).

## What each notebook teaches

| # | Model | New PyTorch / concepts | The "aha" result |
|---|---|---|---|
| 01 | `nn.Linear` classifier | train/val split, eval mode, `no_grad`, learning curves | clean fit, train ≈ val |
| 02 | MLP (`nn.Sequential`) | nonlinearity, overfitting, weight decay / dropout / early stopping | linear fails on circles; val loss rises then reg tames it |
| 03 | `nn.Embedding` + mean-pool | tokens, embeddings, pooling (bag of words) | nails a presence task; blind to order |
| 04 | `nn.LSTM` | recurrence, hidden state, sequence classification + generation | solves an order task bag-of-words can't; generates text |
| 05 | `nn.MultiheadAttention` | self-attention, positional embeddings, permutation-invariance | solves order **only** with positional info |

## How to use

Open a notebook and run cells top to bottom, reading each output/plot. Everything is self-contained (synthetic data generated inline or a tiny inline corpus — no downloads) and device-aware (uses CUDA/MPS if present, else CPU). Change the numbers — dataset size, model width, epochs, the task rule — and re-run to build intuition. All five run end-to-end in well under a minute each on a laptop.

## Target time

Half a day to a day for all five — they're deliberately short and fast.

## Scope

On the repo's NLP→LLM track: notebooks 1–2 use 2-D synthetic data purely to establish the workflow and the overfitting lesson; 3–5 are sequence/token models leading into the Transformer. **No CNNs / vision** (out of scope) and **no advanced tricks** (mixed precision, GQA, RoPE, FlashAttention) — those arrive, with motivation, in the main curriculum. The models here are the simple, honest baselines everything else builds on.
