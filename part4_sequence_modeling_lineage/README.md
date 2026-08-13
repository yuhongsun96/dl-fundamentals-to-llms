# Part 4 — Sequence Modeling Lineage (Brief)

The short chapter. You lived through RNNs, LSTMs, and seq2seq — this part isn't here to teach them, it's here to **restore the mental model and answer one question per idea: why did it lose?** Everything modern (Part 5's Transformer, Part 7's efficient-attention and state-space revival) is a reaction to the specific failures catalogued here, so the "why it died" framing is what makes the rest of the curriculum legible.

Two things make this worth more than a nostalgia tour:

1. **The Transformer is defined by what it removed.** Attention-only makes no sense until you feel the exact pain — the sequential-compute wall and the fixed-vector bottleneck — that recurrence and seq2seq imposed. This part supplies that pain.
2. **The lineage is a loop, not a line.** The RNN's compress-history-into-a-fixed-state idea didn't die; it came back as state-space models / Mamba (Part 7.3), and the LSTM's additive gated carry is the same trick as the residual stream (Part 3). You'll meet all of these again wearing new clothes.

## Structure

- **4.1 RNNs and LSTMs** — recurrence, why its gradients broke, and why it lost the hardware lottery:
  - `01` Hidden-state recurrence & BPTT — the shared-weights-across-time picture, and why backprop-through-time is a *product of Jacobians over the sequence* (the same math as the depth story in Part 3).
  - `02` Vanishing gradients → gating — why vanilla RNNs can't learn long range, and how the LSTM's additive cell-state carry (the "constant error carousel") is the skip connection's across-*time* twin.
  - `03` Why they lost — sequential compute, the fixed-state context ceiling, and scaling difficulty; the trade the Transformer made to beat them, and the trade Part 7.3 makes to win some of it back.
- **4.2 Seq2seq and the First Attention** — the bottleneck and the idea that dissolved it:
  - `01` Encoder–decoder with a bottleneck — cramming a whole source into one fixed vector, and why quality collapsed with length.
  - `02` Bahdanau/Luong attention as learned alignment — the decoder gets direct, weighted access to *all* encoder states; this is query–key–value in embryo.
  - `03` The seed for Transformers — the leap from "attention bolted onto an RNN" to "attention *is* the model," and exactly what attention had to grow to stand alone (→ Part 5).

## How to use

Fast. If the recurrence math is muscle memory, skim `4.1/01` and spend your time on the *why-it-died* file (`4.1/03`) and the attention-as-alignment file (`4.2/02`), which are the two that pay off directly in Parts 5 and 7. Read `4.2/03` last — it's the on-ramp to the Transformer rebuild.

## Target time

Half a day to a day. This part leans entirely on things you already know; its value is the connective tissue, not new mechanics.

## What's deliberately omitted

- **Cell-by-cell LSTM/GRU derivations and variant zoo** (peephole connections, coupled gates, etc.). We take the *one* structural idea that mattered — the additive gated carry — and leave the wiring diagrams to the original papers.
- **Convolutional sequence models** (WaveNet, ByteNet, ConvS2S). A real branch of the lineage, but a detour off the NLP→LLM path; mentioned only where it explains a design choice.
- **The Transformer itself.** Everything from self-attention mechanics onward is Part 5. This part stops exactly at the water's edge — the moment recurrence gets deleted.
