# Part 8 — Post-Training

Everything that happens to a checkpoint after pretraining: instruction tuning, preference optimization, reasoning RL, parameter-efficient adaptation, and repurposing the checkpoint for non-generative uses (embeddings). This is the part of the field that has moved most since the BERT era — in 2019 "fine-tuning" meant one supervised pass on a task dataset; today it is a multi-stage discipline of its own.

## Structure

- **8.1 Supervised Fine-Tuning (SFT)** — instruction tuning, chat templates, data quality over quantity.
- **8.2 Preference Optimization** — RLHF (reward model + PPO), DPO and its variants, RLAIF.
- **8.3 Reasoning & RL** — chain-of-thought, process vs. outcome rewards, RLVR, GRPO.
- **8.4 Parameter-Efficient Fine-Tuning** — LoRA, QLoRA, the adapter zoo.
- **8.5 Embedding Models from LM Checkpoints** — turning a generative checkpoint into a representation model: pooling, anisotropy, contrastive adaptation.

Only **8.5** has content so far — it was written ahead of reading order. The other subsections will be filled in when this part is reached in sequence.

## How to use

Same as earlier parts: primary files numbered in reading order, ~10 min per file, don't move on until the self-check answers feel trivial. 8.5 is self-contained and readable early: it depends only on knowing what a Transformer forward pass produces and on the InfoNCE material in [1.3 mutual information](../part1_math_foundations/1.3_information_theory/02_mutual_information.md).
