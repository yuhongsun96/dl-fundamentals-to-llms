# Part 1 — Mathematical Foundations

Review material for an NLP practitioner returning to DL. Not a textbook — assumes you've seen this before and just need the mental model restored.

## Structure

- **1.1 Linear Algebra** — the shape and flow of computation
- **1.2 Calculus & Probability** — how learning happens and what we're optimizing
- **1.3 Information Theory** — why next-token prediction is the whole game
- **[1.4 Optional Deeper Knowledge](1.4_optional_deeper_knowledge/)** — a vocabulary layer for the math language used casually elsewhere (subspaces and intrinsic dimension, "energy," bilinear forms, aliasing, "to first order"). **Optional and non-linear** — not a prerequisite for anything; read one file when its terms are what's slowing you down.

Each subsection folder contains:

- **Primary study files** at the top level, numbered in reading order (`01_...md`, `02_...md`, ...). These are the core material — the canonical reading path.
- **`supplementary/`** — optional sidecar artifacts. Mixed kinds, all numbered to match the primary file they attach to:
  - `0K_<topic>.md` — deep dive on a concept (e.g. `02_swiglu.md` expands on SwiGLU from primary file 02).
  - `0K_<topic>_quiz.ipynb` — runnable self-quiz drilling primary file `0K`.
  - `0K_<topic>.ipynb` — runnable hands-on walkthrough (e.g. `04_low_rank_approximation.ipynb` for the SVD exercise from file 04).

Supplementary content is additive: skippable on a first pass, valuable when reviewing or going deep. The primary file always links to its sidecars.

## How to use

Read each file (~5–10 min each). For anything that feels shaky, do the tiny exercise at the bottom on paper or in a Jupyter cell. Don't move on from a file until the "self-check" questions feel trivial.

## Target time

2–3 days of focused work. Don't linger — Part 2 (backprop) is where things get real.
