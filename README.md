# OCM

[![arXiv](https://img.shields.io/badge/arXiv-2607.02846-b31b1b.svg)](https://arxiv.org/abs/2607.02846)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation for **Object-Centric Environment Modeling for Agentic
Tasks**.

<p align="center">
  <img src="ocm.jpg" alt="OCM overview" width="850">
</p>

## Overview

Large language model agents can improve through accumulated experience, but
free-form textual memories become difficult to maintain, validate, and reuse as
interactions grow. OCM organizes experience into an executable object-centric
environment model.

During evaluation, OCM maintains two connected code bases:

- object knowledge in `knowledge/object_knowledge.py`, which defines environment
  entities and mechanisms as Python classes;
- procedural knowledge under `knowledge/procedural_knowledge/`, which records
  reusable interaction patterns that import and use the object model.

The canonical `ocm` agent combines OCM's structured code memory with a
WorldEvolver-style predict-observe-update loop. The policy uses OCM to choose a
primitive action, then the same object/procedural code is used to predict the
action's immediate observable outcome before execution. The prediction records
which OCM entries support it, turning otherwise passive memory into a falsifiable
hypothesis. The policy then sees that fallible foresight and may keep or revise
its proposed action. If it revises, OCM predicts the final action as well, so the
prediction later used for learning always corresponds to the action actually
executed.

After execution, the prediction is paired with the real observation, reward,
success, and validity signals. This residual is visible to the next policy step
and, at episode end, updates the same object and procedural code used for future
predictions. This gives one closed loop: act from OCM, predict, observe, and
revise OCM. Reflection receives the existing knowledge implementation alongside
the executed transitions, so it can revise an incorrect claim rather than only
append new lessons. Patch retries include the rejected output and error against
the unchanged source.

The predictor receives the complete executed action/observation history, without
the policy's historical thoughts or predictions. The policy retains its full
history and compares its proposed action and reasoning with the predicted outcome.
This keeps an intended result from serving as evidence for the world model.
The shorter predictor history reduces repeated context processing. Every action
is still reviewed without a validity or effect gate.
The logs retain proposed and executed actions, predicted outcomes, knowledge
references, and observed transitions for mechanism analysis.

The runner currently supports:

- `sciworld`: ScienceWorld task environments.
- `alfworld`: ALFWorld household instruction-following tasks.
- `plancraft`: Minecraft-style crafting tasks.
- `webshop`: WebShop text shopping tasks.

Raw benchmark repositories, cached datasets, model outputs, and analysis
artifacts are intentionally not included in this source release.

## Setup

Create a Python environment and install the dependencies:

```bash
conda create -n oom python=3.10
conda activate oom
python -m pip install -r requirements.txt
```

Benchmark data and runtime notes:

- ScienceWorld and PlanCraft work from the Python dependencies above.
- ALFWorld needs its data downloaded with `alfworld-download`.

Set one API key for the provider you plan to use:

```bash
export OPENAI_API_KEY=...
# or, if using OpenRouter
export OPENROUTER_API_KEY=...
```

## Quick Start

Run a small OCM evaluation:

```bash
python main.py \
  --provider openai \
  --model-name gpt-4.1-mini \
  --agent ocm \
  --benchmark sciworld \
  --split agentboard_test \
  --task-limit 3 \
  --run-name ocm-smoke \
  --output-dir outputs
```

By default the runner uses provider `openai`, model `gpt-4.1-mini`,
temperature `0.0`, and writes results under `outputs/`.

## Benchmark Splits

- `sciworld`: `agentboard_test` (90), `train` (1483), `dev` (194),
  `test` (149; legacy evaluation protocol), `test_full` (1819)
- `alfworld`: `train` (3553), `eval_in_distribution` (140),
  `eval_out_of_distribution` (134)
- `plancraft`: `train` (1145), `val` (570), `val.small` (110),
  `val.small.easy` (100), `val.repeated` (570), `test` (580),
  `test.small` (117), `test.small.easy` (100)
- `webshop`: `train` (1824), `test` (200), `test500` (500), `full` (12087)

Default step limits:

- `sciworld`: `100`
- `alfworld`: `50`
- `plancraft`: `30`
- `webshop`: `15`

## License

This repository is released under the MIT License. See `LICENSE`.
