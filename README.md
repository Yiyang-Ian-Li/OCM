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

After each episode, OCM reflects on the trajectory, updates both knowledge
bases, and verifies that procedures execute against the updated object model.
During future interaction, the agent uses progressive disclosure to inspect
compact code signatures first and read source only when needed.

Each environment action also carries a prediction of its immediate observable
outcome and cites the object and procedure symbols that support it. The realized
transition then provides localized evidence for confirming or revising those
knowledge units during the episode update.

The runner currently supports:

- `sciworld`: ScienceWorld task environments.
- `alfworld`: ALFWorld household instruction-following tasks.
- `plancraft`: Minecraft-style crafting tasks.
- `webshop`: WebShop text shopping tasks.
- `intercode`: InterCode Bash tasks.

Raw benchmark repositories, cached datasets, model outputs, and analysis
artifacts are intentionally not included in this source release.

## Setup

Create a Python environment and install the dependencies:

```bash
conda create -n ocm python=3.10
conda activate ocm
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
  --split test \
  --task-limit 3 \
  --output-dir outputs/gpt-4.1-mini
```

By default the runner uses provider `openai`, model `gpt-4.1-mini`,
temperature `0.0`, and writes results under `outputs/`.

## Benchmark Splits

- `sciworld`: `train` (1483), `dev` (194), `test` (211)
- `alfworld`: `train` (3553), `eval_in_distribution` (140),
  `eval_out_of_distribution` (134)
- `plancraft`: `train` (1145), `val` (570), `val.small` (110),
  `val.small.easy` (100), `val.repeated` (570), `test` (580),
  `test.small` (117), `test.small.easy` (100)
- `webshop`: `train` (1824), `test` (200), `test500` (500), `full` (12087)
- `intercode`: `bash` (200), `bash_fs1` (60), `bash_fs2` (53),
  `bash_fs3` (60), `bash_fs4` (27), `bash_quickstart` (24)

Default step limits:

- `sciworld`: `100`
- `alfworld`: `50`
- `plancraft`: `30`
- `webshop`: `15`
- `intercode`: `10`

## License

This repository is released under the MIT License. See `LICENSE`.
