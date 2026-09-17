---
name: vision-eval
description: Use when measuring detection, PPE, attribute or VLM accuracy; comparing a pipeline, prompt, crop, gate, policy or model change against a baseline; working with tools/vision_eval, tools/p9_dataset, datasets/ manifests or vision-os-data; or when a result looks like a regression.
---

# Vision Evaluation

## Overview

A perception change is judged by a **measured run against a saved baseline on the same annotated
dataset**, never by eyeballing frames or by a unit test passing. Runs are compared side by side and
never averaged — each describes one configuration somebody actually ran.

## Offline evaluation (`tools/vision_eval`)

```bash
PY=.venv/Scripts/python.exe
$PY -m tools.vision_eval.run_baseline datasets/kitchen-01 --tag baseline             # reference
$PY -m tools.vision_eval.run_baseline datasets/kitchen-01 --tag <change> --cache      # candidate
$PY -m tools.vision_eval.compare datasets/kitchen-01 baseline <change>
```

- Results land in `<dataset>/results/<tag>.json`; `compare` prints report tables straight from them,
  so no number in a report is transcribed by hand.
- Defaults: `--policy config/policies/kitchen-safety.example.json`, `--provider nvidia`,
  API key from `--env-file .env` (never from source).
- Ablations: `--no-regions` (one whole-subject crop), `--no-gate` (ask even for rejected crops),
  `--crop-size N`, `--limit N` for a smoke run. `--cache` resumes an interrupted identical run.
- VLM calls are real and slow (~11 s each). Smoke with `--limit 5` before a full run; run full runs in
  the background.
- Change **one** variable per tag. Two changes in one run cannot be attributed.

## Reading a regression

Compare per attribute and per state, not overall accuracy:

| Movement | Severity |
|---|---|
| `not_visible` / `unknown` answered as `absent` | **blocker** — fabricates a violation |
| `absent` answered as `present` | major — misses a real violation |
| more `not_visible` on frames annotated as decided | minor — honest loss of coverage |
| fewer model calls for the same answers | improvement (check the gate is not dropping real evidence) |

Regression suites that must stay green alongside: `pytest tests/compliance/test_dataset_regression.py
tests/compliance/test_ppe_uncertainty.py tests/compliance/test_pose_gate_effect.py tests/tools`.

## Datasets and the external store

- The rule is that `datasets/` commits only the citable record — manifests, digests, provenance,
  annotations, results — and **no recoverable image of any person**. The older datasets predate
  that rule: `git ls-files datasets | grep -ciE '\.(jpg|jpeg|png|mp4|webp)$'` currently counts 153
  tracked images (e.g. `datasets/kitchen-01/frames/`). Never add to them; new imagery goes only to
  the external store.
- Imagery lives in `vision-os-data/` beside the repos (never inside one), resolved from
  `$VISION_OS_DATA_ROOT` or `../vision-os-data`. `DatasetStore.resolve()` refuses a root under a
  working tree.

```bash
$PY -m tools.p9_dataset.store status          # where am I pointed, and is it safe
$PY -m tools.p9_dataset.store init
$PY -m tools.p9_dataset.store verify --layer candidates --dataset p9-live --version v1
```

Layers stay apart: `raw/`, `candidates/`, `annotations/`, `benchmarks/`, `traces/`.

## Rules

- `tools/` is migrated verbatim and lint-excluded; the `protect` entry in `.claude/team.conf` blocks edits. Add a new
  script alongside only with explicit user confirmation.
- Never add an HTTP endpoint that triggers evaluation — it would let a user spend the model budget.
- Never copy frames into the repository to "make a test self-contained".
- Report with `unityworks-team:evidence-report`: dataset, both tags, the exact commands, the
  `compare` table.
