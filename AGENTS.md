# AGENTS.md

## Project Overview

This repository contains a compact CNN project on CIFAR-10 for adversarial robustness and Grad-CAM analysis.

The project is organized by Work Packages. Codex must follow `WP_PLAN.md` before starting any implementation task and must follow `TESTING.md` before marking a Work Package as completed.

## Language Rules

* Explain plans, reasoning, and summaries in Chinese.
* Keep code, comments, filenames, terminal commands, error messages, commit messages, and technical identifiers in English.
* Do not translate terminal output or error messages.
* When reporting changes, use Chinese explanations with English filenames and commands.

## Current Project Status

* WP0 is mostly completed.
* WP1 is mostly completed.
* WP2 is completed.
* WP3 is in progress.
* `Linear.backward` is implemented and tested.
* The next WP3 step is `ReLU.backward`.
* Training, adversarial attacks, Grad-CAM, and final analysis have not been started.

## Current Development Priority

The immediate priority is WP3 manual backward implementation.

Before continuing WP3, Codex must:

1. Read `WP_PLAN.md`.
2. Read `TESTING.md`.
3. Inspect the current repository structure.
4. Inspect the existing forward layers and completed backward components.
5. Determine the next incomplete WP3 layer.
6. Make a small implementation plan before editing code.

Implement and validate one backward component at a time. Do not start training,
adversarial attacks, Grad-CAM, or final analysis before WP3 is completed and
validated.

## General Working Rules

Before making code changes:

1. Read `WP_PLAN.md`.
2. Read `TESTING.md`.
3. Inspect the relevant files in `src/`, `tests/`, `configs/`, and `experiments/`.
4. Summarize the current repository state.
5. Propose a small implementation plan.
6. Wait for confirmation if the task is large or ambiguous.

During implementation:

1. Work in small steps.
2. Do not rewrite the whole project.
3. Do not modify unrelated files.
4. Keep the existing project structure and coding style.
5. Prefer minimal, testable changes.
6. If something is unclear, ask before changing large parts.
7. Do not introduce new dependencies unless necessary and clearly justified.

After implementation:

1. Run the relevant tests listed in `TESTING.md`.
2. Report which commands were run.
3. Report which files were modified.
4. Report whether tests passed or failed.
5. Explain remaining risks or TODOs.
6. Do not start the next Work Package before the current one is validated.

## Git Rules

Before starting a new Work Package or larger task:

```bash
git status
```

After a stable Work Package step is completed and validated, suggest a Git commit, but do not commit automatically unless explicitly instructed.

Suggested commit style:

```text
wp2: implement compact cnn forward pass
wp2: add forward pass tests
wp3: implement and test linear backward
docs: add project workflow files
```

## Security and Credentials

* Never write passwords, tokens, API keys, SSH passwords, or private keys into project files.
* Never include credentials in Git commits.
* Never store cluster passwords in scripts, README files, AGENTS.md, WP_PLAN.md, TESTING.md, or prompts.
* Use placeholders such as `<username>`, `<host>`, or `<path>` in documentation.
* SSH passwords must be entered manually by the user when needed.

## Cluster / GPU Rules

The project may later be run on the university GPU cluster.

Current known cluster username:

```text
gpu04
```

Do not store or ask for the password.

For cluster work:

1. Local macOS development is used for small tests, code editing, and debugging.
2. The cluster is used for longer training runs, GPU experiments, and final measurements.
3. Do not run heavy training jobs on the login/head node.
4. Use the head node only for auxiliary tasks such as editing, environment setup, and job submission.
5. Use Slurm for compute jobs.
6. Prefer reproducible scripts under `scripts/`.
7. Prefer saving outputs under `results/`, `logs/`, or `deliverables/`.

## Expected Final Report Format After Each Codex Task

At the end of every task, report:

```text
Status:
Files inspected:
Files modified:
Commands run:
Test results:
Remaining TODOs:
Risks / unclear points:
```
