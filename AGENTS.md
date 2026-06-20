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
* WP3 is completed.
* Manual layer backward passes, `CompactCNN.backward`,
  `SoftmaxCrossEntropyLoss.backward`, and loss-to-model integration are
  implemented and tested.
* WP4 is completed.
* Numerical gradient checks for `Linear`, `Conv2D`, and
  `SoftmaxCrossEntropyLoss` pass with a relative-error threshold of `1e-4`.
* The `CompactCNN` input-gradient pipeline sanity check passes.
* WP5 has not been started yet.
* The next target is WP5: Baseline training and clean evaluation.
* Training, adversarial attacks, Grad-CAM, and final analysis have not been started.

## Current Development Priority

The immediate priority is WP5 baseline training and clean evaluation.

Before starting WP5, Codex must:

1. Read `WP_PLAN.md`.
2. Read `TESTING.md`.
3. Inspect the current repository structure.
4. Inspect the completed data pipeline, model, loss, backward pipeline, and
   gradient-check tests.
5. Confirm the WP5 validation criteria, dependencies, optimizer choice, and
   training configuration.
6. Make a small implementation plan before editing code.

Implement and validate one training component at a time. Start with the
optimizer and parameter-update API. Do not start adversarial attacks, Grad-CAM,
or final analysis before WP5 is completed and validated.

## General Working Rules

Before starting a new Work Package:

1. Review the project requirements, the source Work Package plan, the current
   repository state, and `TESTING.md`.
2. Complete or update the Work Package's `Relevant folders/files`,
   `Validation`, and `Dependencies` entries in `WP_PLAN.md` before
   implementation begins.
3. Base those entries on documented requirements and existing repository
   evidence. If an item is still unclear, keep it as `TBD` instead of guessing.

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

After completing any implementation, test, documentation update, or refactor
task, do not automatically run `git add`, `git commit`, or `git push` unless
the user explicitly requests those operations.

At the end of each task, report:

1. files inspected,
2. files modified,
3. commands run,
4. test results,
5. scope confirmation,
6. current Git status.

If the working tree contains only the expected changes and the relevant tests
pass, ask whether the user wants to proceed with Git operations. Provide the
exact recommended commands and prefer explicit file paths instead of
`git add .`.

Suggested Git workflow:

```bash
git status --short
git diff --check
git add <modified-file-1> <modified-file-2>
git commit -m "<clear commit message>"
git push
git status -sb
```

After a stable Work Package step is completed and validated, suggest a Git
commit, but do not stage, commit, or push automatically unless explicitly
instructed.

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
