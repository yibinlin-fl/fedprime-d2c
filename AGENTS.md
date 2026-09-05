# FedPRIME-D2C Agent Entry

This is the lightweight entry point for future Codex sessions.

## Safety

Do not batch-delete files or directories. Never use `del /s`, `rd /s`,
`rmdir /s`, `Remove-Item -Recurse`, or `rm -rf`. Delete at most one explicit
file path after the user has approved deletion.

## Default Read Order

Read only these files when resuming ordinary work:

```text
docs/handoffs/latest.md
docs/README_ZH.md
```

Do not automatically read the large historical logs. Open them only when the
current task needs exact old results:

```text
docs/project/CURRENT_PROJECT_MEMORY.md
docs/project/PROJECT_STATE.md
docs/project/TODO_NEXT.md
docs/project/ARCHITECTURE.md
docs/experiments/guides/EXPERIMENT_GUIDE_ZH.md
```

`docs/archive/legacy/AGENT.md` is an obsolete early D2C instruction file and is
not authoritative.

## Current Formal Result

The strict 12-round CLE-HFL v2 A/B completed for matched training seeds 0/1/2
on 2026-08-04:

```text
control   = AugMix/JSD/DCL + strict AsymHFL-val
candidate = AugMix/JSD/DCL + calibrated PEW/BER+CDep + strict AsymHFL-val
```

All seeds kept the CLE scenario and persisted fit/audit split fixed at
`seed0_split0`; only matched training randomness changed. Local gradients used
`fit`, AsymHFL routing used client-private `audit`, and final-test labels were
reporting-only. Independently recomputed last-five deltas were:

```text
seed   Avg       Worst     WCCA      CFG
0      +3.9377   +3.9040   +5.0500   -6.3200
1      +4.7977   +3.8893   +4.3500   -8.3000
2      +5.0287   +4.8573   +7.2500   -5.5250
mean   +4.5880   +4.2169   +5.5500   -6.7150
```

All three seeds passed the original full gate, and all pre-registered
multi-seed gates passed. Verdict: `GO` for training-seed stability on the fixed
CLE scenario. This is not yet a cross-scenario claim.
Never treat smoke accuracy as evidence.

Entry and guide:

```text
scripts/openi_strict_pew_asymhfl_entry.py --mode=both
docs/experiments/archive/STRICT_PEW_ASYMHFL_VAL_OPENI_RUN_ZH.md
docs/experiments/archive/STRICT_PEW_ASYMHFL_VAL_MULTISEED_OPENI_RUN_ZH.md
```

All three seeds passed the frozen single-seed gates:

```text
Avg >= +1.5, Worst >= +1.0, WCCA >= 0, CFG <= -1.0
```

The 40-round training-seed 0 durability result also passed all eight frozen
last-10/last-5 gates. Its last-10 delta was:

```text
Avg +4.9292, Worst +3.2987, WCCA +9.8750, CFG -5.4700
```

The user has explicitly requested matched 40-round repeats for training seeds
1/2. Freeze PEW/BER/CDep parameters and the CLE `seed0_split0` scenario. The
only formal entry is:

```text
scripts/openi_strict_pew_asymhfl_40round_entry.py --mode=both --train_seed=1/2
scripts/openi_strict_pew_asymhfl_40round_entry.py --mode=both --train_seed=all
docs/experiments/current/STRICT_PEW_ASYMHFL_VAL_40ROUND_OPENI_RUN_ZH.md
```

`train_seed=all` means pending seeds `[1, 2]` only. It packages and uploads
seed1 before starting seed2 so a later failure does not discard the completed
seed1 artifact.

Its pre-registered decision uses the original thresholds on last-10 means plus
positive-direction last-5 anti-collapse gates. Do not change these windows or
gates after the result arrives.

## Frozen Negative Results

Do not revive these by tuning only ranks, thresholds, or loss weights:

```text
D2C / Oracle D2C
FedPRIME-PAIR / CPAD
PRAC-HFL communication
FedCARA v1 communication
FedCLEAR v0.1 / PCCD
EBST / EBST-v2
FedFalsify v0.2/v0.3
FedCIS-v0
handcrafted taxonomy-free continuous witness
class-conditional counterfactual regret (C3R)
```

Use `docs/README_ZH.md` to locate their evidence before discussing them.
Active implementations for several frozen methods were removed on 2026-08-09;
use `docs/archive/methods/NEGATIVE_CODE_REMOVAL_INDEX_ZH.md`, archived evidence,
deliverables, and Git history instead of recreating them in the current runner.
Preserve the dirty worktree and never revert unrelated user changes.

## Research Workflow

Use this minimal sequence for ordinary tasks:

```text
read current handoff
-> read the documentation index
-> locate one relevant config/script/module
-> make the smallest scoped change
-> run focused verification
-> update the handoff only when project state actually changed
```

Do not recursively scan the repository, read all source files, or load the
large project logs merely to understand the project. Start with at most five
directly relevant files. If an `rg` search returns more than 30 useful hits,
narrow the query before reading files. Read long logs only around the relevant
heading or result.

When documentation and implementation disagree, verify the relevant code,
config, and current Git diff, then update the current documentation. Do not
revalidate unrelated historical conclusions.

## Documentation Placement

The repository root contains only stable entry documents:

```text
README.md
AGENTS.md
```

Place new documentation by purpose:

```text
docs/README_ZH.md              documentation map and source-of-truth order
docs/handoffs/latest.md        compact current objective, status, next action
docs/project/                  architecture, long-term memory, state, TODO, practices
docs/experiments/current/      guides for experiments that are running or awaiting results
docs/experiments/guides/       reusable benchmark and platform instructions
docs/experiments/archive/      completed or superseded experiment guides
docs/research/status/          current research synthesis and dated progress reports
docs/research/baselines/       baseline implementation readings
docs/archive/methods/          rejected, frozen, or superseded method evidence
docs/archive/legacy/           obsolete instructions retained only for provenance
```

Do not add new Markdown files to the repository root. Do not place raw logs,
checkpoints, datasets, archives, or generated figures under `docs/`.

Use these artifact locations:

```text
outputs/       raw experiment outputs and downloaded archives; normally untracked
deliverables/  parsed tables, plots, and result reports
local_runs/    local datasets, checkpoints, caches, and temporary runs; untracked
```

Before moving a document, find and update its live references. Historical
result snapshots under `outputs/` and `deliverables/` should not be rewritten
just to modernize paths. Move files explicitly; never use recursive bulk
delete or cleanup commands.

## Documentation Maintenance

- Keep `docs/handoffs/latest.md` short and factual: current objective, running
  experiment, verified status, pending decision, next entry point, and frozen
  constraints. Do not paste chat history or long logs.
- Update `docs/README_ZH.md` whenever a document is created, moved, archived,
  or promoted to the current experiment.
- Append exact historical results to `docs/project/CURRENT_PROJECT_MEMORY.md`
  only when they are needed for future scientific decisions.
- Keep `docs/project/PROJECT_STATE.md` for implementation state and
  `docs/project/TODO_NEXT.md` for actionable work; do not duplicate the full
  handoff in both.
- Name dated reports with `YYYY_MM_DD`. Name experiment guides after the method
  and benchmark, not after a chat session.
- Archive a failed method with its evidence; do not erase negative results or
  silently turn them back into active TODOs.

## Experiment Discipline

- Smoke tests validate execution only. Never report smoke accuracy as formal
  evidence.
- Do not start paid, long-running, multi-seed, or 40-round experiments unless
  the user requests them or the documented promotion gate has passed.
- For A/B attribution, keep seeds, initialization, partitions, fit/audit/test
  roles, public batches, evaluation cadence, and reporting metrics matched.
- Final-test labels are reporting-only unless a protocol explicitly states
  otherwise. Never use them for routing, selection, early stopping, or tuning.
- Record the exact config, entry command, archive name, seed, rounds, and gate
  decision when a formal result arrives.
- Prefer focused unit tests, one-round smoke tests, and analyzer dry-runs. Do
  not default to full test suites or long training jobs.

## OpenI User Handoff

After an OpenI-capable implementation is complete and locally verified, always
give the user one self-contained launch card before asking them to create a
task. It must state:

```text
git commit and whether it has been pushed
whether an existing dataset is reused or a new upload is required
dataset display name
exact absolute local upload-file path, byte size, and SHA256
OpenI startup file
every parameter name and exact value (including boolean values)
recommended GPU type/count and whether the run is smoke/benchmark/formal
expected runtime or a benchmark-first requirement
expected output archive name
exact absolute local folder where the downloaded result must be placed
metrics, frozen gates, and what the run is allowed to prove
```

Explicitly warn about obsolete input packages when more than one similarly
named archive exists. Do not say an OpenI run is ready until the entry, input
integrity checks, and a focused smoke have passed. A benchmark is never method
evidence, and Formal remains subject to the experiment-approval rules above.

## Working Tree And Git

- Inspect the relevant Git status before editing. Existing dirty files belong
  to the user unless the current task clearly owns them.
- After completing a coherent implementation or research-stage objective,
  create a scoped local commit without waiting for a separate user reminder,
  unless the user has requested no commit. Commit only the files owned by that
  completed objective; keep unrelated dirty changes and generated artifacts
  out of the commit. Do not push unless the user explicitly requests it.
- Never overwrite, revert, format, stage, or commit unrelated changes.
- Keep datasets, checkpoints, raw outputs, and local test artifacts untracked.
- Outside the standing stage-completion rule above, commit only when the user
  explicitly asks. A commit must contain only the current task's intended
  files, and pushing always requires an explicit user request.
