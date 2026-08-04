# FedPRIME-D2C Agent Entry

This is the lightweight entry point for future Codex sessions.

## Safety

Do not batch-delete files or directories. Never use `del /s`, `rd /s`,
`rmdir /s`, `Remove-Item -Recurse`, or `rm -rf`. Delete at most one explicit
file path after the user has approved deletion.

## Default Read Order

Read only these files when resuming ordinary work:

```text
SESSION_HANDOFF.md
DOCUMENTATION_INDEX_ZH.md
```

Do not automatically read the large historical logs. Open them only when the
current task needs exact old results:

```text
CURRENT_PROJECT_MEMORY.md
PROJECT_STATE.md
TODO_NEXT.md
ARCHITECTURE.md
EXPERIMENT_GUIDE_ZH.md
```

`AGENT.md` is an obsolete early D2C instruction file and is not authoritative.

## Current Experiment

The active paid experiment is the 12-round strict CLE-HFL v2 A/B probe:

```text
control   = AugMix/JSD/DCL + strict AsymHFL-val
candidate = AugMix/JSD/DCL + calibrated PEW/BER+CDep + strict AsymHFL-val
```

Both arms use the same persisted fit/audit split. Local gradients use `fit`,
AsymHFL routing uses client-private `audit`, and final-test labels are
reporting-only. The implementation and RTX 3050 smoke tests are complete; the
formal result is currently pending. Never treat smoke accuracy as evidence.

Entry and guide:

```text
scripts/openi_strict_pew_asymhfl_entry.py --mode=both
STRICT_PEW_ASYMHFL_VAL_OPENI_RUN_ZH.md
```

Do not run 40 rounds unless candidate-minus-control last-five satisfies:

```text
Avg >= +1.5, Worst >= +1.0, WCCA >= 0, CFG <= -1.0
```

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
```

Use `DOCUMENTATION_INDEX_ZH.md` to locate their evidence before discussing
them. Preserve the dirty worktree and never revert unrelated user changes.
