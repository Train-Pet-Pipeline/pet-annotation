## Summary

<!-- 1-3 bullets: what this PR does and why -->

## Plugin-contract impact

<!-- DEV_GUIDE §11.8 retro guardrail — pick exactly one -->

- [ ] **N/A**: This PR does NOT modify plugin contracts (docs / typo / formatting / non-plugin internal refactor).
- [ ] **fixture-real test added/updated**: Production code path verified end-to-end (file existence checks, round-trip data flow, orchestrator dispatch, or shape-specific fixture). MUST be checked for any change to:
  - `src/pet_infra/orchestrator/runner.py` / `hooks.py` / `replay.py`
  - `src/pet_infra/experiment_logger/` (any logger backend)
  - `src/pet_*/plugins/*.py` (any TRAINERS/EVALUATORS/CONVERTERS/DATASETS/OTA registration)
  - new plugin contract producers/consumers

If neither box can be honestly checked, the PR is incomplete. See DEV_GUIDE §11.8 for the F008-F027 retro pattern (11 instances of "shipped + unit-test mock-only + no real path tested" — same root cause every time).

## Test plan

<!-- Bulleted checklist of what was tested + how (commands, real outputs) -->

## Findings cross-reference

<!-- If this PR closes a finding, link it -->

- Closes: pet-infra/docs/ecosystem-validation/2026-04-25-findings/F0XX-...md
- Or: N/A (no finding closure)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
