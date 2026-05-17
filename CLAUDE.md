# CLAUDE.md — StereoComplex Phase 8 Status

## Current state (2026-05-17)

All infrastructure code is implemented. 109 tests pass (37 deselected).

| Phase | Status | Notes |
|---|---|---|
| Phase 1 — Simulator densification | ✅ DONE | 8 tests, SamplingDiagnostics, rejection sampling |
| Phase 2 — Pipeline A on CMO | ⚠️ PARTIAL | Code exists. Missing 2 tests: `test_direct_fit_recovers_brown_on_brown_oracle`, `test_direct_fit_converges_on_cmo_oracle_with_three_candidates` |
| Phase 3 — Zernike from observations | ✅ DONE | 3 tests (2 pass, 1 skipped — CMO oracle requires rayfield_cache) |
| Phase 4 — Schur diagnostics | ✅ DONE | 5 tests, Schur complement, condition numbers |
| Phase 5 — Documentation | ⚠️ PARTIAL | DIRECT_VS_RAYFIELD_INVERSION.md exists (212 lines). Missing: CHANGELOG v0.5.3, git tag, notebook 08 FAST mode validation |

## Remaining work

1. **Phase 2: 2 missing tests** — `test_direct_fit_recovers_brown_on_brown_oracle`, `test_direct_fit_converges_on_cmo_oracle_with_three_candidates`
2. **Phase 5: CHANGELOG + tag v0.5.3 + notebook validation**
3. **Run notebook 08 in FAST mode (≤60s)**

## Key session

Always use `--resume c7c56802-1828-4013-a380-be256e554caa` for deep-claude on this project.
