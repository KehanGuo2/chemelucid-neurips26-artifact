# Audit-18 Partial-Autonomous Diagnostic Leaderboard

MolPuzzle-Gym is a behavior-diagnostic assessment, not a one-number leaderboard. EM is sorted for convenience, but the intended readout is the diagnostic profile across final-answer quality, tool use, and process metrics.

L1 evidence-acquisition breadth, L2 expert-roadmap coverage, L3 observation-conditioned dependency.

**Scope.** Audit-18 only; partial-information access; autonomous prompting; cached diagnostic outputs only. Transfer-24 and hidden-transfer results are intentionally excluded from this main leaderboard.

**Anonymous-release hygiene.** The table reports public model names and aggregate metrics only. It does not include submitter names, institutions, private molecule identities, prompts, traces, or hidden answers.

| Rank | Model | EM ↑ | Tanimoto ↑ | Calls ↓ | Cost ↓ | L1 ↑ | L2 ↑ | L3 ↑ | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | GPT-5.5 | 15/18 (83.3%) | 0.959 | 10.3 | 13.8 | 0.821 | 0.600 | 0.799 | high EM; compare process profile |
| 2 | Gemini 3.1 Pro | 15/18 (83.3%) | 0.905 | 9.6 | 15.3 | 0.754 | 0.643 | 0.687 | high EM; compare process profile |
| 3 | Claude Opus 4.7 | 14/18 (77.8%) | 0.895 | 4.4 | 5.9 | 0.531 | 0.598 | 0.390 | high EM; compare process profile; compact trace |
| 4 | Doubao Seed 2.0 Lite | 13/18 (72.2%) | 0.832 | 6.7 | 8.0 | 0.645 | 0.600 | 0.725 | diagnostic profile |
| 5 | Claude Opus 4.6 | 12/18 (66.7%) | 0.797 | 8.0 | 11.1 | 0.740 | 0.639 | 0.754 | diagnostic profile |
| 6 | DeepSeek V4 Pro | 11/18 (61.1%) | 0.752 | 12.7 | 19.6 | 0.744 | 0.632 | 0.604 | diagnostic profile |
| 7 | Claude Sonnet 4.6 | 11/18 (61.1%) | 0.724 | 11.1 | 18.4 | 0.745 | 0.677 | 0.707 | diagnostic profile |
| 8 | o4-mini | 8/18 (44.4%) | 0.645 | 4.3 | 5.7 | 0.370 | 0.536 | 0.463 | compact trace; 4/18 low-tool run(s) |
| 9 | Qwen3.5 Plus | 7/18 (38.9%) | 0.526 | 13.7 | 20.7 | 0.901 | 0.667 | 0.756 | broad evidence, weaker outcome |
| 10 | GPT-5.4 | 6/18 (33.3%) | 0.500 | 10.6 | 16.4 | 0.756 | 0.654 | 0.658 | broad evidence, weaker outcome |
| 11 | Qwen3.5 Flash | 6/18 (33.3%) | 0.401 | 12.0 | 21.8 | 0.603 | 0.637 | 0.552 | diagnostic profile |
| 12 | DeepSeek V3 | 5/18 (27.8%) | 0.472 | 7.2 | 11.6 | 0.635 | 0.556 | 0.561 | diagnostic profile |
| 13 | GPT-5.2 | 3/18 (16.7%) | 0.353 | 9.0 | 12.8 | 0.651 | 0.590 | 0.663 | 1/18 low-tool run(s) |

## How To Read This Table

- `EM` is exact-match final-answer accuracy over `18` Audit-18 molecules.
- `Tanimoto` is the mean fingerprint similarity of submitted structures to targets.
- `Calls` counts pre-submit non-submit tool calls; lower is not automatically better because sparse evidence can indicate shortcut behavior.
- `Cost` is the cached total tool cost when present in the diagnostic JSON.
- `L1`, `L2`, and `L3` are process diagnostics, not optimization targets in isolation.

## Data Source

- Snapshot CSV: `analysis/broad_pa_audit18_diagnostic_snapshot.csv`
- Snapshot JSON: `analysis/broad_pa_audit18_diagnostic_snapshot.json`
- Source scope: `Broad PA Audit-18 diagnostic snapshot`

Regenerate with:

```bash
python scripts/build_leaderboard.py
```
