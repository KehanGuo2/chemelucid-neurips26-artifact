# Reviewer Docker Quickstart

This is the no-API verification path for the anonymous ChemElucid /
MolPuzzle-Gym artifact. It lets reviewers inspect the public benchmark
interface, public task bundles, episode logging, scoring definitions, and
sanitized toy HRE alignment without installing RDKit on the host.

The container intentionally does **not** include provider API keys, full private
HRE templates, private grader labels, withheld-probe annotations, or complete
expert-curated reasoning graphs.

## 1. Build

```bash
docker build -t chemelucid-gym:reviewer .
```

Expected build time: about 3-5 minutes on a recent laptop. The image installs
Python 3.11, RDKit, and no-API dependencies.

## 2. Review-Mode Smoke Test

The default command lists public review tasks:

```bash
docker run --rm chemelucid-gym:reviewer
```

The review-mode smoke test exercises public task loading, tool logging, L1/L3
diagnostics, and the sanitized toy L2 path:

```bash
docker run --rm chemelucid-gym:reviewer \
    python -m interactive.experiments.smoke_test
```

Expected result: `REVIEW-MODE SMOKE TEST PASSED`.

## 3. Sanitized Toy HRE Scoring

```bash
docker run --rm chemelucid-gym:reviewer \
    python examples/hre_toy/run_toy_scoring.py
```

This prints a small JSON report containing L1, toy L2 coverage, and L3 fields.
It demonstrates trajectory alignment and scoring invocation without exposing
the private grader layer.

## 4. Ingest And Validate A Task Bundle

```bash
docker run --rm \
    -v "$(pwd)/tmp:/tmp" \
    chemelucid-gym:reviewer \
    python -m interactive.task_bundle ingest \
        --smiles "CC(=O)Nc1ccc(O)cc1" \
        --formula "C8H9NO2" \
        --cnmr examples/acetaminophen_cnmr_peaks.csv \
        --source example --license example-only \
        --out /tmp/chemelucid_task_candidate

docker run --rm \
    -v "$(pwd)/tmp:/tmp" \
    chemelucid-gym:reviewer \
    python -m interactive.task_bundle validate /tmp/chemelucid_task_candidate
```

Expected: `OK: bundle passes all hard checks.` with `issues: 0`.

Custom DAG and rubric YAMLs ship as examples:

```bash
docker run --rm chemelucid-gym:reviewer \
    python -m interactive.task_bundle validate-dag examples/custom_dag.yaml

docker run --rm chemelucid-gym:reviewer \
    python -m interactive.task_bundle validate-rubric examples/custom_rubric.yaml
```

Both must succeed with the not-comparable note: custom rubric outputs are local
extension diagnostics and are not comparable to official scores.

## 5. Run The Test Suite

```bash
docker run --rm chemelucid-gym:reviewer pytest interactive/tests/ -q
```

Tests that require the full private HRE assets skip in the anonymous review
package. The public task, leakage, task-bundle, toy scoring, and metric tests
run without API keys.

## 6. Where Data Lives

| Path | Visibility | Purpose |
|---|---|---|
| `data_split/public_task_data/<mol_id>/task_*.json` | agent-visible | Formula, peaks, solvent, frequency, and prompt. No SMILES/InChI/CAS/source IDs. |
| `examples/hre_toy/` | public/sanitized | Toy HRE schema, toy trajectory, and L1/L2/L3 scoring invocation. |
| `data_split/private_grader_data/` | omitted placeholder | Expected private grader location for controlled releases. |
| `data/C_NMR/`, `data/H_NMR/`, `data/combo/` | omitted placeholder | Legacy private HRE locations, not shipped in review package. |
| `data/withheld_probe/` | omitted placeholder | Withheld-probe spectra/labels are not shipped in review package. |

The public surface and private-grader surface are deliberately separated. In
review mode the private surface contains only placeholder READMEs.

## 7. What Agents Must Not See

- canonical SMILES of benchmark molecules
- InChI or InChI key of benchmark molecules
- CAS numbers, IUPAC names, or chemical names of benchmark molecules
- withheld-probe source identifiers or hidden labels
- complete chemistry-specific HRE templates or expert-curated reasoning graphs

The `task_bundle validate` command and leakage tests enforce these invariants
for agent-facing files.

## 8. Expected Runtime

| Command | Approximate runtime |
|---|---|
| `docker build` | 3-5 min |
| `python -m interactive.cli list-molecules` | < 1 s |
| `python -m interactive.experiments.smoke_test` | < 1 s |
| `python examples/hre_toy/run_toy_scoring.py` | < 1 s |
| `python -m interactive.task_bundle ingest ...` | < 1 s |
| `python -m interactive.task_bundle validate <bundle>` | < 1 s |
| `pytest interactive/tests/ -q` | about 1 min |

## 9. Known Limitations

- **Private HRE assets omitted.** Exact private-grader evaluation used for
  hidden probes requires controlled-release assets that are intentionally
  withheld during double-blind review.
- **No live LLM episodes inside the container by default.** Provider keys are
  not passed in. Live runs require host-provided keys.
- **Java / `py2opsin` is not installed.** IUPAC-to-SMILES resolution is outside
  the no-API review path.
- **Multi-architecture wheels.** The image builds on linux/amd64 by default;
  Apple Silicon Docker Desktop may use emulation.

## 10. Official Versus Custom

The harness lets authors and reviewers attach custom DAGs and custom rubrics
for local extension. These are not official benchmark scores. In the anonymous
review package, official hidden-grader scoring is represented by scoring code
and sanitized examples; exact private-grader evaluation requires the withheld
private HRE assets that will be released in controlled form after public paper
release.
