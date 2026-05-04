# Reviewer Docker quickstart

This is the no-API verification path for ChemElucid-Gym, suitable for E&D
reviewers who want to confirm the artifact is reproducible without setting up
a full Python / RDKit toolchain on the host.

The container ships only the minimum needed for verification: it intentionally
does **not** bundle provider API keys, a web UI, a leaderboard, or live LLM
inference. Live model episodes are reproducible from the dev tree on a host
that already has provider keys; the container exists for review verification.

## 1. Build

```bash
docker build -t chemelucid-gym:reviewer .
```

Expected build time: ~3–5 minutes on a recent laptop. The image installs
Python 3.11, RDKit (via the official `rdkit` PyPI wheel), and the project's
no-API dependencies. `py2opsin` (and its Java dependency) is intentionally
omitted; the verification path does not exercise it.

A Compose file is provided as well:

```bash
docker compose build
```

## 2. No-API smoke test

The container's default command lists the 18 process-graded core molecules,
which exercises the package import path:

```bash
docker run --rm chemelucid-gym:reviewer
```

Expected output: 18 lines (`2_4_dimethyl_aniline`, `acetaminophen`, …) printed
to stdout. If this fails, the install is broken; nothing else will work.

The full pipeline smoke test runs an end-to-end episode with mocked LLM calls:

```bash
docker run --rm chemelucid-gym:reviewer \
    python -m interactive.experiments.smoke_test
```

Expected runtime: under 1 second on this host. Expected output: pipeline
checks for tool server, blackboard, DAG grader, environment.

## 3. Grade a cached trajectory

The hidden grader can be exercised on a cached non-error trajectory written
by one of the dummy baselines. No API key is needed:

```bash
docker run --rm \
    -v "$(pwd)/tmp:/tmp" \
    chemelucid-gym:reviewer \
    python -m interactive.grade \
        --episode-log interactive/results/dummy_baselines/random_tool_roulette/partial_autonomous/Benzil.json \
        --manifest core \
        --out /tmp/metrics.json
```

The grader prints `wrote /tmp/metrics.json` and exits 0. Inspect
`./tmp/metrics.json`: it must contain `coverage`, `outcome`, `failure_hist`
fields, and **must not** contain any of `smiles`, `inchi`, `gt_smiles`,
`canonical_smiles`. The `test_grade_cached_trajectory_emits_no_private_fields`
test in `interactive/tests/test_task_bundle.py` verifies this invariant.

## 4. Ingest and validate a task bundle

The reviewer can construct an example task bundle from CLI args (no API):

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

Both must succeed with the not-comparable note ("custom rubric outputs are
local extension diagnostics and are NOT comparable to official scores").

## 5. Run the full test suite

```bash
docker run --rm chemelucid-gym:reviewer pytest interactive/tests/ -q
```

Expected: hundreds of tests pass; the harness-specific tests live at
`interactive/tests/test_task_bundle.py` (10 cases).

## 6. Where data lives

| Path | Visibility | Purpose |
|---|---|---|
| `data_split/public_task_data/<mol_id>/task_*.json` | **agent-visible** | Formula, peaks, solvent, frequency. No SMILES/InChI/CAS. |
| `data/withheld_probe/spectra/probe_*_CNMR.json` | **agent-visible** | 48 hidden-answer probe spectra. |
| `data_split/private_grader_data/<mol_id>/grader_*.json` | grader-only | Ground-truth SMILES + expert DAG nodes. Read by `interactive/grade.py`. |
| `data/withheld_probe/probe_48_manifest.csv` | grader-only | Hidden SMILES + provenance for the probes. Reviewer-private but shipped because reviewers must be able to grade without a personal request. |
| `manifests/*.jsonl` | grader-only | Frozen scope manifests (`core18_full_matrix.jsonl`, `private_canary_manifest.jsonl`). |

Inside the container, the public surface and the private grader surface are
on the same disk; the leakage CI test
(`interactive/tests/test_no_leakage.py`) and the per-bundle `validate`
command enforce that the agent path never reads the private surface.

## 7. What agents must NOT see

- canonical SMILES of any task molecule
- InChI / InChI key of any task molecule
- CAS numbers, IUPAC names, or chemical names of any task molecule
- the source database identifier (`nmrshiftdb` row id, DOI of the published
  spectrum) for the withheld probes
- the legacy internal alias (`canary_NNN`) of any probe (a private mapping
  in the manifest)

The `task_bundle validate` command and the `test_task_bundle.py` regression
tests enforce these invariants by RDKit canonicalization (any token that
canonicalizes to the hidden answer is a leak) plus a keyed-string scan
(`"smiles":`, `"inchi":`, `"cas":`, …).

## 8. Expected runtime

| Command | Approximate runtime |
|---|---|
| `docker build` | 3–5 min |
| `python -m interactive.cli list-molecules` | < 1 s |
| `python -m interactive.experiments.smoke_test` | < 1 s |
| `python -m interactive.grade --episode-log <cached> --manifest core --out /tmp/metrics.json` | < 1 s |
| `python -m interactive.task_bundle ingest ...` | < 1 s |
| `python -m interactive.task_bundle validate <bundle>` | < 1 s |
| `pytest interactive/tests/ -q` | ~1 min |

## 9. Known limitations

- **No live LLM episodes inside the container.** Provider keys are not passed
  in, and the live `run` / `run-all` / `run-probe` paths intentionally fail
  fast if `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` are unset.
- **Java / `py2opsin` is not installed.** The IUPAC→SMILES code path in
  `chem_defense/utils/smiles_utils.py:resolve_molecule_to_smiles` requires a
  JRE and is not on the verification path. Authors who want to ingest a
  molecule from an IUPAC name should use a host install with `pip install
  py2opsin` and a JRE.
- **Multi-architecture wheels.** The image builds on linux/amd64 by default;
  on Apple Silicon Docker Desktop it runs under emulation. `rdkit` ships
  arm64 wheels since 2024, so a native arm64 build is also feasible.
- **Compose volume mount.** The provided `compose.yaml` bind-mounts the
  working tree so reviewers can edit code and re-run without a rebuild;
  for a strictly frozen image, comment out the `volumes:` block.

## 10. What official-vs-custom means

The harness lets authors and reviewers attach **custom DAGs** and **custom
rubrics** for local extension. **These are not part of the official
ChemElucid-Gym score.** The official score is computed only by the frozen
grader (`interactive/grade.py`) against the frozen manifests in
`manifests/*.jsonl` and `data_split/private_grader_data/`. Custom-rubric
outputs are local diagnostics and must be reported under their own header in
any analysis output. See `docs/TASK_BUNDLE_FORMAT.md`.
