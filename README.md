# ChemElucid

> **Anonymous submission to NeurIPS 2026, Evaluations & Datasets track.**

ChemElucid is an interactive evaluation environment for AI agents on molecular structure elucidation from NMR spectroscopy. An agent receives a molecular formula and an NMR spectrum, calls costed chemistry tools under a budget (spectrum query, substructure check, NMR prediction, ...), and submits a SMILES string. Episodes are scored on (a) final-answer accuracy (InChI exact match + Tanimoto), (b) expert-DAG reasoning checkpoint coverage, and (c) tool-call dependency topology.

This artifact pairs:

- **18 process-graded core molecules** with expert reasoning DAGs across CNMR / HNMR / combined modes.
- **48 withheld-probe molecules** sampled from nmrshiftdb with hidden canonical SMILES, used for the outcome-only transfer experiment.

## Quick start

### Reviewer Docker harness (no API keys required)

A containerized reviewer harness ships with this artifact for evaluators
who want to verify the public/private split, grade cached trajectories,
and run the test suite without setting up Python and RDKit on the host.

```bash
docker build -t chemelucid-gym:reviewer .
docker run --rm chemelucid-gym:reviewer                                         # list-molecules
docker run --rm chemelucid-gym:reviewer python -m interactive.experiments.smoke_test
docker run --rm chemelucid-gym:reviewer pytest interactive/tests/ -q
```

Full reviewer runbook: [`docs/REVIEWER_DOCKER_QUICKSTART.md`](docs/REVIEWER_DOCKER_QUICKSTART.md).
Task-bundle format (for authors contributing new molecules):
[`docs/TASK_BUNDLE_FORMAT.md`](docs/TASK_BUNDLE_FORMAT.md).

### Install on host

```bash
pip install -e ".[all]"
```

Optional environment variables for the providers used in the paper:

```bash
export OPENAI_API_KEY=...        # gpt-5.2, o4-mini
export ANTHROPIC_API_KEY=...     # claude-sonnet-4, claude-opus
export OPENROUTER_API_KEY=...    # gemini-2.5-pro, deepseek-r1, qwen3-235b
export DEEPINFRA_API_KEY=...     # llama-4-maverick
```

### Run one withheld-probe episode

```bash
python -m interactive.cli run-probe \
  --probe-id probe_001 \
  --model gpt-5.2 \
  --info-mode partial \
  --budget 5
```

This loads `data/withheld_probe/spectra/probe_001_CNMR.json` (agent surface only — the SMILES is never visible to the agent), runs the multi-turn tool loop, and writes a diagnostic JSON to `interactive_results/withheld_probe/`. The hidden ground-truth SMILES is read by the grader from `data/withheld_probe/probe_48_manifest.csv` and is never returned to the agent loop.

### Reproduce the 4-model × 8-probe × 2-info-mode batch

```bash
python scripts/run_probe_batch.py --dry-run                    # enumerate cells
python scripts/run_probe_batch.py --models gpt-5.2 o4-mini     # run the 32 new cells
python scripts/run_probe_batch.py --models all                 # 4-model crossed
                                                                # (32 reusable cells skipped)
```

Pre-existing `claude-sonnet-4` and `claude-opus` episodes for the eight probes are mirrored under `interactive/results/canary_probe/` and reused via the `legacy_internal_id` mapping in the probe manifest. The runner detects them and only enqueues genuinely new (model, probe, info_mode) cells.

### List available molecules and models

```bash
python -m interactive.cli list-molecules           # 18 core
python -c "from interactive.config import MODELS; print(list(MODELS))"
```

## Repository layout

```
.
├── README.md                     # this file
├── croissant.json                # MLCommons Croissant 1.1 dataset metadata
├── LICENSE                       # MIT
├── pyproject.toml                # Python package metadata
├── interactive/                  # core package: env, agent loop, grader, metrics
│   ├── environment.py            # ChemElucidEnv: orchestrates one episode
│   ├── tool_server.py            # 14 chemistry tools with cost tracking
│   ├── agent_loop.py             # multi-turn LLM dispatch (4 providers)
│   ├── dag_grader.py             # post-hoc reasoning-DAG grader (L2 coverage)
│   ├── probe_grader.py           # outcome-only grader for withheld probes
│   ├── data_loader.py            # public/private data loading
│   ├── metrics/                  # L1 (information acquisition), L3 (causal topology)
│   ├── cli.py                    # entry point: run / run-all / run-probe / analyze
│   └── tests/                    # 133 core + 33 metric tests
├── chem_defense/utils/           # shared helpers: SMILES, formula, HNMR parsing
├── data/
│   ├── C_NMR/, H_NMR/, combo/    # core-18 ground-truth files (back-compat layout)
│   ├── withheld_probe/
│   │   ├── probe_48_manifest.csv # hidden SMILES + provenance (grader-private)
│   │   ├── spectra/              # 48 agent-facing CNMR JSON files
│   │   └── README.md             # provenance + sampling rule
│   └── dynamic/
│       └── nmrshiftdb_scale150.csv  # 150-row source pool (provenance only)
├── data_split/
│   ├── public_task_data/         # agent-facing 18 core molecules (CNMR/HNMR/COMBO)
│   └── private_grader_data/      # hidden grader DAGs + canonical SMILES
├── scripts/
│   ├── run_probe_batch.py        # 4-model × 8-probe runner (T3 in the paper)
│   ├── build_withheld_probe.py   # rebuild the 48-mol artifact from data_split
│   ├── audit_leakage.py          # repo-history leakage audit
│   ├── fix_cnmr_ground_truth.py  # RDKit-based DAG motif validator
│   ├── offline_submit_regrade.py # re-grade an external trajectory
│   ├── run_dummy_baselines.py    # random-tool baselines for L3 calibration
│   └── run_always_chain_dummy.py # always-chain baseline for L3 construct validity
├── interactive/results/
│   ├── canary_probe/             # 32 pre-computed Anthropic probe episodes
│   └── dummy_baselines/          # always-chain / random-ppm / random-tool baselines
└── tests/                        # see `interactive/tests/`
```

## Reviewer-private files

The following files contain hidden ground-truth answers and **must not be served to evaluation agents**. They are visible to reviewers because grading requires them; the code paths under `interactive/` enforce that they are only loaded by the grader:

- `data/withheld_probe/probe_48_manifest.csv` — hidden canonical SMILES for the 48 withheld-probe molecules.
- `data_split/private_grader_data/{molecule_id}/grader_*.json` — hidden DAG + SMILES for the 18 core molecules.

The agent-facing surface is:

- `data/withheld_probe/spectra/probe_*_CNMR.json` — formula + peaks + solvent + frequency only.
- `data_split/public_task_data/{molecule_id}/task_*.json` — same fields for the core-18.

A leakage audit (`scripts/audit_leakage.py`) verifies that none of the agent-facing files contain SMILES, InChI, IUPAC names, or source identifiers.

## Reproducibility

- **Dataset metadata.** A Croissant 1.1 file at `croissant.json` describes both record sets (core-18 process-graded, withheld-probe-48 outcome-only), including the NeurIPS 2026 minimal Responsible AI fields. Validate with:
  ```bash
  pip install mlcroissant
  mlcroissant validate --jsonld croissant.json
  ```
- **Tests.** `pytest interactive/tests/` runs 359 tests in under a minute (356 pass, 3 skip when optional `interactive/results/v3_multimodel` is absent). None require API keys.
- **Smoke test (no API).** `python -m interactive.experiments.smoke_test` exercises the env end-to-end with a dummy provider.
- **Pre-computed results.** Thirty-two probe episodes for `claude-sonnet-4` and `claude-opus` are included under `interactive/results/canary_probe/` so the four-model probe analysis can be reproduced without re-running those Anthropic agents. The `legacy_internal_id` column of the probe manifest documents the one-to-one mapping `probe_NNN ↔ canary_NNN` (canary is the original internal alias of the curation pipeline). Note: the `molecule_id` field inside each pre-computed diagnostic JSON still reads `canary_NNN`; new diagnostics produced by `scripts/run_probe_batch.py` use `probe_NNN`. Aggregation code joining the two batches must normalize the molecule ID via the manifest before merging.
- **Random seeds.** Withheld-probe sampling uses `seed = 42`; bootstrap CIs use `B = 10,000`.

## License

The code is released under MIT (see `LICENSE`). Source data licenses:

- nmrshiftdb-derived peak lists (used for the 48 withheld-probe molecules): CC-BY-SA 3.0.
- Core-18 molecules and reasoning DAGs: derived from open-access chemistry references; redistributed under MIT.

## Citation

```bibtex
@misc{chemelucid_2026,
  title  = {ChemElucid: Diagnosing Scientific Agents in Molecular Structure Elucidation},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {NeurIPS 2026 Evaluations and Datasets track submission.}
}
```
