# ChemElucid / MolPuzzle-Gym

> **Anonymous submission to NeurIPS 2026, Evaluations & Datasets track.**

ChemElucid / MolPuzzle-Gym is an interactive evaluation environment for AI
agents on molecular structure elucidation from NMR spectroscopy. An agent
receives a molecular formula and an NMR spectrum, calls costed chemistry tools
under a budget, and submits a SMILES string. The benchmark reports a diagnostic
profile: final-answer quality when private grader assets are available, L1
evidence-acquisition breadth, L2 expert-roadmap coverage, and L3
observation-conditioned dependency.

This anonymous review package focuses on reproducibility of the public
benchmark interface, task bundle format, episode logging schema, scoring
definitions, cached aggregate tables, and sanitized examples.

## HRE Asset Release Boundary

This anonymous review package releases the public benchmark interface, task
bundle format, episode logging schema, scoring scripts, and reproducibility
harness needed to inspect and run the benchmark.

The full private HRE assets are intentionally not included in this review
package. This includes complete chemistry-specific HRE templates, private
grader labels, withheld probe annotations, and expert-curated reasoning graphs.
These assets are part of the private grader layer and are withheld during
double-blind review to preserve hidden-evaluation integrity and avoid premature
public release of follow-up chemistry-specific assets.

To make the evaluation logic inspectable, this repository includes sanitized
examples / toy HRE templates that demonstrate the data format and scoring
pipeline without exposing the full private grader layer.

The full HRE assets will be released in a controlled form after the paper is
publicly released, subject to maintaining the integrity of hidden evaluation
and follow-up domain studies.

**Artifact note.** This review package supports inspection and reproduction of
the benchmark interface and scoring pipeline on public/sanitized assets. Exact
private-grader evaluation used for hidden probes requires private HRE assets
that are intentionally withheld during double-blind review and will be released
in controlled form after public release.

## Leaderboard

A lightweight static [Audit-18 Partial-Autonomous Diagnostic Leaderboard](leaderboard/audit18_pa.md)
reports cached MolPuzzle-Gym broad PA results. EM is sorted for convenience,
but MolPuzzle-Gym reports a diagnostic profile rather than a single target
score: L1 evidence-acquisition breadth, L2 expert-roadmap coverage, and L3
observation-conditioned dependency. Transfer-24 and hidden-transfer results are
intentionally kept out of the main leaderboard.

## Quick Start

### Reviewer Docker harness, no API keys

```bash
docker build -t chemelucid-gym:reviewer .
docker run --rm chemelucid-gym:reviewer
docker run --rm chemelucid-gym:reviewer python -m interactive.experiments.smoke_test
docker run --rm chemelucid-gym:reviewer python examples/hre_toy/run_toy_scoring.py
docker run --rm chemelucid-gym:reviewer pytest interactive/tests/ -q
```

Full reviewer runbook: [`docs/REVIEWER_DOCKER_QUICKSTART.md`](docs/REVIEWER_DOCKER_QUICKSTART.md).
Task-bundle format: [`docs/TASK_BUNDLE_FORMAT.md`](docs/TASK_BUNDLE_FORMAT.md).

### Install on host

```bash
pip install -e ".[all]"
python -m interactive.cli list-molecules
python -m interactive.experiments.smoke_test
python examples/hre_toy/run_toy_scoring.py
```

Optional provider keys are only needed for live LLM episodes:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPENROUTER_API_KEY=...
export DEEPINFRA_API_KEY=...
```

## Repository Layout

```text
.
├── README.md
├── croissant.json
├── leaderboard/                 # static anonymous diagnostic leaderboard
├── interactive/                  # env, agent loop, tools, graders, metrics, tests
│   ├── environment.py
│   ├── tool_server.py
│   ├── dag_grader.py
│   ├── grade.py                 # private-grader CLI; soft-fails in review mode
│   ├── metrics/                 # L1 and L3 scoring definitions
│   └── tests/
├── data_split/
│   ├── public_task_data/         # agent-visible public task bundles
│   └── private_grader_data/      # placeholder only in anonymous review package
├── data/
│   ├── C_NMR/, H_NMR/, combo/    # placeholder legacy private-HRE locations
│   └── withheld_probe/           # placeholder withheld-probe location
├── examples/
│   ├── hre_toy/                  # sanitized HRE schema and scoring example
│   ├── custom_dag.yaml
│   └── custom_rubric.yaml
└── docs/
```

## Review-Mode Behavior

- Public task bundles under `data_split/public_task_data/` contain formula,
  spectra, solvent/frequency metadata, and prompts only.
- Full private HRE assets, private grader labels, withheld probe annotations,
  and expert-curated reasoning graphs are absent by design.
- Runtime code uses public task bundles by default. When private assets are
  missing, exact private-grader metrics are reported as unavailable rather than
  fabricated.
- `examples/hre_toy/` provides the inspectable toy HRE alignment path:
  ```bash
  python examples/hre_toy/run_toy_scoring.py
  ```

## Reproducibility

- **Public interface smoke test.**
  ```bash
  python -m interactive.experiments.smoke_test
  ```
- **Toy HRE scoring.**
  ```bash
  python examples/hre_toy/run_toy_scoring.py
  ```
- **Task-bundle validation.**
  ```bash
  python -m interactive.task_bundle validate-dag examples/custom_dag.yaml
  python -m interactive.task_bundle validate-rubric examples/custom_rubric.yaml
  ```
- **Dataset metadata.**
  ```bash
  pip install mlcroissant
  mlcroissant validate --jsonld croissant.json
  ```

## License

The code is released under MIT (see `LICENSE`). Public/sanitized review assets
are included for anonymous artifact inspection. Full private HRE assets and
withheld-probe labels are not included in this anonymous package.

## Citation

```bibtex
@misc{chemelucid_2026,
  title  = {ChemElucid: Diagnosing Scientific Agents in Molecular Structure Elucidation},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {NeurIPS 2026 Evaluations and Datasets track submission.}
}
```
