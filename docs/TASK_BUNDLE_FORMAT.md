# Task bundle format

A *task bundle* is a self-contained directory that lets a chemistry author
contribute a new molecule to ChemElucid-Gym, and lets a reviewer or CI verify
that the contribution preserves the benchmark's invariants:

- the agent path sees only formula + peaks + tool metadata
- the grader path holds the ground-truth answer and the audit provenance
- official scores are computed only by the frozen grader; custom rubric and
  DAG outputs are local extensions and are kept distinct from the official
  score

The bundle is the *unit of contribution*. In controlled releases, a frozen
subset of audited bundles becomes the official private-grader manifest;
everything outside that subset is local-only. The anonymous review package
ships the public format and sanitized examples, not the full private HRE assets.

## Directory layout

```
<bundle_dir>/
  public/
    task.json                # agent-visible task: formula, peaks, prompt, tool list
    tool_schema.json         # advertised tool whitelist + costs (frozen-style)
    prompt.md                # human-readable task instructions
  private/
    answer.json              # ground-truth SMILES + InChI (grader-only)
    provenance.json          # source / license / ingest timestamp
    expert_graph.yaml        # OPTIONAL: chemist-authored reasoning DAG
  grader/
    rubric.yaml              # OPTIONAL: custom rubric (local extension only)
  tests/
    test_no_leakage.py       # public/* must not contain SMILES/InChI/CAS
    test_task_validity.py    # bundle is internally consistent
    test_grader_smoke.py     # GT SMILES grades to InChI exact match
```

`python -m interactive.task_bundle ingest` produces this layout from CLI
arguments. `python -m interactive.task_bundle validate <bundle_dir>` checks
every invariant below.

## File-by-file contract

### `public/task.json` — agent-visible

Required keys:

```json
{
  "molecule_id": "<string>",
  "spectrum_type": "CNMR" | "HNMR" | "COMBO",
  "molecular_formula": "<string>",
  "spectrum_peaks": [<float>, ...],
  "task_prompt": "<string>"
}
```

Optional keys: `solvent`, `mhz`.

Forbidden keys (validator rejects): `smiles`, `canonical_smiles`, `inchi`,
`inchi_key`, `answer`, `ground_truth`, `gt_smiles`, `cas`, `source_id`,
`source_doi`, `doi`, `nmrshiftdb_id`, `private_id`, `legacy_internal_id`.
The validator also rejects any free-text token that RDKit canonicalizes to
the hidden answer's InChI (the leakage scan is canonicalization-aware, not
just string-match).

### `public/tool_schema.json` — agent-visible

Frozen advertised tool list and per-tool costs. The grader does not read
this file; it is for the agent runtime and for reviewers who want a
declarative view of the action space without reading
`interactive/tool_server.py`.

### `public/prompt.md` — agent-visible

Human-readable rendering of the task suitable for direct paste into a chat
interface. Must not contain ground-truth SMILES, InChI, CAS, or chemical
name; the `validate` command sweeps this file with the same regex used on
`task.json`.

### `private/answer.json` — grader-only

Required keys:

```json
{
  "molecule_id": "<string>",
  "smiles": "<canonical SMILES>",
  "molecular_formula": "<string>"
}
```

Optional but recommended: `inchi`, `inchi_key`. The validator parses
`smiles` with RDKit and confirms `molecular_formula` agrees with the
formula RDKit derives from the SMILES; a mismatch is a hard issue (the
task would be unsolvable as posed because the agent is told one formula
and graded against another structure).

### `private/provenance.json` — grader-only

```json
{
  "molecule_id": "<string>",
  "source": "<string>",
  "license": "<string>",
  "ingested_at_utc": "<ISO 8601>",
  "schema_version": 1
}
```

Recommended fields: `source_doi`, `journal`, `collection_date`. The
validator emits a warning if `provenance.json` is missing; the canonical
benchmark requires it before promotion into the official manifest.

### `private/expert_graph.yaml` — grader-only, OPTIONAL

A chemist-authored reasoning DAG with the same schema accepted by
`validate-dag`. If present, the grader can use it to compute $L_2$
coverage on this molecule. Schema:

```yaml
schema_version: 1
name: "<descriptive name>"
nodes:
  - id: "<string>"
    type: "<root | F-stage | sub-node | motif-detection | ppm-range>"
    weight: <number, 0 if unscored>
    label: "<question text>"
    answer: "<expected answer; required for weighted nodes>"
edges:
  - {from: "<id>", to: "<id>"}
```

Nodes whose `id` matches `F2` or `F2.1.1` may carry
`deterministic_source: rdkit_canonical_rank_v1` to indicate that the
answer is computed from `Chem.CanonicalRankAtoms(mol, breakTies=False)`,
matching the agent-facing `check_symmetry` tool. Author-supplied
`F2` answers are checked against the deterministic recipe; mismatches are
reported.

### `grader/rubric.yaml` — grader-only, OPTIONAL, NON-OFFICIAL

A custom rubric for local extension. Schema:

```yaml
schema_version: 1
name: "<descriptive name>"
applies_to: "<scope of this rubric>"
description: "<paragraph>"
criteria:
  - id: "<string>"
    description: "<paragraph>"
    weight: <number>
    rubric: {0: "...", 1: "...", 2: "..."}
non_official: true
```

**Output discipline.** Custom rubric outputs are local extension
diagnostics and **must not be aggregated with the official score**. The
harness emits a no-aggregation warning if a custom rubric appears next to
an official metric file; analysts are expected to keep the two streams
visually distinct in any downstream report. The official ChemElucid-Gym
score is the value reported by `interactive/grade.py` against the frozen
manifest, period.

### `tests/test_no_leakage.py` — Docker-runnable

Pre-emitted stub that the validator regenerates if missing. Asserts that
no hard-leak key (`smiles:`, `inchi:`, `cas:`, …) appears in any file
under `public/`. Re-uses
`interactive.task_bundle._scan_text_for_leaks`.

### `tests/test_task_validity.py` — Docker-runnable

Asserts that `public/task.json:molecule_id == private/answer.json:molecule_id`
and that `molecular_formula` agrees on both sides. Independent of RDKit
state.

### `tests/test_grader_smoke.py` — Docker-runnable

Asserts that submitting the GT SMILES grades to InChI exact match using
`chem_defense.utils.smiles_utils.smiles_are_equivalent`. This is an
internal sanity check; it does not exercise the agent loop.

## Official vs custom

The benchmark's headline private-grader scores are computed only against
controlled-release assets:

- the frozen official manifest for audited benchmark molecules,
- `data_split/private_grader_data/<mol_id>/grader_*.json` when supplied in a
  controlled private-grader release,
- withheld-probe labels when supplied in a controlled hidden-evaluation release,
- `interactive/grade.py` and the associated scoring definitions.

The anonymous review package does not include the full private HRE templates,
private grader labels, withheld probe annotations, or expert-curated reasoning
graphs. It includes public task bundles and `examples/hre_toy/` so reviewers can
inspect the schema and scoring pipeline without exposing the private grader
layer.

A bundle authored under this format is **not automatically** part of the
official manifest. Promotion follows the chemist-audit pipeline in
`docs/REVIEWER_DOCKER_QUICKSTART.md` §6 / `Release Protocol` in
`paper/5-appendix.tex` (Step 4: chemist agreement on $\geq$ 90% of nodes
by weight). Until promoted, bundles are local extension data and:

- their grader output is reported under a separate header
- they are not added to the official Croissant metadata
- they cannot be cited as a ChemElucid-Gym benchmark score

This separation is what makes the harness reviewer-proof: a contributor
can build a bundle, validate it, and run a grader against it, but they
cannot inflate the official score with locally-authored data.

## Invariants the validator enforces

| Invariant | Mechanism | Severity |
|---|---|---|
| `public/task.json` is well-formed | JSON parse + required keys | issue |
| `private/answer.json` is well-formed | JSON parse + required keys | issue |
| GT SMILES parses with RDKit | `Chem.MolFromSmiles` | issue |
| `public.molecular_formula == RDKit-from-SMILES formula` | `_formulas_equivalent` | issue |
| no hard-leak key in any `public/*` file | regex sweep | issue |
| no SMILES token canonicalizes to the GT InChI in `public/*` | RDKit InChI compare | issue |
| `provenance.json` exists | file presence | warning |
| recommended test stubs present | file presence | warning |

The validator's exit code is 0 when all hard issues are zero. Warnings do
not block CI but should be addressed before promotion.

## Custom DAG / custom rubric YAML schema (validators)

```bash
python -m interactive.task_bundle validate-dag <path>.yaml
python -m interactive.task_bundle validate-rubric <path>.yaml
```

The validator confirms top-level structure, node id uniqueness, weight
typing, and the presence of an `edges` list (DAG) or a `criteria` list
(rubric). Reference examples ship at:

- `examples/custom_dag.yaml`
- `examples/custom_rubric.yaml`

Both validators print the not-comparable warning at the end. Reviewers
running the harness in Docker see the same notice.

## Round-trip example

```bash
# Authoring side: ingest from CLI args
python -m interactive.task_bundle ingest \
    --smiles "CC(=O)Nc1ccc(O)cc1" \
    --formula "C8H9NO2" \
    --cnmr examples/acetaminophen_cnmr_peaks.csv \
    --source "example" \
    --license "example-only" \
    --out /tmp/chemelucid_task_candidate

# Reviewer side: validate
python -m interactive.task_bundle validate /tmp/chemelucid_task_candidate

# Local extension diagnostics (NOT official scores):
python -m interactive.task_bundle validate-dag examples/custom_dag.yaml
python -m interactive.task_bundle validate-rubric examples/custom_rubric.yaml
```

The first command produces the layout shown at the top of this file; the
second prints `OK: bundle passes all hard checks.` when the bundle is
clean. The third and fourth print their respective `OK` messages with the
not-comparable note.
