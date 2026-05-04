# Withheld-probe layer (N=48)

This directory holds the 48-molecule withheld-probe layer of ChemElucid, used
for the outcome-only generalization evaluation. The probe layer is graded only
on final-answer correctness (InChI exact match + Morgan/2/2048 Tanimoto). It has
no expert-annotated reasoning DAG and therefore receives no L2/L3 metrics.

## Layout

- `probe_48_manifest.csv` — Hidden ground-truth manifest (grader-private,
  reviewer-facing only as a private artifact; do NOT serve to evaluation
  agents). Holds canonical SMILES, source NMR id, heavy-atom count, peak count,
  sampling seed. Includes a `legacy_internal_id` column that records the
  one-to-one mapping back to the curation-pipeline alias used to generate this
  set; this column is private and is not exposed in any agent-facing manifest.
- `spectra/probe_NNN_CNMR.json` and (when available) `spectra/probe_NNN_HNMR.json`
  — Agent-facing spectrum files. Contain only molecular formula, peaks, solvent,
  MHz, and the task prompt. SMILES, InChI, source ids, and legacy aliases are
  excluded by construction.

## Reviewer-facing usage

- **Reviewers**: this manifest is the private grader artifact. Use it together
  with `interactive/probe_grader.py:score_outcome_only(submitted, hidden)` to
  reproduce the outcome-only grading (InChI exact match + Morgan/2/2048
  Tanimoto). The agent surface — under `spectra/` — is what the evaluated
  agent sees and contains no hidden answer.
- **Agents**: must read only from `spectra/probe_NNN_*.json`. The CLI's
  `run-probe` subcommand routes through `interactive/data_loader.py`, which
  loads spectra from the public surface and the hidden SMILES from this
  manifest only on the grader-side code path.

## Sampling rule

- **Source pool:** 150-row NMRShiftDB-derived candidate pool at
  `data/dynamic/nmrshiftdb_scale150.csv`.
- **Curation workflow (frozen):** quotas matched to the 18-core descriptor
  distribution along three axes — heavy-atom count (HAC), degree of
  unsaturation (DBE), and ring count. Distribution caps applied: HAC<=30,
  stereocenters<=3, rings<=4. The workflow lives at
  `scripts/curate_canary_set.py` (legacy filename retained from the original
  curation pipeline; the public layer this script produces is the
  withheld-probe layer); the bins are HAC=
  [(0,10),(10,15),(15,20),(20,30),(30,60)], DBE=[(0,1),(1,3),(3,5),(5,8),(8,20)],
  RING=[(0,1),(1,2),(2,3),(3,6)].
- **Quality filter:** RDKit-parseable SMILES, peaks>=3, 5<=HAC<=60, DAG nodes
  F1..F8 (>=5 of 7) present in the source nmrshiftdb file.
- **Random seed:** 42 (deterministic; same seed -> same 48).
- **Core-overlap exclusion:** verified at build time against the canonical 18
  SMILES from `interactive/config.py:ALL_MOLECULES`. Overlap detected: 0.

## Provenance and license

The 48 molecules come from the nmrshiftdb2 public archive. nmrshiftdb is licensed
CC-BY-SA 3.0; source ids are recorded in `probe_48_manifest.csv:source_id` (e.g.,
`nmrshiftdb2:2500`) and in the build-time provenance file
`data_split/private_grader_data/_canary_provenance.json` (legacy filename;
this is the curation-pipeline provenance file kept under the original
filename for code compatibility, and it carries the same 48 entries under
the legacy internal alias).

## Counts

- Total probe molecules: 48
- With companion 1H NMR spectrum: 0
- Mean heavy-atom count: 21.1
- Mean peak count: 15.3

## Scope of this layer

The withheld-probe layer is the generalization layer; the 18-core is the
calibration set. The probe distribution runs heavier than the 18-core in HAC,
DBE, and ring count because the nmrshiftdb pool lacks small (HAC<11) molecules.
This is a known unfixable mismatch; treat the probe as
"core 18 + a controlled difficulty step up" rather than as i.i.d. with core 18.
