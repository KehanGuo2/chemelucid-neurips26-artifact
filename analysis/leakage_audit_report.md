# Leakage Audit Report

- Canonical answers loaded: 66 (by InChI)
- Mol IDs tracked: 66
- Total candidate findings (incl. unmatched SMILES tokens): 85
- Real leaks (match a canonical answer / InChI / CAS): 70

## Counts by severity
- HARD_LEAK: 0
- SOFT_LEAK: 70
- PUBLIC_DOC: 0

## HARD_LEAK (0)
(none)

## SOFT_LEAK (70)
### analysis/leakage_audit_report.md
- L18 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L21 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L22 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L23 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L24 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L27 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L28 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L29 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L30 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L31 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L34 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L35 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L36 [smiles] [matches `2_4_dimethyl_aniline`] `CC1=CC(C)=CC=C1N`
- L37 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L38 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L39 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L40 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L43 [smiles] [matches `acetaminophen`] `CC(=O)Nc1ccc(O)cc1`
- L44 [smiles] [matches `phenacetin`] `CCOc1ccc(NC(C)=O)cc1`
- L47 [smiles] [matches `acetaminophen`] `CC(=O)Nc1ccc(O)cc1`
- ... and 15 more

### interactive/tests/test_agent_loop.py
- L11 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`

### interactive/tests/test_blackboard.py
- L33 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L34 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L41 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L42 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`

### interactive/tests/test_dag_grader.py
- L183 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L193 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L194 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L690 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L691 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`

### interactive/tests/test_environment.py
- L24 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L24 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L30 [smiles] [matches `2_4_dimethyl_aniline`] `CC1=CC(C)=CC=C1N`
- L30 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L35 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L41 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L46 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`

### interactive/tests/test_probe_grader.py
- L37 [smiles] [matches `acetaminophen`] `CC(=O)Nc1ccc(O)cc1`
- L38 [smiles] [matches `phenacetin`] `CCOc1ccc(NC(C)=O)cc1`

### interactive/tests/test_task_bundle.py
- L58 [smiles] [matches `acetaminophen`] `CC(=O)Nc1ccc(O)cc1`
- L94 [smiles] [matches `acetaminophen`] `CC(=O)Nc1ccc(O)cc1`
- L128 [smiles] [matches `acetaminophen`] `CC(=O)Nc1ccc(O)cc1`
- L132 [smiles] [matches `acetaminophen`] `CC(=O)Nc1ccc(O)cc1`

### interactive/tests/test_tool_server.py
- L4 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L134 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L141 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L148 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L163 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L171 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L187 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L199 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L259 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L261 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L272 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`
- L277 [smiles] [matches `2_4_dimethyl_aniline`] `Cc1ccc(N)c(C)c1`

## PUBLIC_DOC (0)
(none)
