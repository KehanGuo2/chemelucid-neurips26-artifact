"""Reasoning scaffolds for skill-augmented experiments.

These are expert-designed investigation protocols injected into the agent's
system prompt as the `reasoning_scaffold` parameter.  They tell the agent
HOW to approach the problem systematically, simulating what a chemistry
instructor might provide as a lab protocol.

Usage:
    from interactive.scaffolds import NMR_ELUCIDATION_PROTOCOL
    env.run_episode(..., reasoning_scaffold=NMR_ELUCIDATION_PROTOCOL)
"""

NMR_ELUCIDATION_PROTOCOL = """\
=== EXPERT NMR STRUCTURE ELUCIDATION PROTOCOL ===

Follow this systematic protocol step-by-step. Do NOT skip steps or jump to guessing.

PHASE 1 — Molecular Formula Analysis
  1. Call compute_unsaturation to get the Degree of Unsaturation (DoU).
  2. Note the DoU value. DoU >= 4 strongly suggests an aromatic ring.
  3. Check the molecular formula for heteroatoms (N, O, S, halogens).

PHASE 2 — Systematic Spectral Region Survey (13C NMR)
  Query EACH of the following regions using query_spectrum(nucleus="13C", ...):
  4. Aromatic/alkene region: ppm_min=100, ppm_max=170
  5. Carbonyl region: ppm_min=170, ppm_max=220
  6. Alkyl region: ppm_min=0, ppm_max=50
  7. C-heteroatom region: ppm_min=50, ppm_max=100

PHASE 3 — Systematic Spectral Region Survey (1H NMR, if available)
  8. Aromatic protons: ppm_min=6.5, ppm_max=8.0
  9. Alkyl protons: ppm_min=0.25, ppm_max=2.5
  10. Heteroatom-adjacent protons: ppm_min=3.0, ppm_max=4.5
  11. For peaks found: call get_multiplicity and get_integration to determine
      splitting patterns and proton counts.

PHASE 4 — Hypothesis Formation & Testing
  12. Based on spectral evidence, propose 1-2 candidate structures.
  13. Use validate_smiles to confirm correct molecular formula.
  14. Use predict_nmr + compare_spectra to check spectral agreement.
  15. If comparison shows low similarity, revise your structure and repeat.

PHASE 5 — Submission
  16. Submit your best candidate via submit(smiles=...).

IMPORTANT: Complete Phases 1-3 BEFORE forming any structural hypothesis.
Do not guess structures based on the molecular formula alone."""
