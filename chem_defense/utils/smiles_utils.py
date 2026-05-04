"""
Utilities for SMILES string comparison and validation using RDKit.

This module provides robust SMILES comparison that handles:
- Canonical form normalization
- Stereochemistry variations
- Tautomers (optional)
- Multiple SMILES in a single response
"""

from typing import Optional, Tuple, List, Dict, Any
import re


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize a SMILES string using RDKit.

    Args:
        smiles: Input SMILES string

    Returns:
        Canonical SMILES string, or None if invalid
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except ImportError:
        # Fallback: basic string normalization if RDKit not available
        return smiles.strip()
    except Exception:
        return None


def smiles_are_equivalent(smiles1: str, smiles2: str) -> bool:
    """
    Check if two SMILES strings represent the same molecule using RDKit.

    Uses InChI comparison for robust structural equivalence checking.
    Falls back to canonical SMILES comparison if InChI generation fails.

    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string

    Returns:
        True if molecules are structurally equivalent, False otherwise
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import inchi

        # Parse both SMILES
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        # Check validity
        if mol1 is None or mol2 is None:
            return False

        # Primary method: Compare InChI (most robust)
        try:
            inchi1 = inchi.MolToInchi(mol1)
            inchi2 = inchi.MolToInchi(mol2)
            if inchi1 and inchi2:
                return inchi1 == inchi2
        except:
            pass  # Fall back to canonical SMILES

        # Fallback method: Compare canonical SMILES
        canonical1 = Chem.MolToSmiles(mol1, canonical=True)
        canonical2 = Chem.MolToSmiles(mol2, canonical=True)
        return canonical1 == canonical2

    except ImportError:
        # If RDKit not available, fall back to string comparison
        return smiles1.strip() == smiles2.strip()
    except Exception:
        return False


def extract_smiles_from_text(text: str) -> List[str]:
    """
    Extract SMILES strings from free-form text.
    
    Common patterns:
    - "SMILES: CCO"
    - "CCO; CC(O)C"
    - "Final answer: CCO"
    - Semicolon-separated list
    
    Args:
        text: Free-form text that may contain SMILES
        
    Returns:
        List of extracted SMILES strings
    """
    text = text.strip()
    
    # Pattern 1: Extract after "Final answer:" or "SMILES:"
    patterns = [
        r"(?i)final\s+answer\s*:\s*(.+)",
        r"(?i)smiles\s*(?:string)?\s*:\s*(.+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            text = match.group(1).strip()
            break
    
    # Pattern 2: Split by semicolons or newlines
    candidates = re.split(r'[;\n]', text)
    
    # Pattern 3: Also try splitting by "or" / "and"
    if len(candidates) == 1:
        candidates = re.split(r'\s+(?:or|and)\s+', text, flags=re.IGNORECASE)
    
    # Clean and filter
    smiles_list = []
    for candidate in candidates:
        # Remove common prefixes/suffixes
        candidate = re.sub(r'(?i)^(smiles|answer|structure)\s*:?\s*', '', candidate)
        candidate = candidate.strip().strip('.,;:')
        
        # Basic SMILES validation: contains typical SMILES characters
        if candidate and re.search(r'[A-Za-z0-9()\[\]=#@+\-]', candidate):
            # Remove trailing explanatory text (e.g., "CCO (ethanol)")
            candidate = re.split(r'\s+\(', candidate)[0]
            smiles_list.append(candidate)
    
    return smiles_list


def compare_smiles(
    predicted: str,
    ground_truth: str,
    allow_tautomers: bool = False,
    exact_match: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare two SMILES strings using RDKit canonicalization.
    
    Args:
        predicted: SMILES string from model
        ground_truth: Ground truth SMILES string
        allow_tautomers: If True, consider tautomers as equivalent
        exact_match: If True, require exact string match after canonicalization
        
    Returns:
        Tuple of (is_match, details_dict)
        details_dict contains:
            - pred_canonical: Canonical form of prediction
            - gold_canonical: Canonical form of ground truth
            - method: Comparison method used
            - exact_match: Whether strings match exactly
    """
    try:
        from rdkit import Chem
        rdkit_available = True
    except ImportError:
        rdkit_available = False
        
    details: Dict[str, Any] = {
        "method": "rdkit" if rdkit_available else "string",
        "rdkit_available": rdkit_available,
    }
    
    if not rdkit_available:
        # Fallback to string comparison
        pred_clean = predicted.strip().lower()
        gold_clean = ground_truth.strip().lower()
        is_match = pred_clean == gold_clean
        details.update({
            "pred_canonical": predicted.strip(),
            "gold_canonical": ground_truth.strip(),
            "exact_match": is_match,
            "warning": "RDKit not available, using string comparison"
        })
        return is_match, details
    
    # RDKit-based comparison
    pred_mol = Chem.MolFromSmiles(predicted)
    gold_mol = Chem.MolFromSmiles(ground_truth)
    
    if pred_mol is None:
        details.update({
            "error": "Invalid predicted SMILES",
            "pred_canonical": None,
            "gold_canonical": Chem.MolToSmiles(gold_mol) if gold_mol else None,
            "exact_match": False
        })
        return False, details
    
    if gold_mol is None:
        details.update({
            "error": "Invalid ground truth SMILES",
            "pred_canonical": Chem.MolToSmiles(pred_mol),
            "gold_canonical": None,
            "exact_match": False
        })
        return False, details
    
    # Canonicalize both
    pred_canonical = Chem.MolToSmiles(pred_mol, canonical=True)
    gold_canonical = Chem.MolToSmiles(gold_mol, canonical=True)
    
    details.update({
        "pred_canonical": pred_canonical,
        "gold_canonical": gold_canonical,
        "exact_match": pred_canonical == gold_canonical
    })
    
    # Primary check: exact canonical match
    if pred_canonical == gold_canonical:
        return True, details
    
    # If exact match required, stop here
    if exact_match:
        return False, details
    
    # Check molecular formula
    from rdkit.Chem import rdMolDescriptors
    pred_formula = rdMolDescriptors.CalcMolFormula(pred_mol)
    gold_formula = rdMolDescriptors.CalcMolFormula(gold_mol)
    details["pred_formula"] = pred_formula
    details["gold_formula"] = gold_formula
    
    if pred_formula != gold_formula:
        details["error"] = "Different molecular formulas"
        return False, details
    
    # Optional: Check tautomers
    if allow_tautomers:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        enumerator = rdMolStandardize.TautomerEnumerator()
        pred_tautomers = {Chem.MolToSmiles(t, canonical=True) 
                         for t in enumerator.Enumerate(pred_mol)}
        gold_tautomers = {Chem.MolToSmiles(t, canonical=True) 
                         for t in enumerator.Enumerate(gold_mol)}
        
        if pred_tautomers & gold_tautomers:  # Intersection
            details["match_type"] = "tautomer"
            return True, details
    
    # No match found
    details["error"] = "Different structures (same formula)"
    return False, details


def evaluate_smiles_answer(
    predicted: str,
    ground_truth: str,
    allow_multiple: bool = True,
    allow_tautomers: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a predicted SMILES answer against ground truth.
    
    This is the main entry point for SMILES evaluation.
    
    Args:
        predicted: Model's predicted SMILES (may be free-form text)
        ground_truth: Ground truth SMILES string
        allow_multiple: If True, check if any predicted SMILES matches
        allow_tautomers: If True, consider tautomers as correct
        
    Returns:
        Evaluation dictionary with:
            - class: "true" or "false"
            - score: 1.0 or 0.0
            - method: "rdkit_smiles"
            - details: Comparison details
    """
    # Extract SMILES from free-form text
    pred_smiles_list = extract_smiles_from_text(predicted)
    
    if not pred_smiles_list:
        return {
            "class": "false",
            "score": 0.0,
            "method": "rdkit_smiles",
            "rationale": "No valid SMILES found in prediction",
            "details": {
                "predicted_text": predicted[:200],
                "ground_truth": ground_truth
            }
        }
    
    # Try to match any predicted SMILES (if allow_multiple)
    all_results = []
    best_match = None
    
    for i, pred_smiles in enumerate(pred_smiles_list):
        is_match, details = compare_smiles(
            pred_smiles, 
            ground_truth,
            allow_tautomers=allow_tautomers
        )
        
        result = {
            "index": i,
            "pred_smiles": pred_smiles,
            "is_match": is_match,
            "details": details
        }
        all_results.append(result)
        
        if is_match and best_match is None:
            best_match = result
        
        if is_match and not allow_multiple:
            break
    
    # Determine final verdict
    num_predicted = len(pred_smiles_list)
    num_matched = sum(1 for r in all_results if r["is_match"])

    if num_matched > 0 and (num_matched == num_predicted or num_predicted == 1):
        # All predicted SMILES match, or single correct prediction
        return {
            "class": "true",
            "score": 1.0,
            "method": "rdkit_smiles",
            "rationale": f"Predicted SMILES matches ground truth (canonical forms match)",
            "details": {
                "matched_smiles": best_match["pred_smiles"],
                "canonical_pred": best_match["details"].get("pred_canonical"),
                "canonical_gold": best_match["details"].get("gold_canonical"),
                "all_predictions": pred_smiles_list,
                "comparison_results": all_results
            }
        }
    elif num_matched > 0 and num_matched < num_predicted:
        # Multiple SMILES predicted but only some match → partial credit
        return {
            "class": "partial",
            "score": 0.5,
            "method": "rdkit_smiles",
            "rationale": (
                f"Partial match: {num_matched}/{num_predicted} predicted SMILES "
                f"match ground truth (canonical forms match for some but not all)"
            ),
            "details": {
                "matched_smiles": best_match["pred_smiles"],
                "canonical_pred": best_match["details"].get("pred_canonical"),
                "canonical_gold": best_match["details"].get("gold_canonical"),
                "all_predictions": pred_smiles_list,
                "comparison_results": all_results,
                "num_matched": num_matched,
                "num_predicted": num_predicted,
            }
        }
    else:
        # No match
        closest = all_results[0] if all_results else None

        return {
            "class": "false",
            "score": 0.0,
            "method": "rdkit_smiles",
            "rationale": f"No predicted SMILES matches ground truth. "
                        f"Predicted: {closest['details'].get('pred_canonical', 'invalid')}. "
                        f"Expected: {closest['details'].get('gold_canonical', ground_truth)}."
                        if closest else "Invalid SMILES format",
            "details": {
                "all_predictions": pred_smiles_list,
                "comparison_results": all_results,
                "ground_truth": ground_truth
            }
        }


def compute_tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    """
    Compute Tanimoto similarity between two SMILES strings using Morgan fingerprints.

    Args:
        smiles_a: First SMILES string
        smiles_b: Second SMILES string

    Returns:
        Tanimoto similarity score (0.0-1.0), or 0.0 if either SMILES is invalid
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs

        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)

        if mol_a is None or mol_b is None:
            return 0.0

        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)

        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except ImportError:
        return 0.0
    except Exception:
        return 0.0


def extract_ranked_smiles_from_text(text: str) -> List[str]:
    """
    Parse LLM output for a ranked list of SMILES strings.

    Handles formats like:
    - "1. CCO\n2. CC(O)C\n..."
    - "Prediction 1: CCO\nPrediction 2: CC(O)C\n..."
    - "1) CCO\n2) CC(O)C\n..."
    - Plain newline-separated SMILES

    Args:
        text: LLM output text containing ranked SMILES

    Returns:
        List of SMILES strings in rank order
    """
    text = text.strip()
    smiles_list = []

    # Try numbered patterns first: "1. SMILES", "1) SMILES", "Prediction 1: SMILES"
    # Use re.MULTILINE so ^ matches line starts, avoiding newline consumption issues
    numbered_patterns = [
        r"^\s*\d+[\.\)]\s*(.+)$",
        r"^\s*(?:Prediction|Rank|Candidate)\s*\d+\s*[:\.]\s*(.+)$",
    ]

    for pattern in numbered_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if len(matches) >= 2:  # At least 2 numbered items
            for m in matches:
                candidate = m.strip().strip('.,;:"`\'')
                # Remove trailing explanatory text
                candidate = re.split(r'\s+[\(\-–]', candidate)[0].strip()
                # Remove markdown bold/backtick wrappers
                candidate = re.sub(r'[`*]', '', candidate).strip()
                if candidate and re.search(r'[A-Za-z]', candidate):
                    smiles_list.append(candidate)
            if smiles_list:
                return smiles_list

    # Fallback: newline-separated, pick lines that look like SMILES
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = re.sub(r'^(?:Prediction|Rank|Candidate)\s*\d+\s*[:\.]\s*', '', line, flags=re.IGNORECASE)
        line = line.strip().strip('.,;:"`\'')
        line = re.sub(r'[`*]', '', line).strip()
        # Remove trailing explanatory text
        line = re.split(r'\s+[\(\-–]', line)[0].strip()
        # Basic SMILES validation: contains typical SMILES chars, no spaces
        if line and re.match(r'^[A-Za-z0-9()\[\]=#@+\-\\/.]+$', line):
            smiles_list.append(line)

    return smiles_list


def evaluate_ranked_smiles(
    predicted_text: str,
    ground_truth: str,
    top_k: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Evaluate ranked SMILES predictions against ground truth.

    Computes exact match at various top-k cutoffs, MRR, and Tanimoto similarities.

    Args:
        predicted_text: LLM output containing ranked SMILES predictions
        ground_truth: Ground truth SMILES string
        top_k: List of k values to evaluate (default: [1, 3, 5])

    Returns:
        Dictionary with:
            - top_1_exact, top_3_exact, top_5_exact: bool
            - mrr: Mean Reciprocal Rank (float)
            - top_1_tanimoto: Tanimoto of top prediction (float)
            - best_tanimoto: Best Tanimoto across all predictions (float)
            - per_prediction_tanimoto: List of (smiles, tanimoto) tuples
            - score: top_1_tanimoto (continuous 0.0-1.0)
    """
    if top_k is None:
        top_k = [1, 3, 5]

    predictions = extract_ranked_smiles_from_text(predicted_text)

    if not predictions:
        return {
            "class": "false",
            "score": 0.0,
            "method": "ranked_tanimoto",
            "rationale": "No SMILES predictions found in output",
            "top_1_exact": False,
            "top_3_exact": False,
            "top_5_exact": False,
            "mrr": 0.0,
            "top_1_tanimoto": 0.0,
            "best_tanimoto": 0.0,
            "per_prediction_tanimoto": [],
            "predictions": [],
        }

    # Validate ground truth SMILES
    try:
        from rdkit import Chem
        gt_valid = Chem.MolFromSmiles(ground_truth) is not None
    except:
        gt_valid = False

    # Compute per-prediction metrics with explicit validation
    per_pred_tanimoto = []
    valid_smiles_count = 0
    invalid_smiles = []
    first_exact_rank = None

    for i, pred_smiles in enumerate(predictions):
        # Validate predicted SMILES
        try:
            pred_mol = Chem.MolFromSmiles(pred_smiles)
            pred_valid = pred_mol is not None
        except:
            pred_valid = False

        if pred_valid:
            valid_smiles_count += 1
        else:
            invalid_smiles.append((i + 1, pred_smiles))

        tanimoto = compute_tanimoto_similarity(pred_smiles, ground_truth)
        per_pred_tanimoto.append((pred_smiles, tanimoto))

        # Check exact match using RDKit-based comparison
        if first_exact_rank is None:
            if smiles_are_equivalent(pred_smiles, ground_truth):
                first_exact_rank = i + 1  # 1-indexed rank

    # Top-k exact match
    top_k_exact = {}
    for k in top_k:
        top_k_exact[f"top_{k}_exact"] = (first_exact_rank is not None and first_exact_rank <= k)

    # MRR
    mrr = 1.0 / first_exact_rank if first_exact_rank is not None else 0.0

    # Tanimoto metrics
    top_1_tanimoto = per_pred_tanimoto[0][1] if per_pred_tanimoto else 0.0
    best_tanimoto = max(t for _, t in per_pred_tanimoto) if per_pred_tanimoto else 0.0

    # Score = top_1_tanimoto
    score = top_1_tanimoto

    # Determine class
    if top_k_exact.get("top_1_exact", False):
        cls = "true"
    elif top_k_exact.get("top_3_exact", False) or top_k_exact.get("top_5_exact", False):
        cls = "partial"  # Correct answer found in top-3 or top-5, but not ranked first
    elif score >= 0.5:
        cls = "partial"
    else:
        cls = "false"

    result = {
        "class": cls,
        "score": score,
        "method": "ranked_tanimoto",
        "rationale": f"Top-1 Tanimoto={top_1_tanimoto:.4f}, Best Tanimoto={best_tanimoto:.4f}, "
                     f"MRR={mrr:.4f}, {len(predictions)} predictions parsed, "
                     f"{valid_smiles_count}/{len(predictions)} valid SMILES",
        "mrr": mrr,
        "top_1_tanimoto": top_1_tanimoto,
        "best_tanimoto": best_tanimoto,
        "per_prediction_tanimoto": per_pred_tanimoto,
        "predictions": predictions,
        "valid_smiles_count": valid_smiles_count,
        "invalid_smiles": invalid_smiles,
        "ground_truth_valid": gt_valid,
    }
    result.update(top_k_exact)

    return result


def _try_opsin(name: str) -> Optional[str]:
    """Attempt IUPAC → SMILES conversion via OPSIN. Returns canonical SMILES or None."""
    try:
        from rdkit import Chem
        from py2opsin import py2opsin as _opsin_convert
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _opsin_convert([name])
        if results and results[0]:
            smi = results[0].strip()
            # OPSIN sometimes returns dot-separated duplicates for multi-word
            # inputs (e.g., "salicylic acid" → "OC(...)=O.OC(...)=O").
            # Take only the first component if duplicated.
            if '.' in smi:
                parts = smi.split('.')
                unique = list(dict.fromkeys(parts))  # deduplicate preserving order
                if len(unique) == 1:
                    smi = unique[0]
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return None


def resolve_molecule_to_smiles(identifier: str) -> Optional[str]:
    """
    Resolve a molecule identifier (SMILES, IUPAC name, common name) to a
    canonical SMILES string.

    Resolution order:
      1. Try as SMILES directly (RDKit)
      2. Handle "Name (SMILES)" format — extract parenthetical SMILES
      3. Try OPSIN (IUPAC → SMILES, requires py2opsin + Java)
      4. Fallback: None

    Args:
        identifier: SMILES string, IUPAC name, or common name

    Returns:
        Canonical SMILES string, or None if resolution fails.
    """
    from rdkit import Chem

    identifier = identifier.strip()
    if not identifier:
        return None

    # 1) Direct SMILES parse
    mol = Chem.MolFromSmiles(identifier)
    if mol is not None:
        return Chem.MolToSmiles(mol)

    # 2) Handle "Name (SMILES)" or "Name (SMILES: ...)" format
    #    Variants seen in practice:
    #      Claude: "N-methyl-2,6-dimethylbenzylamine (Cc1cccc(C)c1CN(C))"
    #      GPT:    "4-acetamidobenzaldehyde (SMILES: CC(=O)NC1=CC=C(C=C1)C=O)"
    #      GPT:    "4-formylacetanilide (p-formylacetanilide tautomer; SMILES: CC(=O)N(...)H)"
    m = re.match(r'^(.+?)\s+\((.+)\)\s*$', identifier)
    if m:
        name_part = m.group(1).strip()
        paren_part = m.group(2).strip()

        # Try extracting SMILES after "SMILES:" label (GPT format)
        smi_label = re.search(r'(?:SMILES\s*:\s*)(.+)$', paren_part, re.IGNORECASE)
        if smi_label:
            smi_candidate = smi_label.group(1).strip()
            mol2 = Chem.MolFromSmiles(smi_candidate)
            if mol2 is not None:
                return Chem.MolToSmiles(mol2)

        # Try full parenthetical as SMILES (Claude format)
        mol2 = Chem.MolFromSmiles(paren_part)
        if mol2 is not None:
            return Chem.MolToSmiles(mol2)

        # Try name part via OPSIN
        resolved = _try_opsin(name_part)
        if resolved:
            return resolved

        # Try parenthetical content as a name via OPSIN
        # Handles: "CH3CH2... (isobutyl ethyl carbonate)" where the condensed
        # formula isn't valid SMILES but the parenthetical name is IUPAC
        # Strip any "SMILES:" prefix and semicolons for multi-part parentheticals
        for paren_candidate in [paren_part] + [s.strip() for s in paren_part.split(";")]:
            cleaned = re.sub(r'(?i)^SMILES\s*:\s*', '', paren_candidate).strip()
            if cleaned and not Chem.MolFromSmiles(cleaned):
                resolved = _try_opsin(cleaned)
                if resolved:
                    return resolved

    # 3) Handle "Name or SMILES" format (Claude variant)
    #    e.g. "(E)-6-methylhept-5-en-2-one or CC(=O)CCC=C(C)C"
    if ' or ' in identifier:
        parts = [p.strip() for p in identifier.split(' or ')]
        for part in parts:
            mol3 = Chem.MolFromSmiles(part)
            if mol3 is not None:
                return Chem.MolToSmiles(mol3)
        for part in parts:
            resolved = _try_opsin(part)
            if resolved:
                return resolved

    # 4) OPSIN (IUPAC → SMILES) on the full identifier
    resolved = _try_opsin(identifier)
    if resolved:
        return resolved

    return None


def extract_ranked_molecules_from_text(text: str) -> List[str]:
    """
    Parse LLM output for a ranked list of molecule identifiers.

    Unlike extract_ranked_smiles_from_text, this accepts ANY molecule
    representation (SMILES, IUPAC names, common names).

    Args:
        text: LLM output text containing ranked molecule predictions

    Returns:
        List of molecule identifier strings in rank order
    """
    text = text.strip()
    molecules = []

    # Try numbered patterns: "1. <molecule>", "1) <molecule>", "Prediction 1: <molecule>"
    numbered_patterns = [
        r"^\s*\d+[\.\)]\s*(.+)$",
        r"^\s*(?:Prediction|Rank|Candidate)\s*\d+\s*[:\.]\s*(.+)$",
    ]

    for pattern in numbered_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if len(matches) >= 2:
            for m in matches:
                candidate = m.strip().strip('.,;:"`\'')
                # Remove markdown bold/backtick wrappers
                candidate = re.sub(r'[`*]', '', candidate).strip()
                # Remove trailing parenthetical explanations but keep molecule names with parens
                # Only strip if explanation is clearly after a dash or long space
                candidate = re.split(r'\s+[-–—]\s+', candidate)[0].strip()
                if candidate:
                    molecules.append(candidate)
            if molecules:
                return molecules

    # Fallback: newline-separated
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = re.sub(r'^(?:Prediction|Rank|Candidate)\s*\d+\s*[:\.]\s*', '', line, flags=re.IGNORECASE)
        line = line.strip().strip('.,;:"`\'')
        line = re.sub(r'[`*]', '', line).strip()
        line = re.split(r'\s+[-–—]\s+', line)[0].strip()
        if line and len(line) > 1:
            molecules.append(line)

    return molecules


def evaluate_ranked_molecules(
    predicted_text: str,
    ground_truth: str,
    top_k: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Evaluate ranked molecule predictions (any representation) against ground truth.

    Like evaluate_ranked_smiles but first resolves each prediction to SMILES via
    resolve_molecule_to_smiles (supports IUPAC names, common names, SMILES).

    Args:
        predicted_text: LLM output with ranked molecule predictions
        ground_truth: Ground truth SMILES string

    Returns:
        Same schema as evaluate_ranked_smiles, plus resolution metadata.
    """
    if top_k is None:
        top_k = [1, 3, 5]

    raw_predictions = extract_ranked_molecules_from_text(predicted_text)

    if not raw_predictions:
        return {
            "class": "false",
            "score": 0.0,
            "method": "ranked_tanimoto",
            "rationale": "No molecule predictions found in output",
            "top_1_exact": False,
            "top_3_exact": False,
            "top_5_exact": False,
            "mrr": 0.0,
            "top_1_tanimoto": 0.0,
            "best_tanimoto": 0.0,
            "per_prediction_tanimoto": [],
            "predictions": [],
            "raw_predictions": raw_predictions,
            "resolution_log": [],
        }

    # Resolve each prediction to SMILES
    resolved = []
    resolution_log = []
    for raw in raw_predictions:
        smi = resolve_molecule_to_smiles(raw)
        resolved.append(smi)
        resolution_log.append({
            "raw": raw,
            "resolved_smiles": smi,
            "resolved": smi is not None,
        })

    # Build SMILES-only list for evaluation (use resolved or original)
    smiles_predictions = []
    for raw, smi in zip(raw_predictions, resolved):
        smiles_predictions.append(smi if smi else raw)

    # Now evaluate using the resolved SMILES
    from rdkit import Chem

    gt_valid = Chem.MolFromSmiles(ground_truth) is not None

    per_pred_tanimoto = []
    valid_count = 0
    first_exact_rank = None

    for i, pred_smi in enumerate(smiles_predictions):
        pred_mol = Chem.MolFromSmiles(pred_smi) if pred_smi else None
        if pred_mol is not None:
            valid_count += 1

        tanimoto = compute_tanimoto_similarity(pred_smi, ground_truth) if pred_smi else 0.0
        per_pred_tanimoto.append((raw_predictions[i], tanimoto))

        if first_exact_rank is None and pred_smi:
            if smiles_are_equivalent(pred_smi, ground_truth):
                first_exact_rank = i + 1

    top_k_exact = {}
    for k in top_k:
        top_k_exact[f"top_{k}_exact"] = (first_exact_rank is not None and first_exact_rank <= k)

    mrr = 1.0 / first_exact_rank if first_exact_rank is not None else 0.0
    top_1_tanimoto = per_pred_tanimoto[0][1] if per_pred_tanimoto else 0.0
    best_tanimoto = max(t for _, t in per_pred_tanimoto) if per_pred_tanimoto else 0.0
    score = top_1_tanimoto

    resolved_count = sum(1 for s in resolved if s is not None)

    if top_k_exact.get("top_1_exact", False):
        cls = "true"
    elif top_k_exact.get("top_3_exact", False) or top_k_exact.get("top_5_exact", False):
        cls = "partial"
    elif score >= 0.5:
        cls = "partial"
    else:
        cls = "false"

    result = {
        "class": cls,
        "score": score,
        "method": "ranked_tanimoto",
        "rationale": (
            f"Top-1 Tanimoto={top_1_tanimoto:.4f}, Best Tanimoto={best_tanimoto:.4f}, "
            f"MRR={mrr:.4f}, {len(raw_predictions)} predictions parsed, "
            f"{resolved_count}/{len(raw_predictions)} resolved to SMILES, "
            f"{valid_count}/{len(raw_predictions)} valid"
        ),
        "mrr": mrr,
        "top_1_tanimoto": top_1_tanimoto,
        "best_tanimoto": best_tanimoto,
        "per_prediction_tanimoto": per_pred_tanimoto,
        "predictions": raw_predictions,
        "resolved_smiles": [s for s in resolved],
        "resolution_log": resolution_log,
        "valid_smiles_count": valid_count,
        "ground_truth_valid": gt_valid,
    }
    result.update(top_k_exact)
    return result


__all__ = [
    "canonicalize_smiles",
    "extract_smiles_from_text",
    "compare_smiles",
    "evaluate_smiles_answer",
    "compute_tanimoto_similarity",
    "extract_ranked_smiles_from_text",
    "evaluate_ranked_smiles",
    "resolve_molecule_to_smiles",
    "extract_ranked_molecules_from_text",
    "evaluate_ranked_molecules",
]
