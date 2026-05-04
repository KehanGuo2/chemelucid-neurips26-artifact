"""Curate a private canary set from nmrshiftdb-imported molecules.

Pipeline:
  1. Inventory all nmrshiftdb_*_CNMR_HRE.json (and HNMR if present)
  2. Quality filter (RDKit parse, peaks>=3, 5<=HAC<=60, DAG sane)
  3. Stratified-sample 50 candidates whose (HAC, DBE, ring) distribution
     roughly matches the core 18 in interactive/config.py:ALL_MOLECULES
  4. Emit anonymized canary_NNN folders under data_split/{public,private}/
  5. Provenance JSON, manifest JSONL, difficulty CSV
  6. Leakage check: public files must not contain SMILES, IUPAC, CAS, or
     'nmrshiftdb' substring

Run: python scripts/curate_canary_set.py
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CNMR_DIR = ROOT / "data" / "C_NMR"
HNMR_DIR = ROOT / "data" / "H_NMR"
SPLIT_PUBLIC = ROOT / "data_split" / "public_task_data"
SPLIT_PRIVATE = ROOT / "data_split" / "private_grader_data"
MANIFESTS_DIR = ROOT / "manifests"
ANALYSIS_DIR = ROOT / "analysis"
PROVENANCE_PATH = SPLIT_PRIVATE / "_canary_provenance.json"

from interactive.config import ALL_MOLECULES  # noqa: E402

RNG_SEED = 42
TARGET_CANARY_COUNT = 50

# Quality thresholds
MIN_PEAKS = 3
MIN_HAC = 5
MAX_HAC = 60

# Distribution-match caps derived from core 18 (max HAC=29, max stereo=2, max rings=4).
# These prevent the canary pool from drifting toward heavy natural-product chemistry
# that the core 18 doesn't contain.
DIST_CAP_HAC = 30
DIST_CAP_STEREO = 3
DIST_CAP_RINGS = 4

# Stratification bins (matched to core 18 spread)
HAC_BINS = [(0, 10), (10, 15), (15, 20), (20, 30), (30, 60)]
DBE_BINS = [(0, 1), (1, 3), (3, 5), (5, 8), (8, 20)]
RING_BINS = [(0, 1), (1, 2), (2, 3), (3, 6)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SPECTRA_RE = re.compile(r"\{Spectra:\s*\u03b4?\s*([0-9.,\s\-]+)\}", re.IGNORECASE)
FORMULA_RE = re.compile(r"\{Molecular Formula:\s*([A-Za-z0-9]+)\s*\}")
SOLVENT_RE = re.compile(r"\{Solvent:\s*([^}]+)\}")
MHZ_RE = re.compile(r"\{MHz Value:\s*([^}]+)\}")


def _bin_idx(value: float, bins: List[Tuple[float, float]]) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= value < hi:
            return i
    return len(bins) - 1


def _find_node(nodes: List[Dict[str, Any]], node_id: str) -> Optional[Dict[str, Any]]:
    for n in nodes:
        if n.get("id") == node_id:
            return n
    return None


def parse_big_question(label: str) -> Tuple[List[float], str, str, str]:
    """Extract (peaks, formula, solvent, mhz) from BIG QUESTION label."""
    peaks: List[float] = []
    m = SPECTRA_RE.search(label)
    if m:
        chunk = m.group(1).replace("}", "").strip()
        for tok in chunk.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                peaks.append(float(tok))
            except ValueError:
                pass
    formula = ""
    m = FORMULA_RE.search(label)
    if m:
        formula = m.group(1).strip()
    solvent = ""
    m = SOLVENT_RE.search(label)
    if m:
        solvent = m.group(1).strip()
    mhz = ""
    m = MHZ_RE.search(label)
    if m:
        mhz = m.group(1).strip()
    return peaks, formula, solvent, mhz


def descriptors_for(smiles: str) -> Optional[Dict[str, Any]]:
    """Return RDKit descriptors. None if SMILES invalid."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "hac": mol.GetNumHeavyAtoms(),
        "dbe": rdMolDescriptors.CalcNumRotatableBonds(mol) * 0  # placeholder, computed below
        ,
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "stereocenters": rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        + rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol),
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
    }


def calc_dbe_from_formula(formula: str) -> Optional[int]:
    """DBE = (2C + 2 + N - H - X) / 2."""
    counts: Dict[str, int] = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if not el:
            continue
        counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    C = counts.get("C", 0)
    H = counts.get("H", 0)
    N = counts.get("N", 0)
    X = counts.get("F", 0) + counts.get("Cl", 0) + counts.get("Br", 0) + counts.get("I", 0)
    if C == 0:
        return None
    dbe_x2 = 2 * C + 2 + N - H - X
    if dbe_x2 < 0 or dbe_x2 % 2 != 0:
        return None
    return dbe_x2 // 2


# ---------------------------------------------------------------------------
# Step 1: inventory
# ---------------------------------------------------------------------------
def inventory_nmrshiftdb() -> List[Dict[str, Any]]:
    out = []
    for path in sorted(CNMR_DIR.glob("nmrshiftdb_*_CNMR_HRE.json")):
        mol_id = path.stem.replace("_CNMR_HRE", "")
        try:
            with open(path) as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        nodes = doc.get("Nodes", [])
        bigq = _find_node(nodes, "BIG QUESTION")
        if bigq is None:
            continue
        gt_smiles = bigq.get("answer", "")
        peaks, formula, solvent, mhz = parse_big_question(bigq.get("label", ""))

        # Source ID parsing
        nm = re.match(r"nmrshiftdb_(\d+)", mol_id)
        nmrshiftdb_id = nm.group(1) if nm else None

        # HNMR companion
        hnmr_path = HNMR_DIR / f"{mol_id}_HNMR_HRE.json"
        hnmr_doc = None
        if hnmr_path.exists():
            try:
                with open(hnmr_path) as f:
                    hnmr_doc = json.load(f)
            except (json.JSONDecodeError, OSError):
                hnmr_doc = None

        out.append(
            {
                "mol_id": mol_id,
                "nmrshiftdb_id": nmrshiftdb_id,
                "cnmr_path": str(path),
                "hnmr_path": str(hnmr_path) if hnmr_doc is not None else None,
                "gt_smiles": gt_smiles,
                "formula": formula,
                "solvent": solvent,
                "mhz": mhz,
                "peaks": peaks,
                "n_peaks": len(peaks),
                "cnmr_doc": doc,
                "hnmr_doc": hnmr_doc,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Step 2: quality filter
# ---------------------------------------------------------------------------
def quality_filter(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    kept: List[Dict[str, Any]] = []
    drop = Counter()
    for r in records:
        smi = (r.get("gt_smiles") or "").strip()
        if not smi or smi.lower() in {"none", "null"}:
            drop["missing_smiles"] += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            drop["bad_smiles"] += 1
            continue
        if r["n_peaks"] < MIN_PEAKS:
            drop["too_few_peaks"] += 1
            continue
        hac = mol.GetNumHeavyAtoms()
        if hac < MIN_HAC:
            drop["hac_too_small"] += 1
            continue
        if hac > MAX_HAC:
            drop["hac_too_big"] += 1
            continue
        # DAG sanity: must have F1..F8 nodes (or at least a few of them)
        node_ids = {n.get("id") for n in r["cnmr_doc"].get("Nodes", [])}
        required = {"F1", "F2", "F3", "F4", "F5", "F6", "F8"}
        if len(node_ids & required) < 5:
            drop["dag_malformed"] += 1
            continue

        dbe = calc_dbe_from_formula(r["formula"])
        if dbe is None:
            drop["bad_formula"] += 1
            continue

        ring_count = rdMolDescriptors.CalcNumRings(mol)
        stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol) + \
            rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
        canon = Chem.MolToSmiles(mol, canonical=True)
        r.update(
            {
                "canonical_smiles": canon,
                "hac": hac,
                "dbe": dbe,
                "ring_count": ring_count,
                "stereocenters": stereo,
                "peaks_per_hac": round(r["n_peaks"] / hac, 4) if hac else 0.0,
            }
        )
        kept.append(r)
    return kept, dict(drop)


# ---------------------------------------------------------------------------
# Core 18 descriptor distribution
# ---------------------------------------------------------------------------
def core_descriptors() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for mol_id in ALL_MOLECULES:
        path = CNMR_DIR / f"{mol_id}_CNMR_HRE.json"
        if not path.exists():
            continue
        with open(path) as f:
            doc = json.load(f)
        bigq = _find_node(doc.get("Nodes", []), "BIG QUESTION")
        if bigq is None:
            continue
        smi = bigq.get("answer", "")
        peaks, formula, _, _ = parse_big_question(bigq.get("label", ""))
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        dbe = calc_dbe_from_formula(formula) or 0
        out.append(
            {
                "mol_id": mol_id,
                "set": "core",
                "hac": mol.GetNumHeavyAtoms(),
                "dbe": dbe,
                "ring_count": rdMolDescriptors.CalcNumRings(mol),
                "stereocenters": rdMolDescriptors.CalcNumAtomStereoCenters(mol)
                + rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol),
                "n_peaks": len(peaks),
                "peaks_per_hac": round(len(peaks) / max(mol.GetNumHeavyAtoms(), 1), 4),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Step 3: stratified sample
# ---------------------------------------------------------------------------
def _bucket_key(rec: Dict[str, Any]) -> Tuple[int, int, int]:
    return (
        _bin_idx(rec["hac"], HAC_BINS),
        _bin_idx(rec["dbe"], DBE_BINS),
        _bin_idx(rec["ring_count"], RING_BINS),
    )


def stratified_sample(
    candidates: List[Dict[str, Any]],
    core: List[Dict[str, Any]],
    target_n: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Sample target_n candidates so that bin frequencies (HAC, DBE, ring) track
    the core-18 distribution as closely as the candidate pool allows.

    Strategy:
      1. Compute proportional quotas from core bin frequencies.
      2. Within each bucket, prefer candidates whose stereocenter count is
         closest to the core mean (avoids drift toward heavily-stereo natural
         products, since core 18 has stereocenter mean ~0.2).
      3. If a bucket is short, redistribute remaining slots to the *nearest*
         bucket along the HAC axis (capped at the per-bucket pool size) so the
         marginal HAC histogram stays close to core.
    """
    core_dist = Counter(_bucket_key(c) for c in core)
    total_core = sum(core_dist.values()) or 1
    core_stereo_mean = sum(c["stereocenters"] for c in core) / max(len(core), 1)

    bins: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in candidates:
        bins[_bucket_key(r)].append(r)

    # Sort each bucket so we pick low-stereocenter candidates first (matches core),
    # then ties broken by lower HAC (also matches core skew).
    for k in bins:
        bins[k].sort(key=lambda r: (abs(r["stereocenters"] - core_stereo_mean), r["hac"], rng.random()))

    # Proportional quotas (no floor: bins absent in pool stay at 0).
    raw_quotas = {b: (core_dist[b] / total_core) * target_n for b in core_dist}
    quotas: Dict[Tuple[int, int, int], int] = {b: int(q) for b, q in raw_quotas.items()}
    # Distribute fractional remainder by largest fractional part first
    remainder = target_n - sum(quotas.values())
    fracs = sorted(((b, raw_quotas[b] - int(raw_quotas[b])) for b in raw_quotas),
                   key=lambda x: -x[1])
    for b, _ in fracs:
        if remainder <= 0:
            break
        quotas[b] = quotas.get(b, 0) + 1
        remainder -= 1

    # Cap quotas at available pool, track shortfalls
    chosen: List[Dict[str, Any]] = []
    shortfall = 0
    for bucket, q in list(quotas.items()):
        pool = bins.get(bucket, [])
        take = min(q, len(pool))
        chosen.extend(pool[:take])
        shortfall += (q - take)

    # Redistribute shortfalls to the nearest non-saturated bucket (by HAC bin distance)
    # so the marginal HAC distribution stays close to core.
    def _used(bucket):
        return sum(1 for r in chosen if _bucket_key(r) == bucket)

    while shortfall > 0:
        # Build candidate buckets that still have leftover, sorted by distance
        # from the core's HAC mode bin
        core_hac_modes = [
            bucket[0] for bucket, _freq in core_dist.most_common()  # HAC-bin index of densest core bucket
        ]
        progressed = False
        for hac_target in core_hac_modes:
            available = [
                (bucket, _used(bucket), len(bins[bucket]))
                for bucket in bins
                if len(bins[bucket]) > _used(bucket)
            ]
            if not available:
                break
            # Sort by HAC-distance from desired mode, then DBE/ring distance
            available.sort(key=lambda t: (abs(t[0][0] - hac_target), t[0][1], t[0][2]))
            best_bucket, used_n, pool_n = available[0]
            chosen.append(bins[best_bucket][used_n])
            shortfall -= 1
            progressed = True
            if shortfall <= 0:
                break
        if not progressed:
            break

    return chosen[:target_n]


# ---------------------------------------------------------------------------
# Step 4-5: emit split files
# ---------------------------------------------------------------------------
def _strip_dag_for_canary(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep DAG node ids+answers; drop free-text reasoning labels that may leak."""
    keep_ids = {
        "BIG QUESTION", "F1", "F1.1.1", "F1.1.2", "F1.2.1", "F1.2.2", "F1.3.1",
        "F1.3.2", "F1.4.1", "F1.4.2", "F1.5.1", "F1.5.2", "F1.6.1", "F1.6.2",
        "F1.7.1", "F1.8.1", "F2", "F2.1.1", "F3", "F3.1.1", "F4", "F5",
        "F5.1.1", "F5.1.2", "F6", "F6.1.1", "F6.1.2", "F8",
    }
    out = []
    for n in nodes:
        if n.get("id") in keep_ids:
            out.append({k: v for k, v in n.items() if k in ("id", "label", "weight", "answer")})
    return out


def _hnmr_peaks_raw(hnmr_doc: Dict[str, Any]) -> str:
    """Best-effort extraction of HNMR peaks_raw from BIG QUESTION."""
    bigq = _find_node(hnmr_doc.get("Nodes", []), "BIG QUESTION")
    if bigq is None:
        return ""
    label = bigq.get("label", "")
    m = re.search(r"\{Spectra:\s*([^}]+)\}", label)
    if m:
        return m.group(1).strip()
    return ""


def emit_split_files(chosen: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Write public + private files; return provenance records."""
    provenance: List[Dict[str, Any]] = []
    for i, rec in enumerate(chosen, 1):
        canary_id = f"canary_{i:03d}"
        pub_dir = SPLIT_PUBLIC / canary_id
        priv_dir = SPLIT_PRIVATE / canary_id
        pub_dir.mkdir(parents=True, exist_ok=True)
        priv_dir.mkdir(parents=True, exist_ok=True)

        # ---- public CNMR ----
        public_cnmr = {
            "molecule_id": canary_id,
            "spectrum_type": "CNMR",
            "molecular_formula": rec["formula"],
            "solvent": rec["solvent"],
            "mhz": rec["mhz"],
            "spectrum_peaks": rec["peaks"],
            "task_prompt": (
                "You are given the molecular formula and a 13C NMR spectrum of an "
                "unknown molecule X. Use the available chemistry tools to determine "
                "its structure and submit a SMILES string."
            ),
        }
        with open(pub_dir / "task_CNMR.json", "w") as f:
            json.dump(public_cnmr, f, indent=2)

        # ---- private CNMR grader ----
        cnmr_dag = _strip_dag_for_canary(rec["cnmr_doc"].get("Nodes", []))
        private_cnmr = {
            "molecule_id": canary_id,
            "spectrum_type": "CNMR",
            "ground_truth": {
                "smiles": rec["canonical_smiles"],
                "molecular_formula": rec["formula"],
            },
            "dag_nodes": cnmr_dag,
        }
        with open(priv_dir / "grader_CNMR.json", "w") as f:
            json.dump(private_cnmr, f, indent=2)

        # ---- HNMR if available ----
        if rec.get("hnmr_doc") is not None:
            hnmr_raw = _hnmr_peaks_raw(rec["hnmr_doc"])
            public_hnmr = {
                "molecule_id": canary_id,
                "spectrum_type": "HNMR",
                "molecular_formula": rec["formula"],
                "solvent": rec["solvent"],
                "mhz": rec["mhz"],
                "spectrum_peaks_raw": hnmr_raw,
                "task_prompt": (
                    "You are given the molecular formula and a 1H NMR spectrum of an "
                    "unknown molecule X. Use the available chemistry tools to determine "
                    "its structure and submit a SMILES string."
                ),
            }
            with open(pub_dir / "task_HNMR.json", "w") as f:
                json.dump(public_hnmr, f, indent=2)
            hnmr_dag = _strip_dag_for_canary(rec["hnmr_doc"].get("Nodes", []))
            private_hnmr = {
                "molecule_id": canary_id,
                "spectrum_type": "HNMR",
                "ground_truth": {
                    "smiles": rec["canonical_smiles"],
                    "molecular_formula": rec["formula"],
                },
                "dag_nodes": hnmr_dag,
            }
            with open(priv_dir / "grader_HNMR.json", "w") as f:
                json.dump(private_hnmr, f, indent=2)

        # ---- provenance ----
        nmr_id = rec.get("nmrshiftdb_id")
        url = (
            f"https://nmrshiftdb.nmr.uni-koeln.de/portal/media-type/html/user/anon/page/default.psml/js_pane/P-Spectrum?nmrshiftdbaction=showSpectrum&spectrumId={nmr_id}"
            if nmr_id else None
        )
        provenance.append(
            {
                "canary_id": canary_id,
                "source": f"nmrshiftdb_{nmr_id}" if nmr_id else "unknown",
                "nmrshiftdb_url": url,
                "canonical_smiles": rec["canonical_smiles"],
                "formula": rec["formula"],
                "hac": rec["hac"],
                "dbe": rec["dbe"],
                "ring_count": rec["ring_count"],
                "stereocenters": rec["stereocenters"],
                "n_peaks": rec["n_peaks"],
                "has_hnmr": rec.get("hnmr_doc") is not None,
                "license_note": (
                    "nmrshiftdb is CC-BY-SA 3.0 - public; treat IDs as semi-public - "
                    "do NOT include nmrshiftdb_id in any agent-facing output"
                ),
            }
        )
    return provenance


# ---------------------------------------------------------------------------
# Step 6/7: manifest + difficulty CSV
# ---------------------------------------------------------------------------
def write_manifest(provenance: List[Dict[str, Any]]) -> Path:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFESTS_DIR / "private_canary_manifest.jsonl"
    with open(path, "w") as f:
        for p in provenance:
            cid = p["canary_id"]
            entry = {
                "canary_id": cid,
                "public_cnmr": str(SPLIT_PUBLIC / cid / "task_CNMR.json"),
                "public_hnmr": str(SPLIT_PUBLIC / cid / "task_HNMR.json")
                if (SPLIT_PUBLIC / cid / "task_HNMR.json").exists() else None,
                "has_hnmr": p["has_hnmr"],
            }
            f.write(json.dumps(entry) + "\n")
    return path


def write_difficulty_csv(
    core: List[Dict[str, Any]], chosen: List[Dict[str, Any]]
) -> Tuple[Path, Dict[str, Any]]:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS_DIR / "canary_vs_core_difficulty.csv"
    rows = []
    for c in core:
        rows.append(
            {
                "mol_id": c["mol_id"],
                "set": "core",
                "HAC": c["hac"],
                "DBE": c["dbe"],
                "ring": c["ring_count"],
                "stereocenter": c["stereocenters"],
                "peaks": c["n_peaks"],
                "peaks_per_HAC": c["peaks_per_hac"],
            }
        )
    for i, r in enumerate(chosen, 1):
        rows.append(
            {
                "mol_id": f"canary_{i:03d}",
                "set": "canary",
                "HAC": r["hac"],
                "DBE": r["dbe"],
                "ring": r["ring_count"],
                "stereocenter": r["stereocenters"],
                "peaks": r["n_peaks"],
                "peaks_per_HAC": r["peaks_per_hac"],
            }
        )
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["mol_id", "set", "HAC", "DBE", "ring", "stereocenter", "peaks", "peaks_per_HAC"],
        )
        w.writeheader()
        w.writerows(rows)

    # Stratification check: mean diff per descriptor
    def _mean(rs, key):
        vals = [r[key] for r in rs]
        return sum(vals) / len(vals) if vals else 0.0
    core_rows = [r for r in rows if r["set"] == "core"]
    canary_rows = [r for r in rows if r["set"] == "canary"]
    flags = {}
    for k in ("HAC", "DBE", "ring", "stereocenter", "peaks", "peaks_per_HAC"):
        cm = _mean(core_rows, k)
        nm = _mean(canary_rows, k)
        denom = abs(cm) if cm else 1.0
        rel = (nm - cm) / denom if denom else 0.0
        flags[k] = {
            "core_mean": round(cm, 3),
            "canary_mean": round(nm, 3),
            "rel_diff": round(rel, 3),
            "flag_gt20pct": abs(rel) > 0.20,
        }
    return path, flags


# ---------------------------------------------------------------------------
# Step 8: leakage check
# ---------------------------------------------------------------------------
LEAK_PATTERNS = [
    re.compile(r"\bnmrshiftdb\b", re.IGNORECASE),
    re.compile(r"\bCAS[\s-]?\d", re.IGNORECASE),
    re.compile(r"\b\d{2,7}-\d{2}-\d\b"),  # CAS numbers
]


def _looks_like_smiles(s: str) -> bool:
    """Heuristic SMILES detector: contains typical SMILES structural chars."""
    if not isinstance(s, str) or len(s) < 4:
        return False
    if any(ch in s for ch in "()[]=#"):
        # Has SMILES connectors AND no whitespace, and at least one bond char
        if " " not in s.strip() and len(s) <= 200:
            mol = Chem.MolFromSmiles(s)
            if mol is not None and mol.GetNumHeavyAtoms() >= 3:
                return True
    return False


def _walk_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)


def leakage_check(provenance: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    canary_smiles = {p["canonical_smiles"] for p in provenance}
    for p in provenance:
        cid = p["canary_id"]
        for fname in ("task_CNMR.json", "task_HNMR.json"):
            f = SPLIT_PUBLIC / cid / fname
            if not f.exists():
                continue
            with open(f) as fh:
                doc = json.load(fh)
            for s in _walk_strings(doc):
                for pat in LEAK_PATTERNS:
                    if pat.search(s):
                        issues.append(f"{cid}/{fname}: matches pattern {pat.pattern!r}: {s[:80]!r}")
                if s in canary_smiles:
                    issues.append(f"{cid}/{fname}: leaks GT SMILES literal {s!r}")
                if _looks_like_smiles(s):
                    issues.append(f"{cid}/{fname}: contains string that parses as SMILES: {s[:80]!r}")
    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    rng = random.Random(RNG_SEED)

    print("[1] Inventorying nmrshiftdb files...")
    raw = inventory_nmrshiftdb()
    print(f"    raw inventory: {len(raw)}")

    print("[2] Quality filter...")
    kept, drop = quality_filter(raw)
    print(f"    kept {len(kept)}, dropped {sum(drop.values())}: {drop}")

    print("[3] Computing core 18 distribution...")
    core = core_descriptors()
    print(f"    core: {len(core)} molecules")

    # Distribution-cap pre-filter: keeps the candidate pool inside the
    # convex hull of the core 18 along the heavy-tail axes. We log how many
    # we drop here so the cap thresholds can be revisited.
    pre_cap = len(kept)
    capped = [
        r for r in kept
        if r["hac"] <= DIST_CAP_HAC
        and r["stereocenters"] <= DIST_CAP_STEREO
        and r["ring_count"] <= DIST_CAP_RINGS
    ]
    print(f"    distribution cap: {len(capped)}/{pre_cap} pass "
          f"(HAC<={DIST_CAP_HAC}, stereo<={DIST_CAP_STEREO}, rings<={DIST_CAP_RINGS})")

    print(f"[4] Stratified sampling target_n={TARGET_CANARY_COUNT}...")
    target = min(TARGET_CANARY_COUNT, len(capped))
    chosen = stratified_sample(capped, core, target, rng)
    print(f"    chosen: {len(chosen)}")

    print("[5] Emitting split files...")
    SPLIT_PUBLIC.mkdir(parents=True, exist_ok=True)
    SPLIT_PRIVATE.mkdir(parents=True, exist_ok=True)
    provenance = emit_split_files(chosen)

    # Distribution diagnostics (used for stratification flag in Step 7 too,
    # but persisted here so the provenance file is self-contained).
    def _mean(rs, key):
        return sum(r[key] for r in rs) / len(rs) if rs else 0.0
    dist_diag = {}
    for k_pool, k_rec in (("hac", "hac"), ("dbe", "dbe"),
                          ("ring_count", "ring_count"),
                          ("stereocenters", "stereocenters"),
                          ("n_peaks", "n_peaks")):
        dist_diag[k_pool] = {
            "core_mean": round(_mean(core, k_pool), 3),
            "canary_mean": round(_mean(chosen, k_rec), 3),
        }

    with open(PROVENANCE_PATH, "w") as f:
        json.dump(
            {
                "schema_version": 1,
                "rng_seed": RNG_SEED,
                "generated_at_utc": "2026-04-28",
                "canary_count": len(provenance),
                "license_note": "nmrshiftdb data is CC-BY-SA 3.0; do not expose ids to agents.",
                "stratification_bins": {
                    "HAC_BINS": HAC_BINS,
                    "DBE_BINS": DBE_BINS,
                    "RING_BINS": RING_BINS,
                },
                "distribution_caps": {
                    "HAC": DIST_CAP_HAC,
                    "stereocenters": DIST_CAP_STEREO,
                    "rings": DIST_CAP_RINGS,
                },
                "candidate_pool_size_after_quality": len(kept),
                "candidate_pool_size_after_distribution_cap": len(capped),
                "distribution_diagnostics": dist_diag,
                "distribution_caveats": (
                    "Even after distribution caps, the canary set runs heavier than "
                    "the core 18 in HAC/DBE/rings/peaks because the nmrshiftdb pool "
                    "lacks small (HAC<11) molecules entirely; core 18 has 2 such mols. "
                    "This is a known unfixable mismatch from this source. Treat the "
                    "canary set as 'core 18 + a controlled difficulty step up' rather "
                    "than as i.i.d. with core 18."
                ),
                "entries": provenance,
            },
            f,
            indent=2,
        )
    print(f"    wrote provenance -> {PROVENANCE_PATH}")

    print("[6] Manifest...")
    mf = write_manifest(provenance)
    print(f"    wrote -> {mf}")

    print("[7] Difficulty CSV + stratification check...")
    csv_path, flags = write_difficulty_csv(core, chosen)
    print(f"    wrote -> {csv_path}")
    for k, v in flags.items():
        marker = " FLAGGED" if v["flag_gt20pct"] else ""
        print(f"    {k}: core={v['core_mean']} canary={v['canary_mean']} rel={v['rel_diff']:+.2f}{marker}")

    print("[8] Leakage check...")
    ok, issues = leakage_check(provenance)
    if ok:
        print("    PASS - no leaks found in public canary files")
    else:
        print(f"    FAIL - {len(issues)} issue(s):")
        for it in issues[:20]:
            print(f"      - {it}")
        return 2

    print("\nSUMMARY")
    print(f"  raw nmrshiftdb files: {len(raw)}")
    print(f"  kept after filter:    {len(kept)}")
    print(f"  canary set written:   {len(chosen)}")
    print(f"  HNMR companions:      {sum(1 for p in provenance if p['has_hnmr'])}")
    print(f"  output dirs:")
    print(f"    public  -> {SPLIT_PUBLIC}")
    print(f"    private -> {SPLIT_PRIVATE}")
    print(f"    manifest -> {mf}")
    print(f"    csv     -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
