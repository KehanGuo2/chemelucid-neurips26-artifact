"""Configuration for interactive structure elucidation experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root (parent of interactive/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# DAG templates
CNMR_DAG_TEMPLATE = DATA_DIR / "CNMR_HRE_v5.json"
HNMR_DAG_TEMPLATE = DATA_DIR / "HNMR_HRE_v5.json"

# Molecule data directories
CNMR_DATA_DIR = DATA_DIR / "C_NMR"
HNMR_DATA_DIR = DATA_DIR / "H_NMR"


# ---------------------------------------------------------------------------
# All available molecules (derived from filenames)
# ---------------------------------------------------------------------------

def get_all_molecule_ids() -> List[str]:
    """Get list of all available molecule IDs from the CNMR data directory."""
    ids = []
    if CNMR_DATA_DIR.exists():
        for f in sorted(CNMR_DATA_DIR.glob("*_CNMR_HRE.json")):
            mol_id = f.stem.replace("_CNMR_HRE", "")
            ids.append(mol_id)
    return ids


ALL_MOLECULES = [
    "2_4_dimethyl_aniline",
    "2_chloro_benzoic_Acid",
    "2_hydroxy_benzoic_acid",
    "2_isopropyl_5_methyl_phenol",
    "3_methylbutyl_2_methyl_propanoate",
    "3_nitro_benzoic_acid__DMSO_",
    "6_methyl_5_hepten_2_one",
    "acetaminophen",
    "Benzil",
    "benzoin",
    "cinnamaldehyde_trans",
    "ethyl_vanillin",
    "ibuprofen",
    "pamoic_acid",
    "phenacetin",
    "saccharine",
    "theophylline",
    "vanillin_acetate",
]


# Medium-difficulty natural products (imported 2026-04-17 from chem-defense/C_NMR_no2_medium).
# Intended for Exp 4 (F7 finding: medium-difficulty accuracy drops to ~0 but L2/L3 still differentiate agents).
ALL_MOLECULES_MEDIUM = [
    "20_Hydroxyl_1_oxo_20R_22R_witha_2_5_24_trienolide_14",
    "Actinomycetoquinone_A",
    "Allianthrone_A",
    "Caffeine",
    "Formicamycin_H",
    "Naringin",
    "Nocarterphenyl_A",
    "Phrymarolin_I",
    "Retinal",
    "Rutin",
    "Streptovertimycin_Z5",
    "Strychnine",
    "Xanthone_B_4",
    "benditerpenoic_acid",
    "chalaniline_A",
    "chalaniline_B",
    "clavulacylide_A",
    "dinemaxanthone_B",
    "kirschsteinin_D",
    "merochlorin_A",
    "porosuphenol_A",
    "streptomycarbazole_A",
    "suncheonoside_G",
    "ustusal_A",
    "zanzibariolide_A",
]


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    provider: str       # "openai", "anthropic", "deepinfra", "openrouter"
    model: str          # Model name/ID
    temperature: float = 0.0
    label: str = ""     # Display label for results

    def __post_init__(self):
        if not self.label:
            self.label = self.model


MODELS: Dict[str, ModelConfig] = {
    # OpenAI
    "gpt-4.1": ModelConfig(provider="openai", model="gpt-4.1"),
    "gpt-4.1-mini": ModelConfig(provider="openai", model="gpt-4.1-mini"),
    "gpt-4.1-nano": ModelConfig(provider="openai", model="gpt-4.1-nano"),
    "gpt-5.2": ModelConfig(provider="openai", model="gpt-5.2"),
    "gpt-5.5": ModelConfig(provider="openai", model="gpt-5.5"),
    "o3": ModelConfig(provider="openai", model="o3"),
    "o4-mini": ModelConfig(provider="openai", model="o4-mini"),
    # Anthropic
    "claude-sonnet": ModelConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
    "claude-sonnet-4": ModelConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
    "claude-haiku": ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001"),
    "claude-opus": ModelConfig(provider="anthropic", model="claude-opus-4-20250514"),
    "claude-opus-4-6": ModelConfig(provider="anthropic", model="claude-opus-4-6"),
    "claude-opus-4-7": ModelConfig(provider="anthropic", model="claude-opus-4-7"),
    # OpenRouter (Gemini, DeepSeek, Qwen)
    "gemini-2.5-pro": ModelConfig(provider="openrouter", model="google/gemini-2.5-pro-preview"),
    "deepseek-r1": ModelConfig(provider="openrouter", model="deepseek/deepseek-r1"),
    "deepseek-v3": ModelConfig(provider="openrouter", model="deepseek/deepseek-chat-v3-0324"),
    "qwen3-235b": ModelConfig(provider="openrouter", model="qwen/qwen3-235b-a22b"),
    # DeepInfra
    "llama-4-maverick": ModelConfig(provider="deepinfra", model="meta-llama/Llama-4-Maverick-17B-128E-Instruct"),
}


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    data_dir: str = str(DATA_DIR)
    cnmr_dag_template: str = str(CNMR_DAG_TEMPLATE)
    hnmr_dag_template: str = str(HNMR_DAG_TEMPLATE)
    budget: int = 25
    max_turns: int = 50
    temperature: float = 0.0
    output_dir: str = str(PROJECT_ROOT / "interactive_results")
    spectrum_type: str = "CNMR"  # "CNMR", "HNMR", or "COMBO"

    @classmethod
    def default(cls) -> "ExperimentConfig":
        return cls()
