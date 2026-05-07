"""Tests for ChemElucidEnv — the orchestrator.

These tests validate the environment wiring without making real LLM calls.
They test:
- SMILES accuracy checking
- Efficiency computation
- Diagnostic report structure
"""

import pytest

from interactive.config import ExperimentConfig, DATA_DIR
from interactive.environment import ChemElucidEnv


PRIVATE_HRE_AVAILABLE = (
    (DATA_DIR / "CNMR_HRE_v5.json").exists()
    and (
        DATA_DIR.parent
        / "data_split"
        / "private_grader_data"
        / "2_4_dimethyl_aniline"
        / "grader_CNMR.json"
    ).exists()
)


@pytest.fixture
def env():
    config = ExperimentConfig(data_dir=str(DATA_DIR))
    return ChemElucidEnv(config)


class TestSmilesAccuracy:
    def test_exact_match(self, env):
        result = env._check_smiles("Cc1ccc(N)c(C)c1", "Cc1ccc(N)c(C)c1")
        assert result["exact_match"] is True
        assert result["tanimoto"] == 1.0

    def test_equivalent_smiles(self, env):
        # Same molecule, different SMILES notation
        result = env._check_smiles("CC1=CC(C)=CC=C1N", "Cc1ccc(N)c(C)c1")
        assert result["exact_match"] is True  # InChI should match
        assert result["tanimoto"] == 1.0

    def test_different_molecule(self, env):
        result = env._check_smiles("c1ccccc1", "Cc1ccc(N)c(C)c1")
        assert result["exact_match"] is False
        assert result["tanimoto"] < 1.0
        assert result["tanimoto"] > 0.0  # Benzene shares aromatic ring

    def test_none_submitted(self, env):
        result = env._check_smiles(None, "Cc1ccc(N)c(C)c1")
        assert result["exact_match"] is False
        assert "error" in result

    def test_invalid_submitted(self, env):
        result = env._check_smiles("INVALID", "Cc1ccc(N)c(C)c1")
        assert result["exact_match"] is False


class TestEfficiency:
    def test_zero_cost(self, env):
        assert env._compute_efficiency(0, 25) == 1.0

    def test_full_cost(self, env):
        assert env._compute_efficiency(25, 25) == 0.0

    def test_half_cost(self, env):
        assert env._compute_efficiency(12, 25) == pytest.approx(0.52, abs=0.01)

    def test_zero_budget(self, env):
        assert env._compute_efficiency(5, 0) == 0.0


class TestGrading:
    def test_grade_empty_trajectory(self, env):
        result = env._grade_trajectory(
            molecule_id="2_4_dimethyl_aniline",
            trajectory=[],
            spectrum_type="CNMR",
        )
        if not PRIVATE_HRE_AVAILABLE:
            assert result is None
            return
        assert result is not None
        assert result["coverage"] == 0.0

    def test_grade_nonexistent_molecule(self, env):
        result = env._grade_trajectory(
            molecule_id="nonexistent_molecule",
            trajectory=[],
            spectrum_type="CNMR",
        )
        assert result is None  # File not found
