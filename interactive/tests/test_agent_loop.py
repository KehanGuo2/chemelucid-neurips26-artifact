"""Tests for AgentLoop. Uses a stub _call_llm so no API keys are needed."""

import pytest

from interactive.agent_loop import AgentLoop, LLMResponse
from interactive.config import DATA_DIR as _DATA_DIR
from interactive.tool_server import ToolServer

DATA_DIR = str(_DATA_DIR)
MOL_ID = "2_4_dimethyl_aniline"
GT_SMILES = "Cc1ccc(N)c(C)c1"


def _make_loop(submit_smiles: str) -> AgentLoop:
    """Return an AgentLoop whose first LLM call is a valid submit."""
    server = ToolServer(molecule_id=MOL_ID, data_dir=DATA_DIR, spectrum_type="CNMR")
    loop = AgentLoop(provider="openai", model="stub", tool_server=server, budget=25)
    # Monkeypatch _call_llm to skip network — return a single submit response.
    def _fake_call_llm():
        return LLMResponse(
            text=None,
            tool_calls=[{"name": "submit", "args": {"smiles": submit_smiles}, "id": "stub_id"}],
            is_submit=True,
            submitted_smiles=submit_smiles,
        )
    loop._call_llm = _fake_call_llm  # type: ignore[assignment]
    return loop


class TestSubmitTrajectoryLogging:
    """Regression for Bug 1: valid submits must appear in trajectory."""

    def test_valid_submit_logged_in_trajectory(self):
        loop = _make_loop(GT_SMILES)
        result = loop.run("Find the molecule.")
        assert result["terminated_by"] == "submit"
        assert result["submitted_smiles"] == GT_SMILES

        traj = result["trajectory"]
        submit_steps = [s for s in traj if s["tool"] == "submit"]
        assert len(submit_steps) == 1, (
            f"Expected exactly one 'submit' entry in trajectory, got {len(submit_steps)}"
        )
        assert submit_steps[0]["args"]["smiles"] == GT_SMILES
        # submit must remain free (cost 0) so it does not eat the budget.
        assert submit_steps[0]["cost"] == 0

    def test_submit_does_not_change_total_cost(self):
        loop = _make_loop(GT_SMILES)
        result = loop.run("Find the molecule.")
        # No other tools were called; total cost must stay at zero.
        assert result["total_cost"] == 0
