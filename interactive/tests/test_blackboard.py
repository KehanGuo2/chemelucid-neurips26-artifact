"""Tests for BlackboardExt — extended blackboard with tool history."""

import pytest

from interactive.blackboard_ext import BlackboardExt


@pytest.fixture
def bb():
    return BlackboardExt()


class TestFactExtraction:
    def test_unsaturation(self, bb):
        bb.add_tool_result(
            "compute_unsaturation",
            {"formula": "C8H11N"},
            {"degree": 4, "formula": "C8H11N"},
        )
        assert any("4" in f and "unsaturation" in f.lower() for f in bb.confirmed_facts)

    def test_spectrum_query(self, bb):
        bb.add_tool_result(
            "query_spectrum",
            {"nucleus": "13C", "ppm_min": 110, "ppm_max": 160},
            {"nucleus": "13C", "peaks": [142.20, 131.28], "count": 2, "range": [110, 160]},
        )
        assert any("13C" in f and "110" in f for f in bb.confirmed_facts)

    def test_substructure_found(self, bb):
        bb.add_tool_result(
            "substructure_check",
            {"smiles": "Cc1ccc(N)c(C)c1", "pattern": "c1ccccc1"},
            {"has_match": True, "pattern": "c1ccccc1", "smiles": "Cc1ccc(N)c(C)c1", "num_matches": 1},
        )
        assert any("contains" in f for f in bb.confirmed_facts)

    def test_substructure_not_found(self, bb):
        bb.add_tool_result(
            "substructure_check",
            {"smiles": "Cc1ccc(N)c(C)c1", "pattern": "[CX3]=O"},
            {"has_match": False, "pattern": "[CX3]=O", "smiles": "Cc1ccc(N)c(C)c1", "num_matches": 0},
        )
        assert any("NOT contain" in f for f in bb.confirmed_facts)

    def test_error_not_extracted(self, bb):
        bb.add_tool_result(
            "compute_unsaturation",
            {"formula": ""},
            {"error": "Could not parse formula: "},
        )
        assert len(bb.confirmed_facts) == 0

    def test_no_duplicate_facts(self, bb):
        bb.add_tool_result(
            "compute_unsaturation",
            {"formula": "C8H11N"},
            {"degree": 4, "formula": "C8H11N"},
        )
        bb.add_tool_result(
            "compute_unsaturation",
            {"formula": "C8H11N"},
            {"degree": 4, "formula": "C8H11N"},
        )
        dbe_facts = [f for f in bb.confirmed_facts if "unsaturation" in f.lower()]
        assert len(dbe_facts) == 1  # No duplicates


class TestToolHistory:
    def test_history_tracking(self, bb):
        bb.add_tool_result("compute_unsaturation", {"formula": "C8H11N"}, {"degree": 4})
        bb.add_tool_result("query_spectrum", {"nucleus": "13C", "ppm_min": 0, "ppm_max": 250}, {"count": 8})
        assert len(bb.tool_history) == 2

    def test_history_contents(self, bb):
        bb.add_tool_result("compute_unsaturation", {"formula": "C8H11N"}, {"degree": 4})
        entry = bb.tool_history[0]
        assert entry["tool"] == "compute_unsaturation"
        assert entry["args"] == {"formula": "C8H11N"}
        assert entry["result"]["degree"] == 4


class TestRender:
    def test_render_empty(self, bb):
        text = bb.render()
        assert "BLACKBOARD" in text
        assert "No confirmed facts" in text

    def test_render_with_facts(self, bb):
        bb.add_tool_result(
            "compute_unsaturation",
            {"formula": "C8H11N"},
            {"degree": 4, "formula": "C8H11N"},
        )
        text = bb.render()
        assert "CONFIRMED FACTS" in text
        assert "unsaturation" in text.lower()
        assert "TOOL CALLS" in text

    def test_render_recent_history(self, bb):
        # Add 12 tool calls
        for i in range(12):
            bb.add_tool_result(
                "compute_unsaturation",
                {"formula": f"C{i}H{i}"},
                {"error": "test"},
            )
        text = bb.render()
        # Should show only last 10
        assert "TOOL CALLS" in text


class TestStructuredSlots:
    def test_dbe_slot(self, bb):
        bb.add_tool_result(
            "compute_unsaturation",
            {"formula": "C8H11N"},
            {"degree": 4, "formula": "C8H11N"},
        )
        assert bb._degree_of_unsaturation == 4
        assert bb._molecular_formula == "C8H11N"

    def test_mw_slot(self, bb):
        bb.add_tool_result(
            "compute_molecular_weight",
            {"formula": "C8H11N"},
            {"molecular_weight": 121.183, "formula": "C8H11N"},
        )
        assert bb._molecular_weight == 121.183
