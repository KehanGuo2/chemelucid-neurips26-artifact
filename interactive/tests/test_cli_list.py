"""Tests for `cli.cmd_list_molecules` --scope flag (Bug 4)."""

from argparse import Namespace

from interactive.cli import cmd_list_molecules
from interactive.config import ALL_MOLECULES


class TestListMoleculesScope:
    def test_default_scope_is_core_18(self, capsys):
        cmd_list_molecules(Namespace(scope="core"))
        out = capsys.readouterr().out
        assert f"({len(ALL_MOLECULES)}, scope=core)" in out
        # Core list should be exactly the 18 canonical molecules.
        assert len(ALL_MOLECULES) == 18

    def test_scope_all_lists_more_than_core(self, capsys):
        cmd_list_molecules(Namespace(scope="all"))
        out = capsys.readouterr().out
        # 'all' may equal core (if no medium-difficulty files present) but
        # never less; assert it is at least as large.
        first_line = out.splitlines()[0]
        # Format: "Available molecules (N, scope=all):"
        n = int(first_line.split("(")[1].split(",")[0])
        assert n >= len(ALL_MOLECULES)

    def test_missing_scope_attribute_defaults_to_core(self, capsys):
        # Robustness: code uses getattr(args, "scope", "core")
        cmd_list_molecules(Namespace())
        out = capsys.readouterr().out
        assert "scope=core" in out
