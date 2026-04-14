"""
Unit tests for B3: deepmt mr batch-generate

- _collect_batch_operators returns correct operators for pytorch
- --category filter works
- --dry-run prints operators without running generation
- --skip-existing skips operators already in repo
- --limit truncates operator list
No LLM or network dependencies.
"""

import pytest
from click.testing import CliRunner

from deepmt.cli import cli
from deepmt.commands.mr import _collect_batch_operators


# ── _collect_batch_operators ──────────────────────────────────────────────────

class TestCollectBatchOperators:
    def test_pytorch_returns_generic_names(self):
        ops = _collect_batch_operators("pytorch")
        names = [op for op, _ in ops]
        assert all("." not in op for op in names), (
            f"Framework-specific operators found: {[o for o in names if '.' in o]}"
        )

    def test_pytorch_includes_known_operators(self):
        ops = _collect_batch_operators("pytorch")
        names = [op for op, _ in ops]
        assert "relu" in names
        assert "sigmoid" in names
        assert "exp" in names

    def test_category_filter_linearity(self):
        ops = _collect_batch_operators("pytorch", category_filter="linearity")
        names = [op for op, _ in ops]
        assert "relu" in names
        assert "exp" in names
        # symmetry operators should be excluded
        assert "sigmoid" not in names

    def test_category_filter_symmetry(self):
        ops = _collect_batch_operators("pytorch", category_filter="symmetry")
        names = [op for op, _ in ops]
        assert "sigmoid" in names
        assert "abs" in names

    def test_unknown_category_returns_empty(self):
        ops = _collect_batch_operators("pytorch", category_filter="nonexistent_category_xyz")
        assert ops == []

    def test_no_duplicates(self):
        ops = _collect_batch_operators("pytorch")
        names = [op for op, _ in ops]
        assert len(names) == len(set(names)), "Duplicate operators found"

    def test_each_entry_has_category(self):
        ops = _collect_batch_operators("pytorch")
        for op, cat in ops:
            assert isinstance(cat, str), f"Category for {op} is not a string"


# ── CLI: dry-run ──────────────────────────────────────────────────────────────

class TestBatchGenerateCLI:
    def test_dry_run_no_error(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["mr", "batch-generate", "--dry-run", "--framework", "pytorch"])
        assert result.exit_code == 0, result.output
        assert "dry-run" in result.output

    def test_dry_run_shows_operators(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["mr", "batch-generate", "--dry-run", "--framework", "pytorch"])
        assert result.exit_code == 0
        assert "relu" in result.output
        assert "exp" in result.output

    def test_dry_run_category_filter(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "mr", "batch-generate", "--dry-run",
            "--framework", "pytorch",
            "--category", "linearity",
        ])
        assert result.exit_code == 0
        assert "relu" in result.output
        # symmetry operators should not appear in linearity category
        assert "abs  [symmetry]" not in result.output

    def test_dry_run_limit(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "mr", "batch-generate", "--dry-run",
            "--framework", "pytorch",
            "--limit", "2",
        ])
        assert result.exit_code == 0
        assert "待处理算子: 2 个" in result.output

    def test_invalid_source_fails(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "mr", "batch-generate", "--dry-run",
            "--sources", "invalid_src",
        ])
        assert result.exit_code != 0

    def test_skip_existing_dry_run(self):
        """--skip-existing with --dry-run: dry-run should still list all operators (no repo check)"""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "mr", "batch-generate", "--dry-run",
            "--framework", "pytorch",
            "--skip-existing",
        ])
        assert result.exit_code == 0
        assert "dry-run" in result.output
