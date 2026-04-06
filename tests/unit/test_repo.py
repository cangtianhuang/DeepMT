"""
Unit tests for MRRepository — B1 (delete) and B2 (applicable_frameworks).
No LLM or network dependencies.
"""

import tempfile
from pathlib import Path

import pytest

from deepmt.ir.schema import MetamorphicRelation
from deepmt.mr_generator.base.mr_repository import MRRepository


def _make_mr(mr_id: str, framework: str | None = None) -> MetamorphicRelation:
    return MetamorphicRelation(
        id=mr_id,
        description=f"MR {mr_id}",
        transform=lambda x: x,
        transform_code="lambda x: x",
        oracle_expr="orig == trans",
        applicable_frameworks=[framework] if framework else None,
    )


@pytest.fixture()
def repo(tmp_path):
    return MRRepository(db_path=str(tmp_path / "test.db"))


# ── B2: applicable_frameworks 序列化/反序列化 ──────────────────────────────────

class TestApplicableFrameworks:
    def test_save_and_load_with_framework(self, repo):
        mr = _make_mr("mr-001", framework="pytorch")
        repo.save("relu", [mr], version=1, framework="pytorch")

        loaded = repo.load("relu", version=1)
        assert len(loaded) == 1
        assert loaded[0].applicable_frameworks == ["pytorch"]

    def test_save_and_load_without_framework(self, repo):
        mr = _make_mr("mr-002")
        repo.save("relu", [mr], version=1)

        loaded = repo.load("relu", version=1)
        assert len(loaded) == 1
        assert loaded[0].applicable_frameworks is None

    def test_framework_set_from_save_param(self, repo):
        """若 MR 本身无 applicable_frameworks，save 时传入 framework 应自动设置"""
        mr = _make_mr("mr-003")
        assert mr.applicable_frameworks is None
        repo.save("relu", [mr], version=1, framework="pytorch")

        loaded = repo.load("relu", version=1)
        assert loaded[0].applicable_frameworks == ["pytorch"]

    def test_load_filter_by_framework_matches(self, repo):
        repo.save("relu", [_make_mr("mr-pt", "pytorch")], version=1)
        loaded = repo.load("relu", version=1, framework="pytorch")
        assert len(loaded) == 1

    def test_load_filter_by_framework_excludes(self, repo):
        repo.save("relu", [_make_mr("mr-pt", "pytorch")], version=1)
        loaded = repo.load("relu", version=1, framework="tensorflow")
        assert len(loaded) == 0

    def test_load_filter_universal_mr_always_included(self, repo):
        """applicable_frameworks=None 的通用 MR 不被框架过滤排除"""
        repo.save("relu", [_make_mr("mr-univ")], version=1)
        loaded = repo.load("relu", version=1, framework="pytorch")
        assert len(loaded) == 1

    def test_get_mr_with_validation_status_framework_filter(self, repo):
        repo.save("relu", [_make_mr("mr-pt", "pytorch"), _make_mr("mr-tf", "tensorflow")], version=1)
        mrs = repo.get_mr_with_validation_status("relu", version=1, framework="pytorch")
        assert len(mrs) == 1
        assert mrs[0].applicable_frameworks == ["pytorch"]

    def test_list_operators_by_framework(self, repo):
        repo.save("relu", [_make_mr("mr-pt", "pytorch")], version=1)
        repo.save("sigmoid", [_make_mr("mr-tf", "tensorflow")], version=1)
        repo.save("tanh", [_make_mr("mr-univ")], version=1)  # 通用

        pt_ops = repo.list_operators_by_framework("pytorch")
        assert "relu" in pt_ops
        assert "tanh" in pt_ops  # 通用 MR 也包含
        assert "sigmoid" not in pt_ops

    def test_schema_field_default_none(self):
        mr = MetamorphicRelation(
            id="x", description="d", transform=lambda x: x
        )
        assert mr.applicable_frameworks is None


# ── B1: delete ────────────────────────────────────────────────────────────────

class TestDelete:
    def test_delete_by_mr_id(self, repo):
        repo.save("relu", [_make_mr("mr-a"), _make_mr("mr-b")], version=1)
        deleted = repo.delete("relu", mr_id="mr-a")
        assert deleted == 1
        remaining = repo.load("relu", version=1)
        assert len(remaining) == 1
        assert remaining[0].id == "mr-b"

    def test_delete_by_version(self, repo):
        repo.save("relu", [_make_mr("mr-v1a"), _make_mr("mr-v1b")], version=1)
        repo.save("relu", [_make_mr("mr-v2a")], version=2)
        deleted = repo.delete("relu", version=1)
        assert deleted == 2
        assert len(repo.load("relu", version=1)) == 0
        assert len(repo.load("relu", version=2)) == 1

    def test_delete_all(self, repo):
        repo.save("relu", [_make_mr("mr-v1")], version=1)
        repo.save("relu", [_make_mr("mr-v2")], version=2)
        deleted = repo.delete("relu")
        assert deleted == 2
        assert not repo.exists("relu")

    def test_delete_nonexistent_returns_zero(self, repo):
        deleted = repo.delete("nonexistent_op")
        assert deleted == 0

    def test_delete_by_id_and_version(self, repo):
        repo.save("relu", [_make_mr("mr-v1"), _make_mr("mr-other")], version=1)
        repo.save("relu", [_make_mr("mr-v1")], version=2)  # same mr_id, different version
        # 只删 version=1 中的 mr-v1
        deleted = repo.delete("relu", version=1, mr_id="mr-v1")
        assert deleted == 1
        # version=2 中的同名 MR 应还在
        assert len(repo.load("relu", version=2)) == 1
