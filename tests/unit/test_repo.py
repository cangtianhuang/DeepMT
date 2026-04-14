"""
Unit tests for MRRepository — checked/proven/verified and delete.
No LLM or network dependencies.
"""

from pathlib import Path

import pytest

from deepmt.ir import MetamorphicRelation
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
    return MRRepository(repo_dir=str(tmp_path / "mr_repo"))


# ── applicable_frameworks 序列化/反序列化 ──────────────────────────────────────

class TestApplicableFrameworks:
    def test_save_and_load_with_framework(self, repo):
        mr = _make_mr("mr-001", framework="pytorch")
        repo.save("relu", [mr], framework="pytorch")

        loaded = repo.load("relu")
        assert len(loaded) == 1
        assert loaded[0].applicable_frameworks == ["pytorch"]

    def test_save_and_load_without_framework(self, repo):
        mr = _make_mr("mr-002")
        repo.save("relu", [mr])

        loaded = repo.load("relu")
        assert len(loaded) == 1
        assert loaded[0].applicable_frameworks is None

    def test_framework_set_from_save_param(self, repo):
        mr = _make_mr("mr-003")
        assert mr.applicable_frameworks is None
        repo.save("relu", [mr], framework="pytorch")

        loaded = repo.load("relu")
        assert loaded[0].applicable_frameworks == ["pytorch"]

    def test_load_filter_by_framework_matches(self, repo):
        repo.save("relu", [_make_mr("mr-pt", "pytorch")])
        loaded = repo.load("relu", framework="pytorch")
        assert len(loaded) == 1

    def test_load_filter_by_framework_excludes(self, repo):
        repo.save("relu", [_make_mr("mr-pt", "pytorch")])
        loaded = repo.load("relu", framework="tensorflow")
        assert len(loaded) == 0

    def test_load_filter_universal_mr_always_included(self, repo):
        repo.save("relu", [_make_mr("mr-univ")])
        loaded = repo.load("relu", framework="pytorch")
        assert len(loaded) == 1

    def test_get_mr_with_validation_status_framework_filter(self, repo):
        repo.save("relu", [_make_mr("mr-pt", "pytorch"), _make_mr("mr-tf", "tensorflow")])
        mrs = repo.get_mr_with_validation_status("relu", framework="pytorch")
        assert len(mrs) == 1
        assert mrs[0].applicable_frameworks == ["pytorch"]

    def test_list_operators_by_framework(self, repo):
        repo.save("relu", [_make_mr("mr-pt", "pytorch")])
        repo.save("sigmoid", [_make_mr("mr-tf", "tensorflow")])
        repo.save("tanh", [_make_mr("mr-univ")])

        pt_ops = repo.list_operators_by_framework("pytorch")
        assert "relu" in pt_ops
        assert "tanh" in pt_ops
        assert "sigmoid" not in pt_ops

    def test_schema_field_default_none(self):
        mr = MetamorphicRelation(id="x", description="d", transform=lambda x: x)
        assert mr.applicable_frameworks is None


# ── checked / proven / verified ───────────────────────────────────────────────

class TestVerificationState:
    def test_checked_proven_roundtrip(self, repo):
        mr = _make_mr("mr-v")
        mr.checked = True
        mr.proven = True
        mr.verified = True
        repo.save("relu", [mr])

        loaded = repo.load("relu")
        assert loaded[0].checked is True
        assert loaded[0].proven is True
        assert loaded[0].verified is True

    def test_checked_only_not_verified(self, repo):
        mr = _make_mr("mr-c")
        mr.checked = True
        mr.proven = None
        mr.verified = False
        repo.save("relu", [mr])

        loaded = repo.load("relu")
        assert loaded[0].checked is True
        assert loaded[0].proven is None
        assert loaded[0].verified is False

    def test_source_roundtrip(self, repo):
        mr = _make_mr("mr-s")
        mr.source = "llm"
        repo.save("relu", [mr])

        loaded = repo.load("relu")
        assert loaded[0].source == "llm"

    def test_analysis_roundtrip(self, repo):
        mr = _make_mr("mr-a")
        mr.analysis = "some explanation"
        repo.save("relu", [mr])

        loaded = repo.load("relu")
        assert loaded[0].analysis == "some explanation"


# ── delete ────────────────────────────────────────────────────────────────────

class TestDelete:
    def test_delete_by_mr_id(self, repo):
        repo.save("relu", [_make_mr("mr-a"), _make_mr("mr-b")])
        deleted = repo.delete("relu", mr_id="mr-a")
        assert deleted == 1
        remaining = repo.load("relu")
        assert len(remaining) == 1
        assert remaining[0].id == "mr-b"

    def test_delete_all(self, repo):
        repo.save("relu", [_make_mr("mr-v1"), _make_mr("mr-v2")])
        deleted = repo.delete("relu")
        assert deleted == 2
        assert not repo.exists("relu")

    def test_delete_nonexistent_returns_zero(self, repo):
        deleted = repo.delete("nonexistent_op")
        assert deleted == 0

    def test_delete_nonexistent_mr_id_returns_zero(self, repo):
        repo.save("relu", [_make_mr("mr-a")])
        deleted = repo.delete("relu", mr_id="does-not-exist")
        assert deleted == 0
        assert len(repo.load("relu")) == 1


# ── 基本增删查 ─────────────────────────────────────────────────────────────────

class TestBasicCRUD:
    def test_exists_false_before_save(self, repo):
        assert not repo.exists("relu")

    def test_exists_true_after_save(self, repo):
        repo.save("relu", [_make_mr("mr-1")])
        assert repo.exists("relu")

    def test_list_operators(self, repo):
        repo.save("relu", [_make_mr("mr-1")])
        repo.save("sigmoid", [_make_mr("mr-2")])
        ops = repo.list_operators()
        assert set(ops) == {"relu", "sigmoid"}

    def test_save_overwrites(self, repo):
        repo.save("relu", [_make_mr("mr-old")])
        repo.save("relu", [_make_mr("mr-new1"), _make_mr("mr-new2")])
        loaded = repo.load("relu")
        assert len(loaded) == 2
        assert {m.id for m in loaded} == {"mr-new1", "mr-new2"}

    def test_get_statistics(self, repo):
        mr_verified = _make_mr("mr-v")
        mr_verified.verified = True
        mr_verified.checked = True
        mr_verified.proven = True
        mr_unverified = _make_mr("mr-u")
        repo.save("relu", [mr_verified, mr_unverified])

        stats = repo.get_statistics("relu")
        assert stats["total_mrs"] == 2
        assert stats["verified_mrs"] == 1
        assert stats["unverified_mrs"] == 1
        assert stats["checked"] == 1
        assert stats["proven"] == 1

    def test_transform_code_roundtrip(self, repo):
        mr = MetamorphicRelation(
            id="mr-rt",
            description="test",
            transform=lambda *args: (args[1], args[0]),
            transform_code="lambda *args: (args[1], args[0])",
            oracle_expr="orig == trans",
        )
        repo.save("op", [mr])
        loaded = repo.load("op")
        assert len(loaded) == 1
        assert loaded[0].transform_code == "lambda *args: (args[1], args[0])"
        result = loaded[0].transform(1, 2)
        assert result == (2, 1)
