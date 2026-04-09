"""
Unit tests for MRLibrary — project-level verified MR storage.
No LLM or network dependencies.
"""

import pytest

from deepmt.ir.schema import MetamorphicRelation
from deepmt.mr_generator.base.mr_library import MRLibrary
from deepmt.mr_generator.base.mr_repository import MRRepository


def _make_verified_mr(mr_id: str, frameworks=None) -> MetamorphicRelation:
    mr = MetamorphicRelation(
        id=mr_id,
        description=f"MR {mr_id}",
        transform=lambda x: x,
        transform_code="lambda x: x",
        oracle_expr="orig == trans",
        applicable_frameworks=frameworks,
        checked=True,
        proven=True,
        verified=True,
    )
    return mr


def _make_unverified_mr(mr_id: str) -> MetamorphicRelation:
    return MetamorphicRelation(
        id=mr_id,
        description=f"MR {mr_id}",
        transform=lambda x: x,
        transform_code="lambda x: x",
        oracle_expr="orig == trans",
        verified=False,
    )


@pytest.fixture()
def lib(tmp_path):
    return MRLibrary(layer="operator", library_dir=str(tmp_path / "mr_library"))


@pytest.fixture()
def repo(tmp_path):
    return MRRepository(repo_dir=str(tmp_path / "mr_repo"))


class TestMRLibrarySave:
    def test_save_verified_mr(self, lib):
        mr = _make_verified_mr("mr-1")
        count = lib.save("torch.add", [mr])
        assert count == 1
        assert lib.exists("torch.add")

    def test_save_filters_unverified(self, lib):
        mrs = [_make_verified_mr("mr-v"), _make_unverified_mr("mr-u")]
        count = lib.save("torch.add", mrs)
        assert count == 1
        loaded = lib.load("torch.add")
        assert len(loaded) == 1
        assert loaded[0].id == "mr-v"

    def test_save_no_verified_returns_zero(self, lib):
        count = lib.save("torch.add", [_make_unverified_mr("mr-u")])
        assert count == 0
        assert not lib.exists("torch.add")

    def test_save_overwrites_existing(self, lib):
        lib.save("torch.add", [_make_verified_mr("mr-old")])
        lib.save("torch.add", [_make_verified_mr("mr-new")])
        loaded = lib.load("torch.add")
        assert len(loaded) == 1
        assert loaded[0].id == "mr-new"


class TestMRLibraryLoad:
    def test_load_by_operator(self, lib):
        lib.save("torch.add", [_make_verified_mr("mr-add")])
        lib.save("torch.mul", [_make_verified_mr("mr-mul")])

        add_mrs = lib.load("torch.add")
        assert len(add_mrs) == 1
        assert add_mrs[0].id == "mr-add"

    def test_load_all(self, lib):
        lib.save("torch.add", [_make_verified_mr("mr-add")])
        lib.save("torch.mul", [_make_verified_mr("mr-mul")])

        all_mrs = lib.load()
        assert len(all_mrs) == 2

    def test_loaded_mr_always_verified(self, lib):
        lib.save("torch.add", [_make_verified_mr("mr-1")])
        loaded = lib.load("torch.add")
        assert loaded[0].verified is True
        assert loaded[0].checked is True
        assert loaded[0].proven is True

    def test_noise_fields_stripped(self, lib):
        mr = _make_verified_mr("mr-1")
        mr.analysis = "some llm text"
        mr.source = "llm"
        lib.save("torch.add", [mr])

        loaded = lib.load("torch.add")
        assert loaded[0].analysis == ""
        assert loaded[0].source == ""

    def test_tolerance_default_not_stored(self, lib, tmp_path):
        import yaml
        mr = _make_verified_mr("mr-1")
        mr.tolerance = 1e-6
        lib.save("torch.add", [mr])

        raw = yaml.safe_load(lib._yaml_path.read_text())
        assert "tolerance" not in raw["torch.add"][0]

    def test_tolerance_nondefault_stored(self, lib, tmp_path):
        import yaml
        mr = _make_verified_mr("mr-1")
        mr.tolerance = 1e-4
        lib.save("torch.add", [mr])

        raw = yaml.safe_load(lib._yaml_path.read_text())
        assert "tolerance" in raw["torch.add"][0]

    def test_applicable_frameworks_preserved(self, lib):
        mr = _make_verified_mr("mr-1", frameworks=["pytorch"])
        lib.save("torch.add", [mr])
        loaded = lib.load("torch.add")
        assert loaded[0].applicable_frameworks == ["pytorch"]

    def test_applicable_frameworks_null_not_stored(self, lib, tmp_path):
        import yaml
        mr = _make_verified_mr("mr-1", frameworks=None)
        lib.save("torch.add", [mr])

        raw = yaml.safe_load(lib._yaml_path.read_text())
        assert "applicable_frameworks" not in raw["torch.add"][0]


class TestMRLibraryPromote:
    def test_promote_verified_from_repo(self, lib, repo):
        mr = _make_verified_mr("mr-pt", frameworks=["pytorch"])
        repo.save("torch.add", [mr])

        count = lib.promote_from_repository("torch.add", repo)
        assert count == 1
        assert lib.exists("torch.add")

    def test_promote_skips_unverified(self, lib, repo):
        mrs = [_make_verified_mr("mr-v"), _make_unverified_mr("mr-u")]
        repo.save("torch.add", mrs)

        count = lib.promote_from_repository("torch.add", repo)
        assert count == 1
        loaded = lib.load("torch.add")
        assert loaded[0].id == "mr-v"

    def test_promote_empty_returns_zero(self, lib, repo):
        repo.save("torch.add", [_make_unverified_mr("mr-u")])
        count = lib.promote_from_repository("torch.add", repo)
        assert count == 0


class TestMRLibraryListOperators:
    def test_list_operators(self, lib):
        lib.save("torch.add", [_make_verified_mr("mr-1")])
        lib.save("torch.relu", [_make_verified_mr("mr-2")])
        ops = lib.list_operators()
        assert set(ops) == {"torch.add", "torch.relu"}
