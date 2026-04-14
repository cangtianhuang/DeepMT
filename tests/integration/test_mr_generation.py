"""MR生成流水线集成测试"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from deepmt.ir import OperatorIR, MetamorphicRelation
from deepmt.mr_generator.operator.operator_mr_generator import OperatorMRGenerator
from deepmt.mr_generator.base.mr_repository import MRRepository


def _mock_mr(mr_id="mr_1", oracle="orig == trans", category="idempotency",
             transform=None, transform_code="lambda k: k"):
    if transform is None:
        transform = lambda k: k
    return MetamorphicRelation(
        id=mr_id,
        description=f"Test MR {mr_id}",
        transform=transform,
        transform_code=transform_code,
        oracle_expr=oracle,
        category=category,
        tolerance=1e-6,
        layer="operator",
        verified=False,
    )


@pytest.fixture
def generator():
    gen = OperatorMRGenerator()
    gen.llm_generator = MagicMock()
    gen.llm_generator.generate_mr_candidates.return_value = []
    return gen


@pytest.fixture
def repo(tmp_path):
    return MRRepository(repo_dir=str(tmp_path / "mr_repo"))


# ---------------------------------------------------------------------------
# ReLU 数学性质测试（无 mock，直接验证 PyTorch）
# ---------------------------------------------------------------------------

class TestReLUProperties:

    def setup_method(self):
        self.relu = torch.nn.functional.relu

    def test_non_negativity(self):
        x = torch.tensor([-3.0, 0.0, 3.0])
        assert torch.all(self.relu(x) >= 0)

    def test_idempotency(self):
        x = torch.tensor([-2.0, 0.0, 2.0])
        assert torch.allclose(self.relu(self.relu(x)), self.relu(x))

    def test_positive_scaling(self):
        x = torch.tensor([1.0, -1.0, 2.0])
        assert torch.allclose(self.relu(2 * x), 2 * self.relu(x))

    def test_monotonicity(self):
        x1 = torch.tensor([1.0, 2.0])
        x2 = torch.tensor([2.0, 3.0])
        assert torch.all(self.relu(x1) <= self.relu(x2))


# ---------------------------------------------------------------------------
# MR 生成阶段（generate_only）
# ---------------------------------------------------------------------------

class TestMRGeneration:

    def test_template_source_returns_list(self, generator):
        op_ir = OperatorIR(name="ReLU", inputs=[])
        mrs = generator.generate_only(
            operator_ir=op_ir,
            framework="pytorch",
            operator_doc="ReLU(x) = max(0, x)",
            auto_fetch_info=False,
            sources=["template"],
        )
        assert isinstance(mrs, list)
        for mr in mrs:
            assert isinstance(mr, MetamorphicRelation)
            assert mr.verified is False

    def test_llm_source_returns_mocked_mrs(self, generator):
        generator.llm_generator.generate_mr_candidates.return_value = [
            _mock_mr("mr_llm", "trans == 2 * orig", "linearity",
                     transform=lambda k: {**k, "input": 2 * k["input"]},
                     transform_code="lambda k: {**k, 'input': 2*k['input']}")
        ]
        op_ir = OperatorIR(name="TestOp", inputs=[])
        mrs = generator.generate_only(
            operator_ir=op_ir,
            framework="pytorch",
            operator_doc="test doc",
            auto_fetch_info=False,
            sources=["llm"],
        )
        assert len(mrs) == 1
        assert mrs[0].verified is False

    def test_auto_fetch_disabled_skips_fetcher(self, generator):
        mock_fetcher = MagicMock()
        generator.info_fetcher = mock_fetcher
        op_ir = OperatorIR(name="ReLU", inputs=[])
        generator.generate_only(
            operator_ir=op_ir,
            framework="pytorch",
            operator_doc="docs",
            auto_fetch_info=False,
            sources=["template"],
        )
        mock_fetcher.fetch_operator_info.assert_not_called()

    def test_auto_fetch_enabled_calls_fetcher(self, generator):
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_operator_info.return_value = {
            "name": "relu", "doc": "ReLU docs", "source_urls": []
        }
        generator.info_fetcher = mock_fetcher
        op_ir = OperatorIR(name="ReLU", inputs=[])
        generator.generate_only(
            operator_ir=op_ir,
            framework="pytorch",
            auto_fetch_info=True,
            sources=["template"],
        )
        mock_fetcher.fetch_operator_info.assert_called_once()


# ---------------------------------------------------------------------------
# MR 验证阶段（verify_mrs）
# ---------------------------------------------------------------------------

class TestMRVerification:

    def test_verify_with_precheck_returns_list(self, generator):
        mr = _mock_mr()
        op_ir = OperatorIR(name="TestOp", inputs=[])
        result = generator.verify_mrs(
            mrs=[mr],
            operator_ir=op_ir,
            framework="pytorch",
            operator_func=lambda x: x,
            use_precheck=True,
            use_sympy_proof=False,
        )
        assert isinstance(result, list)

    def test_verify_with_sympy_returns_list(self, generator):
        mr = _mock_mr()
        op_ir = OperatorIR(name="TestOp", inputs=[])
        result = generator.verify_mrs(
            mrs=[mr],
            operator_ir=op_ir,
            framework="pytorch",
            operator_code="def f(x):\n    return x\n",
            use_precheck=False,
            use_sympy_proof=True,
        )
        assert isinstance(result, list)

    def test_verify_empty_mrs_returns_empty(self, generator):
        op_ir = OperatorIR(name="TestOp", inputs=[])
        result = generator.verify_mrs(
            mrs=[],
            operator_ir=op_ir,
            framework="pytorch",
            use_precheck=False,
            use_sympy_proof=False,
        )
        assert result == []


# ---------------------------------------------------------------------------
# 完整工作流：生成 → 保存 → 加载 → 验证
# ---------------------------------------------------------------------------

class TestMRWorkflow:

    def test_save_and_load_mrs(self, generator, repo):
        generator.llm_generator.generate_mr_candidates.return_value = [_mock_mr()]
        op_ir = OperatorIR(name="WorkflowOp", inputs=[])
        mrs = generator.generate_only(
            operator_ir=op_ir,
            framework="pytorch",
            operator_doc="test",
            auto_fetch_info=False,
            sources=["llm"],
        )
        saved = repo.save("WorkflowOp", mrs)
        assert saved == len(mrs)
        loaded = repo.load("WorkflowOp")
        assert len(loaded) == len(mrs)
        assert all(not mr.verified for mr in loaded)

    def test_full_generate_pipeline(self, generator):
        op_ir = OperatorIR(name="ReLU", inputs=[])
        mrs = generator.generate(
            operator_ir=op_ir,
            framework="pytorch",
            operator_func=torch.nn.functional.relu,
            operator_doc="ReLU activation",
            auto_fetch_info=False,
            use_precheck=False,
            use_sympy_proof=False,
            sources=["template"],
        )
        assert isinstance(mrs, list)


# ---------------------------------------------------------------------------
# 模块导入健全性
# ---------------------------------------------------------------------------

class TestModuleImports:

    def test_operator_mr_generator_importable(self):
        from deepmt.mr_generator.operator.operator_mr_generator import OperatorMRGenerator
        assert OperatorMRGenerator is not None

    def test_operator_info_fetcher_importable(self):
        from deepmt.tools.web_search import OperatorInfoFetcher
        assert OperatorInfoFetcher is not None

    def test_web_search_modules_exist(self):
        base = Path(__file__).parent.parent.parent / "deepmt" / "tools" / "web_search"
        assert (base / "search_agent.py").exists()
        assert (base / "sphinx_search.py").exists()
        assert (base / "search_tool.py").exists()
        assert (base / "operator_fetcher.py").exists()

    def test_fetch_operator_info_via_generator(self):
        gen = OperatorMRGenerator()
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_operator_info.return_value = {
            "name": "relu", "doc": "ReLU docs", "source_urls": []
        }
        gen.info_fetcher = mock_fetcher
        result = gen.fetch_operator_info("relu", framework="pytorch")
        assert result["doc"] == "ReLU docs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
