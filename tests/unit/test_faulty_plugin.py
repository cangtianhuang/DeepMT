"""
FaultyPyTorchPlugin 单元测试。

覆盖：
  - BUILTIN_FAULT_CATALOG 结构正确
  - _parse_env: 解析 "all" / 显式规格 / 空值
  - _resolve_specs: 直接传入 fault_specs
  - active_faults_from_env: 静态方法
  - _resolve_operator: 返回有缺陷的函数（实际数值验证）
  - backend_override: BatchTestRunner 接受并使用 FaultyPyTorchPlugin
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from deepmt.plugins.faulty_pytorch_plugin import (
    BUILTIN_FAULT_CATALOG,
    FaultyPyTorchPlugin,
)


# ── 辅助 fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def no_env(monkeypatch):
    """确保 DEEPMT_INJECT_FAULTS 未设置"""
    monkeypatch.delenv("DEEPMT_INJECT_FAULTS", raising=False)


@pytest.fixture
def env_all(monkeypatch):
    monkeypatch.setenv("DEEPMT_INJECT_FAULTS", "all")


@pytest.fixture
def env_explicit(monkeypatch):
    monkeypatch.setenv("DEEPMT_INJECT_FAULTS", "torch.nn.functional.relu:negate,torch.exp:scale")


# ── BUILTIN_FAULT_CATALOG ─────────────────────────────────────────────────────

class TestFaultCatalog:
    def test_catalog_not_empty(self):
        assert len(BUILTIN_FAULT_CATALOG) > 0

    def test_catalog_entries_have_correct_structure(self):
        for op, entry in BUILTIN_FAULT_CATALOG.items():
            assert len(entry) == 3, f"{op}: entry should be (mutant_type, kwargs, description)"
            mt, kwargs, desc = entry
            assert isinstance(mt, str)
            assert isinstance(kwargs, dict)
            assert isinstance(desc, str)

    def test_catalog_mutant_types_are_valid(self):
        from deepmt.analysis.reporting.mutation_tester import MutantType
        valid_types = {m.value for m in MutantType}
        for op, (mt, _, _) in BUILTIN_FAULT_CATALOG.items():
            assert mt in valid_types, f"{op}: invalid mutant type {mt!r}"

    def test_relu_in_catalog(self):
        assert "torch.nn.functional.relu" in BUILTIN_FAULT_CATALOG


# ── 环境变量解析 ──────────────────────────────────────────────────────────────

class TestEnvParsing:
    def test_no_env_returns_empty(self, no_env):
        plugin = FaultyPyTorchPlugin()
        assert plugin._active_faults == {}

    def test_env_all_activates_all_catalog(self, env_all):
        plugin = FaultyPyTorchPlugin()
        assert len(plugin._active_faults) == len(BUILTIN_FAULT_CATALOG)
        for op in BUILTIN_FAULT_CATALOG:
            assert op in plugin._active_faults

    def test_env_explicit_spec(self, env_explicit):
        plugin = FaultyPyTorchPlugin()
        assert "torch.nn.functional.relu" in plugin._active_faults
        assert "torch.exp" in plugin._active_faults
        assert plugin._active_faults["torch.nn.functional.relu"][0] == "negate"
        assert plugin._active_faults["torch.exp"][0] == "scale"

    def test_env_single_op_uses_catalog_default(self, monkeypatch):
        monkeypatch.setenv("DEEPMT_INJECT_FAULTS", "torch.nn.functional.relu")
        plugin = FaultyPyTorchPlugin()
        assert "torch.nn.functional.relu" in plugin._active_faults
        # 应使用目录默认变异类型
        catalog_mt = BUILTIN_FAULT_CATALOG["torch.nn.functional.relu"][0]
        assert plugin._active_faults["torch.nn.functional.relu"][0] == catalog_mt


# ── fault_specs 构造参数 ──────────────────────────────────────────────────────

class TestFaultSpecs:
    def test_explicit_specs_override_env(self, env_all):
        # fault_specs 传入空字典，即使 env=all 也不激活缺陷
        plugin = FaultyPyTorchPlugin(fault_specs={})
        assert plugin._active_faults == {}

    def test_explicit_specs_sets_active_faults(self, no_env):
        specs = {"torch.nn.functional.relu": "negate"}
        plugin = FaultyPyTorchPlugin(fault_specs=specs)
        assert "torch.nn.functional.relu" in plugin._active_faults
        assert plugin._active_faults["torch.nn.functional.relu"][0] == "negate"

    def test_multiple_specs(self, no_env):
        specs = {
            "torch.nn.functional.relu": "negate",
            "torch.exp": "add_const",
        }
        plugin = FaultyPyTorchPlugin(fault_specs=specs)
        assert len(plugin._active_faults) == 2


# ── _resolve_operator (数值验证) ──────────────────────────────────────────────

class TestResolveOperator:
    def test_clean_plugin_returns_correct_relu(self, no_env):
        plugin = FaultyPyTorchPlugin(fault_specs={})
        relu = plugin._resolve_operator("torch.nn.functional.relu")
        x = torch.tensor([-1.0, 0.0, 1.0])
        result = relu(input=x)
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(result, expected)

    def test_negate_fault_negates_output(self, no_env):
        plugin = FaultyPyTorchPlugin(fault_specs={"torch.nn.functional.relu": "negate"})
        faulty_relu = plugin._resolve_operator("torch.nn.functional.relu")
        x = torch.tensor([1.0, 2.0, 3.0])
        result = faulty_relu(input=x)
        # 正常 relu 应返回 [1, 2, 3]，取反后 [-1, -2, -3]
        assert (result < 0).all(), "Negate fault should produce negative outputs for positive inputs"

    def test_add_const_fault_adds_bias(self, no_env):
        plugin = FaultyPyTorchPlugin(fault_specs={"torch.exp": "add_const"})
        faulty_exp = plugin._resolve_operator("torch.exp")
        x = torch.tensor([0.0])
        result = faulty_exp(input=x)
        real_exp = torch.exp(x)
        # 结果应该与真实值不同（有偏置）
        assert not torch.allclose(result, real_exp)

    def test_unaffected_operator_returns_real_func(self, no_env):
        plugin = FaultyPyTorchPlugin(fault_specs={"torch.exp": "negate"})
        # relu 不在 fault_specs 中，应返回真实函数
        relu = plugin._resolve_operator("torch.nn.functional.relu")
        x = torch.tensor([-1.0, 1.0])
        result = relu(input=x)
        assert torch.allclose(result, torch.tensor([0.0, 1.0]))

    def test_invalid_operator_raises_value_error(self, no_env):
        plugin = FaultyPyTorchPlugin(fault_specs={})
        with pytest.raises(ValueError, match="Cannot resolve"):
            plugin._resolve_operator("torch.nonexistent.operator.xyz")


# ── active_faults_from_env 静态方法 ──────────────────────────────────────────

class TestActiveFaultsFromEnv:
    def test_returns_empty_when_no_env(self, no_env):
        result = FaultyPyTorchPlugin.active_faults_from_env()
        assert result == {}

    def test_returns_all_catalog_when_all(self, env_all):
        result = FaultyPyTorchPlugin.active_faults_from_env()
        assert len(result) == len(BUILTIN_FAULT_CATALOG)

    def test_explicit_spec_parsed_correctly(self, monkeypatch):
        monkeypatch.setenv("DEEPMT_INJECT_FAULTS", "torch.relu:negate")
        result = FaultyPyTorchPlugin.active_faults_from_env()
        assert result == {"torch.relu": "negate"}


# ── list_catalog 静态方法 ─────────────────────────────────────────────────────

class TestListCatalog:
    def test_returns_dict_with_correct_format(self):
        catalog = FaultyPyTorchPlugin.list_catalog()
        assert isinstance(catalog, dict)
        for op, (mt, desc) in catalog.items():
            assert isinstance(mt, str)
            assert isinstance(desc, str)

    def test_same_operators_as_builtin(self):
        catalog = FaultyPyTorchPlugin.list_catalog()
        assert set(catalog.keys()) == set(BUILTIN_FAULT_CATALOG.keys())


# ── BatchTestRunner backend_override ─────────────────────────────────────────

class TestBatchTestRunnerBackendOverride:
    """验证 BatchTestRunner 能接受 FaultyPyTorchPlugin 作为 backend_override。"""

    def test_backend_override_is_used(self, no_env):
        from deepmt.engine.batch_test_runner import BatchTestRunner
        from deepmt.ir.schema import MetamorphicRelation

        # 构造 MR：negate MR 对正常 relu 会失败，但对含负值缺陷的 relu 也可能失败
        mr = MetamorphicRelation(
            id="test-mr",
            description="non-negativity: trans >= 0",
            transform_code="lambda k: {**k, 'input': -k['input']}",
            oracle_expr="trans >= 0",
            verified=True,
        )

        mock_repo = MagicMock()
        mock_repo.load.return_value = [mr]
        mock_rm = MagicMock()

        faulty_plugin = FaultyPyTorchPlugin(fault_specs={"torch.nn.functional.relu": "negate"})

        runner = BatchTestRunner(
            repo=mock_repo,
            results_manager=mock_rm,
            backend_override=faulty_plugin,
        )

        def faulty_relu(**kwargs):
            return -torch.relu(kwargs["input"])

        summary = runner.run_operator(
            operator_name="torch.nn.functional.relu",
            framework="pytorch",
            n_samples=3,
            operator_func=faulty_relu,
        )

        # 不论通过与否，流程应正常完成
        assert summary.operator == "torch.nn.functional.relu"
        assert summary.n_samples == 3
