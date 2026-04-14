"""
Demo D5：开放测试（FaultyPyTorchPlugin + BatchTestRunner）

演示三种缺陷注入方式：
  A. 构造参数 fault_specs（代码级精确控制）
  B. 环境变量 DEEPMT_INJECT_FAULTS（命令行/CI 控制）
  C. 内置缺陷目录（all 模式，最大覆盖）

运行方式：
    source .venv/bin/activate
    PYTHONPATH=$(pwd) python demo/demo_open_test.py

    # 也可通过环境变量控制（会覆盖脚本内的 fault_specs）：
    DEEPMT_INJECT_FAULTS=all python demo/demo_open_test.py
"""

import os
import tempfile
from unittest.mock import MagicMock

import torch

from deepmt.analysis.reporting.evidence_collector import EvidenceCollector
from deepmt.engine.batch_test_runner import BatchTestRunner
from deepmt.ir.schema import MetamorphicRelation
from deepmt.plugins.faulty_pytorch_plugin import (
    BUILTIN_FAULT_CATALOG,
    FaultyPyTorchPlugin,
)


def _make_nonneg_mr() -> MetamorphicRelation:
    """非负性 MR：relu 输出 >= 0。对取反缺陷必然失败。"""
    return MetamorphicRelation(
        id="demo-mr-nonneg",
        description="relu output should be non-negative",
        transform_code="lambda k: {**k, 'input': -k['input']}",
        oracle_expr="trans >= 0",
        verified=True,
    )


def _run_with_plugin(plugin, mr, n_samples=3):
    """用给定插件跑批量测试，返回 OperatorTestSummary。"""
    mock_repo = MagicMock()
    mock_repo.load.return_value = [mr]
    mock_rm = MagicMock()

    runner = BatchTestRunner(
        repo=mock_repo,
        results_manager=mock_rm,
        backend_override=plugin,   # ← 注入缺陷插件
    )

    def relu_via_plugin(**kwargs):
        """使用插件的 _resolve_operator 取得（可能有缺陷的）函数。"""
        faulty_func = plugin._resolve_operator("torch.nn.functional.relu")
        return faulty_func(**kwargs)

    return runner.run_operator(
        operator_name="torch.nn.functional.relu",
        framework="pytorch",
        n_samples=n_samples,
        operator_func=relu_via_plugin,
    )


def main():
    print("=" * 65)
    print("Demo D5：开放测试 — FaultyPyTorchPlugin 缺陷注入")
    print("=" * 65)

    mr = _make_nonneg_mr()

    # ── A. 构造参数 fault_specs（精确控制，适合脚本化实验） ──────────────────
    print("\n[A] fault_specs 参数注入（代码级控制）")
    plugin_a = FaultyPyTorchPlugin(
        fault_specs={"torch.nn.functional.relu": "negate"}   # 显式指定：取反输出
    )
    summary_a = _run_with_plugin(plugin_a, mr)
    print(f"    缺陷: relu→negate  失败={summary_a.failed}/{summary_a.total_cases}  "
          f"检出={'✓' if summary_a.failed > 0 else '✗'}")

    # ── B. 环境变量控制（适合 CLI 和 CI 管道） ──────────────────────────────
    print("\n[B] 环境变量 DEEPMT_INJECT_FAULTS 控制")
    os.environ["DEEPMT_INJECT_FAULTS"] = "torch.nn.functional.relu:add_const"
    plugin_b = FaultyPyTorchPlugin()   # 无参数，自动读取环境变量
    del os.environ["DEEPMT_INJECT_FAULTS"]  # 还原环境

    summary_b = _run_with_plugin(plugin_b, mr)
    print(f"    缺陷: relu→add_const  失败={summary_b.failed}/{summary_b.total_cases}  "
          f"检出={'✓' if summary_b.failed > 0 else '✗'}")

    # ── C. 不注入缺陷（对照组：正常插件应全部通过） ──────────────────────────
    print("\n[C] 无缺陷对照组（fault_specs={}）")
    plugin_c = FaultyPyTorchPlugin(fault_specs={})   # 空字典 = 正常行为

    def clean_relu(**kwargs):
        return torch.relu(kwargs["input"])

    mock_repo = MagicMock()
    mock_repo.load.return_value = [mr]
    mock_rm = MagicMock()
    runner_c = BatchTestRunner(repo=mock_repo, results_manager=mock_rm)
    summary_c = runner_c.run_operator(
        "torch.nn.functional.relu", "pytorch", n_samples=3, operator_func=clean_relu
    )
    print(f"    无缺陷对照  失败={summary_c.failed}/{summary_c.total_cases}  "
          f"（期望全部通过：正确 relu 的输出满足 trans >= 0）")

    # ── D. 内置缺陷目录展示 ──────────────────────────────────────────────────
    print("\n[D] 内置缺陷目录（BUILTIN_FAULT_CATALOG）")
    catalog = FaultyPyTorchPlugin.list_catalog()
    print(f"    共 {len(catalog)} 个算子缺陷预设：")
    for op, (mt, desc) in catalog.items():
        print(f"      {op:<45}  [{mt}]")

    print("\n" + "=" * 65)
    print("Demo 完成。实际使用：")
    print("  DEEPMT_INJECT_FAULTS=all deepmt test open --collect-evidence")
    print("  deepmt test open --inject-faults torch.exp:scale --n-samples 20")
    print("=" * 65)


if __name__ == "__main__":
    main()
