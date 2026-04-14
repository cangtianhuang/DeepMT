"""
Demo D7：实验数据组织器（ExperimentOrganizer）

演示：
  1. 收集 RQ1 数据（MR 自动生成质量）
  2. 收集 RQ2 数据（缺陷检测能力）
  3. 收集 RQ3 数据（跨框架一致性）
  4. 收集 RQ4 数据（覆盖度与自动化程度）
  5. 整合全部 RQ 并格式化文本报告
  6. JSON 导出（可直接用于论文数据表）

运行方式：
    source .venv/bin/activate
    PYTHONPATH=$(pwd) python demo/demo_experiment.py
"""

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock

from deepmt.experiments.organizer import ExperimentOrganizer
from deepmt.analysis.qa.cross_framework_tester import (
    CrossConsistencyResult,
    CrossFrameworkTester,
    CrossSessionResult,
)
from deepmt.core.results_manager import ResultsManager
from deepmt.ir.schema import MetamorphicRelation, OracleResult, OperatorIR


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _make_mr(op_name="relu", category="activation", source="llm"):
    return MetamorphicRelation(
        id=str(uuid.uuid4()),
        description=f"{op_name} test MR",
        transform_code="lambda k: k",
        oracle_expr="orig == trans",
        category=category,
        source=source,
        verified=True,
    )


def _populate_db(db_path: str) -> None:
    """向数据库写入模拟测试结果（relu 7pass/3fail，exp 全pass，log 全pass）。"""
    rm = ResultsManager(db_path=db_path)

    def make_result(passed):
        return OracleResult(
            passed=passed,
            expr="orig == trans",
            actual_diff=0.0 if passed else 0.5,
            tolerance=1e-6,
            detail="" if passed else "NUMERICAL_DEVIATION: max_abs=0.500",
        )

    mr = MetamorphicRelation(id="mr-01", description="test", transform_code="", oracle_expr="")

    # relu: 10 cases, 7 pass, 3 fail  → 有缺陷线索
    relu_ir = OperatorIR(name="torch.nn.functional.relu", input_specs=[])
    rm.store_result(relu_ir, [(mr, make_result(True))] * 7 + [(mr, make_result(False))] * 3, "pytorch")

    # exp: 8 cases, all pass
    exp_ir = OperatorIR(name="torch.exp", input_specs=[])
    rm.store_result(exp_ir, [(mr, make_result(True))] * 8, "pytorch")

    # log: 6 cases, 5 pass, 1 fail
    log_ir = OperatorIR(name="torch.log", input_specs=[])
    rm.store_result(log_ir, [(mr, make_result(True))] * 5 + [(mr, make_result(False))], "pytorch")


def _make_cross_session(tmp_dir: str) -> None:
    """向 tmp_dir 写入两条跨框架实验结果。"""
    tester = CrossFrameworkTester(results_dir=tmp_dir)

    def make_ccr(op, both_pass, only_f1, only_f2, both_fail, diff):
        return CrossConsistencyResult(
            operator=op,
            framework1="pytorch",
            framework2="numpy",
            mr_id=str(uuid.uuid4()),
            mr_description="test MR",
            oracle_expr="orig == trans",
            n_samples=both_pass + only_f1 + only_f2 + both_fail,
            both_pass=both_pass,
            only_f1_pass=only_f1,
            only_f2_pass=only_f2,
            both_fail=both_fail,
            errors=0,
            output_max_diff=diff,
            output_mean_diff=diff * 0.5,
            output_close=diff < 1e-3,
        )

    # relu: 高一致率
    s1 = CrossSessionResult(
        session_id="demo-relu-001",
        timestamp="2026-04-11T10:00:00",
        operator="torch.nn.functional.relu",
        framework1="pytorch",
        framework2="numpy",
        n_samples=10,
        mr_results=[
            make_ccr("torch.nn.functional.relu", 9, 0, 0, 1, 1.2e-7),
            make_ccr("torch.nn.functional.relu", 8, 1, 0, 1, 3.1e-7),
        ],
    )
    tester.save(s1)

    # exp: 中等一致率（浮点差异略大）
    s2 = CrossSessionResult(
        session_id="demo-exp-001",
        timestamp="2026-04-11T10:05:00",
        operator="torch.exp",
        framework1="pytorch",
        framework2="numpy",
        n_samples=10,
        mr_results=[
            make_ccr("torch.exp", 7, 2, 1, 0, 4.5e-7),
        ],
    )
    tester.save(s2)


def main():
    print("=" * 65)
    print("Demo D7：实验数据组织器（ExperimentOrganizer）")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        ev_dir = str(Path(tmp) / "evidence")
        cr_dir = str(Path(tmp) / "cross_results")
        Path(ev_dir).mkdir()
        Path(cr_dir).mkdir()

        # 写入模拟数据
        _populate_db(db_path)
        _make_cross_session(cr_dir)

        # 构造 mock MR 仓库（模拟 RQ1 数据来源）
        mock_repo = MagicMock()
        mock_repo.get_statistics.return_value = {
            "total_mrs": 18,
            "verified_mrs": 15,
            "by_operator": {
                "torch.nn.functional.relu": {"total": 6},
                "torch.exp": {"total": 5},
                "torch.log": {"total": 4},
                "torch.tanh": {"total": 3},
            },
        }
        mock_repo.list_operators.return_value = [
            "torch.nn.functional.relu", "torch.exp", "torch.log", "torch.tanh"
        ]
        mock_repo.load.side_effect = lambda op: {
            "torch.nn.functional.relu": [_make_mr("relu", "activation", "llm")] * 4
                                       + [_make_mr("relu", "activation", "template")] * 2,
            "torch.exp":  [_make_mr("exp", "arithmetic", "llm")] * 3
                         + [_make_mr("exp", "arithmetic", "template")] * 2,
            "torch.log":  [_make_mr("log", "arithmetic", "llm")] * 3
                         + [_make_mr("log", "arithmetic", "manual")],
            "torch.tanh": [_make_mr("tanh", "activation", "llm")] * 3,
        }.get(op, [])

        org = ExperimentOrganizer(
            db_path=db_path,
            evidence_dir=ev_dir,
            cross_results_dir=cr_dir,
            mr_repo=mock_repo,
        )

        # ── 1. RQ1 ──────────────────────────────────────────────────────────
        print("\n[1] RQ1 — MR 自动生成质量")
        rq1 = org.collect_rq1()
        print(f"  MR 总数:          {rq1['total_mr_count']}")
        print(f"  已验证:           {rq1['verified_mr_count']}")
        print(f"  验证率:           {rq1['verification_rate']:.1%}")
        print(f"  覆盖算子数:       {rq1['operators_with_mr']}")
        print(f"  平均每算子 MR:    {rq1['avg_mr_per_operator']}")
        print(f"  分类分布:         {rq1['category_distribution']}")
        print(f"  来源分布:         {rq1['source_distribution']}")

        # ── 2. RQ2 ──────────────────────────────────────────────────────────
        print("\n[2] RQ2 — 缺陷检测能力")
        rq2 = org.collect_rq2()
        print(f"  总测试用例:       {rq2['total_test_cases']}")
        print(f"  通过:             {rq2['total_passed']}")
        print(f"  失败:             {rq2['total_failed']}")
        print(f"  通过率:           {rq2['overall_pass_rate']:.1%}")
        print(f"  被测算子数:       {rq2['operators_tested']}")
        print(f"  有失败的算子数:   {rq2['operators_with_failure']}")
        print(f"  证据包数量:       {rq2['evidence_pack_count']}")

        # ── 3. RQ3 ──────────────────────────────────────────────────────────
        print("\n[3] RQ3 — 跨框架一致性")
        rq3 = org.collect_rq3()
        print(f"  跨框架实验次数:   {rq3['cross_session_count']}")
        print(f"  对比算子数:       {rq3['operators_compared']}")
        print(f"  平均一致率:       {rq3.get('overall_consistency_rate', 0):.1%}")
        print(f"  框架对:           {rq3['framework_pairs']}")
        for s in rq3.get("sessions", []):
            print(f"    {s['operator']:<45}  一致率={s['consistency_rate']:.0%}")

        # ── 4. RQ4 ──────────────────────────────────────────────────────────
        print("\n[4] RQ4 — 覆盖度与自动化程度")
        rq4 = org.collect_rq4()
        print(f"  覆盖算子总数:     {rq4['operators_covered']}")
        print(f"  平均每算子 MR 数: {rq4['avg_mrs_per_operator']}")
        print(f"  平均每算子用例数: {rq4['test_density']}")
        print(f"  自动化范围:\n    {rq4['automation_scope']}")

        # ── 5. 完整报告 ──────────────────────────────────────────────────────
        print("\n[5] 完整格式化报告")
        data = org.collect_all()
        print(org.format_text(data))

        # ── 6. JSON 导出 ────────────────────────────────────────────────────
        print("[6] JSON 导出（节选）")
        json_str = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=True)
        # 只打印前 20 行
        for line in json_str.splitlines()[:20]:
            print(f"  {line}")
        print("  ...")
        print(f"  （完整 JSON 共 {len(json_str.splitlines())} 行）")

    print("\n" + "=" * 65)
    print("Demo 完成。实际使用：")
    print("  deepmt test experiment")
    print("  deepmt test experiment --rq 2")
    print("  deepmt test experiment --json > experiment_data.json")
    print("=" * 65)


if __name__ == "__main__":
    main()
