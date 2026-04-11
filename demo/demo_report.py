"""
Demo D1：测试报告生成（ReportGenerator）

演示：
  1. 向临时数据库写入模拟测试结果（3 个算子，共 20 条记录）
  2. 生成全量报告（文本格式）
  3. 按算子过滤报告
  4. 查询失败案例列表
  5. 获取单算子 MR 级通过率分解
  6. 导出 JSON 格式摘要

运行方式：
    source .venv/bin/activate
    PYTHONPATH=$(pwd) python demo/demo_report.py
"""

import json
import tempfile
from pathlib import Path

from deepmt.analysis.report_generator import ReportGenerator
from deepmt.core.results_manager import ResultsManager
from deepmt.ir.schema import MetamorphicRelation, OracleResult, OperatorIR


def _make_mr(id: str, desc: str) -> MetamorphicRelation:
    return MetamorphicRelation(
        id=id,
        description=desc,
        transform_code="lambda k: k",
        oracle_expr="orig == trans",
    )


def _make_result(passed: bool, diff: float = 0.0, detail: str = "") -> OracleResult:
    return OracleResult(
        passed=passed,
        expr="orig == trans",
        actual_diff=diff,
        tolerance=1e-6,
        detail=detail,
    )


def _seed_results(rm: ResultsManager) -> None:
    """写入三个算子的模拟测试结果（共 20 条）。"""
    relu_ir = OperatorIR(name="torch.nn.functional.relu", input_specs=[])
    relu_mr1 = _make_mr("relu-mr-01", "relu output is non-negative")
    relu_mr2 = _make_mr("relu-mr-02", "relu is piecewise linear")

    # relu: mr1 — 3 pass / 2 fail; mr2 — 3 pass / 2 fail（共 10 条）
    rm.store_result(
        relu_ir,
        [(relu_mr1, _make_result(True))] * 3
        + [(relu_mr1, _make_result(False, 0.5, "NUMERICAL_DEVIATION: max_abs=0.500"))] * 2
        + [(relu_mr2, _make_result(True))] * 3
        + [(relu_mr2, _make_result(False, 0.1, "NUMERICAL_DEVIATION: max_abs=0.100"))] * 2,
        "pytorch",
    )

    # exp: 4 pass / 1 fail（共 5 条）
    exp_ir = OperatorIR(name="torch.exp", input_specs=[])
    exp_mr = _make_mr("exp-mr-01", "exp(a+b) == exp(a) * exp(b)")
    rm.store_result(
        exp_ir,
        [(exp_mr, _make_result(True))] * 4
        + [(exp_mr, _make_result(False, 0.01, "NUMERICAL_DEVIATION: max_abs=0.010"))],
        "pytorch",
    )

    # log: 5 pass / 0 fail（共 5 条）
    log_ir = OperatorIR(name="torch.log", input_specs=[])
    log_mr = _make_mr("log-mr-01", "log(a*b) == log(a) + log(b)")
    rm.store_result(
        log_ir,
        [(log_mr, _make_result(True))] * 5,
        "pytorch",
    )


def main():
    print("=" * 65)
    print("Demo D1：测试报告生成（ReportGenerator）")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "demo_results.db")
        rm = ResultsManager(db_path=db_path)

        # ── 1. 写入模拟测试结果 ───────────────────────────────────────────────
        print("\n[1] 写入模拟测试结果（3 算子，共 20 条）")
        _seed_results(rm)
        print("    relu: 10 条（6 pass / 4 fail）")
        print("    exp:   5 条（4 pass / 1 fail）")
        print("    log:   5 条（5 pass / 0 fail）")

        gen = ReportGenerator(results_manager=rm)

        # ── 2. 全量报告（文本） ───────────────────────────────────────────────
        print("\n[2] 全量报告（文本格式）")
        report = gen.generate()
        print(gen.format_text(report))

        # ── 3. 按算子过滤 ─────────────────────────────────────────────────────
        print("[3] 过滤：仅查看 relu")
        report_relu = gen.generate(operator="torch.nn.functional.relu")
        s = report_relu["summary"]
        print(f"    total={s['total_cases']}  pass={s['total_passed']}  "
              f"fail={s['total_failed']}  rate={s['pass_rate']:.1%}")

        # ── 4. 失败案例列表 ───────────────────────────────────────────────────
        print("\n[4] 失败案例列表（前 5 条）")
        failures = gen.get_failures(limit=5)
        print(gen.format_failure_text(failures))

        # ── 5. MR 级通过率分解 ────────────────────────────────────────────────
        print("[5] relu MR 级通过率分解")
        for m in gen.get_mr_breakdown("torch.nn.functional.relu"):
            print(f"    {m['description']:<40}  pass_rate={m['pass_rate']:.1%}")

        # ── 6. JSON 格式摘要 ──────────────────────────────────────────────────
        print("\n[6] JSON 格式（summary 字段）")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    print("\n" + "=" * 65)
    print("Demo 完成。实际使用：deepmt test report [--operator OP] [--json]")
    print("=" * 65)


if __name__ == "__main__":
    main()
