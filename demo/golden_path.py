"""
DeepMT 黄金演示路径（Golden Demo Path）

演示目标：
  在一次运行中展示 DeepMT 的核心研究价值：
  "自动生成蜕变关系 → 批量执行测试 → 检测框架缺陷 → 输出可复现证据"

演示主链（5步）：
  Step 1: 算子目录与 MR 知识库  — 展示已有算子和已验证 MR
  Step 2: 正常批量测试           — 正常 PyTorch 全部通过（基线）
  Step 3: 开放测试（缺陷注入）  — FaultyPyTorchPlugin 暴露预设缺陷
  Step 4: 测试报告               — 汇总通过率、失败分布
  Step 5: 可复现证据包           — 完整复现脚本，可直接粘贴运行

前提条件：
  - 不需要 LLM API（使用 data/knowledge/mr_repository/ 中已生成的 MR）
  - 不需要网络（所有依赖离线可用）
  - 需要 PyTorch（pip install torch）

运行方式：
  source .venv/bin/activate
  PYTHONPATH=$(pwd) python demo/golden_path.py

等价 CLI 命令序列见：docs/demo_golden_path.md
"""

import tempfile
from pathlib import Path

import torch

from deepmt.analysis.evidence_collector import EvidenceCollector
from deepmt.analysis.report_generator import ReportGenerator
from deepmt.core.results_manager import ResultsManager
from deepmt.engine.batch_test_runner import BatchTestRunner
from deepmt.mr_generator.base.mr_repository import MRRepository
from deepmt.plugins.faulty_pytorch_plugin import BUILTIN_FAULT_CATALOG, FaultyPyTorchPlugin

# ── 演示算子集（使用知识库中已有 MR 的算子）──────────────────────────────────
DEMO_OPERATORS = [
    "torch.nn.functional.relu",
    "torch.exp",
    "torch.abs",
]

# ── 演示使用的缺陷算子（FaultyPlugin 中有内置缺陷）─────────────────────────────
FAULTY_OPERATORS = [op for op in DEMO_OPERATORS if op in BUILTIN_FAULT_CATALOG]

_SEP = "─" * 68


def _header(step: str, title: str) -> None:
    print(f"\n{'═' * 68}")
    print(f"  {step}  {title}")
    print(f"{'═' * 68}")


def _sub(text: str) -> None:
    print(f"\n▶ {text}")


def step1_catalog_and_repository() -> None:
    """Step 1: 展示算子目录与 MR 知识库概况"""
    _header("Step 1 / 5", "算子目录与 MR 知识库")

    repo = MRRepository()

    _sub("演示算子集（已有 MR 的算子）:")
    total_mr = 0
    for op in DEMO_OPERATORS:
        mrs = repo.load(op)
        verified = sum(1 for m in mrs if m.verified)
        print(f"  {op:<45}  MR数={len(mrs)}  已验证={verified}")
        total_mr += len(mrs)
        for m in mrs:
            status = "✓" if m.verified else "·"
            print(f"      [{status}] {m.description}")

    print(f"\n  合计: {len(DEMO_OPERATORS)} 个算子，{total_mr} 条 MR")

    _sub("内置缺陷目录（FaultyPyTorchPlugin 预设缺陷）:")
    for op, (mutant_type, _, description) in BUILTIN_FAULT_CATALOG.items():
        marker = " ← 演示算子" if op in DEMO_OPERATORS else ""
        print(f"  {op:<45}  [{mutant_type}]{marker}")
        print(f"      {description[:65]}")


def step2_normal_batch_test(tmp_dir: str) -> ResultsManager:
    """Step 2: 正常批量测试（基线，全部应通过）"""
    _header("Step 2 / 5", "正常批量测试（基线验证）")
    _sub(f"框架: pytorch（正常实现）  算子: {len(DEMO_OPERATORS)} 个  每条MR样本数: 8")

    rm = ResultsManager(db_path=str(Path(tmp_dir) / "demo_results.db"))
    runner = BatchTestRunner(results_manager=rm)

    all_pass = True
    for op in DEMO_OPERATORS:
        summary = runner.run_operator(op, "pytorch", n_samples=8)
        if summary.mr_count == 0:
            print(f"  [SKIP] {op}  (知识库无 MR，请先运行 deepmt mr batch-generate)")
            continue
        status = "PASS" if summary.failed == 0 and summary.errors == 0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"  [{status}] {op}"
            f"  MR={summary.mr_count}"
            f"  pass={summary.passed}/{summary.total_cases}"
        )
        for m in summary.mr_summaries:
            mark = "✓" if m.failed == 0 else "✗"
            print(f"      [{mark}] {m.description[:55]}  {m.passed}/{m.total}")

    print(_SEP)
    if all_pass:
        print("  结论：正常 PyTorch 实现在所有 MR 上全部通过 ✓")
    else:
        print("  警告：正常实现出现失败，请检查 MR 定义或 PyTorch 版本。")

    return rm


def step3_open_test_faulty(tmp_dir: str) -> ResultsManager:
    """Step 3: 开放测试——向算子注入预设缺陷，验证 MR 检测能力"""
    _header("Step 3 / 5", "开放测试（缺陷注入 + 检测）")
    _sub(f"框架: FaultyPyTorchPlugin（注入预设缺陷）  算子: {len(FAULTY_OPERATORS)} 个")

    rm_faulty = ResultsManager(db_path=str(Path(tmp_dir) / "demo_faulty_results.db"))
    ec = EvidenceCollector(evidence_dir=str(Path(tmp_dir) / "evidence"))
    faulty_plugin = FaultyPyTorchPlugin(
        fault_specs={op: mt for op, (mt, _, _) in BUILTIN_FAULT_CATALOG.items()}
    )
    runner = BatchTestRunner(
        results_manager=rm_faulty,
        evidence_collector=ec,
        backend_override=faulty_plugin,
    )

    detected_total = 0
    for op in FAULTY_OPERATORS:
        summary = runner.run_operator(op, "pytorch", n_samples=8, collect_evidence=True)
        if summary.mr_count == 0:
            print(f"  [SKIP] {op}  (无 MR)")
            continue
        detected = summary.failed > 0
        if detected:
            detected_total += 1
        mark = "DETECTED" if detected else "MISSED"
        print(
            f"  [{mark}] {op}"
            f"  pass={summary.passed}/{summary.total_cases}"
            f"  failed={summary.failed}"
            + (f"  证据: {summary.evidence_ids}" if summary.evidence_ids else "")
        )
        for m in summary.mr_summaries:
            mark2 = "✗ 缺陷暴露" if m.failed > 0 else "· 未触发"
            print(f"      [{mark2}] {m.description[:55]}")

    print(_SEP)
    print(
        f"  结论：{detected_total}/{len(FAULTY_OPERATORS)} 个缺陷算子被 MR 成功检出"
        + (" ✓" if detected_total > 0 else " — MR 未能覆盖缺陷，需补充 MR")
    )

    return rm_faulty


def step4_report(rm_normal: ResultsManager, rm_faulty: ResultsManager) -> None:
    """Step 4: 测试报告汇总"""
    _header("Step 4 / 5", "测试报告")

    _sub("正常实现报告:")
    gen_normal = ReportGenerator(results_manager=rm_normal)
    report_normal = gen_normal.generate(framework="pytorch")
    s = report_normal["summary"]
    print(f"  算子数={s['total_operators']}  总用例={s['total_cases']}"
          f"  通过={s['total_passed']}  失败={s['total_failed']}"
          f"  通过率={s['pass_rate']:.1%}")

    _sub("缺陷注入报告:")
    gen_faulty = ReportGenerator(results_manager=rm_faulty)
    report_faulty = gen_faulty.generate(framework="pytorch")
    s2 = report_faulty["summary"]
    print(f"  算子数={s2['total_operators']}  总用例={s2['total_cases']}"
          f"  通过={s2['total_passed']}  失败={s2['total_failed']}"
          f"  通过率={s2['pass_rate']:.1%}")

    _sub("MR 级通过率分解（缺陷注入 — relu）:")
    breakdown = gen_faulty.get_mr_breakdown("torch.nn.functional.relu")
    if breakdown:
        for m in breakdown:
            print(f"  {m['description']:<50}  pass_rate={m['pass_rate']:.1%}")
    else:
        print("  (relu 无测试记录，可能未注入缺陷)")

    _sub("失败案例（缺陷注入，前 3 条）:")
    failures = gen_faulty.get_failures(limit=3)
    if failures:
        for rec in failures:
            print(f"  [{rec.get('ir_name','?')}] MR={rec.get('mr_description','?')[:40]}")
            if rec.get("defect_details"):
                print(f"    详情: {rec['defect_details'][:70]}")
    else:
        print("  (无失败记录，缺陷未被 MR 捕获)")


def step5_evidence(tmp_dir: str) -> None:
    """Step 5: 可复现证据包"""
    _header("Step 5 / 5", "可复现证据包（Reproducible Evidence）")

    ec = EvidenceCollector(evidence_dir=str(Path(tmp_dir) / "evidence"))
    packs = ec.list_all()

    if not packs:
        print("  ⚠  本次演示未产生证据包（所有 MR 均通过，或知识库无可用 MR）。")
        print("  提示: 可在 Step 3 中查看到的失败记录对应算子运行:")
        print("        deepmt test open --operator torch.nn.functional.relu \\")
        print("                         --collect-evidence --n-samples 20")
        return

    _sub(f"发现 {len(packs)} 个证据包:")
    for p in packs[:3]:
        print(f"  ID={p.evidence_id[:8]}…  算子={p.operator}  MR={p.mr_description[:40]}")
        print(f"  缺陷描述: {p.detail[:70]}")

    # 展示第一个证据包的可复现脚本
    first = packs[0]
    script = first.reproduce_script
    if script:
        _sub(f"证据包 {first.evidence_id[:8]}… 的可复现 Python 脚本（可直接粘贴运行）:")
        print(_SEP)
        # 只打印脚本前 35 行，避免输出过长
        lines = script.strip().splitlines()
        for line in lines[:35]:
            print(line)
        if len(lines) > 35:
            print(f"  … (共 {len(lines)} 行，完整脚本见证据包文件)")
        print(_SEP)


def main() -> None:
    print("╔" + "═" * 66 + "╗")
    print("║  DeepMT 黄金演示路径 (Golden Demo Path)                       ║")
    print("║  Deep Metamorphic Testing for DL Frameworks                  ║")
    print(f"║  PyTorch {torch.__version__:<57}║")
    print("╚" + "═" * 66 + "╝")
    print()
    print("演示主链：算子目录 → MR 知识库 → 批量测试 → 缺陷检测 → 证据输出")
    print("无需 LLM API，使用已预先生成并验证的蜕变关系。")

    with tempfile.TemporaryDirectory(prefix="deepmt_demo_") as tmp_dir:
        # Step 1
        step1_catalog_and_repository()

        # Step 2
        rm_normal = step2_normal_batch_test(tmp_dir)

        # Step 3
        rm_faulty = step3_open_test_faulty(tmp_dir)

        # Step 4
        step4_report(rm_normal, rm_faulty)

        # Step 5
        step5_evidence(tmp_dir)

    print(f"\n{'═' * 68}")
    print("  演示完成。")
    print()
    print("  下一步：")
    print("  ① 生成更多 MR:   deepmt mr batch-generate --framework pytorch")
    print("  ② 全量批量测试:  deepmt test batch --framework pytorch")
    print("  ③ 开放缺陷测试:  deepmt test open --inject-faults all --collect-evidence")
    print("  ④ 查看报告:      deepmt test report")
    print("  ⑤ 查看证据包:    deepmt test evidence list")
    print("  ⑥ Web 仪表盘:    deepmt ui start")
    print(f"{'═' * 68}\n")


if __name__ == "__main__":
    main()
