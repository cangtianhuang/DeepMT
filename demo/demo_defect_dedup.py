"""
Demo D2：缺陷线索去重（DefectDeduplicator）

演示：
  1. 模拟多批次测试产生的重复证据包（同一失败模式出现多次）
  2. 用 DefectDeduplicator 聚类去重
  3. 查看缺陷线索输出与可读报告

运行方式：
    source .venv/bin/activate
    PYTHONPATH=$(pwd) python demo/demo_defect_dedup.py
"""

import tempfile
from pathlib import Path

import numpy as np

from deepmt.analysis.qa.defect_deduplicator import DefectDeduplicator
from deepmt.analysis.reporting.evidence_collector import EvidenceCollector


def _seed_evidence(collector: EvidenceCollector):
    """
    模拟三类失败模式，每类出现多次：
      A. relu / 非负性 MR / 数值偏差（出现 4 次）
      B. relu / 线性 MR    / 数值偏差（出现 2 次）
      C. exp  / 线性 MR    / 数值偏差（出现 3 次）
    共 9 个证据包 → 应去重为 3 条缺陷线索
    """
    arr = np.array([1.0, -1.0, 0.5])

    # 模式 A：4 次
    for i in range(4):
        pack = collector.create(
            operator="torch.nn.functional.relu",
            framework="pytorch",
            mr_id="mr-relu-nonneg-001",
            mr_description="relu output should be non-negative",
            transform_code="lambda k: {**k, 'input': -k['input']}",
            oracle_expr="trans >= 0",
            input_tensor=arr * (i + 1),
            actual_diff=float(0.5 + i * 0.1),
            tolerance=1e-6,
            detail=f"NUMERICAL_DEVIATION: max_abs={0.5 + i * 0.1:.3f}",
        )
        collector.save(pack)

    # 模式 B：2 次
    for i in range(2):
        pack = collector.create(
            operator="torch.nn.functional.relu",
            framework="pytorch",
            mr_id="mr-relu-linear-002",
            mr_description="relu is piecewise linear",
            transform_code="lambda k: {**k, 'input': 2*k['input']}",
            oracle_expr="trans == 2 * orig",
            input_tensor=arr,
            actual_diff=float(0.1 + i * 0.05),
            tolerance=1e-6,
            detail=f"NUMERICAL_DEVIATION: max_abs={0.1 + i * 0.05:.3f}",
        )
        collector.save(pack)

    # 模式 C：3 次
    for i in range(3):
        pack = collector.create(
            operator="torch.exp",
            framework="pytorch",
            mr_id="mr-exp-additive-003",
            mr_description="exp(a+b) == exp(a) * exp(b)",
            transform_code="lambda k: {**k, 'input': k['input'] + 1}",
            oracle_expr="trans == orig * exp(1)",
            input_tensor=arr,
            actual_diff=float(0.001 + i * 0.0005),
            tolerance=1e-6,
            detail=f"NUMERICAL_DEVIATION: max_abs={0.001 + i * 0.0005:.5f}",
        )
        collector.save(pack)


def main():
    print("=" * 65)
    print("Demo D2：缺陷线索去重")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ev_dir = Path(tmp_dir) / "evidence"
        collector = EvidenceCollector(evidence_dir=str(ev_dir))

        # ── 1. 模拟多批次测试产生重复证据包 ──────────────────────────────────
        print("\n[1] 模拟 3 类失败模式（共 9 个证据包）")
        _seed_evidence(collector)
        print(f"    已保存证据包数: {collector.count()}")

        # ── 2. 去重 ──────────────────────────────────────────────────────────
        print("\n[2] 运行 DefectDeduplicator.deduplicate()")
        dedup = DefectDeduplicator(evidence_dir=str(ev_dir))
        leads = dedup.deduplicate()
        print(f"    9 个证据包 → {len(leads)} 条独立缺陷线索（期望 3）")

        # ── 3. 逐条展示 ──────────────────────────────────────────────────────
        print("\n[3] 缺陷线索明细")
        for i, lead in enumerate(leads, 1):
            print(f"\n  [{i}] {lead.lead_id}")
            print(f"       算子:  {lead.operator}")
            print(f"       MR:    {lead.mr_description}")
            print(f"       类型:  {lead.error_bucket.upper()}")
            print(f"       出现:  {lead.occurrence_count} 次")
            print(f"       证据:  {lead.representative_evidence_id}")

        # ── 4. 格式化报告 ────────────────────────────────────────────────────
        print("\n[4] 格式化文本报告")
        print(dedup.format_text(leads))

        # ── 5. 按算子过滤 ────────────────────────────────────────────────────
        print("\n[5] 按算子过滤（只看 relu）")
        relu_leads = dedup.deduplicate(operator="torch.nn.functional.relu")
        print(f"    relu 相关缺陷线索: {len(relu_leads)} 条（期望 2）")

    print("\n" + "=" * 65)
    print("Demo 完成。实际使用：deepmt test dedup [--operator OP]")
    print("=" * 65)


if __name__ == "__main__":
    main()
