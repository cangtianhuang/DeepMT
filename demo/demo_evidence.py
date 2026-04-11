"""
Demo D3：可复现缺陷证据包（EvidenceCollector + BatchTestRunner）

演示：
  1. 用 BatchTestRunner 对 relu 批量测试，collect_evidence=True 捕获失败
  2. 查询证据包列表
  3. 展示证据包详情与可复现脚本

运行方式：
    source .venv/bin/activate
    PYTHONPATH=$(pwd) python demo/demo_evidence.py
"""

import tempfile
from pathlib import Path

import torch

from deepmt.analysis.evidence_collector import EvidenceCollector
from deepmt.engine.batch_test_runner import BatchTestRunner
from deepmt.ir.schema import MetamorphicRelation


def _make_failing_mr() -> MetamorphicRelation:
    """构造一条对正确 relu 必然失败的 MR（用于演示无需真实知识库）。

    变换：输入取反（-x）
    Oracle：relu(-x) == relu(x)  ← 对非零输入必然违反
      e.g. x=1: relu(1)=1, relu(-1)=0, 1 != 0
    """
    return MetamorphicRelation(
        id="demo-mr-relu-negate",
        description="relu(-x) should equal relu(x) [intentionally wrong MR for demo]",
        # 变换：输入取反
        transform_code="lambda k: {**k, 'input': -k['input']}",
        # oracle：relu(x) == relu(-x)，非零输入下必然失败
        oracle_expr="orig == trans",
        verified=True,
    )


def main():
    print("=" * 65)
    print("Demo D3：可复现缺陷证据包")
    print("=" * 65)

    # 使用临时目录存储证据包（不污染项目数据）
    with tempfile.TemporaryDirectory() as tmp_dir:
        ev_dir = Path(tmp_dir) / "evidence"
        collector = EvidenceCollector(evidence_dir=str(ev_dir))

        # ── 1. 注入必然失败的 MR，执行批量测试并捕获证据 ────────────────────
        print("\n[1] 运行批量测试（collect_evidence=True）")

        from unittest.mock import MagicMock
        mock_repo = MagicMock()
        mock_repo.load.return_value = [_make_failing_mr()]
        mock_rm = MagicMock()

        runner = BatchTestRunner(
            repo=mock_repo,
            results_manager=mock_rm,
            evidence_collector=collector,
        )

        def real_relu(**kwargs):
            return torch.relu(kwargs["input"])

        summary = runner.run_operator(
            operator_name="torch.nn.functional.relu",
            framework="pytorch",
            n_samples=3,
            operator_func=real_relu,
            collect_evidence=True,   # ← 显式开启证据捕获
        )

        print(f"    passed={summary.passed}  failed={summary.failed}  errors={summary.errors}")
        print(f"    证据包 ID: {summary.evidence_ids}")

        # ── 2. 查询证据包列表 ────────────────────────────────────────────────
        print("\n[2] 列出已保存的证据包")
        packs = collector.list_all()
        print(f"    共 {len(packs)} 个证据包")
        for p in packs:
            print(f"    {p.evidence_id}  diff={p.actual_diff:.4g}  {p.detail[:50]}")

        if not packs:
            print("    （无证据包：测试全部通过，或 collect_evidence=False）")
            return

        # ── 3. 展示第一个证据包详情 ──────────────────────────────────────────
        pack = packs[0]
        print("\n[3] 证据包详情")
        print(f"    evidence_id : {pack.evidence_id}")
        print(f"    算子        : {pack.operator}")
        print(f"    框架版本    : {pack.framework} {pack.framework_version}")
        print(f"    MR 描述     : {pack.mr_description}")
        print(f"    实测差值    : {pack.actual_diff:.6g}  (容忍: {pack.tolerance:.6g})")
        print(f"    输入形状    : {pack.input_summary.get('shape')}  "
              f"dtype={pack.input_summary.get('dtype')}")

        # ── 4. 打印可复现脚本 ────────────────────────────────────────────────
        print("\n[4] 可复现 Python 脚本（前 20 行）")
        for line in pack.reproduce_script.splitlines()[:20]:
            print(f"    {line}")

        # ── 5. 序列化往返验证 ────────────────────────────────────────────────
        print("\n[5] 序列化往返验证")
        from deepmt.analysis.evidence_collector import EvidencePack
        restored = EvidencePack.from_dict(pack.to_dict())
        assert restored.evidence_id == pack.evidence_id
        print(f"    ✓ to_dict → from_dict 往返正确（id={restored.evidence_id}）")

    print("\n" + "=" * 65)
    print("Demo 完成。实际使用：deepmt test batch --collect-evidence")
    print("=" * 65)


if __name__ == "__main__":
    main()
