"""
Demo D6：跨框架一致性测试（CrossFrameworkTester）

演示：
  1. 单算子跨框架对比（pytorch vs numpy）
  2. 查看 MR 级一致性统计与输出差异
  3. 保存并重新加载实验结果
  4. 批量对比多个算子
  5. 查看算子等价性声明表

运行前确保 MR 知识库中已有 relu/exp 等算子的 MR：
    deepmt mr generate torch.nn.functional.relu
    deepmt mr generate torch.exp

运行方式：
    source .venv/bin/activate
    PYTHONPATH=$(pwd) python demo/demo_cross_framework.py
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from deepmt.analysis.cross_framework_tester import (
    CrossConsistencyResult,
    CrossFrameworkTester,
    CrossSessionResult,
)
from deepmt.ir.schema import MetamorphicRelation
from deepmt.plugins.numpy_plugin import OPERATOR_EQUIVALENCE_MAP, NumpyPlugin


def _make_mr(desc: str, transform: str, oracle: str) -> MetamorphicRelation:
    return MetamorphicRelation(
        id=f"demo-{desc[:8]}",
        description=desc,
        transform_code=transform,
        oracle_expr=oracle,
        verified=True,
    )


def _mock_repo_with_mrs(mrs):
    """构造 mock MRRepository，返回指定 MR 列表。"""
    repo = MagicMock()
    repo.load.return_value = mrs
    return repo


def main():
    print("=" * 65)
    print("Demo D6：跨框架一致性测试（PyTorch vs NumPy）")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cr_dir = Path(tmp_dir) / "cross_results"

        # ── 1. 单算子对比（relu，两条 MR） ──────────────────────────────────
        print("\n[1] relu 跨框架对比（pytorch vs numpy）")

        relu_mrs = [
            _make_mr(
                "relu output is non-negative",
                "lambda k: {**k, 'input': k['input'] * 2}",
                "trans >= 0",
            ),
            _make_mr(
                "relu is piecewise linear for positive x",
                "lambda k: {**k, 'input': k['input'] * 2}",
                "trans == 2 * orig",
            ),
        ]

        tester = CrossFrameworkTester(
            repo=_mock_repo_with_mrs(relu_mrs),
            results_dir=str(cr_dir),
        )
        session = tester.compare_operator(
            "torch.nn.functional.relu",
            framework1="pytorch",
            framework2="numpy",
            n_samples=10,
        )

        print(f"\n  整体一致率: {session.overall_consistency_rate:.1%}")
        print(f"  输出最大差: {session.output_max_diff:.4g}  (float32 vs numpy float32)")
        print(f"  不一致 MR:  {session.inconsistent_mr_count}/{session.mr_count}")

        for r in session.mr_results:
            print(f"\n  MR: {r.mr_description}")
            print(f"    一致率   = {r.consistency_rate:.0%}")
            print(f"    f1 通过率 = {r.f1_pass_rate:.0%}  (pytorch)")
            print(f"    f2 通过率 = {r.f2_pass_rate:.0%}  (numpy)")
            print(f"    both_pass={r.both_pass}  only_f1={r.only_f1_pass}  "
                  f"only_f2={r.only_f2_pass}  both_fail={r.both_fail}")
            print(f"    输出差    = {r.output_max_diff:.4g}")

        # ── 2. 保存结果 ──────────────────────────────────────────────────────
        print("\n[2] 保存实验结果")
        path = tester.save(session)
        print(f"    已保存: {path.name}")

        # ── 3. 重新加载 ──────────────────────────────────────────────────────
        print("\n[3] 从磁盘加载历史实验结果")
        loaded_sessions = tester.load_all()
        print(f"    加载了 {len(loaded_sessions)} 条实验记录")
        for s in loaded_sessions:
            print(f"    {s.session_id}  {s.operator}  "
                  f"一致率={s.overall_consistency_rate:.0%}")

        # ── 4. 格式化文本报告 ────────────────────────────────────────────────
        print("\n[4] 格式化报告")
        print(tester.format_text(loaded_sessions))

        # ── 5. NumPy 与 PyTorch 数值精度差异展示 ────────────────────────────
        print("[5] 直接对比 PyTorch vs NumPy 数值差异（exp）")
        import torch
        x_np = np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float32)
        x_pt = torch.tensor(x_np)

        out_pt = torch.exp(x_pt).detach().numpy().astype(float)
        out_np = np.exp(x_np.astype(float))

        print(f"    input:    {x_np.tolist()}")
        print(f"    pytorch:  {list(map(lambda x: f'{x:.8f}', out_pt.tolist()))}")
        print(f"    numpy:    {list(map(lambda x: f'{x:.8f}', out_np.tolist()))}")
        print(f"    max_diff: {float(np.max(np.abs(out_pt - out_np))):.2e}")

        # ── 6. 算子等价性声明表 ──────────────────────────────────────────────
        print("\n[6] 算子等价性声明表（论文依据）")
        print(f"    共 {len(OPERATOR_EQUIVALENCE_MAP)} 个算子有 NumPy 等价实现：")
        for op, equiv in list(OPERATOR_EQUIVALENCE_MAP.items())[:6]:
            print(f"    {op:<45}  ↔  {equiv}")
        if len(OPERATOR_EQUIVALENCE_MAP) > 6:
            print(f"    ...（共 {len(OPERATOR_EQUIVALENCE_MAP)} 个）")

    print("\n" + "=" * 65)
    print("Demo 完成。实际使用：")
    print("  deepmt test cross torch.nn.functional.relu --save")
    print("  deepmt test cross torch.exp --n-samples 30 --json")
    print("=" * 65)


if __name__ == "__main__":
    main()
