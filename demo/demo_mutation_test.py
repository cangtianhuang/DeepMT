"""
Demo D4：变异测试（MutationTester）

演示对 torch.nn.functional.relu 注入 5 种变异，验证 MR 能否检出各类缺陷。
运行前确保 MR 知识库中已有 relu 的 MR（可先运行 deepmt mr generate torch.nn.functional.relu）。

运行方式：
    source .venv/bin/activate
    PYTHONPATH=$(pwd) python demo/demo_mutation_test.py
"""

from deepmt.analysis.reporting.mutation_tester import MutantType, MutationTester


def main():
    print("=" * 65)
    print("Demo D4：变异测试 — torch.nn.functional.relu")
    print("=" * 65)

    tester = MutationTester()

    # ── 指定变异类型运行单次测试 ──────────────────────────────────────────────
    print("\n[1] 单次变异测试（negate：输出取反）")
    result = tester.run(
        operator_name="torch.nn.functional.relu",
        mutant_type=MutantType.NEGATE_OUTPUT,
        framework="pytorch",
        n_samples=5,
    )
    print(f"    检出率: {result.detection_rate:.1%}  ({result.detected_cases}/{result.total_cases})")
    print(f"    已检出: {result.detected}")

    # ── 运行全部变异类型 ──────────────────────────────────────────────────────
    print("\n[2] 全部变异类型扫描（n_samples=5）")
    results = tester.run_all_mutants(
        operator_name="torch.nn.functional.relu",
        framework="pytorch",
        n_samples=5,
    )

    print(f"\n{'变异类型':<14}  {'检出':<6}  {'检出率':>7}  {'总用例':>6}")
    print("-" * 45)
    for r in results:
        mark = "✓" if r.detected else "✗"
        print(
            f"  {mark} {r.mutant_type.value:<12}  "
            f"{r.detected_cases}/{r.total_cases:<6}  "
            f"{r.detection_rate:>6.1%}  "
            f"err={r.errors}"
        )

    detected_count = sum(1 for r in results if r.detected)
    print("-" * 45)
    print(f"  总检出变异数: {detected_count}/{len(results)}")

    # ── 自定义参数（缩放系数可控） ────────────────────────────────────────────
    print("\n[3] 可控参数测试（scale 变异，k=3.0）")
    result_scale = tester.run(
        operator_name="torch.nn.functional.relu",
        mutant_type=MutantType.SCALE_WRONG,
        framework="pytorch",
        n_samples=5,
        scale=3.0,   # 可显式控制缩放系数
    )
    print(f"    scale=3.0  检出率: {result_scale.detection_rate:.1%}")

    print("\n" + "=" * 65)
    print("Demo 完成。若 MR 知识库为空，检出率均为 0（符合预期）。")
    print("=" * 65)


if __name__ == "__main__":
    main()
