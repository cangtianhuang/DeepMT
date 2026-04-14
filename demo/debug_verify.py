"""
debug_verify.py — 调试 MR 验证过程（阶段 3-4）

在 debug_generate.py 生成的候选 MR 基础上，分两步调试验证：
  Step 1 — Pre-check（数值随机测试，阶段3）：逐条显示每个 MR 的测试结果
  Step 2 — SymPy 证明（形式化验证，阶段4）：尝试符号证明并输出结果

示例算子：torch.nn.functional.relu
无 LLM / 无网络依赖

运行方式：
    source .venv/bin/activate
    PYTHONPATH=$(pwd) python demo/debug_verify.py
"""

import torch
import torch.nn.functional as F

from deepmt.ir.schema import OperatorIR
from deepmt.mr_generator.base.mr_templates import MRTemplatePool
from deepmt.analysis.verification.mr_prechecker import MRPreChecker
from deepmt.mr_generator.operator.operator_mr_generator import OperatorMRGenerator
from deepmt.mr_generator.operator.sympy_prover import SymPyProver
from deepmt.mr_generator.operator.sympy_translator import SympyTranslator

OPERATOR_NAME = "torch.nn.functional.relu"
OPERATOR_FUNC = F.relu

# ── 准备：生成候选 MR（与 debug_generate.py 相同，模板来源） ─────────────────
print("=" * 64)
print("准备: 生成 relu 的 MR 候选（模板来源，不验证）")
print("=" * 64)

pool = MRTemplatePool()
templates = pool.get_applicable_templates(OPERATOR_NAME, operator_func=OPERATOR_FUNC)
candidates = [pool.create_mr_from_template(t) for t in templates]

operator_ir = OperatorIR(name=OPERATOR_NAME)

print(f"候选 MR: {len(candidates)} 个")
for i, mr in enumerate(candidates, 1):
    print(f"  [{i}] {mr.description}  (verified={mr.verified})")

# ── Step 1: Pre-check — 逐条数值测试 ────────────────────────────────────────
print("\n" + "=" * 64)
print("Step 1: Pre-check — 每条 MR 随机数值测试（5 组，通过率 ≥ 80% 保留）")
print("=" * 64)

prechecker = MRPreChecker()
passed_mrs = []

for i, mr in enumerate(candidates, 1):
    ok, msg = prechecker.check_mr(
        operator_func=OPERATOR_FUNC,
        mr=mr,
        operator_ir=operator_ir,
        framework="pytorch",
    )
    status = "PASS ✓" if ok else "FAIL ✗"
    print(f"\n  [{i}] {mr.description}")
    print(f"       transform_code: {mr.transform_code}")
    print(f"       oracle_expr:    {mr.oracle_expr}")
    print(f"       结果: {status} — {msg}")
    if ok:
        mr.verified = True
        passed_mrs.append(mr)

print(f"\nPre-check 结果: {len(passed_mrs)}/{len(candidates)} 通过")

# ── Step 2: SymPy 形式化证明 ────────────────────────────────────────────────
print("\n" + "=" * 64)
print("Step 2: SymPy 证明 — 尝试对通过 pre-check 的 MR 进行符号验证")
print("=" * 64)

if not passed_mrs:
    print("  无 MR 通过 pre-check，跳过 SymPy 证明")
else:
    import inspect
    operator_code = inspect.getsource(OPERATOR_FUNC)

    translator = SympyTranslator()
    prover = SymPyProver(code_translator=translator)

    # 尝试将算子代码转换为 SymPy 表达式
    print(f"\n  算子源码长度: {len(operator_code)} chars")
    sympy_expr = None
    try:
        sympy_expr = translator.translate(
            code=operator_code,
            func=OPERATOR_FUNC,
            doc=None,
        )
        if sympy_expr is not None:
            print(f"  SymPy 转换成功: {sympy_expr}")
        else:
            print("  SymPy 转换返回 None（算子结构复杂，无法符号化）")
    except Exception as e:
        print(f"  SymPy 转换失败: {e}")

    if sympy_expr is not None:
        proven = prover.prove_mrs(
            mrs=passed_mrs,
            operator_func=OPERATOR_FUNC,
            operator_code=operator_code,
            operator_doc=None,
            operator_name=OPERATOR_NAME,
            num_inputs=1,
            sympy_expr=sympy_expr,
        )
        proven_ids = {mr.id for mr in proven}
        print(f"\n  SymPy 证明结果: {len(proven)}/{len(passed_mrs)} 通过\n")
        for mr in passed_mrs:
            proved = mr.id in proven_ids
            print(f"  {'PROVED ✓' if proved else 'UNPROVED ✗'}  {mr.description}")
    else:
        print("\n  跳过 SymPy 证明（无法转换为符号表达式）")
        print("  → 结论：pre-check 通过的 MR 视为最终验证结果")

# ── 汇总 ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("汇总")
print("=" * 64)
final_verified = [mr for mr in candidates if mr.verified]
print(f"候选: {len(candidates)}  pre-check通过: {len(passed_mrs)}  最终verified: {len(final_verified)}")
for mr in final_verified:
    print(f"  ✓ {mr.description}  [category={mr.category}]")
