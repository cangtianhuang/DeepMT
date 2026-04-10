"""
demo_sympy_proof.py
演示 SymPy 证明流水线对 torch.exp 的三层问题：
  1. inspect.getsource(torch.exp) 失败（C 扩展，无 Python 源码）
  2. 当前流水线静默跳过证明而不报告原因
  3. 直接提供 SymPy 表达式后，证明可以运行——但 oracle 近似值暴露第二层问题

运行：
    cd <project-root>
    source .venv/bin/activate && PYTHONPATH=$(pwd) python demo/demo_sympy_proof.py
"""

import inspect
import sympy as sp
import torch

from deepmt.ir.schema import MetamorphicRelation
from deepmt.mr_generator.operator.sympy_prover import SymPyProver


def sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


# ── 1. inspect.getsource 失败 ─────────────────────────────────────────────────

def demo_inspect_failure() -> None:
    sep("1. inspect.getsource(torch.exp) 失败")
    try:
        src = inspect.getsource(torch.exp)
        print(f"[OK] 获取到源码 ({len(src)} chars)")
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        print("→ torch.exp 是 C 扩展函数（pybind11 绑定 CUDA kernel），Python 层无源码。")
        print("→ operator_mr_generator._prepare_info 捕获此异常后 operator_code 保持 None。")
        print("→ _apply_sympy_proof 条件 'use_sympy_proof and operator_code' 为 False，证明跳过。")


# ── 2. 当前流水线静默跳过 ────────────────────────────────────────────────────────

def demo_proof_skipped() -> None:
    sep("2. 修复后：映射表命中，code_to_sympy(func=torch.exp) 直接返回 sp.exp(x0)")

    prover = SymPyProver()

    # 修复后：translate() 优先查映射表，不再因 inspect.getsource 失败而返回 None
    expr = prover.code_to_sympy(code=None, func=torch.exp)
    print(f"  code_to_sympy(func=torch.exp) → {expr!r}")
    print()
    if expr is not None:
        print("  修复有效：映射表命中，无需 LLM，无需 inspect.getsource。")
        print("  _apply_sympy_proof 现在可以进入证明流程（门控条件改为 operator_func 可用即可）。")
    else:
        print("  [意外] 未命中映射表，请检查 _KNOWN_SYMPY_EXPRS 是否包含 torch.exp。")


# ── 3. 直接提供 SymPy 表达式 → 证明可运行，暴露 oracle 近似值问题 ─────────────────

def demo_direct_proof() -> None:
    sep("3. 直接提供 sp.exp(x0) 后，SymPy 证明可运行")

    x0 = sp.Symbol("x0")
    sympy_expr = sp.exp(x0)   # torch.exp 的精确数学定义
    prover = SymPyProver()

    cases = [
        # (描述, oracle_expr, 预期结果, 说明)
        (
            "exp_additive（近似 e）",
            "trans == orig * 2.718281828",
            "❌ UNPROVED",
            "2.718281828 ≠ sp.E，SymPy 不认为近似值与 e 恒等",
        ),
        (
            "exp_additive（精确 e）",
            "trans == orig * exp(1)",
            "✅ PROVED",
            "exp(x+1) == exp(x)*exp(1) 可符号化简",
        ),
        (
            "exp_positive（不等式）",
            "all(orig > 0)",
            "❌ UNPROVED",
            "SymPy 不支持一般性不等式符号证明",
        ),
    ]

    transform_fn = lambda k: {**k, "input": k["input"] + 1.0}  # noqa: E731

    for desc, oracle, expected, note in cases:
        mr = MetamorphicRelation(
            id=f"demo_{desc}",
            description=desc,
            transform=transform_fn,
            transform_code="lambda k: {**k, 'input': k['input'] + 1.0}",
            oracle_expr=oracle,
            tolerance=1e-6,
        )
        ok, msg = prover.prove_mr_with_expr(
            mr=mr, sympy_expr=sympy_expr, num_inputs=1, operator_name="torch.exp"
        )
        status = "✅ PROVED" if ok else "❌ UNPROVED"
        marker = "" if status == expected else "  ← 预期不符！"
        print(f"  {status}  [{oracle}]{marker}")
        print(f"           说明: {note}")
        if not ok and msg:
            # 只打印原因首行
            print(f"           reason: {msg.splitlines()[0][:120]}")
        print()


# ── 结论汇总 ─────────────────────────────────────────────────────────────────

def summary() -> None:
    sep("结论")
    print(
        """
  1. torch.exp 是"算子本身"需要被翻译（而非翻译产物），
     且因是 C 扩展而无法通过 inspect.getsource 获取源码。

  2. 已修复（日志）：
     _prepare_info 的 except 块升级为 logger.warning；
     _apply_sympy_proof 的 else 分支在无 func/code 时输出 warning。

  3. 已修复（SymPy 翻译）：
     _KNOWN_SYMPY_EXPRS 映射表覆盖常见 C 扩展算子，优先级最高；
     无源码时允许 LLM 按算子名推断（no_source 路径）；
     持久化缓存（data/sympy_cache/）减少重复 LLM 调用；
     oracle_expr 近似值修正：2.718281828 → exp(1)（精确 e）。
    """
    )


if __name__ == "__main__":
    demo_inspect_failure()
    demo_proof_skipped()
    demo_direct_proof()
    summary()
