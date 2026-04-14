"""
debug_generate.py — 调试 MR 生成过程（阶段 1-2）

示例算子：torch.nn.functional.relu
来源：仅使用模板池（template），无 LLM / 无网络依赖

运行方式：
    source .venv/bin/activate
    PYTHONPATH=$(pwd) python demo/debug_generate.py
"""

import torch
import torch.nn.functional as F

from deepmt.ir import OperatorIR
from deepmt.mr_generator.base.mr_templates import MRTemplatePool
from deepmt.mr_generator.operator.operator_mr_generator import OperatorMRGenerator

OPERATOR_NAME = "torch.nn.functional.relu"
OPERATOR_FUNC = F.relu

# ── Step 1: 查看模板池中该算子的可用模板 ────────────────────────────────────────
print("=" * 64)
print("Step 1: 查看模板池中该算子的可用模板")
print("=" * 64)

pool = MRTemplatePool()
templates = pool.get_applicable_templates(OPERATOR_NAME, operator_func=OPERATOR_FUNC)
print(f"算子: {OPERATOR_NAME}")
print(f"模板总数: {len(pool.templates)}, 映射算子数: {len(pool.operator_mr_mapping)}")
print(f"适用模板: {len(templates)} 个\n")

for t in templates:
    print(f"  [{t.name}]")
    print(f"    描述:          {t.description}")
    print(f"    类别:          {t.category}")
    print(f"    transform_code: {t.transform_code}")
    print(f"    oracle_expr:    {t.oracle_expr}")
    print()

# ── Step 2: 调用 generate_only()，生成 MR 候选（不验证） ─────────────────────
print("=" * 64)
print("Step 2: generate_only() — 生成 MR 候选（模板来源，无 LLM / 无网络）")
print("=" * 64)

generator = OperatorMRGenerator()
operator_ir = OperatorIR(
    name=OPERATOR_NAME,
    inputs=[torch.randn(4, 4, dtype=torch.float32)],
)

candidates = generator.generate_only(
    operator_ir=operator_ir,
    framework="pytorch",
    operator_func=OPERATOR_FUNC,
    auto_fetch_info=False,   # 不联网
    sources=["template"],    # 仅模板，不使用 LLM
)

print(f"\n生成候选 MR 数量: {len(candidates)}\n")
for i, mr in enumerate(candidates, 1):
    print(f"  [{i}] {mr.description}")
    print(f"       id:             {mr.id}")
    print(f"       category:       {mr.category}")
    print(f"       transform_code: {mr.transform_code}")
    print(f"       oracle_expr:    {mr.oracle_expr}")
    print(f"       verified:       {mr.verified}")
    print()
