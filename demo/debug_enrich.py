"""
最小 debug 脚本：对 torch.Tensor.argmin 执行 OperatorEnricher 三层丰富，
逐步打印每层输出，便于调试。

运行方式：
    source .venv/bin/activate
    PYTHONPATH=$(pwd) python demo/debug_enrich.py

也可以用 pdb 逐行调试：
    PYTHONPATH=$(pwd) python -m pdb demo/debug_enrich.py
"""

import json
from deepmt.mr_generator.base.operator_enricher import OperatorEnricher

NAME = "torch.Tensor.argmin"
DOC_URL = "https://docs.pytorch.org/docs/stable/generated/torch.Tensor.argmin.html"

enricher = OperatorEnricher()

# ── Step 1: inspect（离线） ─────────────────────────────────────────────────
print("=" * 60)
print("Step 1: inspect（离线）")
inspect_result = enricher._from_inspect(NAME, api_type="function")
print(json.dumps(inspect_result, indent=2, ensure_ascii=False))

# ── Step 2: HTML 解析（需网络） ────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: HTML 解析（需网络）")
html = enricher._fetch_html(DOC_URL)
if html:
    print(f"  HTML 获取成功，长度 {len(html)} 字符")
    updates_copy = dict(inspect_result)  # 不修改原始结果
    import copy
    updates_copy = copy.deepcopy(inspect_result)
    if enricher._has_empty_dtype(updates_copy):
        enricher._enrich_from_html(html, updates_copy)
        print("  HTML 丰富后 input_specs:")
        print(json.dumps(updates_copy.get("input_specs", []), indent=2, ensure_ascii=False))
    else:
        print("  inspect 已填满 dtype，跳过 HTML")
else:
    print("  HTML 获取失败（无网络？）")

# ── Step 3: 完整 enrich（无 LLM） ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: 完整 enrich（无 LLM）")
result = enricher.enrich(
    name=NAME,
    api_type="function",
    doc_url=DOC_URL,
    use_llm=False,
)
print(json.dumps(result, indent=2, ensure_ascii=False))
