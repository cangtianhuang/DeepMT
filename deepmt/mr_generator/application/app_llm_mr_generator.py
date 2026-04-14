"""应用层 LLM MR 候选生成器（J3）。

从 ApplicationScenario + 上下文字符串出发，调用 LLM 生成候选语义 MR，
再解析为标准 MetamorphicRelation 对象列表（layer="application"）。

LLM 输出的 MR 字段：
  - category:             变换类别（如 "noise_robustness"、"invariance"）
  - description:          自然语言描述
  - rationale:            为何该变换应保持输出不变的推理
  - transform_description: 变换的自然语言描述（保留为来源文档）
  - transform_code:       Python lambda 字符串
  - oracle_expr:          验证器可识别的 oracle 类型字符串
  - risk_level:           "low" | "medium" | "high"

支持的 oracle_expr（SemanticMRValidator 解析）：
  - "label_consistent"            预测标签相同
  - "label_consistent_soft:{k}"  原始标签在 top-k 预测中
  - "confidence_acceptable:{d}"  置信度下降不超过 d

transform_code 约定：
  - 接受 dict（含 'input' 或 'text' 键）
  - 仅使用 Python 内置操作（list comprehension、str 方法），不依赖 numpy/torch
  - 示例：
      图像：  "lambda s: {**s, 'input': [x * 0.9 for x in s['input']]}"
      文本：  "lambda s: {**s, 'text': s['text'].lower()}"
"""

import json
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional

from deepmt.mr_generator.application.scenario import ApplicationScenario
from deepmt.core.logger import logger
from deepmt.ir import MetamorphicRelation
from deepmt.tools.llm.client import LLMClient

# ── Prompt 模板 ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a Metamorphic Testing Expert specializing in application-layer semantic testing of deep learning systems.

Your goal is to generate high-quality Metamorphic Relations (MRs) for the given application scenario.

## What is an Application-Layer MR?

An application-layer MR describes an input transformation that should **preserve the application's high-level output** (e.g., predicted class label, sentiment polarity). The transformation operates on the application's input (image pixels, text strings), not internal model activations.

## MR Structure

Each MR must have:
- `category`: One of "noise_robustness", "invariance", "semantic_preservation", "linguistic_robustness", "label_consistency"
- `description`: Clear natural language description of the MR
- `rationale`: Why this transformation should preserve the output (domain reasoning)
- `transform_description`: Natural language description of the transformation
- `transform_code`: Python lambda string — **must use only Python built-ins, no imports**
  - For image inputs (list of floats): `"lambda s: {**s, 'input': [x * 0.9 for x in s['input']]}"`
  - For text inputs (str): `"lambda s: {**s, 'text': s['text'].lower()}"`
- `oracle_expr`: Exactly one of:
  - `"label_consistent"` — predicted label must be the same
  - `"label_consistent_soft:3"` — original label must appear in top-3 predictions
  - `"confidence_acceptable:0.1"` — confidence drop must not exceed 0.1
- `risk_level`: `"low"` (very likely holds), `"medium"` (usually holds), or `"high"` (uncertain)

## Rules

1. transform_code MUST be a valid Python lambda that can be eval'd without imports
2. For image inputs, assume the input is a flat list of float values in [0, 1]
3. For text inputs, assume the input is a Python string
4. Only generate MRs where the transformation genuinely preserves the semantic output
5. Prefer `"label_consistent"` for clear, low-risk transformations
6. Use `"risk_level": "high"` for transformations that might not always hold

## Output Format

Return a single JSON object:

```json
{
  "mrs": [
    {
      "category": "invariance",
      "description": "Lowercasing text should not change sentiment",
      "rationale": "Sentiment is not case-sensitive; 'GOOD' and 'good' have the same meaning",
      "transform_description": "Convert all characters to lowercase",
      "transform_code": "lambda s: {**s, 'text': s['text'].lower()}",
      "oracle_expr": "label_consistent",
      "risk_level": "low"
    }
  ]
}
```
"""


def _build_user_prompt(scenario: ApplicationScenario, context: str, top_k: int) -> str:
    return f"""Please generate {top_k} high-quality Metamorphic Relations for the following application scenario.

{context}

Generate exactly {top_k} MRs. Focus on transformations that are:
1. Clearly semantics-preserving (prefer low risk_level)
2. Simple enough to express as a Python lambda without imports
3. Diverse in category (mix noise_robustness, invariance, semantic_preservation, etc.)

Return only the JSON object.
"""


# ── 生成器主类 ─────────────────────────────────────────────────────────────────


class ApplicationLLMMRGenerator:
    """应用层 LLM MR 候选生成器。

    调用 LLM 为给定场景生成语义 MR 候选，并解析为 MetamorphicRelation 列表。

    用法::

        gen = ApplicationLLMMRGenerator()
        mrs = gen.generate_mr_candidates(scenario, context_str, top_k=5)
    """

    def __init__(self) -> None:
        self._llm: Optional[LLMClient] = None

    def _get_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    def generate_mr_candidates(
        self,
        scenario: ApplicationScenario,
        context: str,
        top_k: int = 5,
    ) -> List[MetamorphicRelation]:
        """调用 LLM 生成候选 MR 列表。

        Args:
            scenario: 应用场景描述
            context:  由 AppContextBuilder 构建的上下文字符串
            top_k:    请求生成的 MR 数量上限

        Returns:
            MetamorphicRelation 列表（layer="application", source="llm"）
        """
        user_prompt = _build_user_prompt(scenario, context, top_k)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            content = self._get_llm().chat_completion(messages, use_model_max=True)
        except Exception as e:
            logger.error(f"[AppLLMGen] LLM 调用失败 ({scenario.name}): {e}")
            return []

        return self._parse_response(content, scenario)

    # ── 解析 ──────────────────────────────────────────────────────────────────

    def _parse_response(
        self, content: str, scenario: ApplicationScenario
    ) -> List[MetamorphicRelation]:
        """解析 LLM 响应，返回 MetamorphicRelation 列表。"""
        # 提取 JSON 块
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"[AppLLMGen] JSON 解析失败: {e}\n内容片段: {content[:200]}")
            return []

        if not isinstance(data, dict) or "mrs" not in data:
            logger.error(f"[AppLLMGen] 响应缺少 'mrs' 键，当前键: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            return []

        mrs = []
        for idx, mr_data in enumerate(data["mrs"]):
            try:
                mr = self._parse_mr_entry(mr_data, scenario)
                if mr:
                    mrs.append(mr)
            except Exception as e:
                logger.warning(f"[AppLLMGen] 解析第 {idx+1} 条 MR 失败: {e}")
                logger.debug(traceback.format_exc())

        logger.info(
            f"[AppLLMGen] {scenario.name}: 从 LLM 解析到 {len(mrs)} 条候选 MR"
        )
        return mrs

    def _parse_mr_entry(
        self, mr_data: Dict[str, Any], scenario: ApplicationScenario
    ) -> Optional[MetamorphicRelation]:
        """解析单条 MR 字典为 MetamorphicRelation 对象。"""
        description = mr_data.get("description", "").strip()
        if not description:
            logger.warning("[AppLLMGen] MR 缺少 description，跳过")
            return None

        transform_code = mr_data.get("transform_code", "").strip()
        if not transform_code:
            logger.warning(f"[AppLLMGen] MR '{description[:40]}' 缺少 transform_code，跳过")
            return None

        oracle_expr = mr_data.get("oracle_expr", "").strip()
        if not oracle_expr:
            logger.warning(f"[AppLLMGen] MR '{description[:40]}' 缺少 oracle_expr，跳过")
            return None

        # 尝试 eval transform_code，验证语法（不带 import）
        transform = _safe_eval_lambda(transform_code)
        if transform is None:
            logger.warning(
                f"[AppLLMGen] transform_code eval 失败，跳过: {transform_code[:60]}"
            )
            return None

        category = mr_data.get("category", "general")
        risk_level = mr_data.get("risk_level", "medium")
        rationale = mr_data.get("rationale", "")
        transform_description = mr_data.get("transform_description", description)

        return MetamorphicRelation(
            id=str(uuid.uuid4()),
            description=description,
            subject_name=scenario.name,
            subject_type="application",
            transform_code=transform_code,
            transform=transform,
            oracle_expr=oracle_expr,
            category=category,
            layer="application",
            source="llm",
            lifecycle_state="pending",
            checked=None,
            proven=None,
            verified=False,
            analysis=f"[risk:{risk_level}] {rationale}\n[transform] {transform_description}",
        )


# ── 工具函数 ──────────────────────────────────────────────────────────────────


_SAFE_BUILTINS = {
    "min": min, "max": max, "sum": sum, "len": len, "abs": abs,
    "round": round, "sorted": sorted, "enumerate": enumerate,
    "zip": zip, "list": list, "dict": dict, "str": str,
    "int": int, "float": float, "bool": bool, "tuple": tuple,
    "set": set, "range": range,
}


def _safe_eval_lambda(code: str) -> Optional[Callable]:
    """安全 eval lambda 字符串（允许常用内置函数，不允许 import）。

    Returns:
        可调用对象，eval 失败则返回 None。
    """
    code = code.strip()
    if not code.startswith("lambda"):
        return None
    try:
        fn = eval(code, {"__builtins__": _SAFE_BUILTINS}, {})  # noqa: S307
        return fn if callable(fn) else None
    except Exception as e:
        logger.debug(f"[AppLLMGen] eval lambda 失败: {e} | code={code[:60]}")
        return None
