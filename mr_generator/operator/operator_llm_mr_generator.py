"""
算子层LLM MR生成器：使用大语言模型生成算子MR猜想（路径A）
专为算子层MR生成设计，因为大模型MR生成方法在后续也可能用于其他层次

"""

import inspect
import json
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional

from core.logger import get_logger, log_error, log_structured
from ir.schema import MetamorphicRelation
from tools.llm.client import LLMClient


class OperatorLLMMRGenerator:
    """算子层LLM MR生成器：使用LLM生成算子MR猜想"""

    def __init__(self):
        """初始化算子层LLM MR生成器"""
        self.logger = get_logger(self.__class__.__name__)
        self.llm_client = LLMClient()

    def _build_user_prompt(
        self,
        operator_name: str,
        operator_signature: str,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
    ) -> str:
        """
        构建LLM提示

        Args:
            operator_name: 算子名称
            operator_signature: 算子签名
            operator_code: 算子代码（可选）
            operator_doc: 算子文档（可选）

        Returns:
            提示字符串
        """
        prompt = f"""Target Operator: `{operator_name}`

### Signature
```python
{operator_signature}
```
"""
        if operator_code:
            prompt += f"""
### Implementation
```python
{operator_code}
```
"""
        if operator_doc:
            prompt += f"""
### Documentation
{operator_doc}
"""
        prompt += (
            "\nPlease analyze this operator and generate "
            "Metamorphic Relations JSON for this operator "
            "following the system instructions."
        )
        return prompt

    def generate_mr_candidates(
        self,
        operator_name: str,
        operator_func: Optional[Callable] = None,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        top_k: int = 5,
    ) -> List[MetamorphicRelation]:
        """
        使用LLM生成MR候选列表

        Args:
            operator_name: 算子名称
            operator_func: 算子函数对象（可选）
            operator_signature: 算子签名字符串（可选）
            operator_code: 算子代码（可选）
            operator_doc: 算子文档（可选）
            top_k: 生成Top-K个MR（默认5）

        Returns:
            MR候选列表
        """
        # 获取算子签名
        operator_signature = "(unknown)"
        if operator_func is not None:
            try:
                sig = inspect.signature(operator_func)
                operator_signature = str(sig)
                self.logger.debug(f"Auto-extracted signature: {operator_signature}")
            except Exception as e:
                self.logger.debug(f"Failed to extract signature from function: {e}")

        # User prompt
        user_prompt = self._build_user_prompt(
            operator_name=operator_name,
            operator_signature=operator_signature,
            operator_code=operator_code,
            operator_doc=operator_doc,
        )

        # System prompt
        system_prompt = """You are a Metamorphic Testing Expert for Deep Learning Operators.

Your goal is to generate 1-5 high-quality Metamorphic Relations (MRs) to verify the mathematical correctness of the given operator.

### PROCESS: CHAIN-OF-THOUGHT (CoT)

Before generating JSON, you MUST perform a **Step-by-Step Analysis** in your response:

1. **Analyze Arguments**: Identify Tensor inputs vs. Configuration parameters (e.g., `bias`, `stride`, `mode`).
2. **Identify Invariants**: What mathematical properties theoretically hold? (Linearity, Symmetry, Periodicity, etc.)
3. **Detect Interference**: Which configuration parameters might BREAK these properties?
* *Reasoning Example*: "Linearity f(ax)=af(x) holds for Matrix Multiplication, but if `bias` is present, it becomes `a(Wx+b) != W(ax)+b`. So I must set `bias=None` or `0`."
4. **Formulate MRs**: Create lambda transformations that handle these interferences.

### OUTPUT FORMAT

1. **Analysis Section**: A brief text block with your reasoning.
2. **JSON Block**: The final result wrapped in ```json ... ```.

### RESPONSE FORMAT (STRICT JSON)

Return a single JSON object containing a list of MRs. No markdown formatting outside the JSON.

```json
{
  "mrs": [
    {
      "category": "linearity|symmetry|idempotency|invariance|boundary",
      "description": "Brief description of the mathematical property",
      "analysis": "Brief analysis explanation for this MR",
      "transform_code": "lambda k: {**k, 'input': ...}",
      "oracle_expr": "trans == ..."
    }
  ]
}
```

### MR STRUCTURE SPECIFICATION

* `category`: One of: "linearity", "symmetry", "idempotency", "invariance", "boundary"
* `description`: Brief human-readable description of the MR
* `analysis`: Brief analysis explaining why this MR holds and any constraints
* `transform_code`: A Python lambda `lambda k: {**k, ...}`.
* Input `k`: Dict of ALL arguments.
* Output: New dict.
* **Rule**: You can and should modify config parameters if needed to satisfy the mathematical property.
* `oracle_expr`: A mathematical assertion using `trans` (transformed output), `orig` (original output), `x` (input).
* Use `all(...)` for tensor comparisons.
* No framework-specific functions (e.g., no `torch.*`).

### FEW-SHOT EXAMPLES (General & Simple)

**Example 1: Sin (Symmetry/Periodicity)**

> Analysis: Sin is an odd function (sin(-x) = -sin(x)) and periodic (sin(x+2pi) = sin(x)). No config args interfere.

```json
{
  "mrs": [
    {
      "category": "symmetry",
      "description": "Sin is an odd function: f(-x) = -f(x)",
      "analysis": "The sine function satisfies the odd function property sin(-x) = -sin(x) due to its symmetry about the origin.",
      "transform_code": "lambda k: {{**k, 'input': -k['input']}}",
      "oracle_expr": "trans == -orig"
    }
  ]
}
```

**Example 2: Linear / Dense (Linearity with Inference)**

> Analysis: A Linear layer is `x @ w.T + bias`. Pure linearity `f(kx) == kf(x)` fails because of bias. I must disable bias for this MR.

```json
{
  "mrs": [
    {
      "category": "linearity",
      "description": "Scaling input scales output (valid only when bias is disabled)",
      "analysis": "For a linear layer without bias, f(kx) = k*f(x). The bias term breaks this property, so it must be disabled.",
      "transform_code": "lambda k: {**k, 'input': 2 * k['input'], 'bias': None}",
      "oracle_expr": "trans == 2 * orig"
    }
  ]
}
```

### FINAL INSTRUCTION

Generate the response for the user's operator. Start with **Analysis**, then provide **JSON**.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        content = self.llm_client.chat_completion(messages, use_model_max=True)

        try:
            # 提取JSON代码块
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # 解析JSON
            data = json.loads(content)

            # 转换为MR对象
            if not isinstance(data, dict) or "mrs" not in data:
                log_error(
                    self.logger,
                    f"Invalid JSON structure from LLM",
                    exception=TypeError(
                        f"Expected dict with 'mrs' key, got {type(data)}"
                    ),
                )
                if isinstance(data, dict):
                    self.logger.debug(f"Available keys: {list(data.keys())}")
                return []

            mrs = []
            mr_list = data.get("mrs", [])
            if not isinstance(mr_list, list):
                log_error(
                    self.logger,
                    f"Invalid 'mrs' field type",
                    exception=TypeError(f"Expected list, got {type(mr_list)}"),
                )
                return []

            log_structured(
                self.logger,
                "GEN",
                f"Parsed {len(mr_list)} MR entries from LLM response",
            )

            for idx, mr_data in enumerate(mr_list[:top_k]):
                try:
                    if not isinstance(mr_data, dict):
                        self.logger.warning(f"MR entry {idx} is not a dict, skipping")
                        continue

                    mr = self._parse_mr_response(mr_data)
                    if mr:
                        mrs.append(mr)
                        self.logger.debug(
                            f"Successfully parsed MR {idx+1}: {mr.description[:50]}..."
                        )
                    else:
                        self.logger.warning(
                            f"Failed to parse MR {idx+1}: returned None"
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to parse MR {idx+1}: {e}")
                    self.logger.debug(traceback.format_exc())
                    continue

            log_structured(
                self.logger,
                "GEN",
                f"Generated {len(mrs)} MR candidates from LLM for {operator_name}",
            )
            return mrs

        except Exception as e:
            log_error(
                self.logger,
                f"LLM MR generation failed for '{operator_name}'",
                exception=e,
            )
            return []

    def _parse_mr_response(
        self, mr_data: Dict[str, Any]
    ) -> Optional[MetamorphicRelation]:
        """解析LLM响应的MR数据"""
        try:
            # 1. 解析基本字段
            description = mr_data.get("description", "")
            if not description:
                self.logger.warning("MR data missing 'description' field")
                return None

            category = mr_data.get("category", "general")
            analysis = mr_data.get("analysis", "").strip()
            oracle_expr = mr_data.get("oracle_expr", "").strip()

            # 2. 解析 transform_code
            transform_code = mr_data.get("transform_code", "").strip()
            transform = self._parse_lambda_code(
                transform_code, code_type="transform", test_input={"input": 1.0}
            )
            if transform is None:
                self.logger.warning(
                    f"Failed to parse transform_code for: {description}"
                )
                return None

            # 3. 验证 oracle_expr
            if not oracle_expr:
                self.logger.warning(f"MR missing 'oracle_expr': {description}")
                return None

            # 4. 创建 MetamorphicRelation 对象
            return MetamorphicRelation(
                id=str(uuid.uuid4()),
                description=description,
                transform=transform,
                transform_code=transform_code,
                oracle_expr=oracle_expr,
                category=category,
                analysis=analysis,
                tolerance=1e-6,
                layer="operator",
                verified=False,
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse MR data: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def _parse_lambda_code(
        self, code: str, code_type: str = "transform", test_input: Any = None
    ) -> Optional[Callable]:
        """
        解析并验证 lambda 表达式代码

        Args:
            code: Lambda 表达式字符串
            code_type: 代码类型 ("transform" 或 "oracle")
            test_input: 测试输入

        Returns:
            可调用对象，如果解析失败则返回 None
        """
        if not code:
            return None

        # 清理代码
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        # 检查是否为 lambda 表达式
        if not code.startswith("lambda"):
            self.logger.warning(
                f"{code_type}_code doesn't start with 'lambda': {code[:50]}"
            )
            return None

        # 安全地执行 lambda 表达式
        try:
            # 创建安全的执行环境
            safe_dict = {}
            func = eval(code, {"__builtins__": {}}, safe_dict)

            # 验证是否可调用
            if not callable(func):
                self.logger.warning(f"{code_type} is not callable: {type(func)}")
                return None

            # 简单测试（可选）
            if test_input is not None:
                try:
                    result = func(test_input)
                    if code_type == "transform" and not isinstance(result, dict):
                        self.logger.warning(
                            f"Transform returned {type(result)}, expected dict"
                        )
                except Exception as e:
                    self.logger.debug(f"{code_type} test failed (may be OK): {e}")

            return func

        except Exception as e:
            self.logger.warning(
                f"Failed to eval {code_type}_code '{code[:50]}...': {e}"
            )
            return None
