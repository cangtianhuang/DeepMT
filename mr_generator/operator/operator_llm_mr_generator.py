"""
算子层LLM MR生成器：使用大语言模型生成算子MR猜想（路径A）
专为算子层MR生成设计，因为大模型MR生成方法在后续也可能用于其他层次
"""

import inspect
import json
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional

from core.logger import get_logger
from ir.schema import MetamorphicRelation
from tools.llm.client import LLMClient


class OperatorLLMMRGenerator:
    """算子层LLM MR生成器：使用LLM生成算子MR猜想"""

    def __init__(self):
        """初始化算子层LLM MR生成器"""
        self.logger = get_logger(self.__class__.__name__)

        self.llm_client = LLMClient()

    def _build_prompt(
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
            operator_code: 算子代码（可选）
            operator_doc: 算子文档（可选）

        Returns:
            提示字符串
        """
        prompt = f"""You are a Deep Learning Metamorphic Testing Expert.
Your task is to generate Metamorphic Relations (MRs) for the operator `{operator_name}`.

### Operator Signature
```python
{operator_signature}
```

"""

        if operator_code:
            prompt += f"""### Operator Code
```python
{operator_code}
```

"""

        if operator_doc:
            prompt += f"""### Documentation
{operator_doc}

"""

        prompt += """### Transformation Protocol (CRITICAL)

The input transformation must be a Python LAMBDA function that:

1. **Receives a single argument `k`** (a dictionary containing ALL function arguments)
2. **Returns a new dictionary** with modified values
3. **Uses `{**k, 'key': new_value}` syntax** to modify specific parameters while keeping others unchanged
4. **Focus on Tensor arguments** (like 'input', 'weight', 'bias'), NOT configuration parameters (like 'stride', 'padding', 'dilation') unless necessary

### Transform Code Format Requirements

- MUST be a valid Python lambda expression (NOT a def function)
- Input: Receives a single dictionary argument `k` containing all function parameters
- Output: MUST return a new dictionary using `{**k, 'key': new_value}` syntax
- Only modify the parameters you want to transform, keep all others unchanged

### CORRECT Examples

1. **Scale input tensor:**
   ```python
   lambda k: {**k, 'input': 2 * k['input']}
   ```
   Explanation: Scales the 'input' parameter by 2, keeps all other parameters unchanged.

2. **Negate input tensor:**
   ```python
   lambda k: {**k, 'input': -k['input']}
   ```
   Explanation: Negates the 'input' parameter, keeps all other parameters unchanged.

3. **Modify multiple tensor parameters:**
   ```python
   lambda k: {**k, 'input': 2 * k['input'], 'weight': k['weight'] * 0.5}
   ```
   Explanation: Scales 'input' by 2 and 'weight' by 0.5, keeps other parameters unchanged.

4. **Swap tensor parameters (if applicable):**
   ```python
   lambda k: {**k, 'input': k.get('other', k['input']), 'other': k['input']}
   ```
   Explanation: Swaps 'input' and 'other' parameters if both exist.

### INCORRECT Examples (DO NOT USE)

- `lambda k: k['input'] * 2`  ❌ WRONG: returns single value, not dictionary
- `lambda x: (x,)`  ❌ WRONG: old args mode, not kwargs mode
- `def transform(k): return {**k, 'input': 2*k['input']}`  ❌ WRONG: not a lambda expression
- `lambda k: {**k, 'stride': 2}`  ❌ WRONG: modifying configuration parameter (usually not part of MR)

### MR Categories & Examples

1. **Affine/Linearity (Scaling):**
   - Description: "Scaling input scales output proportionally"
   - Transform: `lambda k: {**k, 'input': 2 * k['input']}`
   - Expected: "proportional"

2. **Negation:**
   - Description: "Negating input negates output"
   - Transform: `lambda k: {**k, 'input': -k['input']}`
   - Expected: "negate"

3. **Identity:**
   - Description: "No transformation preserves output"
   - Transform: `lambda k: k`
   - Expected: "equal"

4. **Idempotent (if applicable):**
   - Description: "f(f(input)) == f(input)"
   - Transform: `lambda k: k`  # Special handling needed
   - Expected: "idempotent"

### Output Format

Return strictly valid JSON (no markdown code blocks, no extra text):

```json
{
    "mrs": [
        {
            "description": "Brief mathematical or natural language description",
            "input_transform": "Text description of the transformation",
            "expected_relation": "equal|proportional|invariant|negate|zero|first_input|idempotent",
            "transform_code": "lambda k: {**k, 'input': k['input'] * 2}"
        }
    ]
}
```

### Valid expected_relation Values

- "equal": Outputs are equal (most common)
- "proportional": Outputs are proportional (e.g., f(x) == k * f(y))
- "invariant": Output remains invariant (same as equal)
- "negate": Outputs are negated (f(x) == -f(y))
- "zero": Output is zero
- "first_input": Output equals the first input
- "idempotent": f(f(x)) == f(x)

### Important Notes

- Focus on transforming TENSOR arguments ('input', 'weight', 'bias'), not configuration parameters
- Use dictionary unpacking syntax: `{**k, 'key': new_value}`
- Keep all non-transformed parameters unchanged
- Return ONLY valid JSON, no markdown code blocks, no explanations
"""
        return prompt

    def generate_mr_candidates(
        self,
        operator_name: str,
        operator_func: Optional[Callable] = None,
        operator_signature: Optional[str] = None,
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
        if operator_signature is None:
            if operator_func is not None:
                try:
                    sig = inspect.signature(operator_func)
                    operator_signature = str(sig)
                    self.logger.info(f"Auto-extracted signature: {operator_signature}")
                except Exception as e:
                    self.logger.error(
                        f"Failed to extract signature from function: {e}. "
                        "Please provide operator_signature explicitly."
                    )
                    return []
            else:
                self.logger.error(
                    "Either operator_func or operator_signature must be provided"
                )
                return []

        try:
            prompt = self._build_prompt(
                operator_name=operator_name,
                operator_signature=operator_signature,
                operator_code=operator_code,
                operator_doc=operator_doc,
            )

            # System prompt for kwargs mode
            system_prompt = """You are an expert in metamorphic testing for deep learning operators.

Your task is to generate Metamorphic Relations (MRs) using the Signature-based Kwargs Transformation protocol.

═══════════════════════════════════════════════════════════════════
CRITICAL REQUIREMENTS for transform_code (Kwargs Mode):
═══════════════════════════════════════════════════════════════════

1. FORMAT: MUST be a valid Python lambda expression (NOT a def function)

2. INPUT: Receives a single argument `k` which is a dictionary containing ALL function parameters
   - Example: lambda k: ...
   - The dictionary `k` contains all arguments (positional and keyword)

3. OUTPUT: MUST return a new dictionary (NOT a tuple or single value)
   - Use dictionary unpacking and merging: {**k, 'key': new_value}
   - Only modify the parameters you want to transform
   - Keep all other parameters unchanged

4. CORRECT Examples (Kwargs Mode):
   - Scale input tensor: "lambda k: {**k, 'input': 2 * k['input']}"
   - Negate input tensor: "lambda k: {**k, 'input': -k['input']}"
   - Modify multiple tensors: "lambda k: {**k, 'input': 2 * k['input'], 'weight': k['weight'] * 0.5}"
   - Swap tensor parameters: "lambda k: {**k, 'input': k.get('other', k['input']), 'other': k['input']}"

5. INCORRECT Examples (DO NOT USE):
   - "lambda k: k['input'] * 2"  ❌ WRONG: returns single value, not dictionary
   - "lambda x: (x,)"  ❌ WRONG: old args mode, not kwargs mode
   - "def transform(k): return {**k, 'input': 2*k['input']}"  ❌ WRONG: not a lambda expression
   - "lambda k: {**k, 'stride': 2}"  ❌ WRONG: modifying configuration parameter

6. Focus on Tensor Parameters:
   - Transform tensor arguments like 'input', 'weight', 'bias'
   - DO NOT modify configuration parameters like 'stride', 'padding', 'dilation' unless necessary
   - Configuration parameters are usually not part of MR transformations

7. MR Description Format:
   - Mathematical: "f(2*input) == 2*f(input)" for scaling property
   - Natural language: "Scaling input scales output proportionally"
   - Be specific about which parameters are transformed

8. Valid expected_relation values (use exactly as shown):
   - "equal": Outputs are equal (most common)
   - "proportional": Outputs are proportional (e.g., f(x) == k * f(y))
   - "invariant": Output remains invariant (same as equal)
   - "negate": Outputs are negated (f(x) == -f(y))
   - "zero": Output is zero
   - "first_input": Output equals the first input
   - "idempotent": f(f(x)) == f(x)

═══════════════════════════════════════════════════════════════════
REMEMBER: transform_code MUST return a dictionary using {**k, ...} syntax!
═══════════════════════════════════════════════════════════════════"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            content = self.llm_client.chat_completion(
                messages, temperature=0.7, max_tokens=3000
            )

            # 提取JSON代码块
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # 解析JSON
            data = json.loads(content)

            # 转换为MR对象
            if not isinstance(data, dict) or "mrs" not in data:
                self.logger.error(
                    f"Invalid JSON structure: expected dict with 'mrs' key, got {type(data)}"
                )
                if isinstance(data, dict):
                    self.logger.debug(f"Available keys: {list(data.keys())}")
                return []

            mrs = []
            mr_list = data.get("mrs", [])
            if not isinstance(mr_list, list):
                self.logger.error(f"Expected 'mrs' to be a list, got {type(mr_list)}")
                return []

            self.logger.info(f"Parsed {len(mr_list)} MR entries from LLM response")

            for idx, mr_data in enumerate(mr_list[:top_k]):
                try:
                    if not isinstance(mr_data, dict):
                        self.logger.warning(f"MR entry {idx} is not a dict, skipping")
                        continue

                    mr = self._parse_mr_response(
                        mr_data, operator_signature=operator_signature
                    )
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
                    import traceback

                    self.logger.debug(traceback.format_exc())
                    continue

            self.logger.info(
                f"Generated {len(mrs)} MR candidates from LLM for {operator_name}"
            )
            return mrs

        except Exception as e:
            self.logger.error(f"LLM MR generation error: {e}")
            self.logger.debug(traceback.format_exc())
            return []

    def _parse_mr_response(
        self,
        mr_data: Dict[str, Any],
        operator_signature: Optional[str] = None,
    ) -> Optional[MetamorphicRelation]:
        """
        解析LLM响应的MR数据（Kwargs模式）

        Args:
            mr_data: MR数据字典
            operator_signature: 算子签名字符串（用于验证）

        Returns:
            MetamorphicRelation对象，如果解析失败则返回None
        """
        try:
            description = mr_data.get("description", "")
            if not description:
                self.logger.warning("MR data missing 'description' field")
                return None

            expected = mr_data.get("expected_relation", "equal")
            # 规范化expected值
            expected = expected.lower().strip()
            valid_expecteds = [
                "equal",
                "proportional",
                "invariant",
                "negate",
                "zero",
                "first_input",
                "idempotent",
            ]
            if expected not in valid_expecteds:
                self.logger.warning(
                    f"Unknown expected_relation '{expected}', using 'equal'"
                )
                expected = "equal"

            # 解析transform_code（Kwargs模式）
            transform_code = mr_data.get("transform_code", "").strip()
            transform = None

            if transform_code:
                # 清理transform_code（移除可能的markdown代码块标记）
                if "```python" in transform_code:
                    transform_code = (
                        transform_code.split("```python")[1].split("```")[0].strip()
                    )
                elif "```" in transform_code:
                    transform_code = (
                        transform_code.split("```")[1].split("```")[0].strip()
                    )

                # 安全地执行lambda表达式（Kwargs模式）
                try:
                    if transform_code.startswith("lambda"):
                        # 创建安全的执行环境
                        safe_dict = {}
                        transform = eval(
                            transform_code, {"__builtins__": {}}, safe_dict
                        )
                    else:
                        # 如果不是lambda，警告
                        self.logger.warning(
                            f"transform_code doesn't look like a lambda: {transform_code[:50]}"
                        )
                        return None
                except Exception as e:
                    self.logger.warning(
                        f"Failed to eval transform_code '{transform_code[:50]}...': {e}"
                    )
                    transform = None

            # Kwargs模式：必须有transform_code，无法从描述推断
            if transform is None:
                self.logger.warning(
                    "Kwargs mode requires transform_code, cannot infer from description"
                )
                return None

            # 验证transform是可调用的
            if not callable(transform):
                self.logger.warning(f"Transform is not callable: {type(transform)}")
                return None

            # 验证transform接收kwargs并返回字典（简单测试）
            try:
                test_kwargs = {"input": 1.0, "weight": 2.0}
                test_result = transform(test_kwargs)
                if not isinstance(test_result, dict):
                    self.logger.warning(
                        f"Transform returned {type(test_result)}, expected dict"
                    )
                    return None
            except Exception as e:
                self.logger.warning(f"Transform validation failed: {e}")
                # 不直接返回None，可能只是测试参数不匹配

            return MetamorphicRelation(
                id=str(uuid.uuid4()),
                description=description,
                transform=transform,
                expected=expected,
                tolerance=1e-6,
                layer="operator",
                verified=False,  # LLM生成的MR默认未验证
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse MR data: {e}")
            self.logger.debug(traceback.format_exc())
            return None
