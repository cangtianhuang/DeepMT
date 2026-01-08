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
            operator_signature: 算子签名
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
            prompt += f"""### Operator Implementation
```python
{operator_code}
```

"""

        if operator_doc:
            prompt += f"""### Documentation
{operator_doc}

"""

        prompt += f"""### CRITICAL: Understanding {operator_name}

Before generating MRs, analyze the mathematical properties of {operator_name}:
- What is the mathematical definition?
- What are the invariant properties?
- What transformations preserve or predictably change the output?
- What common mistakes should be avoided?

### New MR Structure

Each MR consists of:

1. **transform_code**: A lambda that transforms inputs (dictionary format)
2. **oracle_expr**: A framework-independent mathematical expression for verification

### Transform Code Format

**transform_code** transforms input:
- Format: lambda expression like "lambda k: {{**k, 'input': modified}}"
- Input: 'k' is dictionary of all function arguments
- Output: New dictionary with modified values
- Focus on tensor arguments, not config parameters
- IMPORTANT: Do NOT use framework-specific functions in transform_code (like torch.relu)
- Instead, use simple mathematical operations: +, -, *, /, abs, etc.
- For operations that need the function itself, use a generic placeholder like apply_operator()

### Oracle Expression Format (NEW)

**oracle_expr** verifies output relationship using a simple mathematical expression:
- Format: Mathematical expression using these variables:
  - `orig`: Original output
  - `trans`: Transformed output
  - `x`: Original input (if single input)
  - `tolerance`: Numerical tolerance value
- NO framework-specific functions (e.g., do NOT use torch.allclose, torch.relu, np.allclose)
- The framework adapter will translate the expression to framework-specific code
- Use simple operators: ==, !=, <, >, <=, >=, +, -, *, /, abs, etc.

### Complete MR Examples

**Example 1: Positive Scaling (for ReLU)**
```json
{{
  "description": "Scaling input by positive factor scales output by same factor",
  "category": "linearity",
  "transform_code": "lambda k: {{**k, 'input': 2 * k['input']}}",
  "oracle_expr": "trans == 2 * orig"
}}
```

**Example 2: Idempotency (for ReLU)**
```json
{{
  "description": "Applying operator twice is same as applying once: f(f(x)) = f(x)",
  "category": "idempotency",
  "transform_code": "lambda k: {{**k, 'input': apply_operator(k['input'])}}",
  "oracle_expr": "orig == trans"
}}
```

**Example 3: Absolute Value Identity (for ReLU)**
```json
{{
  "description": "Operator(x) + Operator(-x) equals |x|",
  "category": "composition",
  "transform_code": "lambda k: {{**k, 'input': -k['input']}}",
  "oracle_expr": "orig + trans == abs(x)"
}}
```

**Example 4: Monotonicity (for ReLU)**
```json
{{
  "description": "If input increases, output does not decrease",
  "category": "monotonicity",
  "transform_code": "lambda k: {{**k, 'input': k['input'] + 1.5}}",
  "oracle_expr": "trans >= orig"
}}
```

**Example 5: Negation (for odd functions like Tanh)**
```json
{{
  "description": "Negating input negates output",
  "category": "symmetry",
  "transform_code": "lambda k: {{**k, 'input': -k['input']}}",
  "oracle_expr": "trans == -orig"
}}
```

**Example 6: Zero Output for Negative Input (for ReLU)**
```json
{{
  "description": "Negative input produces zero output",
  "category": "boundary",
  "transform_code": "lambda k: {{**k, 'input': -abs(k['input']) - 1.0}}",
  "oracle_expr": "all(trans == 0)"
}}
```

### MR Categories

Choose appropriate category:
- `linearity`: Linear scaling properties (trans == k * orig)
- `monotonicity`: Monotonic transformations
- `idempotency`: f(f(x)) = f(x) (orig == trans)
- `composition`: Relationships involving multiple operations (orig + trans == ...)
- `invariance`: Transformation-invariant properties
- `symmetry`: Symmetric properties (trans == -orig)
- `boundary`: Boundary value properties

### Common Mistakes to Avoid

❌ **WRONG**: Hard-coded coefficients (e.g., "trans == 3.5 * orig")
   - Use generic factors like "trans == 2 * orig" or describe as "trans == k * orig"
   - Specify constraints in description (e.g., "for k > 0")

❌ **WRONG**: "No transformation preserves output" or identity transformation
   - This is trivial and useless for testing

❌ **WRONG**: Using framework-specific functions in transform_code or oracle_expr
   - Do NOT use: torch.relu, torch.allclose, np.allclose, etc.
   - Use: simple math operations +, -, *, /, abs, apply_operator(), etc.

❌ **WRONG**: Complex nested functions in oracle_expr
   - Keep it simple and mathematical
   - Examples: "orig == trans", "trans == 2*orig", "trans >= orig"

✓ **CORRECT**: Write simple mathematical expressions
   - "orig == trans" (equality)
   - "trans == 2 * orig" (proportional)
   - "trans >= orig" (monotonicity)
   - "all(trans == 0)" (zero output)
   - "orig + trans == abs(x)" (composition)

### Output Format

Return ONLY valid JSON (no markdown, no extra text):

```json
{{
  "mrs": [
    {{
      "description": "Natural language or mathematical description",
      "category": "linearity|monotonicity|idempotency|composition|invariance|symmetry|boundary",
      "transform_code": "lambda k: {{**k, 'input': ...}}",
      "oracle_expr": "trans == 2 * orig"
    }}
  ]
}}
```

### Requirements

1. Generate 3-5 HIGH-QUALITY MRs (quality over quantity)
2. Each MR must be mathematically sound for {operator_name}
3. Avoid trivial or incorrect MRs
4. Use SIMPLE mathematical expressions in oracle_expr (no framework-specific functions)
5. Always consider numerical tolerance via the tolerance variable
6. Return ONLY JSON object, no markdown blocks

### QUALITY REQUIREMENTS (CRITICAL)

✓ **GENERICITY**: MRs should work for ANY valid input, not just specific values
  - Example: "Scaling by positive factor" (generic) vs "Scaling by 3.5" (specific)
  - Use simple numbers (2, -1, etc.) in transform_code for demonstration
  - Describe the general property in description

✓ **CORRECTNESS**: Ensure mathematical correctness for the operator
  - Double-check properties before including them
  - Consider edge cases (zero, negative values, etc.)
  - Add necessary constraints in description (e.g., "for positive scaling factor")

✓ **FRAMEWORK INDEPENDENCE**:
  - NO framework-specific functions anywhere (torch, tensorflow, numpy, etc.)
  - Use: apply_operator() to denote applying the operator itself
  - Use: simple math operations (+, -, *, /, abs, etc.)
  - Use: comparison operators (==, !=, <, >, <=, >=)
  - Use: all() for element-wise checks on arrays

✓ **SIMPLICITY**:
  - Keep oracle_expr concise and readable
  - Prefer direct relationships over complex compositions
  - Use standard mathematical notation
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

            # System prompt
            system_prompt = """You are an expert in metamorphic testing for deep learning operators.

Your task is to generate HIGH-QUALITY Metamorphic Relations (MRs) with:
1. transform_code: Lambda to transform inputs
2. oracle_expr: Simple mathematical expression for verification

═══════════════════════════════════════════════════════════════════
CRITICAL REQUIREMENTS:
═══════════════════════════════════════════════════════════════════

1. QUALITY OVER QUANTITY
   - Generate 3-5 SOUND MRs
   - Each MR must be mathematically correct
   - Avoid trivial or obviously wrong MRs
   - Ensure MRs are GENERIC, not tied to specific values

2. transform_code FORMAT (Framework Independent):
   - Lambda: "lambda k: {{**k, 'input': modified}}"
   - Input: 'k' is dictionary of all function arguments
   - Output: New dictionary with modified values
   - Focus on tensor arguments, not config parameters
   - NO framework-specific functions (torch.relu, np.abs, etc.)
   - Use simple math: +, -, *, /, abs()
   - For applying the operator itself, use: apply_operator()

3. oracle_expr FORMAT (Framework Independent):
   - Simple mathematical expression, NOT a lambda function
   - Available variables: orig, trans, x, tolerance
   - NO framework-specific functions (torch, tensorflow, numpy, etc.)
   - Use standard operators: ==, !=, <, >, <=, >=, +, -, *, /, abs
   - Use all() for element-wise checks on arrays
   - Examples:
     * "orig == trans"  # Equality
     * "trans == 2 * orig"  # Proportional (factor 2)
     * "trans == -orig"  # Negation
     * "trans >= orig"  # Monotonicity
     * "all(trans == 0)"  # Zero output
     * "orig + trans == abs(x)"  # Composition

4. HIGH-QUALITY EXAMPLES:
   
   ✓ Positive Scaling (ReLU):
   {{
     "description": "Scaling input by positive factor scales output by same factor",
     "category": "linearity",
     "transform_code": "lambda k: {{**k, 'input': 2 * k['input']}}",
     "oracle_expr": "trans == 2 * orig"
   }}
   
   ✓ Idempotency (ReLU):
   {{
     "description": "Applying operator twice is same as applying once: f(f(x)) = f(x)",
     "category": "idempotency",
     "transform_code": "lambda k: {{**k, 'input': apply_operator(k['input'])}}",
     "oracle_expr": "orig == trans"
   }}
   
   ✓ Absolute Value (ReLU):
   {{
     "description": "f(x) + f(-x) = |x|",
     "category": "composition",
     "transform_code": "lambda k: {{**k, 'input': -k['input']}}",
     "oracle_expr": "orig + trans == abs(x)"
   }}

   ✓ Monotonicity (ReLU):
   {{
     "description": "If input increases, output does not decrease",
     "category": "monotonicity",
     "transform_code": "lambda k: {{**k, 'input': k['input'] + 1.0}}",
     "oracle_expr": "trans >= orig"
   }}

   ✓ Zero Output (ReLU):
   {{
     "description": "Negative input always produces zero output",
     "category": "boundary",
     "transform_code": "lambda k: {{**k, 'input': -abs(k['input']) - 1.0}}",
     "oracle_expr": "all(trans == 0)"
   }}

5. COMMON MISTAKES TO AVOID:
   ❌ Hard-coded specific values: "trans == 3.5 * orig" (should be generic)
   ❌ Using framework-specific functions: torch.relu, torch.allclose, np.allclose
   ❌ Trivial: "Identity transformation" or no transformation is useless
   ❌ Complex: Keep oracle_expr simple and mathematical

6. FRAMEWORK INDEPENDENCE IS MANDATORY:
   - NEVER use torch.relu, torch.max, torch.where, etc.
   - NEVER use np.relu, np.maximum, etc.
   - Use apply_operator() to denote applying the operator
   - Use abs(x) for absolute value
   - Use all(expr) for element-wise checks on arrays

═══════════════════════════════════════════════════════════════════
OUTPUT: Return ONLY JSON with {{"mrs": [...]}}, no markdown blocks
═══════════════════════════════════════════════════════════════════"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            content = self.llm_client.chat_completion(
                messages, temperature=0.7, max_tokens=3000, use_model_max=True
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
        解析LLM响应的MR数据

        Args:
            mr_data: MR数据字典
            operator_signature: 算子签名字符串

        Returns:
            MetamorphicRelation对象，如果解析失败则返回None
        """
        try:
            # 1. 解析基本字段
            description = mr_data.get("description", "")
            if not description:
                self.logger.warning("MR data missing 'description' field")
                return None

            category = mr_data.get("category", "general")
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
