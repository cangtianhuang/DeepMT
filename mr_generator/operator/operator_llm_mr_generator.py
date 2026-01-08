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

For ReLU specifically:
- ReLU(x) = max(0, x)
- ReLU(-x) ≠ -ReLU(x)  ❌ INCORRECT
- ReLU(ax) = a·ReLU(x) ONLY when a > 0
- ReLU(x) + ReLU(-x) = |x|  ✓ CORRECT
- ReLU is idempotent: ReLU(ReLU(x)) = ReLU(x)

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

### Oracle Expression Format (NEW)

**oracle_expr** verifies output relationship using a simple mathematical expression:
- Format: Mathematical expression using these variables:
  - `orig`: Original output
  - `trans`: Transformed output
  - `x`: Original input (if single input)
  - `tolerance`: Numerical tolerance value
- NO framework-specific functions (e.g., do NOT use torch.allclose)
- The framework adapter will translate the expression to framework-specific code

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
  "description": "Applying ReLU twice is same as applying once: f(f(x)) = f(x)",
  "category": "idempotency",
  "transform_code": "lambda k: {{**k, 'input': torch.relu(k['input'])}}",
  "oracle_expr": "orig == trans"
}}
```

**Example 3: Absolute Value Identity (for ReLU)**
```json
{{
  "description": "ReLU(x) + ReLU(-x) equals |x|",
  "category": "composition",
  "transform_code": "lambda k: {{**k, 'input': -k['input']}}",
  "oracle_expr": "orig + trans == abs(x)"
}}
```

**Example 4: Addition Invariance (for some operators)**
```json
{{
  "description": "Adding constant to inputs adds same constant to output",
  "category": "invariance",
  "transform_code": "lambda k: {{**k, 'input': k['input'] + 5.0}}",
  "oracle_expr": "trans == orig + 5.0"
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

❌ **WRONG for ReLU**: "Negating input negates output"
   - ReLU(-x) ≠ -ReLU(x) because ReLU is not odd function

❌ **WRONG**: "No transformation preserves output"
   - This is trivial and useless

❌ **WRONG**: Using framework-specific functions in oracle_expr
   - Do NOT use: torch.allclose, torch.equal, etc.
   - Use: ==, <, >, +, -, *, /, abs, etc.

❌ **WRONG**: Complex nested functions in oracle_expr
   - Keep it simple and mathematical
   - Examples: "orig == trans", "trans == 2*orig", "orig + trans == abs(x)"

✓ **CORRECT**: Write simple mathematical expressions
   - "orig == trans"
   - "trans == 2 * orig"
   - "orig + trans == abs(x)"

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

2. transform_code FORMAT:
   - Lambda: "lambda k: {{**k, 'input': modified}}"
   - Input: 'k' is dictionary of all function arguments
   - Output: New dictionary with modified values
   - Focus on tensor arguments, not config parameters

3. oracle_expr FORMAT (NEW - Framework Independent):
   - Simple mathematical expression, NOT a lambda function
   - Available variables: orig, trans, x, tolerance
   - NO framework-specific functions (torch, tensorflow, etc.)
   - Examples:
     * "orig == trans"  # Equality
     * "trans == 2 * orig"  # Proportional (factor 2)
     * "trans == -orig"  # Negation
     * "orig + trans == abs(x)"  # Composition

4. EXAMPLES:
   
   ✓ Positive Scaling (ReLU):
   {{
     "description": "f(2x) = 2f(x) for positive scaling",
     "category": "linearity",
     "transform_code": "lambda k: {{**k, 'input': 2 * k['input']}}",
     "oracle_expr": "trans == 2 * orig"
   }}
   
   ✓ Idempotency (ReLU):
   {{
     "description": "f(f(x)) = f(x)",
     "category": "idempotency",
     "transform_code": "lambda k: {{**k, 'input': torch.relu(k['input'])}}",
     "oracle_expr": "orig == trans"
   }}
   
   ✓ Absolute Value (ReLU):
   {{
     "description": "f(x) + f(-x) = |x|",
     "category": "composition",
     "transform_code": "lambda k: {{**k, 'input': -k['input']}}",
     "oracle_expr": "orig + trans == abs(x)"
   }}

5. COMMON MISTAKES TO AVOID:
   ❌ For ReLU: "f(-x) = -f(x)" is WRONG
   ❌ Using torch.allclose, np.allclose in oracle_expr
   ❌ Trivial: "Identity transformation" is useless
   ❌ Complex: Keep oracle_expr simple and mathematical

6. OPERATOR-SPECIFIC KNOWLEDGE:
   - ReLU: f(x) = max(0,x), NOT odd function, idempotent
   - Sigmoid: f(x) bounded [0,1], f(-x) = 1-f(x)
   - Tanh: f(x) odd function, f(-x) = -f(x)
   - Softmax: permutation invariant, translation invariant

═══════════════════════════════════════════════════════════════════
OUTPUT: Return ONLY JSON with {{"mrs": [...]}}, no markdown blocks
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
