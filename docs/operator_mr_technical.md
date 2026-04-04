# 算子层MR生成与测试技术文档

## 1. 概述

本文档详细说明DeepMT项目中算子层蜕变关系（Metamorphic Relation, MR）的自动生成和测试技术实现。

**设计理念**：
- 对于复杂算子（如softmax），预定义性质验证（交换律、单位元等）无法推导出特有的蜕变关系
- LLM 能够理解算子语义，猜想如 "softmax 具有平移不变性" 等复杂MR
- SymPy 作为统一的形式化验证工具，确保MR的正确性
- **框架无关的MR结构**：同一MR可在不同深度学习框架中复用

**核心架构**：LLM猜想 + SymPy验证 + 框架无关MR

---

## 2. 系统架构

### 2.1 整体流程

```
输入：算子IR + (可选) operator_func / operator_code / operator_doc
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段1：信息准备                                                   │
│   - 自动信息获取（网络搜索算子文档/代码）                            │
│   - 代码提取（从 operator_code 参数或网络获取）                     │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段2：MR猜想生成（框架无关的MR结构）                             │
│                                                                 │
│   ┌───────────────────────┐  ┌───────────────────────┐          │
│   │ 来源1：LLM猜想（主要） │  │ 来源2：模板池（辅助） │          │
│   │ 理解算子语义生成MR    │  │ 提供常见数学变换模板  │          │
│   │ → 候选MR              │  │ → 候选MR              │          │
│   └───────────┬───────────┘  └───────────┬───────────┘          │
│               │                          │                      │
│               └──────────────────────────┘                      │
│                              ↓                                  │
│                        合并 & 去重（基于描述）                        │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段3：快速筛选（Pre-check，可选）                                 │
│   - 需要 operator_func 参数                                      │
│   - 用随机输入执行算子，验证MR是否满足                              │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段4：SymPy形式化验证（可选）                                     │
│   - 需要代码或SymPy表达式                                         │
│   - 对所有候选MR进行形式化验证                                     │
│   - 优化：只转换一次代码，所有MR复用同一表达式                         │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│ 输出：MR列表（框架无关结构，标记验证状态 verified=True/False）         │
│   - transform_code: lambda表达式（字典格式）                       │
│   - oracle_expr: 框架无关的数学表达式                             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

```
OperatorMRGenerator (主生成器)
├── OperatorInfoFetcher (信息获取)
├── SympyTranslator (代码到SymPy转换)
├── OperatorLLMMRGenerator (LLM猜想，主要来源)
├── MRTemplatePool (模板池，辅助来源)
├── MRPreChecker (快速筛选)
└── SymPyProver (形式化验证)
```

### 2.3 初始化

`OperatorMRGenerator` 采用无参数初始化，内部自动创建所有必需组件：

```python
generator = OperatorMRGenerator()
```

### 2.4 generate() 方法参数

| 参数 | 类型 | 默认值 | 用途 |
|------|------|--------|------|
| `operator_ir` | OperatorIR | 必需 | 算子IR对象，包含名称和输入 |
| `operator_func` | Callable | None | 用于快速筛选（Pre-check） |
| `operator_code` | str | None | 用于自动推导和SymPy证明 |
| `operator_doc` | str | None | 用于LLM猜想生成MR |
| `auto_fetch_info` | bool | True | 是否自动从网络获取算子信息 |
| `framework` | str | "pytorch" | 框架名称 |
| `sources` | List[str] | None | MR生成来源，可选：`["llm", "template"]`，None表示全部 |
| `use_precheck` | bool | True | 是否进行快速筛选 |
| `use_sympy_proof` | bool | True | 是否进行SymPy证明 |

**关于 `operator_func` 的重要说明**：

```python
# 对于纯Python函数，可以提取源码
def my_add(x, y):
    return x + y
# inspect.getsource(my_add) → 成功

# 对于PyTorch内置算子（C++实现），无法提取源码
import torch.nn.functional as F
# inspect.getsource(F.relu) → 失败！

# 但 operator_func 仍可用于快速筛选（Pre-check）
# 因为快速筛选只需要执行函数，不需要源码
```

### 2.5 框架无关的MR结构

**核心创新**：MR使用框架无关的结构，可在多个深度学习框架中复用

```python
MetamorphicRelation(
    id=str(uuid.uuid4()),
    description="Scaling input by positive factor scales output by same factor",
    transform=lambda k: {**k, 'input': 2 * k['input']},  # 输入变换
    transform_code="lambda k: {**k, 'input': 2 * k['input']}",
    oracle_expr="trans == 2 * orig",  # 框架无关的验证表达式
    category="linearity",
    tolerance=1e-6,
    layer="operator",
    verified=False,
)
```

**关键特性**：

1. **transform_code**: lambda表达式，使用字典格式传递所有函数参数
   - 输入：`k` 是字典，包含所有函数参数
   - 输出：新字典，包含修改后的参数
   - 示例：`"lambda k: {**k, 'input': 2 * k['input']}"`

2. **oracle_expr**: 框架无关的数学表达式
   - 可用变量：`orig`（原始输出）、`trans`（变换后输出）、`x`（原始输入）、`tolerance`（容差）
   - 不使用框架特定函数（如 `torch.allclose`、`np.allclose`）
   - 示例：`"trans == 2 * orig"`、`"trans >= orig"`、`"all(trans == 0)"`

3. **复用性**：同一MR可在PyTorch、TensorFlow、PaddlePaddle等多个框架中使用

---

## 3. 技术实现详解

### 3.1 自动信息获取

#### 3.1.1 技术方案

**组件**：`tools/web_search/operator_fetcher.py`

**功能**：自动从网络搜索获取算子的代码、文档、签名等信息。

**技术栈**：
- `requests`：HTTP请求
- `BeautifulSoup`：HTML解析
- 多源搜索：PyTorch文档、GitHub、Stack Overflow

**实现流程**：

```python
class OperatorInfoFetcher:
    def fetch_operator_info(self, operator_name, framework="pytorch"):
        # 1. 规范化算子名称
        normalized_name = self._normalize_operator_name(operator_name)

        # 2. 从多个源搜索
        search_results = self.search_tool.search_operator(
            operator_name=normalized_name,
            framework=framework,
            sources=["pytorch_docs", "github", "stackoverflow"]
        )

        # 3. 提取算子信息
        operator_info = self.search_tool.extract_operator_info(search_results)

        # 4. 返回信息字典
        return {
            "code": "...",      # 算子代码
            "doc": "...",       # 文档字符串
            "signature": "...", # 函数签名
            "examples": [...]   # 代码示例
        }
```

**搜索源**：
1. **PyTorch官方文档**：直接访问文档页面，提取代码和文档
2. **GitHub仓库**：搜索PyTorch源码中的算子实现
3. **Stack Overflow**：搜索相关问题和答案
4. **技术博客**：搜索相关技术文章

**信息提取**：
- 代码提取：从HTML中提取代码块（```python ... ```）
- 文档提取：提取函数文档字符串
- 签名提取：使用正则表达式提取函数签名

---

### 3.2 代码到SymPy转换

#### 3.2.1 技术方案

**组件**：`mr_generator/operator/sympy_translator.py`

**功能**：将任意Python代码转换为SymPy表达式，支持符号计算。

**技术栈**：
- **LLM翻译**：使用OpenAI GPT将代码翻译为SymPy代码
- **AST解析**：直接解析Python AST并转换为SymPy表达式

**实现流程**：

```python
class SympyTranslator:
    def translate(self, code, func=None, doc=None, signature=None):
        # 1. 获取代码和文档
        if func is not None:
            code = inspect.getsource(func)
            doc = inspect.getdoc(func)

        # 2. 尝试代理路径：LLM → Python参考实现 → AST → SymPy
        if use_proxy_path:
            result = self._try_proxy_path(code, doc, signature)
            if result is not None:
                return result

        # 3. 回退到直接路径：LLM → SymPy表达式代码
        result = self._try_direct_path(code, doc, signature)
        if result is not None:
            return result

        # 4. 最终回退到纯AST解析
        return self.ast_parser.parse_to_sympy(code)

    def _try_proxy_path(self, code, doc, signature):
        """代理路径：LLM → Python参考 → AST → SymPy"""
        python_ref = self._llm_to_python_reference(code, doc, signature)
        if python_ref:
            return self.ast_parser.parse_to_sympy(python_ref)

    def _try_direct_path(self, code, doc, signature):
        """直接路径：LLM → SymPy代码"""
        sympy_code = self._llm_to_sympy_code(code, doc, signature)
        if sympy_code:
            return self._execute_sympy_code(sympy_code)
```

#### 3.2.2 LLM翻译

**提示工程（代理路径）**：

```python
prompt = f"""请将以下Python代码转换为一个清晰的Python参考实现。

函数签名：{signature}

原始代码：
```python
{code}
```

文档：{doc if doc else "无"}

要求：
1. 输出一个清晰的Python函数，参数命名为 x0, x1, x2, ...
2. 使用标准Python数学操作（+, -, *, /, **, max, min, abs等）
3. 条件表达式使用三元表达式 (a if condition else b)
4. 只返回代码，不要包含说明文字

输出格式：
```python
def reference_impl(x0, x1, ...):
    return <表达式>
```"""
```

**提示工程（直接路径）**：

```python
prompt = f"""请将以下Python代码转换为SymPy表达式。

函数签名：{signature}

代码：
```python
{code}
```

文档：{doc if doc else "无"}

要求：
1. 使用符号变量 x0, x1, x2, ... (用 sp.Symbol('x0') 等)
2. 将操作转换为SymPy操作
3. 条件表达式使用 sp.Piecewise
4. 最终表达式赋值给变量 result
5. 只返回代码，不要包含说明文字

输出格式：
```python
import sympy as sp
x0 = sp.Symbol('x0')
x1 = sp.Symbol('x1')
result = <SymPy表达式>
```"""
```

**输出格式**：
```python
import sympy as sp
x, y = sp.symbols('x y')
result = <SymPy表达式>
```

---

### 3.3 LLM猜想（主要来源）

#### 3.3.1 技术方案

**组件**：`mr_generator/operator/operator_llm_mr_generator.py`

**功能**：使用LLM生成Top-5 MR猜想。这是主要的MR来源，能够理解复杂算子语义。

**核心创新：框架无关的MR结构**
- **transform_code**: lambda表达式，使用字典格式传递所有函数参数
- **oracle_expr**: 框架无关的数学表达式，用于验证输出关系
- **优势**：同一MR可在不同深度学习框架（PyTorch、TensorFlow、PaddlePaddle）中复用

**优势**：
- 能够理解算子的数学语义（如softmax的概率归一化）
- 能够猜想复杂的蜕变关系（如softmax的平移不变性）
- 不受预定义性质列表的限制
- **框架无关性**：生成的MR可在多个框架中复用

**技术栈**：
- OpenAI GPT-4
- JSON格式输出
- 提示工程
- 框架无关的数学表达式

**实现流程**：

```python
class OperatorLLMMRGenerator:
    def generate_mr_candidates(self, operator_name, operator_code=None,
                              operator_doc=None, operator_func=None, top_k=5):
        # 1. 自动提取函数签名
        operator_signature = ""
        if operator_func is not None:
            sig = inspect.signature(operator_func)
            operator_signature = str(sig)

        # 2. 构建提示
        prompt = self._build_prompt(operator_name, operator_signature,
                                    operator_code, operator_doc)

        # 3. 调用LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        content = self.llm_client.chat_completion(messages, temperature=0.7)

        # 4. 解析JSON响应
        data = json.loads(content)

        # 5. 转换为MR对象
        mrs = []
        for mr_data in data.get("mrs", [])[:top_k]:
            mr = self._parse_mr_response(mr_data, operator_signature)
            if mr:
                mrs.append(mr)

        return mrs  # 这些是候选MR，需要后续验证
```

#### 3.3.2 MR结构定义

**框架无关的MR结构**：

```json
{
  "description": "MR的数学描述",
  "category": "linearity|monotonicity|idempotency|composition|invariance|symmetry|boundary",
  "transform_code": "lambda k: {**k, 'input': modified_input}",
  "oracle_expr": "trans == 2 * orig"
}
```

**关键特点**：

1. **transform_code**（输入变换）
   - 格式：lambda表达式
   - 输入：`k` 是字典，包含所有函数参数
   - 输出：新字典，包含修改后的参数
   - 示例：`"lambda k: {**k, 'input': 2 * k['input']}"`

2. **oracle_expr**（输出验证）
   - 格式：框架无关的数学表达式
   - 可用变量：`orig`（原始输出）、`trans`（变换后输出）、`x`（原始输入）、`tolerance`（容差）
   - 不使用框架特定函数（如 `torch.allclose`、`np.allclose`）
   - 示例：`"trans == 2 * orig"`、`"trans >= orig"`、`"all(trans == 0)"`

3. **category**（MR分类）
   - `linearity`: 线性性质（trans == k * orig）
   - `monotonicity`: 单调性（trans >= orig）
   - `idempotency`: 幂等性（f(f(x)) = f(x)）
   - `composition`: 复合关系（orig + trans == abs(x)）
   - `invariance`: 不变性（orig == trans）
   - `symmetry`: 对称性（trans == -orig）
   - `boundary`: 边界值性质（all(trans == 0)）

#### 3.3.3 提示工程

**System Prompt**（系统提示）：
```
You are an expert in metamorphic testing for deep learning operators.

Your task is to generate HIGH-QUALITY Metamorphic Relations (MRs) with:
1. transform_code: Lambda to transform inputs
2. oracle_expr: Simple mathematical expression for verification

CRITICAL REQUIREMENTS:
1. QUALITY OVER QUANTITY: Generate 3-5 SOUND MRs
2. transform_code FORMAT (Framework Independent):
   - Lambda: "lambda k: {**k, 'input': modified}"
   - Input: 'k' is dictionary of all function arguments
   - NO framework-specific functions
3. oracle_expr FORMAT (Framework Independent):
   - Simple mathematical expression
   - Variables: orig, trans, x, tolerance
   - NO framework-specific functions (torch, tensorflow, numpy, etc.)
```

**User Prompt**（用户提示）：
```
You are a Deep Learning Metamorphic Testing Expert.
Your task is to generate Metamorphic Relations (MRs) for operator `{operator_name}`.

### Operator Signature
```python
{operator_signature}
```

### Operator Implementation
```python
{operator_code}
```

### Documentation
{operator_doc}

### CRITICAL: Understanding {operator_name}

Before generating MRs, analyze the mathematical properties:
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
- Format: lambda expression like "lambda k: {**k, 'input': modified}"
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
{
  "description": "Scaling input by positive factor scales output by same factor",
  "category": "linearity",
  "transform_code": "lambda k: {**k, 'input': 2 * k['input']}",
  "oracle_expr": "trans == 2 * orig"
}
```

**Example 2: Idempotency (for ReLU)**
```json
{
  "description": "Applying operator twice is same as applying once: f(f(x)) = f(x)",
  "category": "idempotency",
  "transform_code": "lambda k: {**k, 'input': apply_operator(k['input'])}",
  "oracle_expr": "orig == trans"
}
```

**Example 3: Absolute Value Identity (for ReLU)**
```json
{
  "description": "Operator(x) + Operator(-x) equals |x|",
  "category": "composition",
  "transform_code": "lambda k: {**k, 'input': -k['input']}",
  "oracle_expr": "orig + trans == abs(x)"
}
```

**Example 4: Monotonicity (for ReLU)**
```json
{
  "description": "If input increases, output does not decrease",
  "category": "monotonicity",
  "transform_code": "lambda k: {**k, 'input': k['input'] + 1.5}",
  "oracle_expr": "trans >= orig"
}
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
{
  "mrs": [
    {
      "description": "Natural language or mathematical description",
      "category": "linearity|monotonicity|idempotency|composition|invariance|symmetry|boundary",
      "transform_code": "lambda k: {**k, 'input': ...}",
      "oracle_expr": "trans == 2 * orig"
    }
  ]
}
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
```

#### 3.3.4 质量保证机制

**1. 提示工程约束**
- 明确禁止框架特定函数
- 强调质量优于数量
- 提供高质量示例

**2. 后处理验证**
- 解析并验证lambda表达式可执行性
- 检查必需字段（description, transform_code, oracle_expr）
- 简单测试transform函数

**3. 框架无关性检查**
- 自动检测并拒绝包含框架特定函数的MR
- 使用简单数学操作符（+, -, *, /, abs等）
- 使用 `apply_operator()` 占位符表示算子本身应用

---

### 3.4 模板池生成（辅助来源）

#### 3.4.1 技术方案

**组件**：`mr_generator/base/mr_templates.py`

**功能**：使用预定义的数学变换模板生成MR。作为LLM猜想的补充，提供常见数学性质。

**定位**：辅助来源，覆盖LLM可能遗漏的基础数学性质。

**核心特性：框架无关的MR结构**
- 模板也使用与LLM生成相同的框架无关结构
- `oracle_expr`: 框架无关的数学表达式
- 支持在多个框架中复用

**技术栈**：
- 20+ 常见数学变换模板
- 算子到模板的映射
- YAML配置文件管理

**模板数据结构**：

```python
@dataclass
class MRTemplate:
    """MR模板数据结构"""
    name: str                          # 模板名称
    description: str                    # MR描述
    transform_func: Callable            # 输入变换函数
    oracle_expr: str                   # 框架无关的验证表达式
    category: str = "general"          # MR类别
    min_inputs: int = 1               # 最小输入数量
    max_inputs: Optional[int] = None  # 最大输入数量（None表示无限制）
```

**配置文件格式**（`config/mr_templates.yaml`）：

```yaml
templates:
  # 交换律模板
  commutative:
    name: "Commutative Law"
    description: "交换律: f(x, y) == f(y, x)"
    transform_code: "lambda x, y: (y, x)"
    oracle_expr: "orig == trans"
    category: "symmetry"
    min_inputs: 2
    max_inputs: 2

  # 幂等性模板
  idempotent:
    name: "Idempotent Property"
    description: "幂等性: f(f(x)) == f(x)"
    transform_code: "lambda x: (apply_operator(x),)"
    oracle_expr: "orig == trans"
    category: "idempotency"
    min_inputs: 1

  # 单位元模板
  identity:
    name: "Identity Element"
    description: "单位元: f(x, 0) == x"
    transform_code: "lambda x: (x, 0)"
    oracle_expr: "orig == trans"
    category: "identity"
    min_inputs: 2

operator_mr_mapping:
  # 算子到模板的映射
  ReLU: ["idempotent", "positive_scaling"]
  Add: ["commutative", "identity", "associative"]
  Multiply: ["commutative", "zero"]
```

**模板类型**：

| 模板名称 | 数学描述 | oracle_expr | 适用算子 |
|---------|---------|-------------|---------|
| commutative | `f(x, y) == f(y, x)` | `orig == trans` | Add, Multiply, Max, Min |
| associative | `f(f(x, y), z) == f(x, f(y, z))` | `orig == trans` | Add, Multiply |
| identity | `f(x, 0) == x` | `orig == trans` | Add |
| zero | `f(x, 0) == 0` | `all(trans == 0)` | Multiply |
| distributive | `f(x, g(y, z)) == g(f(x, y), f(x, z))` | `orig == trans` | Multiply, Add |
| idempotent | `f(f(x)) == f(x)` | `orig == trans` | ReLU, Abs |
| transpose | `f(A^T) == f(A)^T` | `orig == trans` | MatMul |

**实现示例**：

```python
class MRTemplatePool:
    def _load_config(self):
        """从配置文件加载模板"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 加载算子到MR的映射
        self.operator_mr_mapping = config.get("operator_mr_mapping", {})

        # 加载模板定义
        templates_config = config.get("templates", {})

        for template_name, template_data in templates_config.items():
            # 解析transform_code为函数
            transform_code = template_data.get("transform_code", "")
            transform_func = eval(transform_code) if transform_code else lambda *args: args

            # 解析 oracle_expr（框架无关的表达式）
            oracle_expr = template_data.get("oracle_expr", "orig == trans")

            template = MRTemplate(
                name=template_data.get("name", template_name),
                description=template_data.get("description", ""),
                transform_func=transform_func,
                oracle_expr=oracle_expr,
                category=template_data.get("category", "general"),
                min_inputs=template_data.get("min_inputs", 1),
                max_inputs=template_data.get("max_inputs"),
            )

            self.templates[template_name] = template
```

**生成流程**：

```python
def generate_mr_candidates(self, operator_name, operator_func=None, num_inputs=None):
    """
    为算子生成MR候选列表（路径B：模板池）

    Returns:
        MR候选列表（使用框架无关的结构）
    """
    # 1. 获取适用的模板
    templates = self.get_applicable_templates(
        operator_name, operator_func=operator_func, num_inputs=num_inputs
    )

    # 2. 为每个模板创建MR
    candidates = []
    for template in templates:
        mr = self.create_mr_from_template(template)
        candidates.append(mr)

    return candidates

def create_mr_from_template(self, template: MRTemplate) -> MetamorphicRelation:
    """从模板创建MR对象（框架无关的结构）"""
    return MetamorphicRelation(
        id=str(uuid.uuid4()),
        description=template.description,
        transform=template.transform_func,
        transform_code=inspect.getsource(template.transform_func).strip(),
        oracle_expr=template.oracle_expr,  # 框架无关的验证表达式
        category=template.category,
        tolerance=1e-6,
        layer="operator",
        verified=False,
    )
```

---

### 3.5 快速筛选（Pre-check）

#### 3.5.1 技术方案

**组件**：`mr_generator/operator/mr_precheck.py`

**功能**：使用随机测试用例快速过滤明显不满足的MR。

**前置条件**：需要 `operator_func` 参数（可调用的函数对象）

**作用**：在SymPy形式化验证之前，快速排除明显错误的MR猜想。

**技术栈**：
- NumPy：数值计算
- PyTorch（可选）：支持Tensor类型
- 随机测试用例生成

**通过率阈值**：**80%**（通过率 ≥ 80% 时保留MR）

**支持的输入类型**：
- 标量（int, float）
- NumPy数组（np.ndarray）
- PyTorch张量（torch.Tensor）
- 列表/元组

**支持的期望关系类型**：

| expected | 含义 | 验证方式 |
|----------|------|----------|
| equal | 输出相等 | `np.allclose(orig, trans)` |
| invariant | 同 equal | - |
| negate | 输出取反 | `np.allclose(orig, -trans)` |
| proportional | 输出成比例 | 检查比值是否为常数 |
| first_input | 输出等于第一个输入 | `np.allclose(trans, input[0])` |
| zero | 输出为零 | `np.allclose(trans, 0)` |
| idempotent | 幂等性 | `np.allclose(f(trans), trans)` |

**实现流程**：

```python
class MRPreChecker:
    PASS_RATE_THRESHOLD = 0.8  # 通过率阈值

    def filter_mrs(self, operator_func, mr_candidates, original_inputs, framework="pytorch"):
        # 1. 生成随机测试用例（支持Tensor）
        test_cases = self.generate_test_inputs(original_inputs, num_cases=5)

        # 2. 对每个MR进行测试
        valid_mrs = []
        for mr in mr_candidates:
            passed = 0
            for test_input in test_cases:
                # 执行原始输入
                original_output = operator_func(*test_input)

                # 执行变换后的输入
                transformed_input = mr.transform(*test_input)
                transformed_output = operator_func(*transformed_input)

                # 检查期望关系（支持新的类型）
                if self._check_expected_relation(
                    orig_output=original_output,
                    trans_output=transformed_output,
                    expected=mr.expected,
                    tolerance=mr.tolerance,
                    orig_input=test_input[0],  # 用于 first_input 检查
                    operator_func=operator_func,  # 用于 idempotent 检查
                ):
                    passed += 1

            # 如果通过率 ≥ 80%，保留该MR
            if passed / len(test_cases) >= self.PASS_RATE_THRESHOLD:
                valid_mrs.append(mr)

        return valid_mrs
```

**测试用例生成**（支持PyTorch Tensor）：

```python
def _generate_random_value(self, original, framework="pytorch"):
    # PyTorch Tensor
    if framework == "pytorch" and isinstance(original, torch.Tensor):
        shape, dtype, device = original.shape, original.dtype, original.device
        if dtype in (torch.float32, torch.float64):
            return torch.randn(shape, dtype=dtype, device=device) * 10.0
        else:
            return torch.randint(-10, 10, shape, dtype=dtype, device=device)

    # NumPy数组
    if isinstance(original, np.ndarray):
        return np.random.uniform(-10.0, 10.0, size=original.shape)

    # 标量
    if isinstance(original, int):
        return np.random.randint(-10, 10)
    if isinstance(original, float):
        return np.random.uniform(-10.0, 10.0)

    return original
```

**关系检查**（扩展版本）：

```python
def _check_expected_relation(self, orig_output, trans_output, expected,
                             tolerance, orig_input=None, operator_func=None):
    orig = self._to_numpy(orig_output)
    trans = self._to_numpy(trans_output)

    if expected == "equal" or expected == "invariant":
        return np.allclose(orig, trans, atol=tolerance)

    elif expected == "negate":
        return np.allclose(orig, -trans, atol=tolerance)

    elif expected == "first_input":
        first = self._to_numpy(orig_input)
        return np.allclose(trans, first, atol=tolerance)

    elif expected == "zero":
        return np.allclose(trans, 0, atol=tolerance)

    elif expected == "idempotent":
        # f(f(x)) == f(x)
        nested = operator_func(trans_output)
        return np.allclose(self._to_numpy(nested), trans, atol=tolerance)

    # ...
```

---

### 3.6 SymPy形式化验证

#### 3.6.1 技术方案

**组件**：`mr_generator/operator/sympy_prover.py`

**功能**：使用SymPy进行形式化数学验证。这是统一的验证工具，对所有MR猜想进行形式化证明。

**前置条件**：需要代码或SymPy表达式

**技术栈**：
- SymPy：符号数学库
- SympyTranslator：代码到SymPy表达式的转换器

**核心价值**：
- **统一验证**：所有MR猜想（无论来自LLM还是模板池）都由SymPy统一验证
- **形式化证明**：使用数学证明而非经验测试，确保MR的正确性
- **复用表达式**：代码只转换一次，所有MR验证复用同一SymPy表达式（性能优化）

**实现流程**：

```python
class SymPyProver:
    def __init__(self, code_translator=None):
        # 接受共享的 code_translator（推荐）
        self.code_translator = code_translator or SympyTranslator()

    def prove_mrs(self, mrs, operator_code=None, sympy_expr=None, ...):
        # 优化：只转换一次代码为 SymPy，对所有 MR 复用
        if sympy_expr is None:
            sympy_expr = self.code_to_sympy(code=operator_code, ...)

        # 对每个 MR 进行证明（复用 sympy_expr）
        proven_mrs = []
        for mr in mrs:
            is_proven, _ = self.prove_mr_with_expr(mr, sympy_expr, num_inputs)
            if is_proven:
                mr.verified = True
                proven_mrs.append(mr)

        return proven_mrs
```

#### 3.6.2 验证方法

**相等关系证明**：

```python
def _prove_equal(self, sympy_expr, mr, num_inputs):
    symbols = [sp.Symbol(f"x{i}") for i in range(num_inputs)]

    # 原始表达式
    original = sympy_expr

    # 变换后的表达式
    transformed_inputs = mr.transform(*symbols)
    # 重要：使用 simultaneous=True 确保所有替换同时进行
    transformed = sympy_expr.subs({
        symbols[i]: transformed_inputs[i]
        for i in range(num_inputs)
    }, simultaneous=True)

    # 证明：simplify(LHS - RHS) == 0
    diff = sp.simplify(original - transformed)
    return diff == 0
```

**比例关系证明**：

```python
def _prove_proportional(self, sympy_expr, mr, num_inputs):
    symbols = [sp.Symbol(f"x{i}") for i in range(num_inputs)]

    original = sympy_expr
    transformed_inputs = mr.transform(*symbols)
    transformed = sympy_expr.subs({
        symbols[i]: transformed_inputs[i]
        for i in range(num_inputs)
    }, simultaneous=True)

    # 证明：transformed / original 是常数
    ratio = sp.simplify(transformed / original)
    return ratio.is_constant()
```

#### 3.6.3 测试覆盖

**测试体系**：完整的单元测试覆盖，确保 SymPy 形式化证明功能的正确性和完备性

1. **AST 解析器测试**（`tests/test_ast_parser.py`，26个测试用例）
   - 基础测试：简单数学表达式（加减乘除、幂运算、一元取反）
   - 中级测试：复合表达式和函数调用（abs, max, min, sqrt, exp, log）
   - 高级测试：条件表达式、三角函数、复杂组合、多项式、有理函数
   - 边界测试：常量函数、恒等函数、无效语法、无return语句

2. **SymPy 翻译器测试**（`tests/test_sympy_translator.py`，20个测试用例）
   - 基础测试：简单函数转换（加法、乘法、abs）
   - 中级测试：复合表达式和数学函数（max, sqrt, power）
   - 高级测试：ReLU、条件表达式、多项式、有理函数、嵌套函数、三角函数
   - 边界测试：无代码、无效代码、常量函数、恒等函数、带签名和文档

3. **SymPy 证明引擎测试**（`tests/test_sympy_prover.py`，22个测试用例）
   - 基础测试：代码转SymPy、加法交换律、乘法交换律、恒等变换
   - 中级测试：ReLU正缩放、abs对称性、平方对称性、加法结合律
   - 高级测试：多项式缩放、SymPy表达式复用、批量证明、复杂表达式
   - 边界测试：无效代码、无代码、错误MR、不变性类型、空MR列表、自动推断输入数量
   - 真实场景测试：ReLU幂等性、abs三角不等式、多项式齐次性

**测试结果**：
- AST 解析器：26/26 通过（100%）
- SymPy 翻译器：20/20 通过（100%）
- SymPy 证明引擎：22/22 通过（100%）

**关键修复**：
1. 修复 `ir/schema.py` 中 `MetamorphicRelation` 缺少 `expected` 字段的问题
2. 修复 `sympy_prover.py` 中替换逻辑，使用 `simultaneous=True` 确保同时替换（解决交换律证明失败问题）

---

## 4. 框架适配器设计

### 4.1 框架适配器的作用

由于MR使用框架无关的结构，需要框架适配器将MR转换为特定框架的测试代码。

**适配器职责**：

1. **transform转换**：将框架无关的transform lambda表达式转换为框架特定的代码
2. **oracle_expr转换**：将框架无关的数学表达式转换为框架特定的验证代码
3. **测试执行**：执行测试并收集结果

### 4.2 框架适配器实现

**PyTorch适配器**（`plugins/pytorch_plugin.py`）：

```python
class PyTorchPlugin:
    def ir_to_code(self, ir_object, mr):
        """
        将IR和MR转换为PyTorch测试代码
        """
        operator_name = ir_object.name

        # 1. 生成原始调用代码
        original_call = f"{operator_name}(**original_kwargs)"

        # 2. 生成变换后的调用代码
        # mr.transform 是字典格式的lambda表达式
        transformed_kwargs_code = self._generate_transform_code(mr.transform)
        transformed_call = f"{operator_name}(**{transformed_kwargs_code})"

        # 3. 生成oracle验证代码
        # mr.oracle_expr 是框架无关的表达式，需要转换为PyTorch代码
        oracle_code = self._generate_oracle_code(mr.oracle_expr)

        return f"""
import torch

# 原始调用
original_output = {original_call}

# 变换后的调用
{transformed_kwargs_code}
transformed_output = {transformed_call}

# 验证
{oracle_code}
"""

    def _generate_transform_code(self, transform_func):
        """生成变换代码（从lambda表达式）"""
        # 获取lambda表达式的源代码
        import inspect
        transform_code = inspect.getsource(transform_func)
        # 处理字典格式：{**k, 'input': 2 * k['input']}
        # 转换为PyTorch代码
        return transform_code

    def _generate_oracle_code(self, oracle_expr):
        """
        将框架无关的表达式转换为PyTorch代码

        示例：
        - "orig == trans" → "torch.allclose(original_output, transformed_output)"
        - "trans == 2 * orig" → "torch.allclose(transformed_output, 2 * original_output)"
        - "trans >= orig" → "torch.all(transformed_output >= original_output)"
        - "all(trans == 0)" → "torch.all(transformed_output == 0)"
        """
        # 使用简单的字符串替换和解析
        # 更复杂的表达式可能需要AST解析

        # 常见模式替换
        replacements = {
            r'\borig\b': 'original_output',
            r'\btrans\b': 'transformed_output',
            r'\b==\b': 'torch.allclose',  # 对于整个表达式
            r'\b>\b': '>',
            r'\b<\b': '<',
            r'\ball\(([^)]+)\)\b': r'torch.all(\1)',
        }

        # 应用替换（简化版）
        code = oracle_expr
        for pattern, replacement in replacements.items():
            code = re.sub(pattern, replacement, code)

        return code
```

**TensorFlow适配器**（`plugins/tensorflow_plugin.py`）：

```python
class TensorFlowPlugin:
    def _generate_oracle_code(self, oracle_expr):
        """
        将框架无关的表达式转换为TensorFlow代码

        示例：
        - "orig == trans" → "tf.reduce_all(tf.equal(original_output, transformed_output))"
        - "trans == 2 * orig" → "tf.reduce_all(tf.equal(transformed_output, 2 * original_output))"
        """
        code = oracle_expr
        replacements = {
            r'\borig\b': 'original_output',
            r'\btrans\b': 'transformed_output',
            r'\b==\b': 'tf.reduce_all(tf.equal',
            r'\b>\b': '>',
            r'\b<\b': '<',
            r'\ball\(([^)]+)\)\b': r'tf.reduce_all(\1)',
        }

        for pattern, replacement in replacements.items():
            code = re.sub(pattern, replacement, code)

        return code
```

### 4.3 oracle_expr转换规则

**常见转换模式**：

| 框架无关表达式 | PyTorch | TensorFlow | PaddlePaddle |
|----------------|----------|-------------|--------------|
| `orig == trans` | `torch.allclose(orig, trans)` | `tf.reduce_all(tf.equal(orig, trans))` | `paddle.allclose(orig, trans)` |
| `trans == 2 * orig` | `torch.allclose(trans, 2 * orig)` | `tf.reduce_all(tf.equal(trans, 2 * orig))` | `paddle.allclose(trans, 2 * orig)` |
| `trans >= orig` | `torch.all(trans >= orig)` | `tf.reduce_all(trans >= orig)` | `paddle.all(trans >= orig)` |
| `all(trans == 0)` | `torch.all(trans == 0)` | `tf.reduce_all(tf.equal(trans, 0))` | `paddle.all(trans == 0)` |
| `abs(x)` | `torch.abs(x)` | `tf.abs(x)` | `paddle.abs(x)` |

---

## 5. 完整示例

### 5.1 基本使用

```python
from ir.schema import OperatorIR
from mr_generator.operator.operator_mr import OperatorMRGenerator
import torch.nn.functional as F

# 1. 创建算子IR（只需要name）
relu_ir = OperatorIR(name="ReLU")

# 2. 创建MR生成器
generator = OperatorMRGenerator()

# 3. 生成MR（默认使用所有来源 + 筛选 + 证明）
mrs = generator.generate(
    operator_ir=relu_ir,
    operator_func=F.relu,
    auto_fetch_info=True,
    framework="pytorch"
)

for mr in mrs:
    print(f"MR: {mr.description}, 已验证: {mr.verified}")
    print(f"  Transform: {mr.transform_code}")
    print(f"  Oracle: {mr.oracle_expr}")
```

### 5.2 仅使用LLM生成MR（无验证）

```python
generator = OperatorMRGenerator()
mrs = generator.generate(
    operator_ir=OperatorIR(name="ReLU"),
    operator_func=F.relu,
    operator_doc="ReLU: f(x) = max(0, x)",
    auto_fetch_info=False,
    sources=["llm"],        # 仅使用LLM
    use_precheck=False,     # 不筛选
    use_sympy_proof=False,  # 不证明
)
```

### 5.3 LLM生成 + 筛选 + 证明

```python
mrs = generator.generate(
    operator_ir=OperatorIR(name="ReLU"),
    operator_func=F.relu,   # 用于筛选
    operator_code="def relu(x): return max(0, x)",  # 用于证明
    operator_doc="ReLU: f(x) = max(0, x)",
    auto_fetch_info=False,
    sources=["llm"],
    use_precheck=True,
    use_sympy_proof=True,
)
```

### 5.4 测试执行（使用框架适配器）

```python
from core.test_runner import TestRunner
from core.plugins_manager import PluginsManager

# 1. 初始化插件管理器
plugins_manager = PluginsManager()
plugins_manager.load_plugins()

# 2. 创建测试执行器
test_runner = TestRunner(plugins_manager, results_manager)

# 3. 执行测试
results = test_runner.run_with_mrs(
    ir_object=relu_ir,
    mrs=mrs,
    target_framework="pytorch"
)

# 4. 查看结果
for result in results:
    print(f"MR: {result['mr'].description}")
    print(f"状态: {result['status']}")
    if result['status'] == 'success':
        print(f"比对结果: {result['comparison']}")
```

---

## 6. 技术特点

### 6.1 LLM猜想 + SymPy验证

- **LLM猜想**：主要来源，能理解复杂算子语义
- **模板池**：辅助来源，提供常见数学变换
- **快速筛选**：使用随机输入快速过滤错误猜想
- **SymPy验证**：统一的形式化验证工具

### 6.2 框架无关的MR结构

**核心创新**：
- MR使用框架无关的结构
- 同一MR可在多个框架中复用
- 框架适配器负责将MR转换为框架特定的代码

**优势**：
- 减少代码重复
- 提高MR的复用性
- 简化多框架支持

### 6.3 验证流程

```
MR猜想（来自LLM/模板池，框架无关结构）
  ↓ 快速筛选（需要operator_func，可选）
  ↓ SymPy验证（需要代码，可选）
  ↓ 输出MR列表（框架无关结构，标记 verified=True/False）
  ↓ 框架适配器（PyTorch/TensorFlow/PaddlePaddle）
  ↓ 框架特定的测试代码
```

### 6.4 自动化

- **自动信息获取**：从网络自动搜索算子信息
- **自动代码转换**：将任意代码转换为SymPy表达式
- **自动MR生成**：从LLM猜想和模板池生成MR
- **自动测试执行**：自动生成和执行测试代码

### 6.5 可扩展性

- **模块化设计**：各组件独立，易于扩展
- **插件化架构**：支持多种框架
- **配置驱动**：模板和映射可通过配置管理

---

## 7. 性能优化

### 7.1 SymPy表达式复用

**优化策略**：代码只转换一次，所有MR验证复用同一SymPy表达式

```python
# 在generate()方法中预先转换
sympy_expr = self.code_translator.translate(code=operator_code, ...)

# 在prove_mrs()中复用
proven_mrs = self.sympy_prover.prove_mrs(
    mrs=candidate_mrs,
    sympy_expr=sympy_expr,  # 传递已转换的表达式
    ...
)
```

**性能提升**：
- 避免重复转换代码
- 减少LLM调用次数
- 加快证明速度

### 7.2 快速筛选优先

- 快速筛选在SymPy验证之前执行
- 使用随机输入快速排除明显错误的MR
- 减少后续SymPy验证的计算量

### 7.3 并行处理

- 多个MR可以并行生成和证明
- 测试用例可以并行执行

---

## 8. 局限性

### 8.1 LLM依赖

- 需要API密钥和网络连接
- 生成质量依赖于提示工程
- 成本考虑（API调用费用）

### 8.2 代码转换限制

- 复杂代码可能无法完全转换为SymPy
- AST解析只支持标准Python语法
- 某些框架特定操作可能无法转换

### 8.3 SymPy验证限制

- SymPy只能验证符号表达式
- 某些性质可能无法用SymPy验证（如数值稳定性）
- 浮点数精度问题

### 8.4 框架适配器复杂度

- oracle_expr转换可能需要复杂的AST解析
- 某些复杂的数学表达式可能难以精确转换
- 需要为每个框架单独实现适配器

---

## 9. 未来改进

1. **提示工程优化**：改进LLM提示，提高MR猜想质量
2. **Z3集成**：使用Z3求解器处理SymPy无法验证的复杂约束
3. **更多算子支持**：扩展对更多深度学习算子的支持
4. **并行优化**：实现并行MR生成和验证
5. **缓存增强**：缓存LLM响应和SymPy转换结果
6. **框架适配器增强**：使用AST解析实现更精确的oracle_expr转换

---

## 10. 参考资源

- **设计文档**：`docs/design.md`
- **开发状态**：`docs/status.md`
- **示例代码**：`examples/test_relu_mr.py`
- **API文档**：`api/deepmt.py`
