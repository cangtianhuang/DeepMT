# DeepMT 配置指南

## 一、配置文件

项目使用 `config.yaml` 进行配置管理。首次使用前，请配置以下内容：

### 1.1 LLM配置（必需）

LLM现在是MR自动生成的必需组件。请配置API密钥：

```yaml
llm:
  provider: "openai"
  api_key: "your-api-key-here"  # 请填入您的OpenAI API密钥
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2000
```

**获取API密钥**：
1. 访问 https://platform.openai.com/api-keys
2. 创建新的API密钥
3. 将密钥填入 `config.yaml` 的 `llm.api_key` 字段

**或者使用环境变量**：
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 1.2 网络搜索配置

网络搜索工具用于自动获取算子信息：

```yaml
web_search:
  enabled: true
  sources:
    - "pytorch_docs"  # PyTorch官方文档
    - "github"  # GitHub仓库
    - "stackoverflow"  # Stack Overflow
    - "blogs"  # 技术博客
  max_results: 5
  timeout: 10
```

### 1.3 MR生成配置

```yaml
mr_generation:
  use_llm: true  # LLM现在是必需的
  use_template_pool: true
  use_precheck: true
  use_sympy_proof: true
  use_auto_derivation: true
```

---

## 二、快速开始

### 2.1 配置API密钥

编辑 `config.yaml`，填入您的OpenAI API密钥：

```yaml
llm:
  api_key: "sk-..."  # 您的API密钥
```

### 2.2 运行ReLU测试

```bash
python examples/test_relu_mr.py
```

这将：
1. 自动从网络搜索ReLU算子的信息
2. 使用LLM将代码转换为SymPy表达式
3. 自动推导MR
4. 使用SymPy进行形式化证明

---

## 三、功能说明

### 3.1 自动算子信息获取

系统会自动从以下源搜索算子信息：
- **PyTorch官方文档**：获取官方文档和API说明
- **GitHub仓库**：获取源代码实现
- **Stack Overflow**：获取使用示例和常见问题
- **技术博客**：获取详细解释和教程

### 3.2 MR自动生成流程

```
算子名称（如 "ReLU"）
    ↓
[网络搜索] 获取代码、文档、示例
    ↓
[LLM翻译] 代码 → SymPy表达式
    ↓
[MR推导] 基于符号表达式自动推导
    ↓
[快速筛选] 用随机数快速验证
    ↓
[SymPy证明] 形式化验证
    ↓
最终MR列表
```

---

## 四、故障排除

### 4.1 API密钥错误

**错误**：`ValueError: LLM API key is required!`

**解决**：
1. 检查 `config.yaml` 中的 `llm.api_key` 是否设置
2. 或设置环境变量 `OPENAI_API_KEY`
3. 确保API密钥有效

### 4.2 网络搜索失败

**错误**：`No search results found`

**解决**：
1. 检查网络连接
2. 检查 `web_search.enabled` 是否为 `true`
3. 尝试手动提供 `operator_code` 和 `operator_doc`

### 4.3 依赖缺失

**错误**：`ModuleNotFoundError`

**解决**：
```bash
pip install -r requirements.txt
```

---

## 五、示例

### 5.1 基本使用

```python
from ir.schema import OperatorIR
from mr_generator.operator_mr import OperatorMRGenerator
import torch.nn.functional as F

# 创建算子IR
relu_ir = OperatorIR(name="ReLU", inputs=[-1.0, 0.0, 1.0], outputs=[], properties={})

# 创建生成器
generator = OperatorMRGenerator(
    use_llm=True,
    use_auto_derivation=True
)

# 生成MR（自动从网络获取信息）
mrs = generator.generate(
    operator_ir=relu_ir,
    operator_func=F.relu,
    auto_fetch_info=True,
    framework="pytorch"
)
```

### 5.2 手动提供代码

```python
mrs = generator.generate(
    operator_ir=operator_ir,
    operator_code="def relu(x): return max(0, x)",
    operator_doc="ReLU activation function",
    auto_fetch_info=False  # 不使用网络搜索
)
```

---

## 六、注意事项

1. **API成本**：LLM调用会产生费用，请注意使用量
2. **网络依赖**：自动获取算子信息需要网络连接
3. **搜索限制**：某些网站可能有访问频率限制
4. **隐私**：API密钥请妥善保管，不要提交到版本控制系统

---

## 七、支持

如有问题，请查看：
- 日志文件：`data/logs/deepmt.log`
- 文档：`docs/` 目录
- 示例：`examples/` 目录

