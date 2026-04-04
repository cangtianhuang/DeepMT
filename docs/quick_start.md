# DeepMT 快速开始指南

## 🚀 最简单的使用方式

### 3行代码完成测试

```python
from deepmt import DeepMT

client = DeepMT()
result = client.test_operator("Add", [1.0, 2.0], "pytorch")
print(result.summary())
```

**输出**：
```
============================================================
DeepMT 测试结果
============================================================
名称: Add
框架: pytorch
总测试数: 3
通过: 3
失败: 0
耗时: 0.12s
============================================================
```

---

## 📚 更多使用示例

### 1. 批量测试多个算子

```python
from deepmt import DeepMT

client = DeepMT()

operators = [
    {"name": "Add", "inputs": [1.0, 2.0]},
    {"name": "Multiply", "inputs": [3.0, 4.0]},
    {"name": "Subtract", "inputs": [10.0, 3.0]},
]

results = client.test_operators(operators, "pytorch")

for result in results:
    print(result.summary())
```

### 2. 使用配置文件

**创建配置文件** `tests/config.yaml`：
```yaml
tests:
  - type: operator
    name: Add
    inputs: [1.0, 2.0]
    framework: pytorch
  - type: operator
    name: Multiply
    inputs: [3.0, 4.0]
    framework: pytorch
```

**运行测试**：
```python
from deepmt import DeepMT

client = DeepMT()
results = client.test_from_config("tests/config.yaml")
```

### 3. 查看测试历史

```python
from deepmt import DeepMT

client = DeepMT()

# 查看特定算子的测试历史
history = client.get_test_history("Add")

# 查看所有失败的测试
failures = client.get_failed_tests(limit=10)
```

### 4. MR生成与测试分离（高级用法）

```python
from deepmt import DeepMT
from mr_generator.operator_mr import OperatorMRGenerator
from mr_generator.knowledge_base import KnowledgeBase
from mr_generator.mr_repository import MRRepository
from ir.converter import IRConverter
from core.test_runner import TestRunner
from core.plugins_manager import PluginsManager
from core.results_manager import ResultsManager

# 步骤1：生成MR并保存
ir_converter = IRConverter()
add_ir = ir_converter.from_operator_name("Add", [1.0, 2.0])

kb = KnowledgeBase()
generator = OperatorMRGenerator(kb)
mrs = generator.generate(add_ir)

mr_repo = MRRepository()
mr_repo.save("Add", mrs)  # 保存到知识库

# 步骤2：从知识库加载MR并测试
loaded_mrs = mr_repo.load("Add")

plugins = PluginsManager()
plugins.load_plugins()
results_manager = ResultsManager()

test_runner = TestRunner(plugins, results_manager)
test_runner.run_with_mrs(add_ir, loaded_mrs, "pytorch")
```

---

## 🎯 关键特性

### ✅ IR完全隐藏
- 用户不需要了解IR的存在
- 系统自动从用户输入创建IR
- 所有IR操作都在内部完成

### ✅ MR生成与测试分离
- MR可以独立生成并保存
- MR可以重用，避免重复生成
- 同一个MR可以测试多个框架

### ✅ 简洁的API
- 3行代码完成测试
- 支持批量测试
- 支持配置文件

---

## 📖 API参考

### DeepMT 类

#### `test_operator(name, inputs, framework, properties=None)`
测试单个算子

**参数**：
- `name`: 算子名称（如 "Add", "Multiply"）
- `inputs`: 输入值列表
- `framework`: 目标框架（"pytorch", "tensorflow", "paddle"）
- `properties`: 算子属性（可选，会自动推断）

**返回**：`TestResult` 对象

#### `test_operators(operators, framework)`
批量测试多个算子

**参数**：
- `operators`: 算子列表，每个元素为 `{"name": str, "inputs": List}`
- `framework`: 目标框架

**返回**：`TestResult` 列表

#### `test_from_config(config_path)`
从配置文件运行测试

**参数**：
- `config_path`: 配置文件路径（YAML格式）

**返回**：`TestResult` 列表

### TestResult 类

#### `summary()`
返回测试摘要字符串

#### `to_dict()`
转换为字典格式

---

## 🔍 内部架构（用户不需要了解）

```
用户输入
  ↓
DeepMT API
  ↓
IR转换器 → IR（内部）
  ↓
MR知识库 ← MRGenerator（独立）
  ↓
TestRunner（使用预生成MR）
  ↓
结果 → 报告
```

---

## 💡 最佳实践

1. **简单测试**：直接使用 `test_operator()`
2. **批量测试**：使用 `test_operators()` 或配置文件
3. **MR重用**：使用MR知识库避免重复生成
4. **结果查询**：使用 `get_test_history()` 查看历史

---

## 🐛 故障排除

### 问题：找不到插件
**解决**：确保已安装对应的框架（如 PyTorch）

### 问题：MR生成失败
**解决**：检查算子名称是否正确，查看日志了解详情

### 问题：测试结果为空
**解决**：检查输入格式是否正确，查看日志了解详情

---

## 📝 更多信息

- 详细文档：`docs/optimization_summary.md`
- 优化计划：`docs/optimization_plan.md`
- 使用示例：`examples/user_friendly_example.py`

