# DeepMT 项目开发状态与重点

## 📊 当前开发状态

根据 `docs/design.md` 的设计思路，项目当前处于**阶段3.5-4**（多源融合MR自动生成引擎完成，模型层开发中）。

### ✅ 已完成的核心模块

#### 1. **IR层（统一中间表示）**
- ✅ `ir/schema.py` - 定义了 `OperatorIR`, `ModelIR`, `ApplicationIR` 和 `MetamorphicRelation` 数据结构
- ✅ `ir/converter.py` - IR转换器（用户输入 -> IR）
  - 已清理 `_infer_operator_properties` 相关设计，算子属性由后续步骤推断
- ✅ `core/ir_manager.py` - 实现了IR的加载、保存、验证功能

#### 2. **多源融合MR生成引擎（算子层）**

##### 2.1 核心生成器
- ✅ `mr_generator/operator/operator_mr.py` - 算子层MR生成器（LLM猜想 + SymPy验证）
  - 支持自动信息获取（网络搜索）
  - 支持LLM猜想（主要来源）
  - 支持模板池生成（辅助来源）
  - 支持快速筛选（Pre-check）
  - 支持SymPy形式化验证

##### 2.2 MR生成组件
- ✅ `mr_generator/operator/operator_llm_mr_generator.py` - LLM MR猜想生成器（主要来源）
  - 使用LLM生成Top-5 MR猜想
  - 能理解复杂算子语义（如softmax的平移不变性）
  - 支持代码、文档、签名作为输入
- ✅ `mr_generator/base/mr_templates.py` + `config/mr_templates.yaml` - MR模板池（辅助来源）
   - 常见数学变换模板
   - 交换律、结合律、单位元、分配律、转置等
- ✅ `mr_generator/operator/mr_precheck.py` - 快速筛选器
  - 使用5组随机输入快速测试
  - 过滤明显不满足的MR
- ✅ `mr_generator/operator/sympy_prover.py` - SymPy形式化验证引擎
  - 统一的MR验证工具
  - 使用 `code_to_sympy` 动态将Python代码转换为SymPy表达式
  - 使用 `simplify(LHS - RHS) == 0` 进行验证
- ❌ `mr_generator/operator/mr_deriver.py` - 已删除
  - 预定义性质验证能力有限，无法处理复杂算子
  - 功能已被LLM猜想 + SymPy验证架构取代

##### 2.3 工具层
- ✅ `tools/llm/client.py` - 通用LLM客户端
  - 支持多种提供商（OpenAI, Anthropic等）
  - 统一接口：`chat_completion()`
- ✅ `mr_generator/operator/sympy_translator.py` - 代码到SymPy转换器（算子相关）
  - LLM翻译 + AST解析
  - 支持任意Python代码转换
  - 已移至operator目录
- ✅ `mr_generator/operator/ast_parser.py` - AST解析器（算子相关）
  - 解析Python AST并转换为SymPy表达式
  - 已移至operator目录
  - ✅ 完整的单元测试覆盖（26个测试用例）
- ✅ `tools/web_search/search_tool.py` - 通用网络搜索工具
  - 支持PyTorch文档、GitHub、Stack Overflow等
- ✅ `tools/web_search/operator_fetcher.py` - 算子信息获取器
  - 自动从网络搜索获取算子代码、文档、签名

##### 2.4 知识库和持久化
- ✅ `mr_generator/base/mr_repository.py` - MR知识库（持久化存储）
  - SQLite数据库存储
  - 支持MR版本管理
- ✅ `mr_generator/base/knowledge_base.py` + `config/knowledge_base.yaml` - 三层知识库
   - 支持算子、模型、应用三个级别的知识
   - 使用配置文件注册，便于后续拓展开发
   - 配置文件位于 `mr_generator/config/` 目录，便于增量开发
   - 已从operator目录重构至base目录

#### 3. **微内核框架**
- ✅ `core/scheduler.py` - 任务调度器，协调整个测试流程
- ✅ `core/test_runner.py` - 测试执行器（MR生成与测试分离）
- ✅ `core/plugins_manager.py` - 插件管理器，支持动态加载框架插件
- ✅ `core/results_manager.py` - 结果管理器，负责结果比对、存储和统计
- ✅ `core/logger.py` - 统一日志系统（使用Python内置logging工具）

#### 4. **框架适配插件**
- ✅ `plugins/pytorch_plugin.py` - PyTorch插件，支持多种算子：
  - Add, Multiply, Subtract, Divide
  - MatMul, Pow, Sum, Mean
  - ReLU, Sigmoid, Tanh

#### 5. **缺陷分析**
- ✅ `analysis/defect_classifier.py` - 缺陷分类器，支持多种比对模式：
  - 相等检查（equal）
  - 比例检查（proportional）
  - 不变性检查（invariant）
  - 单调性检查（monotonic）

#### 6. **用户友好API**
- ✅ `api/deepmt.py` - 主API类
  - 隐藏IR细节
  - 自动MR生成和测试
  - 简洁的方法接口

#### 7. **示例和演示代码**
- ✅ `examples/test_relu_mr.py` - ReLU MR生成完整示例
- ✅ `examples/auto_mr_generation_example.py` - 自动MR生成示例
- ✅ `examples/auto_derivation_example.py` - 自动推导示例
- ✅ `demo/quick_demo.py` - 完整的端到端演示

#### 8. **配置和文档**
- ✅ `config.yaml.example` - 配置文件示例
- ✅ `README_CONFIG.md` - 配置指南
- ✅ 完整的文档体系

#### 9. **测试体系**
- ✅ `tests/test_ast_parser.py` - AST解析器单元测试（26个测试用例）
  - 基础测试：简单数学表达式
  - 中级测试：复合表达式和函数调用
  - 高级测试：条件表达式、三角函数和复杂组合
  - 边界测试：特殊情况和错误处理
- ✅ `tests/test_sympy_translator.py` - SymPy翻译器单元测试（20个测试用例）
  - 基础测试：简单函数转换
  - 中级测试：复合表达式和数学函数
  - 高级测试：复杂函数和特殊情况
  - 边界测试：特殊情况和错误处理
- ✅ `tests/test_sympy_prover.py` - SymPy证明引擎单元测试（22个测试用例）
  - 基础测试：简单的MR证明
  - 中级测试：复杂的MR证明
  - 高级测试：复杂MR和批量证明
  - 边界测试：特殊情况和错误处理
  - 真实场景测试：实际算子的MR证明

---

## 🎯 当前应该重点做的开发

### **优先级1：完善算子层功能**

#### 1.1 代码重构和优化
- ✅ 将 `tools/llm/ast_parser.py` 和 `tools/llm/code_translator.py` 移至 `mr_generator/operator/`
- ✅ 将 `mr_generator/base/mr_templates.py` 改为配置文件（YAML）
- ✅ 重构 `mr_generator/operator/knowledge_base.py` 至 `mr_generator/base/`，支持三层知识库
- ✅ 改进 `mr_generator/operator/sympy_prover.py`，使用 `code_to_sympy` 动态转换而非硬编码映射
- ✅ 重命名 `llm_mr_generator.py` 为 `operator_llm_mr_generator.py`，专为算子MR生成
- ✅ 清理 `ir/converter.py` 中的 `_infer_operator_properties` 相关设计
- ✅ 将 `core/logger.py` 改为使用Python内置的logging工具
- ✅ 修复 `ir/schema.py` 中 `MetamorphicRelation` 缺少 `expected` 字段的问题
- ✅ 修复 `sympy_prover.py` 中替换逻辑，使用 `simultaneous=True` 确保同时替换

#### 1.2 测试覆盖
- ✅ 完成 AST 解析器的完整单元测试（26个测试用例，100%通过）
- ✅ 完成 SymPy 翻译器的完整单元测试（20个测试用例，100%通过）
- ✅ 完成 SymPy 证明引擎的完整单元测试（22个测试用例，100%通过）
- ✅ 测试覆盖从简单到复杂的各种场景
- ✅ 验证了 SymPy 形式化证明功能的正确性和完备性

#### 1.3 扩展算子支持
- [ ] 在模板池中添加更多常见算子：
  - 矩阵运算：`Transpose`, `Inverse`, `Eigenvalue`
  - 激活函数：`ReLU`, `Sigmoid`, `Tanh`（部分已完成）
  - 归一化：`BatchNorm`, `LayerNorm`
- [ ] 在 `PyTorchPlugin` 中添加对应的算子映射

#### 1.4 增强MR生成能力
- [ ] 实现基于Z3的复杂约束求解
- [ ] 添加数值稳定性相关的MR（如浮点数精度问题）
- [ ] 实现MR的组合和链式应用

### **优先级2：完善结果分析与报告**

#### 2.1 缺陷分类细化
- [ ] 扩展缺陷类型：
  - 数值偏差（Numerical Deviation）
  - 梯度错误（Gradient Error）
  - API不一致性（API Inconsistency）
  - 性能问题（Performance Issue）
- [ ] 实现缺陷最小化算法（Minimizer）

#### 2.2 报告生成
- [ ] 实现 `analysis/report_generator.py`：
  - HTML报告生成
  - 缺陷统计图表
  - 复现代码生成
- [ ] 实现 `analysis/visualizer.py`：
  - 测试结果可视化
  - 缺陷分布图

### **优先级3：扩展到模型层**

#### 3.1 模型IR扩展
- [ ] 完善 `ModelIR` 数据结构：
  - 支持层类型定义
  - 支持连接关系描述
  - 支持模型参数

#### 3.2 模型层MR生成
- [ ] 实现 `mr_generator/model/model_mr.py`：
  - 基于网络拓扑的MR生成
  - 数据增强策略集成（使用Albumentations）
  - 结构不变性MR

#### 3.3 模型测试支持
- [ ] 扩展插件以支持模型执行
- [ ] 实现模型加载和推理功能

### **优先级4：应用层LLM MR生成**

#### 4.1 应用层MR生成
- [ ] 完善 `mr_generator/application/app_mr.py`：
  - 自然语言MR生成
  - MR描述到代码的转换
  - 语义验证

---

## 🔧 关键技术实现

### 1. LLM猜想 + SymPy验证（算子层）

**设计理念**：
- 对于复杂算子（如softmax），预定义性质验证无法推导出特有的蜕变关系
- LLM 能够理解算子语义，猜想复杂的MR（如softmax的平移不变性）
- SymPy 作为统一的形式化验证工具

**流程**：
1. **自动信息获取**：从网络搜索获取算子代码、文档、签名
2. **LLM猜想**（主要来源）：使用LLM生成Top-5 MR猜想
3. **模板池**（辅助来源）：使用常见数学变换模板
4. **快速筛选**：5组随机输入快速过滤
5. **SymPy验证**：`simplify(LHS - RHS) == 0`

**实现示例**：
```python
from mr_generator.operator.operator_mr import OperatorMRGenerator

generator = OperatorMRGenerator()

mrs = generator.generate(
    operator_ir=relu_ir,
    operator_func=torch.nn.functional.relu,
    auto_fetch_info=True,
    framework="pytorch",
    sources=["llm", "template"],  # LLM猜想 + 模板池
    use_precheck=True,            # 快速筛选
    use_sympy_proof=True,         # SymPy验证
)
```

### 2. 代码到SymPy转换

**技术栈**：
- LLM翻译：将任意Python代码翻译为SymPy表达式
- AST解析：直接解析Python AST转换为SymPy
- 统一验证：所有MR猜想由SymPy统一验证

**实现示例**：
```python
from mr_generator.operator.code_translator import CodeToSymPyTranslator

translator = CodeToSymPyTranslator(llm_client=llm)
sympy_expr = translator.translate(code="def relu(x): return max(0, x)")
```

### 3. 自动信息获取

**技术栈**：
- 网络搜索：PyTorch文档、GitHub、Stack Overflow
- HTML解析：BeautifulSoup
- 智能提取：代码、文档、签名提取

**实现示例**：
```python
from tools.web_search.operator_fetcher import OperatorInfoFetcher

fetcher = OperatorInfoFetcher()
info = fetcher.fetch_operator_info("ReLU", framework="pytorch")
# 返回：{"code": "...", "doc": "...", "signature": "..."}
```

### 4. MR生成与测试分离

**架构**：
- MR生成阶段：独立生成MR并保存到知识库
- 测试执行阶段：从知识库加载MR并执行测试

**实现示例**：
```python
# 生成阶段
generator = OperatorMRGenerator()
mrs = generator.generate(operator_ir)
mr_repository.save(operator_name="ReLU", mrs=mrs)

# 测试阶段
mrs = mr_repository.load(operator_name="ReLU")
test_runner.run_with_mrs(operator_ir, mrs, "pytorch")
```

---

## 📝 下一步行动建议

1. **立即执行**：
   - 运行 `python examples/test_relu_mr.py` 验证当前实现
   - 修复发现的任何错误
   - 完成代码重构任务（优先级1.1）

2. **本周完成**：
   - 将模板池改为配置文件
   - 重构知识库至base层
   - 改进SymPy证明引擎

3. **本月完成**：
   - 完善算子层功能
   - 开始模型层MR生成模块
   - 实现缺陷最小化算法

---

## 🐛 已知问题

1. **LLM依赖**：
   - LLM猜想质量依赖于提示工程
   - 需要API密钥和网络连接
   - 成本考虑（API调用费用）

2. **SymPy验证限制**：
   - 某些复杂性质可能无法用SymPy验证
   - 浮点数精度问题

3. **错误处理**：
   - 需要增强异常处理和错误恢复机制
   - 需要更好的网络搜索错误处理

---

## ✅ 最新完成的改进

### 框架类型统一管理（2026-01-02）

1. **新增 `core/framework.py` 模块**
   - 定义 `FrameworkType` 类型别名用于类型提示
   - 支持的框架：`pytorch`, `tensorflow`, `paddlepaddle`
   - 框架别名映射 `FRAMEWORK_ALIASES`

2. **更新的模块**
   - 所有涉及 `framework` 参数的方法现已使用 `FrameworkType` 类型提示

3. **使用方式**
   ```python
   from core.framework import FrameworkType
   
   def my_func(framework: FrameworkType = "pytorch"):
       # framework 参数现在有类型提示
       ...
   ```

### 项目健康监控系统（2026-01-02）

实现了轻量级项目健康监控工具 `monitoring/`:

```bash
python -m monitoring check     # 运行健康检查
python -m monitoring progress  # 查看开发进度
python -m monitoring all       # 运行所有检查
```

详见 `docs/project_health_monitoring.md`

---

## 📚 参考资源

- **设计文档**：`docs/design.md`
- **算子MR技术文档**：`docs/operator_mr_technical.md`
- **快速开始**：`docs/quick_start.md`
- **配置指南**：`README_CONFIG.md`
- **示例代码**：`examples/` 目录

---

## 📈 开发进度统计

- ✅ **已完成**：算子层多源融合MR自动生成引擎（100%）
- ✅ **已完成**：工具层（LLM、网络搜索）（100%）
- ✅ **已完成**：用户友好API（100%）
- 🚧 **进行中**：模型层MR生成（30%）
- 📋 **计划中**：应用层MR生成（0%）
- 📋 **计划中**：缺陷分析与报告增强（50%）

---

## 🎉 主要成就

1. **LLM猜想 + SymPy验证架构**：能够处理复杂算子（如softmax）的MR生成
2. **代码到SymPy转换**：支持将任意Python代码转换为SymPy表达式
3. **自动信息获取**：实现了从网络自动获取算子信息的功能
4. **模块化架构**：清晰的目录结构，易于扩展和维护
5. **用户友好API**：隐藏了IR细节，提供了简洁的使用接口
6. **框架类型管理**：`core/framework.py` 提供 `FrameworkType` 类型定义
7. **项目健康监控**：`monitoring/` 模块提供健康检查和进度追踪

## ⚠️ 架构变更（2026-01）

**移除 `mr_deriver.py`**：
- 原因：预定义性质验证（交换律、单位元等）能力有限
- 问题：对于复杂算子（如softmax、batchnorm）无法推导出特有的蜕变关系
- 解决方案：采用 LLM 猜想 + SymPy 验证的架构
- LLM 能够理解算子语义，猜想如 "softmax 具有平移不变性" 等复杂MR
- SymPy 作为统一的形式化验证工具
