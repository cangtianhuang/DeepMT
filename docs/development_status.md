# DeepMT 项目开发状态与重点

## 📊 当前开发状态

根据 `docs/design.md` 的设计思路和 `.tmp/prompt.md` 的实施计划，项目当前处于**阶段1-3**（算子层闭环开发阶段）。

### ✅ 已完成的核心模块

#### 1. **IR层（统一中间表示）**
- ✅ `ir/schema.py` - 定义了 `OperatorIR`, `ModelIR`, `ApplicationIR` 和 `MetamorphicRelation` 数据结构
- ✅ `core/ir_manager.py` - 实现了IR的加载、保存、验证功能

#### 2. **MR生成引擎**
- ✅ `mr_generator/knowledge_base.py` - 算子知识库，支持多种MR类型：
  - 交换律（Commutative）
  - 结合律（Associative）
  - 单位元（Identity）
  - 反交换律（Anti-commutative）
  - 倒数关系（Reciprocal）
  - 转置关系（Transpose）
- ✅ `mr_generator/operator_mr.py` - 算子层MR生成器

#### 3. **微内核框架**
- ✅ `core/scheduler.py` - 任务调度器，协调整个测试流程
- ✅ `core/plugins_manager.py` - 插件管理器，支持动态加载框架插件
- ✅ `core/results_manager.py` - 结果管理器，负责结果比对、存储和统计
- ✅ `core/logger.py` - 统一日志系统

#### 4. **框架适配插件**
- ✅ `plugins/pytorch_plugin.py` - PyTorch插件，支持多种算子：
  - Add, Multiply, Subtract, Divide
  - MatMul, Pow, Sum, Mean

#### 5. **缺陷分析**
- ✅ `analysis/defect_classifier.py` - 缺陷分类器，支持多种比对模式：
  - 相等检查（equal）
  - 比例检查（proportional）
  - 不变性检查（invariant）
  - 单调性检查（monotonic）

#### 6. **演示代码**
- ✅ `demo/quick_demo.py` - 完整的端到端演示

---

## 🎯 当前应该重点做的开发

### **优先级1：完善算子层闭环测试**

#### 1.1 修复和测试现有代码
- [ ] 运行 `demo/quick_demo.py`，确保端到端流程正常工作
- [ ] 修复可能存在的导入错误（如 `yaml` 模块）
- [ ] 测试各种算子的MR生成和执行

#### 1.2 扩展算子支持
- [ ] 在 `KnowledgeBase` 中添加更多常见算子：
  - 矩阵运算：`Transpose`, `Inverse`, `Eigenvalue`
  - 激活函数：`ReLU`, `Sigmoid`, `Tanh`
  - 归一化：`BatchNorm`, `LayerNorm`
- [ ] 在 `PyTorchPlugin` 中添加对应的算子映射

#### 1.3 增强MR生成能力
- [ ] 实现基于形式化规约的MR推导（使用SymPy/Z3）
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
- [ ] 实现 `mr_generator/model_mr.py`：
  - 基于网络拓扑的MR生成
  - 数据增强策略集成（使用Albumentations）
  - 结构不变性MR

#### 3.3 模型测试支持
- [ ] 扩展插件以支持模型执行
- [ ] 实现模型加载和推理功能

### **优先级4：应用层LLM MR生成**

#### 4.1 LLM工具
- [ ] 实现 `mr_generator/llm_utils.py`：
  - LLM客户端封装（支持GPT/LLaMA）
  - 提示模板管理
  - 代码生成和解析

#### 4.2 应用层MR生成
- [ ] 实现 `mr_generator/app_mr.py`：
  - 自然语言MR生成
  - MR描述到代码的转换
  - 语义验证

---

## 🔧 关键技术实现建议

### 1. 形式化MR推导（算子层）

```python
# 使用SymPy进行符号计算
from sympy import symbols, simplify, Eq

def derive_commutative_mr(operator_expr):
    """推导交换律MR"""
    x, y = symbols('x y')
    expr1 = operator_expr(x, y)
    expr2 = operator_expr(y, x)
    return simplify(Eq(expr1, expr2))
```

### 2. 数据增强MR（模型层）

```python
# 使用Albumentations
import albumentations as A

def generate_rotation_mr(model_ir):
    """生成旋转不变性MR"""
    transform = A.Rotate(limit=90, p=1.0)
    return MetamorphicRelation(
        id="rotation_invariance",
        description="Model output should be invariant to 90° rotation",
        transform=lambda img: transform(image=img)['image'],
        expected="invariant",
        layer="model"
    )
```

### 3. LLM MR生成（应用层）

```python
# 使用LangChain
from langchain.llms import OpenAI

def generate_app_mr(app_ir):
    """使用LLM生成应用层MR"""
    prompt = f"""
    Generate metamorphic relations for this application:
    Purpose: {app_ir.purpose}
    Input: {app_ir.input_format}
    Output: {app_ir.output_format}
    
    Provide MRs in the format: description, transform, expected
    """
    llm = OpenAI()
    response = llm(prompt)
    return parse_llm_response(response)
```

---

## 📝 下一步行动建议

1. **立即执行**：
   - 运行 `python demo/quick_demo.py` 验证当前实现
   - 修复发现的任何错误
   - 添加必要的依赖到 `requirements.txt`

2. **本周完成**：
   - 扩展 `KnowledgeBase` 支持至少10种常见算子
   - 完善 `PyTorchPlugin` 的算子映射
   - 实现基础的报告生成功能

3. **本月完成**：
   - 完成算子层完整闭环
   - 开始模型层MR生成模块
   - 实现缺陷最小化算法

---

## 🐛 已知问题

1. **依赖缺失**：需要添加 `pyyaml` 到 `requirements.txt`
2. **MR变换函数**：当前 `KnowledgeBase` 中的MR变换函数需要适配不同输入格式
3. **错误处理**：需要增强异常处理和错误恢复机制

---

## 📚 参考资源

- 设计文档：`docs/design.md`
- 实施计划：`.tmp/prompt.md`
- 演示代码：`demo/quick_demo.py`


