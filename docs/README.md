# DeepMT 项目介绍

> **面向深度学习框架的蜕变关系自动生成与分层测试体系研究**
> Research on Automatic Metamorphic Relation Generation and a Hierarchical Testing System for Deep Learning Frameworks

---

## 一、项目概述

### 1.1 研究背景

深度学习框架（如 PyTorch、TensorFlow、PaddlePaddle）作为现代人工智能应用的核心计算基础设施，其可靠性与正确性直接影响上层智能系统的稳健性与可信度。然而，深度学习框架测试过程中普遍存在的**测试预言机问题（Oracle Problem）**对其质量保障构成了重大挑战。

**蜕变测试（Metamorphic Testing, MT）**通过构建并验证软件在特定输入变换下输出所应满足的蜕变关系（Metamorphic Relations, MRs），为缓解测试预言机问题提供了理论基础与实践路径。

### 1.2 研究目标

本项目致力于建立一个面向主流深度学习框架的通用分层测试体系架构，主要目标包括：

1. **构建分层测试体系**：将复杂的深度学习框架测试任务按照抽象层次划分为算子（Operator）层、模型（Model）层与应用（Application）层三个相互正交的测试维度
2. **开发多源融合 MR 自动生成引擎**：集成基于形式化规约的演绎方法、基于深度学习模型内在特性的启发式方法，以及基于大语言模型的知识驱动生成方法
3. **提供跨框架自动化测试执行框架**：微内核 + 插件化架构，支持 PyTorch、TensorFlow、PaddlePaddle 等主流框架
4. **支持大规模缺陷检测与报告生成**：自动化缺陷检测、分类、最小化、可视化

### 1.3 关键词

深度学习框架；软件测试；蜕变测试；自动化测试；深度学习；PyTorch；TensorFlow；PaddlePaddle

---

## 二、系统架构

### 2.1 总体架构

DeepMT 采用**微内核 + 插件化**架构，核心模块包括：

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepMT 主框架（微内核）                    │
│  ┌──────────────────┐      ┌──────────────────────────┐     │
│  │  任务调度器       │      │  统一中间表示(IR)管理     │     │
│  │  TaskScheduler   │◄────►│  IRManager               │     │
│  └──────────────────┘      └──────────────────────────┘     │
│           │                            │                      │
│           ▼                            ▼                      │
│  ┌──────────────────────────┐  ┌──────────────────────┐     │
│  │  MR生成引擎               │  │  插件管理器           │     │
│  │  ┌────────────────────┐  │  │  ┌────────────────┐  │     │
│  │  │ 算子层MR生成器      │  │  │  │ PyTorch插件    │  │     │
│  │  │ OperatorMRGenerator│  │  │  │ TensorFlow插件 │  │     │
│  │  ├────────────────────┤  │  │  │ Paddle插件     │  │     │
│  │  │ 模型层MR生成器      │  │  │  └────────────────┘  │     │
│  │  │ ModelMRGenerator   │  │  └──────────────────────┘     │
│  │  ├────────────────────┤  │                                │
│  │  │ 应用层MR生成器      │  │                                │
│  │  │ ApplicationMRGen   │  │                                │
│  │  └────────────────────┘  │                                │
│  └──────────────────────────┘                                │
│           │                            │                      │
│           ▼                            ▼                      │
│  ┌──────────────────────────┐  ┌──────────────────────┐     │
│  │  测试执行器               │  │  结果管理器           │     │
│  │  TestRunner              │  │  ResultsManager      │     │
│  └──────────────────────────┘  └──────────────────────┘     │
│           │                            │                      │
│           └────────────┬───────────────┘                      │
│                        ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  缺陷分析与报告系统                                    │   │
│  │  DefectClassifier + ReportGenerator                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块说明

#### （1）统一中间表示（IR）层
- **目标**：屏蔽不同深度学习框架的差异，提供跨框架可执行描述
- **层次划分**：
  - **算子层 IR**（OperatorIR）：数学算子与数值计算描述
  - **模型层 IR**（ModelIR）：网络拓扑结构与数据流描述
  - **应用层 IR**（ApplicationIR）：应用接口与语义描述
- **关键文件**：
  - `ir/schema.py`：IR 数据结构定义（OperatorIR、ModelIR、ApplicationIR、MetamorphicRelation）
  - `ir/converter.py`：IR 转换器

#### （2）MR 生成引擎
分为三个子模块，分别对应三个测试层次：

**a. 算子层 MR 生成器（OperatorMRGenerator）**
- **输入**：算子名称、框架类型
- **方法**：多源融合自动生成
  - **路径 A（LLM 生成）**：使用大语言模型基于算子文档生成 MR 猜想
  - **路径 B（模板池）**：从预定义的数学变换模板库匹配 MR
  - **快速筛选（Pre-check）**：使用少量随机测试用例快速过滤明显错误的 MR
  - **形式化证明（SymPy Proof）**：使用符号计算验证 MR 的数学正确性
- **关键文件**：
  - `mr_generator/operator/operator_mr.py`：算子层 MR 生成器主类
  - `mr_generator/operator/operator_llm_mr_generator.py`：LLM 生成器
  - `mr_generator/operator/mr_prechecker.py`：快速筛选器
  - `mr_generator/operator/sympy_prover.py`：SymPy 形式化证明引擎
  - `mr_generator/operator/sympy_translator.py`：代码到 SymPy 表达式转换器
  - `mr_generator/operator/ast_parser.py`：AST 解析器
  - `mr_generator/base/mr_templates.py`：MR 模板池
  - `mr_generator/base/mr_repository.py`：MR 知识库（持久化存储）

**b. 模型层 MR 生成器（ModelMRGenerator）**
- **输入**：模型拓扑结构（来自 IR）
- **方法**：静态图分析 + 数据增强策略库
- **关键文件**：
  - `mr_generator/model/model_mr.py`：模型层 MR 生成器

**c. 应用层 MR 生成器（ApplicationMRGenerator）**
- **输入**：API 文档、技术博客等文本
- **方法**：LLM 语义生成
- **关键文件**：
  - `mr_generator/application/app_mr.py`：应用层 MR 生成器

#### （3）工具层（Tools）
提供通用工具支持，可复用于多个模块：

**a. LLM 工具**
- `tools/llm/client.py`：通用 LLM 客户端（支持 OpenAI、Anthropic 等）
- `tools/llm/ocr_client.py`：OCR 客户端（使用百度千帆 API 识别公式图片）

**b. 网络搜索工具**
- `tools/web_search/search_tool.py`：网络搜索工具（支持多源搜索）
- `tools/web_search/search_agent.py`：智能搜索代理（使用 LLM 进行搜索和内容理解）
- `tools/web_search/operator_fetcher.py`：算子信息获取器（自动从网络获取算子文档）
- `tools/web_search/sphinx_search.py`：Sphinx 文档搜索索引解析器

#### （4）执行框架（Core）
微内核 + 插件化架构，实现框架无关的测试执行：

- `core/scheduler.py`：任务调度器（TaskScheduler）
- `core/test_runner.py`：测试执行器（TestRunner）
- `core/ir_manager.py`：IR 管理器（IRManager）
- `core/plugins_manager.py`：插件管理器（PluginsManager）
- `core/results_manager.py`：结果管理器（ResultsManager）
- `core/oracle_evaluator.py`：Oracle 表达式评估器（框架无关的 MR 验证）
- `core/framework.py`：框架类型定义
- `core/config_loader.py`：配置加载器
- `core/logger.py`：统一日志系统

#### （5）插件系统（Plugins）
支持多个深度学习框架的适配器插件：

- `plugins/framework_adapter.py`：框架适配器基类
- `plugins/pytorch_plugin.py`：PyTorch 插件
- （待实现）`plugins/tensorflow_plugin.py`：TensorFlow 插件
- （待实现）`plugins/paddle_plugin.py`：PaddlePaddle 插件

#### （6）分析系统（Analysis）
缺陷检测、分类、最小化、可视化：

- `analysis/defect_classifier.py`：缺陷分类器（DefectClassifier）

#### （7）用户 API（API）
用户友好的接口，隐藏 IR 和内部实现细节：

- `api/deepmt.py`：DeepMT 主 API 类

---

## 三、算子层 MR 生成技术详解

### 3.1 多源融合生成流程

算子层 MR 生成是 DeepMT 的核心技术，采用**多源融合**策略：

```
┌─────────────────────────────────────────────────────────────┐
│                    算子层 MR 生成流程                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  1. 信息准备阶段                       │
        │  - 自动获取算子信息（网络搜索）         │
        │  - 提取算子代码                        │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  2. 多源 MR 生成阶段                   │
        │  ┌─────────────────────────────────┐  │
        │  │ 路径 A: LLM 生成                 │  │
        │  │ - 基于算子文档生成 MR 猜想        │  │
        │  └─────────────────────────────────┘  │
        │  ┌─────────────────────────────────┐  │
        │  │ 路径 B: 模板池匹配               │  │
        │  │ - 从预定义模板库匹配 MR          │  │
        │  └─────────────────────────────────┘  │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  3. 快速筛选阶段（Pre-check）          │
        │  - 使用少量随机测试用例               │
        │  - 过滤明显错误的 MR                  │
        │  - 通过率阈值：80%                    │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  4. 形式化验证阶段（SymPy Proof）      │
        │  - 将代码转换为 SymPy 表达式          │
        │  - 使用符号计算验证 MR 正确性         │
        │  - simplify(LHS - RHS) == 0          │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  5. MR 持久化存储                      │
        │  - 保存到 MR 知识库（SQLite）         │
        │  - 支持后续测试复用                   │
        └───────────────────────────────────────┘
```

### 3.2 关键技术组件

#### （1）算子信息自动获取（OperatorInfoFetcher）
- 自动从 PyTorch/TensorFlow/PaddlePaddle 官方文档搜索算子信息
- 支持多源搜索：官方文档、GitHub、网络搜索
- 使用智能搜索代理（SearchAgent）进行内容理解和重排

#### （2）LLM 生成器（OperatorLLMMRGenerator）
- 基于算子文档和代码，使用大语言模型生成 MR 猜想
- 支持多种 LLM 提供商（OpenAI、Anthropic 等）
- 生成框架无关的 MR 描述

#### （3）MR 模板池（MRTemplatePool）
- 预定义常见数学变换模板（如交换律、结合律、分配律等）
- 从配置文件读取模板定义
- 支持动态匹配和实例化

#### （4）快速筛选器（MRPreChecker）
- 使用少量随机测试用例（默认 5 个）快速验证 MR
- 通过率阈值：80%（即失败率 < 20%）
- 框架无关，通过插件系统支持多个框架

#### （5）SymPy 证明引擎（SymPyProver）
- 将 Python 代码动态转换为 SymPy 符号表达式
- 使用符号计算验证 MR 的数学正确性
- 支持复杂的数学运算和函数

#### （6）MR 知识库（MRRepository）
- 使用 SQLite 持久化存储 MR
- 支持 MR 的增删改查
- 实现 MR 生成与测试的分离

---

## 四、项目目录结构

```
DeepMT/
│
├── README.md                          # 项目说明
├── requirements.txt                   # 依赖包列表
├── config.yaml                        # 配置文件（API 密钥等）
├── config.yaml.example                # 配置示例
│
├── core/                              # 微内核框架
│   ├── scheduler.py                   # 任务调度器（TaskScheduler）
│   ├── test_runner.py                 # 测试执行器（TestRunner）
│   ├── ir_manager.py                  # IR 管理器（IRManager）
│   ├── plugins_manager.py             # 插件管理器（PluginsManager）
│   ├── results_manager.py             # 结果管理器（ResultsManager）
│   ├── oracle_evaluator.py            # Oracle 表达式评估器
│   ├── framework.py                   # 框架类型定义
│   ├── config_loader.py               # 配置加载器
│   └── logger.py                      # 统一日志系统
│
├── ir/                                # 统一中间表示（IR）
│   ├── schema.py                      # IR 数据结构定义
│   └── converter.py                   # IR 转换器
│
├── mr_generator/                      # MR 生成引擎
│   ├── operator/                      # 算子层 MR 生成
│   │   ├── operator_mr.py             # 算子层 MR 生成器主类
│   │   ├── operator_llm_mr_generator.py  # LLM 生成器
│   │   ├── mr_prechecker.py           # 快速筛选器
│   │   ├── sympy_prover.py            # SymPy 形式化证明引擎
│   │   ├── sympy_translator.py        # 代码到 SymPy 转换器
│   │   └── ast_parser.py              # AST 解析器
│   ├── model/                         # 模型层 MR 生成
│   │   └── model_mr.py                # 模型层 MR 生成器
│   ├── application/                   # 应用层 MR 生成
│   │   └── app_mr.py                  # 应用层 MR 生成器
│   └── base/                          # 基础组件
│       ├── mr_templates.py            # MR 模板池
│       ├── mr_repository.py           # MR 知识库
│       └── knowledge_base.py          # 知识库管理
│
├── tools/                             # 通用工具
│   ├── llm/                           # LLM 工具
│   │   ├── client.py                  # LLM 客户端
│   │   └── ocr_client.py              # OCR 客户端
│   └── web_search/                    # 网络搜索工具
│       ├── search_tool.py             # 网络搜索工具
│       ├── search_agent.py            # 智能搜索代理
│       ├── operator_fetcher.py        # 算子信息获取器
│       └── sphinx_search.py           # Sphinx 文档搜索
│
├── plugins/                           # 框架适配器插件
│   ├── framework_adapter.py           # 框架适配器基类
│   ├── pytorch_plugin.py              # PyTorch 插件
│   ├── tensorflow_plugin.py           # TensorFlow 插件（待实现）
│   └── paddle_plugin.py               # PaddlePaddle 插件（待实现）
│
├── analysis/                          # 缺陷分析与报告
│   └── defect_classifier.py           # 缺陷分类器
│
├── api/                               # 用户 API
│   └── deepmt.py                      # DeepMT 主 API 类
│
├── demo/                              # 示例代码
│   └── quick_demo.py                  # 快速演示
│
├── docs/                              # 文档
│   ├── README.md                      # 项目介绍（本文档）
│   ├── design.md                      # 技术设计文档
│   ├── development_status.md          # 开发状态
│   ├── operator_mr_technical.md       # 算子 MR 技术文档
│   └── quick_start.md                 # 快速开始
│
└── data/                              # 数据目录
    ├── logs/                          # 日志文件
    ├── mr_knowledge_base.db           # MR 知识库（SQLite）
    └── results/                       # 测试结果
```

---

## 五、使用示例

### 5.1 基本使用

```python
from api.deepmt import DeepMT

# 初始化 DeepMT
deepmt = DeepMT()

# 测试算子
result = deepmt.test_operator(
    operator_name="torch.add",
    framework="pytorch",
    num_tests=10
)

# 打印测试结果
print(result.summary())
```

### 5.2 生成 MR

```python
from mr_generator.operator.operator_mr import OperatorMRGenerator

# 初始化 MR 生成器
mr_generator = OperatorMRGenerator()

# 生成 MR
mrs = mr_generator.generate(
    operator_name="torch.add",
    framework="pytorch",
    use_llm=True,
    use_template_pool=True,
    use_precheck=True,
    use_sympy_proof=True
)

# 打印生成的 MR
for mr in mrs:
    print(f"MR: {mr.description}")
```

### 5.3 使用预生成的 MR 进行测试

```python
from core.test_runner import TestRunner
from core.plugins_manager import PluginsManager
from core.results_manager import ResultsManager
from mr_generator.base.mr_repository import MRRepository

# 初始化组件
plugins_manager = PluginsManager()
results_manager = ResultsManager()
test_runner = TestRunner(plugins_manager, results_manager)
mr_repository = MRRepository()

# 从知识库加载 MR
mrs = mr_repository.get_by_operator("torch.add", "pytorch")

# 执行测试
from ir.schema import OperatorIR
operator_ir = OperatorIR(name="torch.add")
results = test_runner.run_with_mrs(
    ir_object=operator_ir,
    mrs=mrs,
    framework="pytorch",
    num_tests=10
)
```

---

## 六、配置说明

配置文件 `config.yaml` 示例：

```yaml
# LLM 配置
llm:
  provider: "openai"  # openai, anthropic
  api_key: "your-api-key"
  model: "gpt-4"
  url: "https://api.openai.com/v1/"

# 网络搜索配置
web_search:
  enabled: true
  sources: ["pytorch_docs", "github", "stackoverflow"]
  timeout: 10
  max_results: 5
  baidu_api_key: "your-baidu-api-key"  # 用于 OCR
  ocr: false  # 是否启用 OCR

# MR 生成配置
mr_generation:
  use_llm: true
  use_template_pool: true
  use_precheck: true
  use_sympy_proof: true

# 日志配置
logging:
  level: "INFO"
  file: "data/logs/deepmt.log"
```

---

## 七、开发状态

### 7.1 已完成功能

- ✅ 统一中间表示（IR）层
- ✅ 算子层 MR 生成引擎（多源融合）
- ✅ LLM 工具（支持 OpenAI、Anthropic）
- ✅ 网络搜索工具（支持多源搜索）
- ✅ 快速筛选器（Pre-check）
- ✅ SymPy 形式化证明引擎
- ✅ MR 知识库（持久化存储）
- ✅ PyTorch 插件
- ✅ 测试执行框架
- ✅ 缺陷分类器
- ✅ 统一日志系统
- ✅ 配置管理系统

### 7.2 进行中功能

- 🚧 模型层 MR 生成器
- 🚧 应用层 MR 生成器
- 🚧 TensorFlow 插件
- 🚧 PaddlePaddle 插件

### 7.3 待开发功能

- ⏳ 缺陷最小化
- ⏳ 可视化报告生成
- ⏳ 大规模实证研究
- ⏳ 性能优化（并行测试、缓存机制）

---

## 八、技术特点

### 8.1 创新点

1. **多源融合 MR 生成**：结合 LLM 生成、模板池匹配、快速筛选、形式化证明，提高 MR 生成的效率和准确性
2. **分层测试体系**：算子层、模型层、应用层三个维度，实现精确化、细粒度测试覆盖
3. **框架无关设计**：通过统一 IR 和插件系统，支持多个深度学习框架
4. **MR 生成与测试分离**：通过 MR 知识库实现生成与测试的解耦，提高复用性

### 8.2 技术优势

1. **自动化程度高**：从算子信息获取、MR 生成到测试执行，全流程自动化
2. **可扩展性强**：微内核 + 插件化架构，易于扩展新框架和新功能
3. **验证严格**：多层验证机制（Pre-check + SymPy Proof），确保 MR 质量
4. **易于使用**：提供用户友好的 API，隐藏内部实现细节

---

## 九、未来改进方向

1. **模型层 MR 生成**：完善结构分析和数据增强 MR
2. **应用层 MR 生成**：LLM 语义生成和验证
3. **缺陷分析增强**：更细粒度的缺陷分类和最小化
4. **性能优化**：并行测试、缓存机制
5. **更多框架支持**：TensorFlow、PaddlePaddle 完整支持
6. **大规模实证研究**：在真实框架上验证有效性

---

## 十、参考文档

- **项目介绍**：`docs/README.md`（本文档）
- **技术设计文档**：`docs/design.md`
- **开发状态**：`docs/development_status.md`
- **算子 MR 技术文档**：`docs/operator_mr_technical.md`
- **快速开始**：`docs/quick_start.md`
- **配置指南**：`README_CONFIG.md`

---

## 十一、联系方式

如有问题或建议，请联系项目维护者。

---

**最后更新时间**：2026-03-02
