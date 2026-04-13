# DeepMT 项目介绍

> **面向深度学习框架的蜕变关系自动生成与分层测试体系研究**

---

## 一、项目概述

DeepMT 针对深度学习框架（PyTorch、TensorFlow、PaddlePaddle）的**测试预言机问题**，通过蜕变测试（Metamorphic Testing）自动生成与验证蜕变关系（MR）。

**核心目标：**
1. 分层测试体系：算子层 → 模型层 → 应用层
2. 多源融合 MR 自动生成：LLM 生成 + 模板池 + 形式化验证
3. 跨框架执行：微内核 + 插件化架构
4. MR 生成与测试分离：通过 SQLite 知识库解耦

---

## 二、系统架构

采用**微内核 + 插件化**架构。MR 生成存入知识库（SQLite），测试执行时复用。

### 三层 MR 生成流水线

| 层次   | 生成器                   | 生成策略                             | oracle 类型                                     |
| ------ | ------------------------ | ------------------------------------ | ----------------------------------------------- |
| 算子层 | `OperatorMRGenerator`    | LLM + 模板池 + 数值预检 + SymPy 证明 | Python 表达式（`orig == trans` 等）             |
| 模型层 | `ModelMRGenerator`       | 结构分析 → 策略库模板                | `prediction_consistent` / `output_close` 等     |
| 应用层 | `ApplicationMRGenerator` | 场景知识 + LLM/模板 + 语义验证       | `label_consistent` / `confidence_acceptable` 等 |

### 算子层四阶段流水线

```
信息准备（CrawlAgent/网络搜索）
    → 候选生成（LLM 猜想 + 模板池匹配）
    → 快速筛选（Pre-check：随机输入数值验证）
    → 形式化验证（SymPy 符号证明）
    → 持久化存储（SQLite）
```

---

## 三、目录结构

```
DeepMT/
├── deepmt/                     # 主包（CLI 入口 + 所有核心模块）
│   ├── ir/                     #   统一 IR（OperatorIR / ModelIR / ApplicationIR / MetamorphicRelation）
│   ├── mr_generator/           #   MR 生成引擎（三层）
│   │   ├── operator/           #     算子层（四阶段流水线）
│   │   ├── model/              #     模型层（模板驱动，Phase I）
│   │   └── application/        #     应用层（LLM + 模板回退，Phase J）
│   ├── model/                  #   模型结构分析（GraphAnalyzer）
│   ├── application/            #   应用层场景定义（ApplicationScenario）
│   ├── benchmarks/             #   基准注册表
│   │   ├── models/             #     ModelBenchmarkRegistry（SimpleMLP / SimpleCNN / SimpleRNN / TinyTransformer）
│   │   └── applications/       #     ApplicationBenchmarkRegistry（ImageClassification / TextSentiment）
│   ├── analysis/               #   验证、报告、实验（三层均有）
│   ├── engine/                 #   测试执行引擎
│   ├── plugins/                #   框架适配器（PyTorch 完整，NumPy/Paddle 部分）
│   ├── tools/                  #   通用工具（LLM / 网络搜索）
│   ├── commands/               #   CLI 子命令（mr / test / repo / catalog / data / health / ui）
│   ├── core/                   #   微内核框架
│   ├── monitoring/             #   健康检查与进度追踪
│   └── ui/                     #   Web 仪表盘（FastAPI + Jinja2）
├── tests/                      # 测试用例（unit/ + integration/），708 个测试
├── demo/                       # 快速演示脚本
├── docs/                       # 开发文档（见下方索引）
├── data/                       # 运行时数据（日志、SQLite、MR YAML）
├── config.yaml                 # 运行配置
└── pyproject.toml
```

---

## 四、使用方式

### CLI

```bash
# 算子层
deepmt mr generate torch.relu --framework pytorch --save
deepmt test batch --operator torch.relu --framework pytorch

# 查看结果
deepmt test report
deepmt repo stats

# Web 仪表盘
deepmt ui start
```

### Python API

```python
from deepmt import DeepMT

client = DeepMT()
result = client.run_batch_test("torch.relu", framework="pytorch", n_samples=10)
print(result.summary())
```

### 应用层 MR 生成（Python API，不依赖 LLM）

```python
from deepmt.mr_generator.application import ApplicationMRGenerator
from deepmt.analysis.semantic_mr_validator import SemanticMRValidator
from deepmt.analysis.application_reporter import ApplicationReporter
from deepmt.benchmarks.applications.app_registry import ApplicationBenchmarkRegistry

registry = ApplicationBenchmarkRegistry()
sc = registry.get("TextSentiment")

gen = ApplicationMRGenerator(use_llm=False)
mrs = gen.generate_from_scenario("TextSentiment")

validator = SemanticMRValidator()
results = validator.validate_batch(mrs, sc.sample_inputs, sc.sample_labels)

reporter = ApplicationReporter()
report = reporter.generate(mrs, results, scenario_name="TextSentiment")
print(reporter.format_text(report))
```

详见 `docs/quick_start.md`。

---

## 五、配置

配置文件 `config.yaml`（参考 `config.yaml.example`）：

```yaml
llm:
  provider: "openai"
  api_key: "your-api-key"
  model_base: "gpt-4o-mini"
  model_max: "gpt-4o"
  url: "https://api.openai.com/v1/"

mr_generation:
  use_llm: true
  use_template_pool: true
  use_precheck: true
  use_sympy_proof: true
```

环境变量：`OPENAI_API_KEY`、`DEEPMT_LOG_LEVEL`、`DEEPMT_LOG_DIR`。详见 `docs/environment_variables.md`。

---

## 六、开发状态

Phase A–J 均已完成，708 个测试通过。详见 `docs/dev/status.md`。

| 层次 / 模块          | 状态                                   |
| -------------------- | -------------------------------------- |
| 算子层 MR 生成       | ✅ 完成（四阶段流水线 + SymPy 证明）    |
| 批量测试执行         | ✅ 完成                                 |
| 缺陷分析报告         | ✅ 完成                                 |
| 跨框架一致性测试     | ✅ 完成（PyTorch vs NumPy）             |
| Web 仪表盘           | ✅ 完成                                 |
| 统一 IR（三层）      | ✅ 完成（Phase G）                      |
| NumPy/PaddlePaddle   | ✅ 完成（Phase H，基础跨框架适配）      |
| 模型层 MR 生成       | ✅ 完成（Phase I，模板驱动）            |
| 应用层 MR 生成与验证 | ✅ 完成（Phase J，图像分类 + 文本情感） |
| 全层 MR 质量保障     | 📋 Phase K，计划中                      |

---

## 七、文档索引

### 用户文档

| 文件                            | 内容                              |
| ------------------------------- | --------------------------------- |
| `docs/quick_start.md`           | 快速上手（安装、CLI、Python API） |
| `docs/cli_reference.md`         | CLI 完整命令参考                  |
| `docs/environment_variables.md` | 环境变量说明                      |
| `docs/demo_golden_path.md`      | 答辩/演示用黄金演示路径           |

### 技术设计

| 文件                            | 内容                         |
| ------------------------------- | ---------------------------- |
| `docs/tech/operator_catalog.md` | 算子目录设计与 YAML 规范     |
| `docs/tech/operator_mr.md`      | 算子层 MR 生成与测试技术细节 |
| `docs/tech/web_dashboard.md`    | Web 仪表盘技术设计           |

### 开发文档

| 文件                                           | 内容                                |
| ---------------------------------------------- | ----------------------------------- |
| `docs/dev/status.md`                           | 开发状态、已完成模块、架构约定      |
| `docs/dev/agent_rules.md`                      | 编码智能体执行规范与开发原则        |
| `docs/dev/archived/`                           | 已完成阶段（Phase A–J）详细规划文档 |
| `docs/dev/11_Phase_K_*.md` ~ `14_Phase_N_*.md` | 待开发阶段规划文档                  |

---

*最后更新：2026-04-13（Phase J 完成，三层 MR 生成闭环）*
