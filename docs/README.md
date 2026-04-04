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

### 算子层 MR 生成四阶段流水线

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
├── deepmt/                     # CLI 入口（python -m deepmt）
│   └── commands/               #   mr / test / repo / catalog / data / health
├── api/                        # 用户 API（DeepMT 主入口）
├── core/                       # 微内核框架
│   ├── config_loader.py        #   配置加载
│   ├── framework.py            #   FrameworkType 定义
│   ├── ir_manager.py           #   IR 管理
│   ├── logger.py               #   日志
│   ├── oracle_evaluator.py     #   oracle_expr 运行时评估
│   ├── plugins_manager.py      #   插件加载
│   ├── results_manager.py      #   结果管理
│   ├── scheduler.py            #   任务调度
│   └── test_runner.py          #   测试执行
├── ir/                         # 统一中间表示
│   ├── schema.py               #   OperatorIR / MetamorphicRelation 等
│   └── converter.py            #   IR 转换器
├── mr_generator/               # MR 生成引擎
│   ├── operator/               #   算子层（核心）
│   │   ├── operator_mr.py      #     主生成器（4阶段流水线）
│   │   ├── operator_llm_mr_generator.py  # LLM 生成
│   │   ├── mr_prechecker.py              # 数值预检
│   │   ├── sympy_prover.py               # 符号证明
│   │   ├── sympy_translator.py           # 代码→SymPy
│   │   └── ast_parser.py                 # AST 解析
│   ├── model/                  #   模型层（开发中）
│   ├── application/            #   应用层（开发中）
│   ├── base/                   #   知识库、模板池、MR 仓库、算子目录
│   │   ├── mr_repository.py    #     MR 知识库（SQLite）
│   │   ├── mr_templates.py     #     MR 模板池
│   │   ├── knowledge_base.py   #     三层知识库
│   │   └── operator_catalog.py #     算子目录管理
│   └── config/                 #   模板/知识库 YAML + 算子目录 YAML
├── tools/                      # 通用工具
│   ├── llm/                    #   LLM 客户端 / OCR
│   ├── agent/                  #   CrawlAgent（ReAct）+ TaskRunner + ToolRegistry
│   │   └── tasks/              #     任务规格 YAML
│   └── web_search/             #   搜索、Sphinx 解析、算子文档获取
├── plugins/                    # 框架适配器（PyTorch 可用）
├── analysis/                   # 缺陷分类器
├── monitoring/                 # 健康检查与进度追踪
├── tests/                      # 测试用例
│   ├── unit/                   #   单元测试
│   └── integration/            #   集成测试
├── examples/                   # 示例脚本
├── demo/                       # 快速演示
├── docs/                       # 开发文档
├── data/                       # 数据（日志、SQLite）
├── config.yaml                 # 运行配置
└── pyproject.toml
```

---

## 四、使用方式

### CLI

```bash
python -m deepmt mr generate --operator torch.relu --framework pytorch
python -m deepmt catalog list --framework pytorch
python -m deepmt health check
```

### Python API

```python
from api.deepmt import DeepMT

deepmt = DeepMT()
result = deepmt.test_operator("torch.add", framework="pytorch", num_tests=10)
```

### 直接调用 MR 生成器

```python
from mr_generator.operator.operator_mr import OperatorMRGenerator

generator = OperatorMRGenerator()
mrs = generator.generate(
    operator_ir=relu_ir,
    operator_func=torch.nn.functional.relu,
    framework="pytorch",
    sources=["llm", "template"],
    use_precheck=True,
    use_sympy_proof=True,
)
```

---

## 五、配置

配置文件 `config.yaml`（参考 `config.yaml.example`）：

```yaml
llm:
  provider: "openai"
  api_key: "your-api-key"
  model: "gpt-4"
  url: "https://api.openai.com/v1/"

agent:
  enabled: false           # 是否启用 CrawlAgent 自动获取算子文档
  max_steps: 10
  cache_ttl_days: 7

web_search:
  enabled: true

mr_generation:
  use_llm: true
  use_template_pool: true
  use_precheck: true
  use_sympy_proof: true
```

环境变量：`OPENAI_API_KEY`、`DEEPMT_LOG_LEVEL`、`DEEPMT_LOG_DIR`。

---

## 六、开发状态

详见 `docs/development_status.md`。

| 层次     | 状态         |
|--------|------------|
| 算子层 MR 生成 | ✅ 完成       |
| CrawlAgent  | ✅ 完成       |
| CLI         | ✅ 完成       |
| 模型层 MR 生成 | 🚧 进行中（30%）|
| 应用层 MR 生成 | 📋 计划中      |
| 缺陷报告增强   | 📋 计划中      |

---

## 七、参考文档

| 文件 | 内容 |
|------|------|
| `docs/development_status.md` | 开发状态与下一步 |
| `docs/agent_crawler_plan.md` | CrawlAgent 设计 |
| `docs/environment_variables.md` | 环境变量说明 |
| `docs/operator_mr_technical.md` | 算子 MR 技术细节 |
| `docs/quick_start.md` | 快速上手 |

---

*最后更新：2026-04-04*
