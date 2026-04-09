# DeepMT 开发状态

> 详细开发规划与任务清单见 `docs/deepmt_dev_docs/`。本文件只记录"已完成"状态与已知限制。

## 阶段进度

| 阶段 | 状态 |
|------|------|
| Phase A：算子数据层完善（A1~A6） | ✅ 完成 |
| Phase B：算子层 MR 生成与知识库（B1~B3） | ✅ 完成 |
| Phase C：测试执行与跨框架适配 | 🔲 当前目标 |
| Phase D：缺陷分析与实验闭环 | ⬜ 待开始 |
| Phase E：演示交付与生产化加固 | ⬜ 待开始 |

---

## 已完成模块

### MR 生成（算子层）

| 模块 | 说明 |
|------|------|
| `mr_generator/operator/operator_mr_generator.py` | 主生成器（4 阶段流水线） |
| `mr_generator/operator/operator_llm_mr_generator.py` | LLM 猜想生成 |
| `analysis/mr_prechecker.py` | 随机输入快速筛选 |
| `mr_generator/operator/sympy_prover.py` | SymPy 符号证明引擎 |
| `mr_generator/operator/sympy_translator.py` | 代码 → SymPy（LLM + AST 双路径） |
| `mr_generator/operator/ast_parser.py` | Python AST → SymPy |
| `mr_generator/base/mr_templates.py` + YAML | MR 模板池 |
| `mr_generator/base/mr_repository.py` | MR 知识库（SQLite，含 `applicable_frameworks` 字段） |
| `mr_generator/base/knowledge_base.py` + YAML | 三层知识库 |
| `mr_generator/base/operator_catalog.py` | 算子目录管理 |
| `mr_generator/base/operator_enricher.py` | input_specs 自动丰富（inspect + HTML + LLM） |

### 工具层

| 模块 | 说明 |
|------|------|
| `tools/llm/client.py` | LLM 客户端（OpenAI/Anthropic） |
| `tools/web_search/search_agent.py` | HTML 解析 + 版本查询 |
| `tools/web_search/operator_fetcher.py` | 算子信息获取器 |
| `tools/web_search/sphinx_search.py` | Sphinx 文档索引解析 |
| `tools/agent/agent_core.py` | CrawlAgent（ReAct 模式） |
| `tools/agent/task_runner.py` | TaskRunner |

### 已实现 CLI 命令

| 命令 | 说明 |
|------|------|
| `deepmt mr generate` | 单算子生成 MR |
| `deepmt mr batch-generate` | 批量生成 MR（含断点续跑、dry-run） |
| `deepmt repo list/stats/info/delete` | MR 知识库查询与管理 |
| `deepmt catalog list/search/info/sync` | 算子目录管理 |
| `deepmt catalog latest-version` | 从 PyPI 获取框架版本 |
| `deepmt catalog fetch-doc` | 获取算子文档 |
| `deepmt catalog update-api` | 拉取 PyTorch API 列表 |
| `deepmt catalog import-api` | 批量导入 API 并丰富 input_specs |
| `deepmt catalog enrich` | 单算子 input_specs 丰富 |
| `deepmt test operator` | 单算子蜕变测试 |
| `deepmt health` | 健康检查 |

### 测试体系

```
tests/
├── unit/
│   ├── test_core.py            6 个（config、framework、IR）
│   ├── test_parsers.py         13 个（ASTParser + SympyTranslator）
│   ├── test_prover.py          8 个（SymPyProver，无 LLM 依赖）
│   ├── test_search.py          搜索工具
│   ├── test_enricher.py        22 个（OperatorEnricher）
│   ├── test_repo.py            14 个（MRRepository）
│   ├── test_mr_generate.py     18 个（模板/oracle/precheck/import）
│   └── test_batch_generate.py  13 个（batch-generate）
└── integration/
    ├── test_mr_generation.py   需 LLM API
    └── test_web_tools.py       需网络
```

**全部 124 个单元测试通过（无 LLM/网络依赖）。**

---

## 架构设计约定

- **框架参数化**：所有框架相关代码以 `FrameworkType` 作为参数，PyTorch 先行实现，其他框架入口抛 `NotImplementedError`。参考：`_SUPPORTED_FRAMEWORKS = {"pytorch"}`。
- **MR 与 IR 框架无关**：`MetamorphicRelation` 通过 `applicable_frameworks` 声明适用范围（`None` = 通用）；`transform_code` 仅用 Python 原生算术；框架 tensor 包装/解包由插件负责。
- **算子双态**：`function`（`torch.nn.functional.relu`）与 `module`（`torch.nn.ReLU`）；当前算子层测试统一用 `function` 形态。
- **input_specs 质量分层**：`confirmed`（人工确认）/ `auto-usable`（自动生成可用）/ `weak`（信息不足），通过 `input_specs_auto` 字段区分。

---

## 已知限制

1. **SymPy 验证限制**：含浮点的复杂性质无法符号证明，仅依赖 pre-check 数值验证
2. **LLM 依赖**：MR 猜想质量依赖提示工程与 API 密钥，单元测试通过 `use_llm=False` 隔离
3. **transform_code 可移植性**：跨框架 MR 要求 `transform_code` 不使用框架特定 API；PyTorch 阶段暂不强制，文档中标注

---

*最后更新：2026-04-09（文件重构同步）*
