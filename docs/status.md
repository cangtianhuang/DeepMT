# DeepMT 开发状态

> 详细开发规划与任务清单见 `docs/deepmt_dev_docs/`。本文件只记录"已完成"状态与已知限制。

## 阶段进度

| 阶段 | 状态 |
|------|------|
| Phase A：算子数据层完善（A1~A6） | ✅ 完成 |
| Phase B：算子层 MR 生成与知识库（B1~B3） | ✅ 完成 |
| Phase C：测试执行与跨框架适配 | ✅ 完成 |
| Phase D：缺陷分析与实验闭环 | ✅ 完成 |
| Phase E：演示交付与生产化加固 | ✅ 完成（E1~E6 全部完成） |
| Phase F：软件工程规范化与包发布准备（F1~F11） | ✅ F1~F11全部完成 |

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
| `mr_generator/base/mr_library.py` + YAML | MR 项目库（git 追踪，只读导出已验证 MR） |
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
| `deepmt test operator` | 单算子蜕变测试（手动指定输入） |
| `deepmt test batch` | 批量蜕变测试（RandomGenerator 自动生成输入，从 MR 知识库加载 MR） |
| `deepmt test mutate` | 变异测试（注入已知错误实现，验证 MR 缺陷检测能力） |
| `deepmt test report` | 生成测试结果报告（按算子/MR/失败聚合） |
| `deepmt test open` | 开放测试（FaultyPyTorchPlugin 注入预设缺陷，`DEEPMT_INJECT_FAULTS` 控制） |
| `deepmt test dedup` | 缺陷线索去重（将证据包聚类为独立缺陷模式） |
| `deepmt test evidence list/show/script` | 证据包管理（列出/展示/打印复现脚本） |
| `deepmt test history` | 查看测试历史摘要 |
| `deepmt test failures` | 查看失败测试用例 |
| `deepmt test cross` | 跨框架一致性测试（PyTorch vs NumPy，MR 级统计，支持持久化） |
| `deepmt test experiment` | 实验数据收集（RQ1-RQ4 结构化报告，支持 JSON 导出） |
| `deepmt health` | 健康检查 |

### 已完成 Phase C 模块

| 模块 | 说明 |
|------|------|
| `analysis/random_generator.py` | 输入生成器（RandomGenerator），解析 input_specs 生成随机张量 |
| `plugins/framework_plugin.py` | 框架插件抽象基类（统一执行接口） |
| `plugins/pytorch_plugin.py` | PyTorch 插件（ir_to_code/allclose/eval_expr/element_compare） |
| `engine/batch_test_runner.py` | 批量测试执行器（BatchTestRunner，dict kwargs 风格，接入 RandomGenerator） |
| `core/results_manager.py` | 测试结果持久化（SQLite） |
| `analysis/mr_verifier.py` | Oracle 验证器 |

### 已完成 Phase D 模块

| 模块 | 说明 |
|------|------|
| `analysis/report_generator.py` | 报告生成器（ReportGenerator），从 SQLite 汇总测试结果 |
| `analysis/mutation_tester.py` | 变异测试器（MutationTester + create_mutant_func），5 种变异类型 |
| `analysis/evidence_collector.py` | 证据包采集器（EvidenceCollector + EvidencePack），含可复现 Python 脚本生成 |
| `analysis/defect_deduplicator.py` | 缺陷去重器（DefectDeduplicator），按签名聚类证据包 |
| `plugins/faulty_pytorch_plugin.py` | 含预设缺陷的 PyTorch 插件（FaultyPyTorchPlugin），8 个算子缺陷目录，env var 控制 |
| `plugins/numpy_plugin.py` | NumPy 参考后端（NumpyPlugin），25+ 算子等价映射表，用于跨框架一致性测试 |
| `analysis/cross_framework_tester.py` | 跨框架一致性测试器（CrossFrameworkTester + CrossConsistencyResult + CrossSessionResult），结果持久化到 `data/cross_results/` |
| `analysis/experiment_organizer.py` | 实验数据组织器（ExperimentOrganizer），收集 RQ1-RQ4 数据，生成可 JSON 导出的结构化报告 |

### 已完成 Phase F-F11 模块（Web 仪表盘）

| 模块 | 说明 |
|------|------|
| `ui/app.py` | FastAPI 应用实例，挂载所有路由与静态文件 |
| `ui/server.py` | uvicorn 启动封装（`deepmt ui start` 调用） |
| `ui/templating.py` | 共享 Jinja2 模板引擎（全局注入版本号） |
| `ui/dependencies.py` | `lru_cache` 数据源单例（供路由复用） |
| `ui/routers/overview.py` | `GET /` — 总览页 |
| `ui/routers/mr_repo.py` | `GET /mr` + `GET /mr/{operator}` — MR 知识库页 |
| `ui/routers/test_results.py` | `GET /tests` — 测试结果页 |
| `ui/routers/cross_framework.py` | `GET /cross` — 跨框架一致性页 |
| `ui/routers/api.py` | `GET /api/**` — JSON 数据 API（10 个端点，TTL 缓存） |
| `ui/templates/*.html` | 6 个 Jinja2 模板（base + 5 页面，Chart.js + Bootstrap 5） |
| `ui/static/` | 本地化静态资源（Bootstrap 5.3.3 / Icons 1.11.3 / Chart.js 4.4.2），离线可用 |
| `commands/ui.py` | `deepmt ui start` CLI 命令 |

### 测试体系

```
tests/
├── unit/
│   ├── test_core.py              6 个（config、framework、IR）
│   ├── test_parsers.py           13 个（ASTParser + SympyTranslator）
│   ├── test_prover.py            8 个（SymPyProver，无 LLM 依赖）
│   ├── test_search.py            搜索工具
│   ├── test_enricher.py          22 个（OperatorEnricher）
│   ├── test_repo.py              14 个（MRRepository）
│   ├── test_mr_generate.py       18 个（模板/oracle/precheck/import）
│   ├── test_batch_generate.py    13 个（batch-generate）
│   ├── test_batch_test_runner.py 9 个（BatchTestRunner）
│   ├── test_mutation_tester.py   14 个（MutationTester）
│   ├── test_report_generator.py  14 个（ReportGenerator）
│   ├── test_evidence_collector.py 26 个（EvidenceCollector + BatchTestRunner 集成）
│   ├── test_faulty_plugin.py      22 个（FaultyPyTorchPlugin + backend_override）
│   ├── test_defect_deduplicator.py 26 个（DefectDeduplicator + DefectLead）
│   ├── test_numpy_plugin.py       30 个（NumpyPlugin 数值正确性与接口）
│   ├── test_cross_framework_tester.py 23 个（CrossConsistencyResult / CrossFrameworkTester）
│   ├── test_experiment_organizer.py   16 个（ExperimentOrganizer RQ1-RQ4）
│   └── test_ui_api.py                 27 个（Web 仪表盘 JSON API 端点，mock 数据源）
└── integration/
    ├── test_mr_generation.py   需 LLM API
    └── test_web_tools.py       需网络
```

**全部 369 个单元测试通过（无 LLM/网络依赖）。**

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

*最后更新：2026-04-13（Phase E 全部完成（E1~E6）：E4 补充完整双语 README（英文主页 + 中文版 README_CN.md）、GPU-capable Dockerfile；E5 新增 .github/workflows/ci.yml（GitHub Actions，三个 Job：Unit Tests × Python 3.10/3.11/3.12 矩阵 + CLI Health Check + Import Check）；E6 声明生产化最低边界达成（批量测试/历史对比/适配器扩展/报告输出全部已实现）；385 个单元测试通过）*
