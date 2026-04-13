# DeepMT 开发状态

> 本文档由 `00_开发总览与执行原则.md` 与旧 `docs/status.md` 合并而成，记录项目目标、交付形态、已完成模块及架构约定。

---

## 项目目标

DeepMT 后续开发围绕以下核心问题展开：

> 如何把"面向深度学习框架的分层蜕变测试体系研究"落成一个既能支撑硕士论文研究结论、又能实际发现框架问题、还能被演示和持续运行的软件系统。

系统必须同时具备三种能力：

- **研究闭环能力**：支持自动生成 MR、批量测试、可统计的实验数据、支撑 RQ1-RQ4；
- **演示交付能力**：短路径展示完整价值链，稳定复现真实缺陷或高价值异常；
- **生产落地能力**：模块边界清晰、核心流程可重复执行、结果可追踪、框架适配可扩展。

---

## 最终交付形态

系统至少支持以下四个稳定入口：

- **`catalog`**：管理算子与文档；
- **`mr`**：生成与治理蜕变关系；
- **`test`**：执行与复现测试；
- **`report`** / `test report`：分析与输出结果。

---

## 阶段进度

| 阶段                                          | 状态                     |
| --------------------------------------------- | ------------------------ |
| Phase A：算子数据层完善（A1~A6）              | ✅ 完成                   |
| Phase B：算子层 MR 生成与知识库（B1~B3）      | ✅ 完成                   |
| Phase C：测试执行与跨框架适配                 | ✅ 完成                   |
| Phase D：缺陷分析与实验闭环                   | ✅ 完成                   |
| Phase E：演示交付与生产化加固（E1~E6）        | ✅ 完成                   |
| Phase F：软件工程规范化与包发布准备（F1~F11） | ✅ F1~F11 全部完成         |
| Phase G：统一IR与三层对象建模                 | ✅ G1~G6 全部完成          |
| Phase H：第二框架落地与真实跨框架适配         | ✅ H1~H7 全部完成          |
| Phase I：模型层MR自动生成引擎                 | ⬜ 未开始                 |
| Phase J：应用层语义MR生成与验证               | ⬜ 未开始                 |
| Phase K：全层MR质量保障与统一知识库治理       | ⬜ 未开始                 |
| Phase L：论文实验基准与自动化数据生产线       | ⬜ 未开始                 |
| Phase M：真实缺陷挖掘与案例沉淀               | ⬜ 未开始                 |
| Phase N：论文交付收口与复现资产封装           | ⬜ 未开始                 |

---

## 已完成模块

### MR 生成（算子层）

| 模块                                                 | 说明                                                 |
| ---------------------------------------------------- | ---------------------------------------------------- |
| `mr_generator/operator/operator_mr_generator.py`     | 主生成器（4 阶段流水线）                             |
| `mr_generator/operator/operator_llm_mr_generator.py` | LLM 猜想生成                                         |
| `analysis/mr_prechecker.py`                          | 随机输入快速筛选                                     |
| `mr_generator/operator/sympy_prover.py`              | SymPy 符号证明引擎                                   |
| `mr_generator/operator/sympy_translator.py`          | 代码 → SymPy（LLM + AST 双路径）                     |
| `mr_generator/operator/ast_parser.py`                | Python AST → SymPy                                   |
| `mr_generator/base/mr_templates.py` + YAML           | MR 模板池                                            |
| `mr_generator/base/mr_repository.py`                 | MR 知识库（SQLite，含 `applicable_frameworks` 字段） |
| `mr_generator/base/mr_library.py` + YAML             | MR 项目库（git 追踪，只读导出已验证 MR）             |
| `mr_generator/base/operator_catalog.py`              | 算子目录管理                                         |
| `mr_generator/base/operator_enricher.py`             | input_specs 自动丰富（inspect + HTML + LLM）         |

### 工具层

| 模块                                   | 说明                           |
| -------------------------------------- | ------------------------------ |
| `tools/llm/client.py`                  | LLM 客户端（OpenAI/Anthropic） |
| `tools/web_search/search_agent.py`     | HTML 解析 + 版本查询           |
| `tools/web_search/operator_fetcher.py` | 算子信息获取器                 |
| `tools/web_search/sphinx_search.py`    | Sphinx 文档索引解析            |
| `tools/agent/agent_core.py`            | CrawlAgent（ReAct 模式）       |
| `tools/agent/task_runner.py`           | TaskRunner                     |

### 已实现 CLI 命令

| 命令                                    | 说明                                                                      |
| --------------------------------------- | ------------------------------------------------------------------------- |
| `deepmt mr generate`                    | 单算子生成 MR                                                             |
| `deepmt mr batch-generate`              | 批量生成 MR（含断点续跑、dry-run）                                        |
| `deepmt repo list/stats/info/delete`    | MR 知识库查询与管理                                                       |
| `deepmt catalog list/search/info/sync`  | 算子目录管理                                                              |
| `deepmt catalog latest-version`         | 从 PyPI 获取框架版本                                                      |
| `deepmt catalog fetch-doc`              | 获取算子文档                                                              |
| `deepmt catalog update-api`             | 拉取 PyTorch API 列表                                                     |
| `deepmt catalog import-api`             | 批量导入 API 并丰富 input_specs                                           |
| `deepmt catalog enrich`                 | 单算子 input_specs 丰富                                                   |
| `deepmt test operator`                  | 单算子蜕变测试（手动指定输入）                                            |
| `deepmt test batch`                     | 批量蜕变测试（RandomGenerator 自动生成输入，从 MR 知识库加载 MR）         |
| `deepmt test mutate`                    | 变异测试（注入已知错误实现，验证 MR 缺陷检测能力）                        |
| `deepmt test report`                    | 生成测试结果报告（按算子/MR/失败聚合）                                    |
| `deepmt test open`                      | 开放测试（FaultyPyTorchPlugin 注入预设缺陷，`DEEPMT_INJECT_FAULTS` 控制） |
| `deepmt test dedup`                     | 缺陷线索去重（将证据包聚类为独立缺陷模式）                                |
| `deepmt test evidence list/show/script` | 证据包管理（列出/展示/打印复现脚本）                                      |
| `deepmt test history`                   | 查看测试历史摘要                                                          |
| `deepmt test failures`                  | 查看失败测试用例                                                          |
| `deepmt test cross`                     | 跨框架一致性测试（PyTorch vs NumPy，MR 级统计，支持持久化）               |
| `deepmt test experiment`                | 实验数据收集（RQ1-RQ4 结构化报告，支持 JSON 导出）                        |
| `deepmt health`                         | 健康检查                                                                  |

### Phase C 模块

| 模块                           | 说明                                                                      |
| ------------------------------ | ------------------------------------------------------------------------- |
| `analysis/random_generator.py` | 输入生成器（RandomGenerator），解析 input_specs 生成随机张量              |
| `plugins/framework_plugin.py`  | 框架插件抽象基类（统一执行接口）                                          |
| `plugins/pytorch_plugin.py`    | PyTorch 插件（ir_to_code/allclose/eval_expr/element_compare）             |
| `engine/batch_test_runner.py`  | 批量测试执行器（BatchTestRunner，dict kwargs 风格，接入 RandomGenerator） |
| `core/results_manager.py`      | 测试结果持久化（SQLite）                                                  |
| `analysis/mr_verifier.py`      | Oracle 验证器                                                             |

### Phase D 模块

| 模块                                 | 说明                                                                                                                         |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `analysis/report_generator.py`       | 报告生成器（ReportGenerator），从 SQLite 汇总测试结果                                                                        |
| `analysis/mutation_tester.py`        | 变异测试器（MutationTester + create_mutant_func），5 种变异类型                                                              |
| `analysis/evidence_collector.py`     | 证据包采集器（EvidenceCollector + EvidencePack），含可复现 Python 脚本生成                                                   |
| `analysis/defect_deduplicator.py`    | 缺陷去重器（DefectDeduplicator），按签名聚类证据包                                                                           |
| `plugins/faulty_pytorch_plugin.py`   | 含预设缺陷的 PyTorch 插件（FaultyPyTorchPlugin），8 个算子缺陷目录，env var 控制                                             |
| `plugins/numpy_plugin.py`            | NumPy 参考后端（NumpyPlugin），25+ 算子等价映射表，用于跨框架一致性测试                                                      |
| `analysis/cross_framework_tester.py` | 跨框架一致性测试器（CrossFrameworkTester + CrossConsistencyResult + CrossSessionResult），结果持久化到 `data/cross_results/` |
| `analysis/experiment_organizer.py`   | 实验数据组织器（ExperimentOrganizer），收集 RQ1-RQ4 数据，生成可 JSON 导出的结构化报告                                       |

### Phase F（Web 仪表盘）

| 模块                            | 说明                                                                        |
| ------------------------------- | --------------------------------------------------------------------------- |
| `ui/app.py`                     | FastAPI 应用实例，挂载所有路由与静态文件                                    |
| `ui/server.py`                  | uvicorn 启动封装（`deepmt ui start` 调用）                                  |
| `ui/templating.py`              | 共享 Jinja2 模板引擎（全局注入版本号）                                      |
| `ui/dependencies.py`            | `lru_cache` 数据源单例（供路由复用）                                        |
| `ui/routers/overview.py`        | `GET /` — 总览页                                                            |
| `ui/routers/mr_repo.py`         | `GET /mr` + `GET /mr/{operator}` — MR 知识库页                              |
| `ui/routers/test_results.py`    | `GET /tests` — 测试结果页                                                   |
| `ui/routers/cross_framework.py` | `GET /cross` — 跨框架一致性页                                               |
| `ui/routers/api.py`             | `GET /api/**` — JSON 数据 API（10 个端点，TTL 缓存）                        |
| `ui/templates/*.html`           | 6 个 Jinja2 模板（base + 5 页面，Chart.js + Bootstrap 5）                   |
| `ui/static/`                    | 本地化静态资源（Bootstrap 5.3.3 / Icons 1.11.3 / Chart.js 4.4.2），离线可用 |
| `commands/ui.py`                | `deepmt ui start` CLI 命令                                                  |

### Phase H（PaddlePaddle 第二框架落地）

| 模块                                                   | 说明                                                                                                                                  |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `plugins/paddle_plugin.py`                             | `PaddlePlugin`：完整 FrameworkPlugin 实现；33 个算子映射（torch.* 名称 → paddle 等价）；支持 paddle.* 原生名称解析；graceful import guard |
| `plugins/plugins.yaml`                                 | 注册 `paddlepaddle` 插件条目                                                                                                          |
| `mr_generator/config/operator_mapping/pytorch_to_paddle.yaml` | PyTorch→PaddlePaddle 算子等价性声明表（论文依据），含 status/note                                              |
| `analysis/cross_framework_tester.py`                   | `DiffType` 差异类型常量；`CrossConsistencyResult` 新增 `diff_type_counts` 字段；`CrossSessionResult` 新增 `diff_type_summary` 属性；`_get_backend` 支持 paddle/paddlepaddle；`_normalize_framework` 别名映射 |
| `pyproject.toml`                                       | 新增 `[paddle]` 可选依赖组                                                                                                           |
| `tests/unit/test_paddle_plugin.py`                     | 75 个单元测试（接口/make_tensor/allclose/eval_expr/element_compare/算子解析/数值精度对比/跨框架集成/DiffType）                         |

### Phase G（统一 IR 底座）

| 模块                              | 说明                                                                                            |
| --------------------------------- | ----------------------------------------------------------------------------------------------- |
| `ir/schema.py`                    | `TestSubject` 统一基类；`OperatorIR/ModelIR/ApplicationIR` 继承；`MetamorphicRelation` 新增 `subject_name`/`lifecycle_state`/`sync_lifecycle()` |
| `core/run_manifest.py`            | `RunManifest` 运行清单（run_id、framework_version、random_seed、env_summary 等），支持序列化/反序列化 |
| `core/subject_registry.py`        | `SubjectRegistry` 三层主体注册表（register/lookup/list_by_type），为后续模型层/应用层接入提供统一入口 |
| `core/results_manager.py`         | `test_results` 新增 `run_id`/`framework_version`/`random_seed` 列；新增 `run_manifests` 表；`store_run_manifest`/`get_run_manifest` 接口 |
| `migrations/migrate_g_phase.py`   | Phase G 数据库迁移脚本（兼容追加列，支持 `--dry-run`，幂等可重复执行）                          |

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
│   ├── test_numpy_plugin.py       30 个（NumpyPlugin 数値正確性与接口）
│   ├── test_cross_framework_tester.py 23 个（CrossConsistencyResult / CrossFrameworkTester）
│   ├── test_experiment_organizer.py   16 个（ExperimentOrganizer RQ1-RQ4）
│   ├── test_ui_api.py                 27 个（Web 仪表盘 JSON API 端点，mock 数据源）
│   ├── test_ir_unified.py             50 个（TestSubject/OperatorIR/ModelIR/AppIR/MR新字段/SubjectRegistry/RunManifest）
│   ├── test_storage_migration.py      15 个（ResultsManager 新列/RunManifest 持久化/迁移脚本）
│   └── test_paddle_plugin.py          75 个（PaddlePlugin 接口/算子解析/数值精度/CrossFrameworkTester+paddle/DiffType）
└── integration/
    ├── test_mr_generation.py   需 LLM API
    └── test_web_tools.py       需网络
```

**全部 525 个单元测试通过（无 LLM/网络依赖）。**

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

*最后更新：2026-04-13（Phase H 全部完成：PaddlePaddle 第二框架落地，525 个单元测试通过；下一步进入 Phase I）*
