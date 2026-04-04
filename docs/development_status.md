# DeepMT 开发状态

## 当前阶段

**阶段 4**：算子层多源融合 MR 自动生成引擎完成，CrawlAgent 完成，CLI 完成，模型层开发中。

---

## 已完成模块

### 核心 MR 生成（算子层）

| 文件 | 状态 | 说明 |
|------|------|------|
| `mr_generator/operator/operator_mr.py` | ✅ | 主生成器（4阶段流水线） |
| `mr_generator/operator/operator_llm_mr_generator.py` | ✅ | LLM 猜想生成器 |
| `mr_generator/operator/mr_prechecker.py` | ✅ | 随机输入快速筛选（5组，80%阈值） |
| `mr_generator/operator/sympy_prover.py` | ✅ | SymPy 符号证明引擎 |
| `mr_generator/operator/sympy_translator.py` | ✅ | 代码→SymPy 转换（LLM+AST双路径） |
| `mr_generator/operator/ast_parser.py` | ✅ | Python AST→SymPy |
| `mr_generator/base/mr_templates.py` + YAML | ✅ | MR 模板池 |
| `mr_generator/base/mr_repository.py` | ✅ | MR 知识库（SQLite） |
| `mr_generator/base/knowledge_base.py` + YAML | ✅ | 三层知识库 |
| `mr_generator/base/operator_catalog.py` | ✅ | 算子目录管理 |

### 工具层

| 文件 | 状态 | 说明 |
|------|------|------|
| `tools/llm/client.py` | ✅ | LLM 客户端（OpenAI/Anthropic） |
| `tools/web_search/operator_fetcher.py` | ✅ | 算子信息获取器 |
| `tools/web_search/search_tool.py` | ✅ | 网络搜索工具 |
| `tools/web_search/sphinx_search.py` | ✅ | Sphinx 文档索引解析 |
| `tools/web_search/search_agent.py` | ✅ | HTML 解析 + 版本查询（无需 LLM） |
| `tools/agent/agent_core.py` | ✅ | CrawlAgent（ReAct 模式） |
| `tools/agent/task_runner.py` | ✅ | TaskRunner（语义 API） |
| `tools/agent/tool_registry.py` | ✅ | 工具注册表 |
| `tools/agent/tasks/*.yaml` | ✅ | 任务规格（get_operator_doc 等） |

### 基础设施

| 文件 | 状态 | 说明 |
|------|------|------|
| `core/` 全部 | ✅ | 微内核（调度、测试执行、IR管理、日志、配置） |
| `core/oracle_evaluator.py` | ✅ | oracle_expr 运行时评估 |
| `plugins/pytorch_plugin.py` | ✅ | PyTorch 插件 |
| `analysis/defect_classifier.py` | ✅ | 缺陷分类器 |
| `ir/schema.py` + `ir/converter.py` | ✅ | 统一 IR |
| `api/deepmt.py` | ✅ | 用户友好 API |
| `deepmt/` CLI | ✅ | 命令行入口（mr/test/repo/catalog/data/health） |
| `monitoring/` | ✅ | 健康检查与进度追踪 |

### 测试体系

```
tests/
├── unit/
│   ├── test_core.py       ✅  6 个用例（config、framework、IR等）
│   ├── test_parsers.py    ✅  13 个用例（ASTParser + SympyTranslator）
│   ├── test_prover.py     ⚠️  部分用例因 SymPy simplify 耗时可能超时
│   └── test_search.py     ✅  搜索工具测试
└── integration/
    ├── test_mr_generation.py   需 LLM API
    └── test_web_tools.py       需网络
```

> ⚠️ **已知问题**：`test_parsers.py::TestSympyTranslator` 和 `test_prover.py::TestProveMR` 中部分测试在无 LLM 配置时调用 `SympyTranslator.translate()`，会等待 LLM 调用超时。需修复：单元测试不应依赖 LLM。

---

## 进行中 / 待做

### 优先级 1：修复单元测试（阻塞CI）

- [ ] `tests/unit/test_parsers.py`：mock 掉 `SympyTranslator` 中的 LLM 调用，或新增 `use_llm=False` 参数
- [ ] `tests/unit/test_prover.py`：同上，确保单元测试不依赖 LLM 或网络

### 优先级 2：扩展算子支持

- [ ] 在 `mr_generator/config/operator_catalog/` 中补充更多算子（归一化：BatchNorm、LayerNorm）
- [ ] 在 `plugins/pytorch_plugin.py` 中添加对应算子映射

### 优先级 3：模型层 MR 生成

- [ ] 完善 `ModelIR` 数据结构（层类型、连接关系、参数）
- [ ] 实现 `mr_generator/model/model_mr.py`（网络拓扑 MR + 数据增强策略）
- [ ] 扩展插件支持模型执行

### 优先级 4：缺陷分析增强

- [ ] 实现 `analysis/report_generator.py`（HTML 报告 + 统计图表）
- [ ] 实现缺陷最小化算法

### 优先级 5：更多框架支持

- [ ] `plugins/tensorflow_plugin.py`
- [ ] `plugins/paddle_plugin.py`

---

## 已知问题

1. **SymPy 验证限制**：某些含浮点数的复杂性质无法用符号证明
2. **LLM 依赖**：猜想质量依赖提示工程，需要 API 密钥和网络
3. **单元测试含 LLM 调用**：见优先级1

---

## 关键设计约定（快速参考）

- `FrameworkType`：`Literal["pytorch", "tensorflow", "paddlepaddle"]`
- `oracle_expr`：使用变量 `orig`（变换后输出）、`trans`（变换后输出）、`x`（**原始**输入张量）；空字符串默认等值检查
- `transform_code`：kwargs 风格 lambda，如 `lambda k: {**k, 'input': 2 * k['input']}`
- MR 类别：linearity、monotonicity、idempotency、composition、invariance、symmetry、boundary
- agent 懒初始化：`TaskRunner` 和 `CrawlAgent` 均为 `@property` 懒加载

---

*最后更新：2026-04-04*
