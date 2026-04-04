# CLAUDE.md

本文件为 Claude Code 提供项目上下文指引。

## 项目简介

DeepMT（Deep Metamorphic Testing）是面向深度学习框架（PyTorch、TensorFlow、PaddlePaddle）的**蜕变关系（MR）自动生成与分层测试系统**。

## 环境与运行

项目使用 `uv` 管理虚拟环境：

```bash
source .venv/bin/activate && PYTHONPATH=/home/lhy/DeepMT python -m pytest tests/
```

配置文件查找顺序：`DEEPMT_CONFIG_PATH` → 当前目录 `config.yaml` → 项目根 `config.yaml` → `~/.config/deepmt/config.yaml`

关键配置项（参考 `config.yaml.example`）：
- `llm.api_key` / `llm.url`：LLM API 密钥与地址
- `agent.enabled`：是否启用 CrawlAgent 自动获取算子文档（默认 false）
- `agent.max_steps` / `agent.cache_ttl_days`：全局覆盖各任务 YAML 的限制

关键环境变量：`OPENAI_API_KEY`、`DEEPMT_LOG_LEVEL`、`DEEPMT_LOG_DIR`。详见 `docs/environment_variables.md`。

## 文档

| 文件                              | 内容                  |
| --------------------------------- | --------------------- |
| `docs/status.md`                  | 开发状态与进度        |
| `docs/agent_crawler_plan.md`      | CrawlAgent 设计与进度 |
| `docs/environment_variables.md`   | 环境变量说明          |
| `docs/operator_catalog_design.md` | 算子目录设计          |
| `docs/operator_mr_technical.md`   | 算子层 MR 技术细节    |
| `docs/quick_start.md`             | 快速上手              |

## 架构概览

采用**微内核 + 插件化**架构。MR 生成与测试执行分离：先生成 MR 存入知识库（SQLite），再复用于多次测试。

### MR 生成四阶段流水线（`mr_generator/operator/operator_mr.py`）

1. **信息准备** — 提取算子代码 + CrawlAgent 获取文档
2. **候选生成** — LLM 猜想 + 模板池匹配
3. **快速筛选（Pre-check）** — 随机输入数值验证
4. **形式化验证** — SymPy 符号证明

### 核心数据结构（`ir/schema.py`）

- `OperatorIR`：算子描述（名称、输入、输出、属性）
- `MetamorphicRelation`：MR 对象，包含 `transform_code`（输入变换 lambda）、`oracle_expr`（输出关系表达式，使用变量 `orig`/`trans`/`x`）、`verified`、`category`

### CrawlAgent（`tools/agent/`）

基于 ReAct 模式的爬取智能体，通过 `TaskRunner` 提供语义 API：

- `TaskRunner.get_operator_doc(name, framework)` — 获取算子文档
- `TaskRunner.sync_catalog(framework)` — 更新算子目录 YAML
- 任务规格在 `tools/agent/tasks/*.yaml` 中声明（支持自定义 `entry_points`）
- 文件系统缓存位于 `data/agent_cache/`（TTL 可配）

## 项目地图

```
.
├── deepmt/                 # CLI 入口（python -m deepmt）
│   └── commands/           #   mr / test / repo / catalog / data / health
├── api/                    # 用户 API（DeepMT 主入口）
├── core/                   # 微内核框架
│   ├── config_loader.py    #   配置加载
│   ├── framework.py        #   FrameworkType 定义
│   ├── ir_manager.py       #   IR 管理
│   ├── logger.py           #   日志（get_logger / log_structured）
│   ├── oracle_evaluator.py #   oracle_expr 运行时评估
│   ├── plugins_manager.py  #   插件加载
│   ├── results_manager.py  #   结果管理
│   ├── scheduler.py        #   任务调度
│   └── test_runner.py      #   测试执行（使用已生成的 MR）
├── ir/                     # 统一中间表示
│   ├── schema.py           #   IR 与 MR 数据结构
│   └── converter.py        #   IR 转换器
├── mr_generator/           # MR 生成引擎
│   ├── operator/           #   算子层（核心）
│   │   ├── operator_mr.py  #     主生成器（generate/verify 流水线）
│   │   ├── operator_llm_mr_generator.py  # LLM 生成
│   │   ├── mr_prechecker.py              # 数值预检
│   │   ├── sympy_prover.py               # 符号证明
│   │   ├── sympy_translator.py           # 代码→SymPy
│   │   └── ast_parser.py                 # AST 解析
│   ├── model/              #   模型层（开发中）
│   ├── application/        #   应用层（开发中）
│   ├── base/               #   知识库、模板池、MR 仓库
│   └── config/             #   模板/知识库 YAML + 算子目录
├── plugins/                # 框架适配器（目前仅 PyTorch 可用）
├── analysis/               # 缺陷分类器
├── tools/                  # 通用工具
│   ├── llm/                #   LLM 客户端 / OCR
│   ├── agent/              #   CrawlAgent（ReAct）+ TaskRunner + ToolRegistry
│   └── web_search/         #   搜索、Sphinx 解析、算子文档获取
├── monitoring/             # 健康检查与进度追踪
├── tests/                  # 测试用例
├── examples/               # 示例脚本
├── demo/                   # 快速演示
├── docs/                   # 开发文档
├── data/                   # 数据（日志、SQLite 数据库）
├── config.yaml             # 运行配置
└── pyproject.toml          # 项目元数据
```

## 关键设计约定

- **`FrameworkType`**：`Literal["pytorch", "tensorflow", "paddlepaddle"]`
- **`oracle_expr`**：使用变量 `orig`、`trans`、`x`；空字符串默认为等值检查
- **`transform_code`**：kwargs 风格 lambda，如 `lambda k: {**k, 'input': 2 * k['input']}`
- **MR 类别**：linearity、monotonicity、idempotency、composition、invariance、symmetry、boundary
- **OperatorCatalog**：`merge_from_agent_result()` 只新增/更新 `doc_url`，不覆盖 `category`/`since`
- **agent 懒初始化**：`TaskRunner` 和 `CrawlAgent` 均为 `@property` 懒加载，导入时不触发 LLM 初始化