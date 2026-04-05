# CLAUDE.md

本文件为 Claude Code 提供项目上下文指引。

## 项目简介

DeepMT（Deep Metamorphic Testing）是面向深度学习框架（PyTorch、TensorFlow、PaddlePaddle）的**蜕变关系（MR）自动生成与分层测试系统**。

## 环境与运行

项目使用 `uv` 管理虚拟环境：

```bash
source .venv/bin/activate && PYTHONPATH=$(pwd) python -m pytest tests/
```

配置文件查找顺序：`DEEPMT_CONFIG_PATH` → 当前目录 `config.yaml` → 项目根 `config.yaml` → `~/.config/deepmt/config.yaml`

关键配置项（参考 `config.yaml.example`）：
- `llm.api_key` / `llm.url`：LLM API 密钥与地址

关键环境变量：`OPENAI_API_KEY`、`DEEPMT_LOG_LEVEL`、`DEEPMT_LOG_DIR`。详见 `docs/environment_variables.md`。

## 文档

| 文件                              | 内容                  |
| --------------------------------- | --------------------- |
| `docs/status.md`                  | 开发状态与进度        |
| `docs/environment_variables.md`   | 环境变量说明          |
| `docs/operator_catalog_design.md` | 算子目录设计          |
| `docs/operator_mr_technical.md`   | 算子层 MR 技术细节    |
| `docs/quick_start.md`             | 快速上手              |

## 架构概览

采用**微内核 + 插件化**架构。MR 生成与测试执行分离：先生成 MR 存入知识库（SQLite），再复用于多次测试。

### MR 生成四阶段流水线（`deepmt/mr_generator/operator/operator_mr.py`）

1. **信息准备** — 提取算子代码与文档（网络搜索）
2. **候选生成** — LLM 猜想 + 模板池匹配
3. **快速筛选（Pre-check）** — 随机输入数值验证
4. **形式化验证** — SymPy 符号证明

### 核心数据结构（`deepmt/ir/schema.py`）

- `OperatorIR`：算子描述（名称、输入、输出、属性）
- `MetamorphicRelation`：MR 对象，包含 `transform_code`（输入变换 lambda）、`oracle_expr`（输出关系表达式，使用变量 `orig`/`trans`/`x`）、`verified`、`category`

## 项目地图

```
.
├── deepmt/                 # 主包（CLI 入口 + 所有核心模块）
│   ├── __init__.py         #   公共 API（导出 DeepMT, TestResult）
│   ├── __main__.py         #   python -m deepmt 入口
│   ├── cli.py              #   CLI 命令组
│   ├── client.py           #   DeepMT / TestResult 高层 API
│   ├── commands/           #   CLI 子命令实现（mr / test / repo / catalog / data / health）
│   ├── core/               #   微内核框架（原 core/）
│   │   ├── config_loader.py    #   配置加载
│   │   ├── framework.py        #   FrameworkType 定义
│   │   ├── ir_manager.py       #   IR 管理
│   │   ├── logger.py           #   日志（get_logger / log_structured）
│   │   ├── oracle_evaluator.py #   oracle_expr 运行时评估
│   │   ├── plugins_manager.py  #   插件加载
│   │   ├── results_manager.py  #   结果管理
│   │   ├── scheduler.py        #   任务调度
│   │   └── test_runner.py      #   测试执行（使用已生成的 MR）
│   ├── ir/                 #   统一中间表示（原 ir/）
│   │   ├── schema.py           #   IR 与 MR 数据结构
│   │   └── converter.py        #   IR 转换器
│   ├── mr_generator/       #   MR 生成引擎（原 mr_generator/）
│   │   ├── operator/           #   算子层（核心）
│   │   │   ├── operator_mr.py  #     主生成器（generate/verify 流水线）
│   │   │   ├── operator_llm_mr_generator.py  # LLM 生成
│   │   │   ├── mr_prechecker.py              # 数值预检
│   │   │   ├── sympy_prover.py               # 符号证明
│   │   │   ├── sympy_translator.py           # 代码→SymPy
│   │   │   └── ast_parser.py                 # AST 解析
│   │   ├── model/              #   模型层（开发中）
│   │   ├── application/        #   应用层（开发中）
│   │   ├── base/               #   知识库、模板池、MR 仓库
│   │   └── config/             #   模板/知识库 YAML + 算子目录
│   ├── plugins/            #   框架适配器（原 plugins/，目前仅 PyTorch 可用）
│   ├── tools/              #   通用工具（原 tools/）
│   │   ├── llm/            #     LLM 客户端 / OCR
│   │   └── web_search/     #     搜索、Sphinx 解析、算子文档获取（缓存位于 data/web_search_cache/）
│   ├── analysis/           #   缺陷分类器（原 analysis/）
│   └── monitoring/         #   健康检查与进度追踪（原 monitoring/）
├── tests/                  # 测试用例
├── demo/                   # 快速演示
├── docs/                   # 开发文档
├── data/                   # 数据（日志、SQLite 数据库）
├── config.yaml             # 运行配置
└── pyproject.toml          # 项目元数据
```

## 开发规范（强制）

每次完成功能开发后，必须同步更新以下内容：

1. **文档同步**
   - 修改了开发进度、模块状态、架构设计 → 更新 `docs/status.md`
   - 新增或修改了 CLI 命令（含命令名、选项、行为）→ 更新 `docs/cli_reference.md`

2. **依赖同步**
   - 新增了 `import` 的第三方包 → 同时更新 `pyproject.toml`（`dependencies`）和 `requirements.txt`，两者保持一致

3. **测试覆盖**
   - 任何新功能必须在 `tests/` 下新增最小测试代码（不要求覆盖完全，只验证功能打通）
   - 测试文件放在对应层级：`tests/unit/` 或 `tests/integration/`
   - 单元测试不得依赖 LLM API 或网络（通过 `use_llm=False` / mock 隔离）

4. **框架参数化（拓展性）**
   - 凡涉及框架相关逻辑，框架名称必须以 `FrameworkType` 参数传入，**不得写死**（包括字符串字面量 `"pytorch"`）
   - PyTorch 先行实现，其他框架入口处抛出 `NotImplementedError`，保留接口占位
   - 参考现有模式：`_SUPPORTED_FRAMEWORKS = {"pytorch"}` + 显式的 `not_implemented_error` 提示
