# CLAUDE.md

本文件为 Claude Code 提供项目上下文指引。

## 项目简介

DeepMT（Deep Metamorphic Testing）是面向深度学习框架（PyTorch、TensorFlow、PaddlePaddle）的**蜕变关系（MR）自动生成与分层测试系统**。

## 当前阶段与下一步

> **在开始任何“阶段开发”任务前，必须先阅读 `docs/deepmt_dev_docs/` 中的规划文档。“代码修复”任务不必查阅。**  
> 入口：`docs/deepmt_dev_docs/README.md` → `00_开发总览与执行原则.md` → 对应阶段文档。  
> 执行规范：`docs/deepmt_dev_docs/06_编码智能体执行规范.md`

| 阶段                            | 状态           | 文档                                                              |
| ------------------------------- | -------------- | ----------------------------------------------------------------- |
| Phase A：算子数据层完善         | ✅ 完成         | `docs/deepmt_dev_docs/01_Phase_A_算子数据层完善.md`               |
| Phase B：算子层 MR 生成与知识库 | ✅ 完成         | `docs/deepmt_dev_docs/02_Phase_B_算子层MR生成与知识库.md`         |
| Phase C：测试执行与跨框架适配   | ✅ 完成         | `docs/deepmt_dev_docs/03_Phase_C_测试执行与跨框架适配.md`         |
| Phase D：缺陷分析与实验闭环     | ✅ 完成         | `docs/deepmt_dev_docs/04_Phase_D_缺陷分析、实验闭环与研究结论.md` |
| Phase E：演示交付与生产化加固   | 🔲 **当前目标** | `docs/deepmt_dev_docs/05_Phase_E_演示交付与生产化加固.md`         |

**当前主链：** 算子目录 → MR 生成 → 批量测试 → 缺陷分析 → **演示交付与生产化加固（Phase E）**

已完成：A1~A6（算子目录与 input_specs）、B1~B3（MR 知识库与批量生成）、C1~C5（RandomGenerator + BatchTestRunner + test batch 命令）、D1~D7（报告生成、变异测试、证据包、跨框架一致性、RQ1-RQ4 数据组织）。全量单元测试 377 个通过。

## 环境与运行

项目使用 `uv` 管理虚拟环境：

```bash
source .venv/bin/activate && PYTHONPATH=$(pwd) python -m pytest tests/
```

关键环境变量：`OPENAI_API_KEY`、`DEEPMT_LOG_LEVEL`、`DEEPMT_LOG_DIR`。详见 `docs/environment_variables.md`。

## 文档

| 文件                              | 内容               |
| --------------------------------- | ------------------ |
| `docs/deepmt_dev_docs/`           | 主规划文档         |
| `docs/status.md`                  | 已完成模块清单     |
| `docs/cli_reference.md`           | CLI 命令参考       |
| `docs/environment_variables.md`   | 环境变量说明       |
| `docs/operator_catalog_design.md` | 算子目录设计       |
| `docs/operator_mr_technical.md`   | 算子层 MR 技术细节 |
| `docs/quick_start.md`             | 快速上手           |

## 架构概览

采用**微内核 + 插件化**架构。MR 生成与测试执行分离：先生成 MR 存入知识库（SQLite），再复用于多次测试。

### MR 生成四阶段流水线（`deepmt/mr_generator/operator/operator_mr_generator.py`）

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
│   ├── core/               #   微内核框架
│   │   ├── config_manager.py   #   配置加载与管理
│   │   ├── ir_manager.py       #   IR 管理
│   │   ├── logger.py           #   日志（get_logger / log_structured）
│   │   ├── plugins_manager.py  #   插件加载
│   │   └── results_manager.py  #   结果管理
│   ├── engine/             #   测试执行引擎
│   │   ├── scheduler.py        #   任务调度
│   │   └── test_runner.py      #   测试执行（使用已生成的 MR）
│   ├── ir/                 #   统一中间表示
│   │   ├── schema.py           #   IR 与 MR 数据结构
│   │   └── converter.py        #   IR 转换器
│   ├── mr_generator/       #   MR 生成引擎
│   │   ├── operator/           #   算子层（核心）
│   │   │   ├── operator_mr_generator.py  #   主生成器（generate/verify 流水线）
│   │   │   ├── operator_llm_mr_generator.py  # LLM 生成
│   │   │   ├── sympy_prover.py               # 符号证明
│   │   │   ├── sympy_translator.py           # 代码→SymPy
│   │   │   └── ast_parser.py                 # AST 解析
│   │   ├── model/              #   模型层（开发中）
│   │   ├── application/        #   应用层（开发中）
│   │   ├── base/               #   知识库、模板池、MR 仓库
│   │   └── config/             #   模板/知识库 YAML + 算子目录
│   ├── plugins/            #   框架适配器（目前仅 PyTorch 可用）
│   ├── tools/              #   通用工具
│   │   ├── llm/            #     LLM 客户端 / OCR
│   │   └── web_search/     #     搜索、Sphinx 解析、算子文档获取（缓存位于 data/cache_web_search/）
│   ├── analysis/           #   输入生成、预检、验证（mr_prechecker.py、mr_verifier.py 等）
│   └── monitoring/         #   健康检查与进度追踪
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
   - 修改了开发进度、模块状态、架构设计 → 更新 `docs/status.md`（已完成模块列表）
   - 完成某阶段中的关键任务 → 在对应 `docs/deepmt_dev_docs/0X_Phase_*.md` 中标记完成状态
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

5. **无向后兼容性负担**
   - 本项目处于初期开发阶段，**不考虑任何向后兼容性**
   - 重命名函数/类/CLI 命令、修改接口签名、调整数据结构时，必须**彻底清理**：删除旧名称、更新所有调用点、移除兼容性代码
   - 禁止保留废弃别名、`_deprecated_` 包装、兼容性注释（如 `# kept for backward compat`）等过渡代码

6. **插件职责边界（反复违规，强制记忆）**
   - `FrameworkPlugin` / `FrameworkAdapter` 只提供**基础、框架专属**的原语接口：
     - ✅ 合法：`make_tensor(shape, dtype, value_range)`、`allclose`、`to_numpy`、`get_shape`、`_execute_operator`
     - ❌ 禁止：任何需要解析项目自定义数据格式（`input_specs`、MR 结构、YAML 字段）的逻辑
   - 需要解析 `input_specs` 或执行生成策略的代码，统一放在 `deepmt/analysis/`（如 `InputGenerator`），再调用插件的基础接口
