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
├── deepmt/                     # 主包（CLI 入口 + 所有核心模块）
│   ├── __init__.py             #   公共 API（导出 DeepMT, TestResult, __version__）
│   ├── __main__.py             #   python -m deepmt 入口
│   ├── cli.py                  #   CLI 命令组
│   ├── client.py               #   DeepMT / TestResult 高层 API
│   ├── commands/               #   CLI 子命令实现（mr / test / repo / catalog / data / health）
│   ├── core/                   #   微内核框架
│   ├── engine/                 #   测试执行引擎
│   ├── ir/                     #   统一中间表示
│   ├── mr_generator/           #   MR 生成引擎
│   ├── tools/                  #   通用工具（LLM / 网络搜索）
│   ├── plugins/                #   框架适配器（PyTorch 可用）
│   ├── analysis/               #   输入生成、预检、验证、报告
│   ├── monitoring/             #   健康检查与进度追踪
│   └── ui/                     #   Web 仪表盘（FastAPI + Jinja2）
├── tests/                      # 测试用例（unit/ + integration/）
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
deepmt mr generate torch.relu --framework pytorch --save
deepmt test batch --operator torch.relu --framework pytorch
deepmt test report
deepmt ui start
```

### Python API

```python
from deepmt import DeepMT

client = DeepMT()
result = client.run_batch_test("torch.relu", framework="pytorch", n_samples=10)
print(result.summary())
```

详见 `docs/quick_start.md`。

---

## 五、配置

配置文件 `config.yaml`（参考 `config.yaml.example`）：

```yaml
llm:
  provider: "openai"
  api_key: "your-api-key"
  model: "gpt-4"
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

所有阶段（Phase A–F）均已完成，385 个单元测试通过。详见 `docs/dev/status.md`。

| 层次           | 状态              |
|----------------|-------------------|
| 算子层 MR 生成  | ✅ 完成            |
| 批量测试执行    | ✅ 完成            |
| 缺陷分析报告    | ✅ 完成            |
| 跨框架一致性    | ✅ 完成（PyTorch vs NumPy）|
| Web 仪表盘      | ✅ 完成            |
| 模型层 MR 生成  | 🚧 接口占位，未实现 |
| 应用层 MR 生成  | 📋 计划中          |

---

## 七、文档索引

### 用户文档

| 文件 | 内容 |
|------|------|
| `docs/quick_start.md` | 快速上手（安装、CLI、Python API） |
| `docs/cli_reference.md` | CLI 完整命令参考 |
| `docs/environment_variables.md` | 环境变量说明 |
| `docs/demo_golden_path.md` | 答辩/演示用黄金演示路径 |

### 技术设计

| 文件 | 内容 |
|------|------|
| `docs/tech/operator_catalog.md` | 算子目录设计与 YAML 规范 |
| `docs/tech/operator_mr.md` | 算子层 MR 生成与测试技术细节 |
| `docs/tech/web_dashboard.md` | Web 仪表盘技术设计 |

### 开发文档

| 文件 | 内容 |
|------|------|
| `docs/dev/status.md` | 开发状态、已完成模块、架构约定 |
| `docs/dev/agent_rules.md` | 编码智能体执行规范与开发原则 |
| `docs/dev/01_Phase_A_*.md` ~ `07_Phase_F_*.md` | 各阶段详细规划文档 |

---

*最后更新：2026-04-13*
