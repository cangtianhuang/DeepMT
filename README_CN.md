<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&pause=1000&color=4F9BF7&center=true&vCenter=true&width=700&lines=DeepMT+%F0%9F%94%AC;%E8%87%AA%E5%8A%A8%E7%94%9F%E6%88%90%E8%9C%95%E5%8F%98%E5%85%B3%E7%B3%BB;%E5%8F%91%E7%8E%B0+PyTorch+%2F+TF+%2F+Paddle+%E4%B8%AD%E7%9A%84%E7%BC%BA%E9%99%B7;%E6%B2%A1%E6%9C%89+Oracle%3F+%E4%B8%8D%E6%80%95%EF%BC%81" alt="Typing SVG" />

**面向深度学习框架的蜕变测试系统**

<table border="0" width="100%">
<tr>
<td width="60%" valign="top">

[![CI](https://github.com/cangtianhuang/DeepMT/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cangtianhuang/DeepMT/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10_%7C_3.11_%7C_3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-385%20passing-brightgreen)](tests/)

[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.9-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-F7DF1E?logo=opensourceinitiative&logoColor=black)](LICENSE)

</td>
<td width="40%" align="center" valign="middle">

```bash
# 无需 LLM，30 秒演示
source .venv/bin/activate
python demo/golden_path.py
```

</td>
</tr>
</table>

[🇺🇸 English](README.md) &nbsp;·&nbsp; [📖 文档](docs/) &nbsp;·&nbsp; [⚡ 快速开始](#-快速开始) &nbsp;·&nbsp; [🎬 演示](#-黄金演示路径) &nbsp;·&nbsp; [🏗️ 架构](#%EF%B8%8F-架构概览)

</div>

---

## 🧠 DeepMT 是什么？

DeepMT 是一个研究系统，能够**自动发现并验证深度学习算子的蜕变关系（MR）**，再用这些 MR 对框架实现进行压力测试——发现传统测试手段难以覆盖的数值错误、精度回归与跨框架不一致性。

<table border="0" width="100%">
<tr>
<td width="50%" valign="top">

### 问题所在

测试深度学习框架很困难——不存在**ground truth oracle**。你无法简单地断言 `torch.conv2d(x)` 的返回值是否"正确"，因为正确性本身依赖于被测实现。

</td>
<td width="50%" valign="top">

### 解决思路

**蜕变关系**通过检验输出之间的*关系*绕过 oracle 问题，例如：

```
relu(2x) == 2 · relu(x)    ∀ x ≥ 0
```

DeepMT 能**自动生成并形式化证明**这类关系，再将其用作测试判定依据。

</td>
</tr>
</table>

### ✨ 核心能力

| 能力 | 说明 |
|---|---|
| 🤖 **MR 自动生成** | LLM 猜想 → 模板池匹配 → SymPy 符号证明 |
| 🧪 **批量蜕变测试** | 从知识库加载已验证 MR，`RandomGenerator` 自动生成测试输入 |
| 🐛 **缺陷注入检测** | `FaultyPyTorchPlugin` 注入已知缺陷，量化 MR 缺陷检测率 |
| 🔀 **跨框架一致性** | PyTorch vs NumPy 算子级输出对比 |
| 📊 **Web 仪表盘** | Chart.js + Bootstrap 5 实时可视化报告 |
| 📦 **证据包** | 每个检测到的缺陷附带可独立运行的 Python 复现脚本 |

---

## 🚀 快速开始

### 安装

<table border="0" width="100%">
<tr>
<td width="33%" valign="top">

**uv（推荐）**
```bash
git clone https://github.com/\
  cangtianhuang/DeepMT.git
cd DeepMT
pip install uv
uv sync --all-extras
source .venv/bin/activate
```

</td>
<td width="33%" valign="top">

**pip**
```bash
git clone https://github.com/\
  cangtianhuang/DeepMT.git
cd DeepMT
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

</td>
<td width="34%" valign="top">

**Docker（GPU 版）**
```bash
git clone https://github.com/\
  cangtianhuang/DeepMT.git
cd DeepMT
docker build -t deepmt .
docker run --gpus all \
  -e OPENAI_API_KEY=sk-... \
  deepmt deepmt health check
```

</td>
</tr>
</table>

> **没有 GPU？** 构建时加上 `--build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cpu`。

### 验证安装

```bash
deepmt health check
```

<details>
<summary>预期输出 ▸</summary>

```
================================================================
DeepMT 系统健康检查报告
================================================================
总体状态: ✅ HEALTHY    通过: 38  警告: 0  错误: 0
...
所有核心模块运行正常。
================================================================
```

</details>

---

## 🎬 黄金演示路径

> 无需 LLM API · 无需网络 · 约 30 秒完成

```bash
source .venv/bin/activate
PYTHONPATH=$(pwd) python demo/golden_path.py
```

完整覆盖核心研究价值：

```
Step 1  算子目录与 MR 知识库展示 ── 已验证 MR 一览
Step 2  正常批量测试 ────────────── PyTorch 基线（全部通过）
Step 3  缺陷注入开放测试 ──────────── FaultyPyTorchPlugin 暴露预设缺陷
Step 4  测试报告生成 ────────────── 通过率、失败分布统计
Step 5  可复现证据包 ────────────── 可直接粘贴运行的 Python 脚本
```

---

## 🔄 核心工作流

### 生成 MR

```bash
deepmt mr generate torch.nn.functional.relu --save   # 单算子
deepmt mr batch-generate --framework pytorch          # 批量（目录中所有算子）
```

### 运行测试

```bash
deepmt test batch   --framework pytorch                       # 批量蜕变测试
deepmt test open    --inject-faults all --collect-evidence    # 缺陷注入测试
deepmt test cross                                             # PyTorch vs NumPy
```

### 分析结果

```bash
deepmt test report                  # 聚合通过/失败报告
deepmt test evidence list           # 证据包索引
deepmt test evidence show <id>      # 单个缺陷详情
deepmt ui start                     # Web 仪表盘 → http://localhost:8000
```

---

## 🏗️ 架构概览

**MR 生成四阶段流水线：**

```
┌────────────┐    ┌────────────────┐    ┌──────────────┐    ┌──────────────┐
│ ① 信息准备 │───▶│  ② 候选生成    │───▶│  ③ 快速预检  │───▶│  ④ 符号证明  │
│  文档/代码 │    │  LLM + 模板    │    │  随机数值验证 │    │  SymPy 形式化│
└────────────┘    └────────────────┘    └──────────────┘    └──────────────┘
```

**包结构：**

```
deepmt/
├── mr_generator/     🧬  MR 生成引擎
│   ├── operator/     │     LLM 猜想 · 模板池 · SymPy 证明
│   └── base/         │     SQLite 知识库 · MR 项目库
├── engine/           ⚙️   批量测试执行器（BatchTestRunner）
├── analysis/         🔍  输入生成 · Oracle 验证 · 报告 · 证据包
├── plugins/          🔌  框架适配器
│   ├── pytorch       │     PyTorch — 主实现
│   ├── numpy         │     NumPy   — 跨框架参考后端
│   └── faulty_pytorch│     缺陷注入后端（8 种算子缺陷类型）
├── ui/               📊  Web 仪表盘（FastAPI + Jinja2 + Chart.js）
├── commands/         💻  CLI 子命令
└── core/             🎛️   配置 · 日志 · 插件管理
```

---

## 🛠️ 配置

```bash
cp config.yaml.example config.yaml
# 编辑 config.yaml：
```

```yaml
llm:
  provider: "openai"
  api_key: "sk-..."        # 或：export OPENAI_API_KEY=sk-...
  model_base: "gpt-4o-mini"
  model_max:  "gpt-4o"
```

关键环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | — | LLM API 密钥（仅 MR 生成时需要） |
| `DEEPMT_LOG_LEVEL` | `INFO` | 日志级别 — `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `DEEPMT_LOG_CONSOLE_STYLE` | `colored` | 终端样式 — `colored` / `file` |
| `DEEPMT_INJECT_FAULTS` | — | 缺陷注入规格 — `all` 或 `op:mutant,...` |

完整说明 → [README_CONFIG.md](README_CONFIG.md) &nbsp;·&nbsp; [docs/environment_variables.md](docs/environment_variables.md)

---

## 🧪 运行测试

```bash
source .venv/bin/activate
# 全量单元测试，无 LLM/网络依赖，共 385 个
PYTHONPATH=$(pwd) python -m pytest tests/unit/ -v

# 生成 HTML 覆盖率报告
python -m pytest tests/unit/ --cov=deepmt --cov-report=html
```

---

## 📚 文档导航

| 文档 | 内容 |
|---|---|
| [README_CONFIG.md](README_CONFIG.md) | 配置文件与所有环境变量说明 |
| [docs/cli_reference.md](docs/cli_reference.md) | CLI 完整命令参考（20+ 条命令） |
| [docs/quick_start.md](docs/quick_start.md) | Python API 快速上手 |
| [docs/operator_mr_technical.md](docs/operator_mr_technical.md) | 算子层 MR 技术细节 |
| [docs/environment_variables.md](docs/environment_variables.md) | 环境变量参考 |
| [docs/status.md](docs/status.md) | 开发状态与已完成模块 |

---

## ⚙️ 环境要求

| 组件 | 要求 |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 1.9.0 · 推荐 GPU 版 |
| LLM API | 仅 MR 生成时需要（`OPENAI_API_KEY`） |
| 浏览器 | 任意现代浏览器（Web 仪表盘） |

---

## 📄 许可证

[MIT License](LICENSE) © 2024 cangtianhuang

<div align="center">
<br>
<sub>⭐ 如果 DeepMT 对您的研究有帮助，欢迎点个 Star，让更多人发现它！</sub>
</div>
