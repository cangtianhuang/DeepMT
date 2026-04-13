# Phase F：软件工程规范化与包发布准备

> **当前状态：✅ F1~F11 全部完成（2026-04-13）**  
> 385 个单元测试通过。

---

## 1. 阶段定位

Phase F 不引入新的研究功能，而是从**软件工程规范性**的角度对 DeepMT 进行系统整理，使其满足作为 Python 软件包发布的基础要求，并为 Phase E 的演示交付奠定可靠的工程底座。

问题来源：经过 A–D 阶段的快速研究迭代，积累了若干工程债——依赖声明不完整、构建配置错误、公开 API 与实现脱节、历史遗留 stub 被导出、开发期 hack 未清理。在进入演示阶段之前，这些问题需要优先解决。

一句话概括：

> Phase F 负责把 DeepMT 从"在作者机器上能跑"整理为"满足工程规范、可被他人安装和使用"。

---

## 2. 本阶段目标

### 目标一：包发布可行性

修复阻止 `pip install deepmt` 正常工作的所有错误：构建配置、依赖声明、包数据包含。

### 目标二：公开 API 可信性

清理对外导出的破损符号，使 `from deepmt import X` 的任何合法调用不会在运行时静默崩溃。

### 目标三：工程惯例对齐

补齐缺失的版本导出、元数据声明、依赖分组，使项目符合 PyPI 发布的标准惯例。

### 目标四：历史遗留 hack 清理

删除开发期遗留的 `sys.path` 手动注入和文档中的错误引用。

### 目标五（可选）：轻量可视化仪表盘

为 Phase E 演示提供一个只读 Web 仪表盘，将 RQ1-RQ4 数据、测试结果、跨框架对比可视化呈现。

---

## 3. 本阶段交付物

1. 可构建的 `pyproject.toml`（build-backend 正确、依赖完整、元数据规范）；
2. 清洁的公开 API（`deepmt/__init__.py`、`deepmt/mr_generator/__init__.py`）；
3. 无 `sys.path` 手动注入的代码库；
4. `requirements.txt` 与 `pyproject.toml` 对齐；
5. 修复的 `__version__` 导出；
6. （可选）轻量 Web 仪表盘。

---

## 4. 开发任务

---

### F1. 修复 `pyproject.toml` 构建配置 ✅ 已完成（2026-04-11）

**问题**：`build-backend` 字段值错误，安装包构建直接失败。

```toml
# 当前（错误）
build-backend = "setuptools.backends.legacy:build"
# 正确值
build-backend = "setuptools.build_meta"
```

**完成标准**：`pip install -e .` 执行成功，无构建错误。

---

### F2. 补齐 `pyproject.toml` 依赖声明 ✅ 已完成（2026-04-11）

**问题**：`click`、`sympy`、`requests`、`aiohttp`、`beautifulsoup4`、`openai` 均未在 `pyproject.toml` 中声明，但被代码导入。用户 `pip install deepmt` 后会立即遇到 `ImportError`。

**修改内容**：

- 将 `click` 加入 `dependencies`（CLI 框架，核心依赖）
- 将可选功能拆分为 optional-dependency 组：
  ```toml
  [project.optional-dependencies]
  llm        = ["openai>=1.0.0"]
  web-search = ["requests>=2.28.0", "aiohttp>=3.8.0",
                "beautifulsoup4>=4.11.0", "pylatexenc>=2.10"]
  sympy      = ["sympy>=1.9.0"]
  all        = ["deepmt[llm,web-search,sympy]"]
  ```
- 同步更新 `requirements.txt`，使其与 `pyproject.toml` 对齐（或将其改为 `requirements-dev.txt`，只含开发工具）

**完成标准**：`pip install deepmt` 后 `deepmt health` 可运行；CLI、核心测试通过。

---

### F3. 声明包内数据文件 ✅ 已完成（2026-04-11）

**问题**：`deepmt/plugins/plugins.yaml` 是运行时必需文件，但未被声明为包数据，打包后丢失，导致 `PluginsManager` 静默失败。

**修改内容**：

```toml
[tool.setuptools.package-data]
"deepmt.plugins" = ["*.yaml"]
```

检查是否还有其他需要随包分发的 YAML/数据文件（如模板配置），一并声明。

**完成标准**：构建后 wheel 中包含 `plugins.yaml`；`plugins_manager.load_plugins()` 不报"文件不存在"警告。

---

### F4. 移除公开 API 中的破损 stub ✅ 已完成（2026-04-11）

**问题**：`deepmt/mr_generator/__init__.py` 导出了 `ModelMRGenerator` 和 `ApplicationMRGenerator`，二者均为非功能性 stub，调用任何方法都会在运行时崩溃（引用了不存在的 `model_ir.get_layers()`、`llm_client.ask()` 等接口）。这是对外公开 API 的陷阱。

**修改内容**：

- 从 `mr_generator/__init__.py` 和 `__all__` 中移除这两个符号
- 在 stub 文件中添加注释，说明"待实现，暂不导出"

**完成标准**：`from deepmt.mr_generator import OperatorMRGenerator` 正常；尝试导入 `ModelMRGenerator` 得到明确的 `ImportError` 或文档说明。

---

### F5. 清除开发期 `sys.path` 手动注入 ✅ 已完成（2026-04-11）

**问题**：`deepmt/cli.py` 和 `deepmt/monitoring/__main__.py` 中有 `sys.path.insert(0, project_root)` 代码。安装后此代码有害无益，可能导致模块解析异常。

**修改内容**：

- 删除 `cli.py` 中的 `sys.path` 注入块
- 删除 `monitoring/__main__.py` 中的 `sys.path` 注入块
- 修正 `monitoring/__main__.py` 的注释（当前注释写的是 `python -m monitoring check`，应为 `python -m deepmt.monitoring check`）

**完成标准**：`deepmt --help` 正常运行；无 `sys.path` 操作代码残留。

---

### F6. 修复 `requires-python` 与补充 `pyproject.toml` 元数据 ✅ 已完成（2026-04-11）

**问题**：`requires-python = ">=3.8"` 已是 EOL 版本，代码实际可能使用了 3.9+ 特性（海象运算符、`str.removeprefix` 等）。`pyproject.toml` 缺少 `authors`、`license`、`keywords`、`classifiers`、`urls`，发布到 PyPI 时呈现不完整。

**修改内容**：

```toml
requires-python = ">=3.10"

authors = [{name = "作者名", email = "邮箱"}]
license = {text = "MIT"}
keywords = ["metamorphic testing", "deep learning", "pytorch", "software testing"]
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Software Development :: Testing",
    "Intended Audience :: Science/Research",
]

[project.urls]
Repository = "https://github.com/..."
```

**完成标准**：`python -m build` 生成的 wheel/sdist 元数据完整；`pip show deepmt` 显示 license 和 author。

---

### F7. 修复 `__version__` 导出与版本硬编码 ✅ 已完成（2026-04-11）

**问题**：`deepmt/__init__.py` 不导出 `__version__`；`cli.py` 中版本字符串 `"0.1.0"` 硬编码，升级时需要多处同步修改。

**修改内容**：

```python
# deepmt/__init__.py
try:
    from importlib.metadata import version
    __version__ = version("deepmt")
except Exception:
    __version__ = "0.1.0"  # fallback for editable installs

__all__ = ["DeepMT", "TestResult", "__version__"]
```

```python
# deepmt/cli.py —— 引用而非硬编码
from deepmt import __version__
@click.version_option(__version__, "-V", "--version", prog_name="deepmt")
```

**完成标准**：`python -c "import deepmt; print(deepmt.__version__)"` 输出正确版本；`deepmt --version` 与包元数据一致。

---

### F8. 同步 `requirements.txt` 与 `pyproject.toml` ✅ 已完成（2026-04-11）

**问题**：`requirements.txt` 包含 `click`、`sympy`、`openai`、`requests` 等，`pyproject.toml` 中缺失，形成双重标准，发布时安装依赖不可预期。

**修改内容**：

- 将 `requirements.txt` 改为**仅含开发工具**（`pytest`、`mypy`、`black` 等，可重命名为 `requirements-dev.txt`）
- 所有用户依赖统一声明在 `pyproject.toml` 的 `dependencies` 或 `optional-dependencies` 中
- 更新 README/快速开始文档中涉及 `pip install -r requirements.txt` 的描述

**完成标准**：`pip install -e ".[all]"` 完整安装所有功能；`requirements.txt` 只含开发依赖。

---

### F9. 对齐 `client.py` 公开 API 与实际主链 ✅ 已完成（2026-04-11）

**问题**：`from deepmt import DeepMT` 是对外的第一印象，但 `DeepMT.test_operator()` 内部使用旧式 `OperatorIR(inputs=inputs)` + `TaskScheduler` 路径，与当前主链（`BatchTestRunner + RandomGenerator + input_specs`）完全不同。用户调用公开 API 的行为与 CLI 不一致，且可能因接口变动而静默出错。

**修改策略**：不要求完全重写——在当前阶段，将 `client.py` 的职责明确为"CLI 主链的 Python 程序化调用包装"：

- 移除对 `TaskScheduler`、`ApplicationIR`、`ModelIR` 的引用（这些暂未实现）
- `test_operator()` 调用 `BatchTestRunner`，与 `deepmt test batch` CLI 行为对齐
- 或暂时在方法上加 `raise NotImplementedError` 并在 docstring 中说明"建议使用 CLI"

**完成标准**：`from deepmt import DeepMT` 不会产生导入错误；docstring 诚实地描述当前实现状态。

---

### F10. 修复文档中的错误引用 ✅ 已完成（2026-04-11）

**问题清单**：

1. `docs/status.md` 提到 `mr_generator/base/knowledge_base.py` + YAML，实际文件是 `mr_library.py`
2. `CLAUDE.md` 项目地图中列出 `ir/converter.py`，实际文件不存在
3. `deepmt/monitoring/__main__.py` 注释中模块路径错误

**修改内容**：逐一修正上述引用，使文档与代码实际结构一致。

**完成标准**：文档中所有文件路径引用均可在代码库中找到对应实体。

---

### F11. 轻量 Web 仪表盘（可选，服务 Phase E 演示） ✅ 已完成（2026-04-13）

**背景**：论文答辩和演示场景中，RQ1-RQ4 数据的可视化呈现（图表、统计摘要）比纯文本 CLI 输出更具说服力。

**方案**：

- 后端：FastAPI（已有 aiohttp，可替代；优先 FastAPI 生态）
- 前端：Jinja2 模板 + Chart.js（无需 Node.js 构建链）
- 作用域：**只读**，不实现表单化 MR 生成（保留在 CLI）
- 功能页：
  - 总览页（RQ1-RQ4 摘要 + 最近测试会话）
  - MR 知识库浏览（算子列表、MR 统计、来源分类分布）
  - 测试结果面板（通过率趋势、失败算子列表、证据包入口）
  - 跨框架一致性（会话列表、算子级一致率对比图）

**CLI 入口**：`deepmt ui start [--port 8080]`

**完成标准**：`deepmt ui start` 可在本地浏览器访问仪表盘；RQ1-RQ4 核心指标可读；不依赖外部网络。

---

## 5. 本阶段建议实现顺序

```
P0 阶段（必须先完成，任何一项失败都会阻止安装/运行）：
  F1 → F2 → F3 → F4 → F5

P1 阶段（规范性，影响发布质量和用户体验）：
  F6 → F7 → F8 → F9

P2 阶段（次要，可在后续迭代完成）：
  F10 → F11（可选）
```

---

## 6. 本阶段完成标志

- `pip install -e .` 和 `pip install -e ".[all]"` 均无错误完成；
- `from deepmt import DeepMT, __version__` 导入无崩溃；
- `deepmt health` 通过；
- 377 个单元测试全部通过；
- `pyproject.toml` 包含完整元数据和依赖声明；
- 代码库中不存在 `sys.path.insert(0, ...)` 形式的路径注入。
