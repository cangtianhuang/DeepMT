# DeepMT CLI 命令大全

DeepMT 提供统一的命令行接口，入口为 `python -m deepmt`。

## 运行方式

```bash
source .venv/bin/activate
PYTHONPATH=/home/lhy/DeepMT python -m deepmt <command> [options]
```

全局选项：

| 选项 | 说明 |
|------|------|
| `-h, --help` | 显示帮助 |
| `-V, --version` | 显示版本号 |

---

## 命令组总览

| 命令组 | 说明 | 实现状态 |
|--------|------|----------|
| `mr` | MR 生成与管理（算子层） | ✅ 已实现 |
| `test` | 测试执行 | ✅ 已实现（仅 pytorch） |
| `repo` | MR 知识库管理 | ✅ 已实现 |
| `catalog` | 算子目录浏览与跨框架查询 | ✅ 已实现 |
| `data` | 数据目录管理（日志清理等） | ✅ 已实现 |
| `health` | 系统健康检查与进度 | ✅ 已实现 |

---

## `deepmt catalog` — 算子目录浏览与查询

### `deepmt catalog list`

列出指定框架的算子，支持按分类（`--category`）和关键字（`--search`）筛选。
数据来源：`mr_generator/config/operator_catalog/<framework>.yaml`。

```
deepmt catalog list [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--framework, -f` | `pytorch` | 目标框架（`pytorch` / `tensorflow` / `paddlepaddle`）|
| `--category, -c` | 全部 | 按分类过滤（如 `activation` / `math` / `pooling` / `convolution` …）|
| `--search, -s` | 无 | 按名称关键字模糊过滤（含别名）|
| `--json` | `False` | 以 JSON 格式输出 |
| `--count-only` | `False` | 仅输出匹配算子数量 |

**示例：**

```bash
deepmt catalog list --framework pytorch                          # PyTorch 全部算子
deepmt catalog list --framework pytorch --category activation    # 仅激活函数
deepmt catalog list --framework tensorflow --search conv         # 搜索含 "conv" 的算子
deepmt catalog list --framework paddlepaddle --count-only        # 总数
deepmt catalog list --framework pytorch --category math --json
```

---

### `deepmt catalog search <keyword>`

跨所有框架搜索算子（模糊匹配名称及别名）。

```
deepmt catalog search <KEYWORD> [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--json` | `False` | 以 JSON 格式输出 |

**示例：**

```bash
deepmt catalog search relu
deepmt catalog search conv
deepmt catalog search batch_norm --json
```

---

### `deepmt catalog info <operator>`

查询算子的跨框架分布 + 知识库 MR 数量（联合视图）。

- 在各框架算子目录中精确/模糊匹配，显示正式名称、分类、引入版本、别名
- 从 SQLite 知识库中查询该算子已保存的 MR 数量和验证情况

```
deepmt catalog info <OPERATOR> [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--json` | `False` | 以 JSON 格式输出 |

**示例：**

```bash
deepmt catalog info relu
deepmt catalog info torch.nn.ReLU
deepmt catalog info conv2d --json
```

**输出示例：**

```
算子: relu

  框架目录分布:
    ✓ pytorch          torch.nn.ReLU  [activation]  since=1.0  → torch.nn.functional.relu
    ✓ tensorflow       tf.keras.layers.ReLU  [activation]  since=2.0
    ✓ paddlepaddle     paddle.nn.ReLU  [activation]  since=2.0

  知识库 MR 情况:
    ✓ 已有 MR   versions=[1]  total=3  verified=2
```

---

## `deepmt data` — 数据目录管理

### `deepmt data clean-logs`

清理 `data/logs/` 目录下的日志文件（`.log`）。

```
deepmt data clean-logs [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--log-dir` | `data/logs` | 日志目录路径 |
| `--before` | 无 | 删除此日期（YYYY-MM-DD，不含）之前的日志 |
| `--keep-days` | 无 | 保留最近 N 天的日志，更早的删除 |
| `--dry-run` | `False` | 仅预览，不实际删除 |
| `--yes, -y` | `False` | 跳过确认提示 |

> `--before` 与 `--keep-days` 互斥，不可同时使用。
> 不指定任何过滤条件时，清理所有日志文件。

**示例：**

```bash
deepmt data clean-logs --dry-run              # 预览要删除的文件
deepmt data clean-logs                        # 清理所有日志（需确认）
deepmt data clean-logs --keep-days 7          # 保留最近 7 天
deepmt data clean-logs --keep-days 7 --dry-run  # 先预览再执行
deepmt data clean-logs --before 2026-01-01 -y   # 删除 2026-01-01 之前，跳过确认
```

---

## `deepmt mr` — MR 生成与管理

### `deepmt mr generate <operator>`

为算子生成蜕变关系，并可选保存至知识库。

**当前支持层次**：`operator`（算子层）。`model` / `application` 层尚未实现，调用会给出明确提示。

```
deepmt mr generate <OPERATOR> [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--layer` | `operator` | 测试层次（`operator` / `model` / `application`）|
| `--framework` | `pytorch` | 目标框架（`pytorch` / `tensorflow` / `paddlepaddle`）|
| `--sources` | `llm,template` | MR 生成来源，逗号分隔（`llm` / `template`）|
| `--precheck / --no-precheck` | `True` | 启用数值预检（5组随机输入快速过滤）|
| `--sympy / --no-sympy` | `False` | 启用 SymPy 符号证明 |
| `--auto-fetch / --no-auto-fetch` | `False` | 自动从网络获取算子文档 |
| `--save / --no-save` | `True` | 将结果保存至知识库 |
| `--version` | `1` | 知识库版本号 |

**示例：**

```bash
# 最简调用（仅模板来源，不触发 LLM）
deepmt mr generate relu --sources template

# LLM + 模板，带数值预检，保存到知识库
deepmt mr generate relu --sources llm,template --precheck --save

# 同时进行 SymPy 符号证明（较慢）
deepmt mr generate relu --sympy

# 自动获取在线文档后再生成
deepmt mr generate relu --auto-fetch --sources llm

# 未实现层次（会给出友好提示并退出）
deepmt mr generate my_model --layer model
```

---

### `deepmt mr verify <operator>`

对知识库中已有的 MR 重新执行验证。

```
deepmt mr verify <OPERATOR> [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--version` | 最新 | 版本号 |
| `--precheck / --no-precheck` | `True` | 启用数值预检 |
| `--sympy / --no-sympy` | `False` | 启用 SymPy 符号证明 |
| `--save / --no-save` | `False` | 将验证结果更新到知识库 |

**示例：**

```bash
deepmt mr verify relu
deepmt mr verify relu --sympy --version 1
deepmt mr verify relu --precheck --save
```

---

### `deepmt mr list [operator]`

列出知识库中算子的蜕变关系。省略 `operator` 时列出所有算子名称。

```
deepmt mr list [OPERATOR] [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--version` | 最新 | 版本号 |
| `--verified-only` | `False` | 仅显示已验证的 MR |
| `--json` | `False` | 以 JSON 格式输出 |

**示例：**

```bash
deepmt mr list                        # 列出所有算子
deepmt mr list relu                   # 列出 relu 的所有 MR
deepmt mr list relu --verified-only   # 仅已验证的 MR
deepmt mr list relu --json            # JSON 输出
```

---

### `deepmt mr stats [operator]`

显示 MR 知识库统计信息。省略 `operator` 时统计全库。

```
deepmt mr stats [OPERATOR] [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--json` | `False` | 以 JSON 格式输出 |

**示例：**

```bash
deepmt mr stats            # 全库统计
deepmt mr stats relu       # 单算子统计
deepmt mr stats --json
```

---

### `deepmt mr delete <operator>` ⚠️ 未实现

删除知识库中算子的 MR 记录。

> **状态**：尚未实现。调用后会显示友好提示并以退出码 2 退出。
> 临时替代方案：直接操作 `data/mr_knowledge_base.db` 文件。

---

## `deepmt test` — 测试执行

> **框架支持**：目前仅 `pytorch` 插件已实现。指定 `tensorflow` / `paddlepaddle` 会给出友好提示。

### `deepmt test operator <operator>`

对算子运行蜕变测试。若知识库中已有 MR 则直接加载；否则自动生成（需要 LLM 配置）。

```
deepmt test operator <OPERATOR> [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--framework` | `pytorch` | 目标框架 |
| `--inputs` | `1.0,2.0` | 逗号分隔的输入值 |
| `--generate / --no-generate` | `True` | 若无 MR 则自动生成 |
| `--json` | `False` | 以 JSON 格式输出结果 |

**示例：**

```bash
deepmt test operator relu
deepmt test operator relu --inputs 1.0,-1.0,0.0
deepmt test operator relu --framework pytorch --json

# 未实现框架（给出提示）
deepmt test operator relu --framework tensorflow
```

---

### `deepmt test from-config <config_path>`

从 YAML 配置文件批量运行测试。

```
deepmt test from-config <CONFIG_PATH> [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--json` | `False` | 以 JSON 格式输出结果 |

**配置文件格式：**

```yaml
tests:
  - type: operator
    name: relu
    inputs: [1.0, -1.0, 0.0]
    framework: pytorch
  - type: operator
    name: add
    inputs: [1.0, 2.0]
    framework: pytorch
```

**示例：**

```bash
deepmt test from-config tests/my_tests.yaml
deepmt test from-config tests/my_tests.yaml --json
```

---

### `deepmt test history [name]`

查看测试历史记录。

```
deepmt test history [NAME] [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--limit` | `20` | 最多显示条数 |
| `--json` | `False` | 以 JSON 格式输出 |

**示例：**

```bash
deepmt test history
deepmt test history relu
deepmt test history --limit 100 --json
```

---

### `deepmt test failures`

查看所有失败的测试用例。

```
deepmt test failures [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--limit` | `50` | 最多显示条数 |
| `--json` | `False` | 以 JSON 格式输出 |

**示例：**

```bash
deepmt test failures
deepmt test failures --limit 100
deepmt test failures --json
```

---

## `deepmt repo` — MR 知识库管理

### `deepmt repo list`

列出知识库中所有有 MR 的算子，含版本、数量摘要。

```
deepmt repo list [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--json` | `False` | 以 JSON 格式输出 |

**示例：**

```bash
deepmt repo list
deepmt repo list --json
```

---

### `deepmt repo stats`

显示知识库整体统计（总数、已验证、Precheck/SymPy 通过数），以及按算子分布表格。

```
deepmt repo stats [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--json` | `False` | 以 JSON 格式输出 |

**示例：**

```bash
deepmt repo stats
deepmt repo stats --json
```

---

### `deepmt repo info <operator>`

显示算子的详细信息：版本列表、MR 数量、各 MR 摘要（类别、oracle 表达式、验证状态）。

```
deepmt repo info <OPERATOR> [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--version` | 所有版本 | 仅查看指定版本 |
| `--json` | `False` | 以 JSON 格式输出 |

**示例：**

```bash
deepmt repo info relu
deepmt repo info relu --version 1
deepmt repo info relu --json
```

---

## `deepmt health` — 系统健康检查

### `deepmt health check`

逐一尝试导入所有核心模块，输出通过/警告/错误统计，快速诊断环境问题。

```bash
deepmt health check
```

**输出示例：**

```
总体状态: ✅ healthy
通过: 22 | 警告: 0 | 错误: 0
```

---

### `deepmt health progress`

显示各模块的开发进度（百分比进度条），来源于 `monitoring/progress_tracker.py`。

```bash
deepmt health progress
```

---

### `deepmt health all`

同时运行进度报告与健康检查（等价于依次执行上面两条命令）。

```bash
deepmt health all
```

---

## 未实现功能速查

以下功能调用后会给出友好错误提示（退出码 2），**不会崩溃**：

| 命令 / 选项 | 未实现原因 |
|-------------|-----------|
| `deepmt mr generate <op> --layer model` | 模型层 MR 生成尚在开发中（进度约 30%）|
| `deepmt mr generate <op> --layer application` | 应用层 MR 生成尚未开始 |
| `deepmt test operator <op> --framework tensorflow` | TensorFlow 插件未实现 |
| `deepmt test operator <op> --framework paddlepaddle` | PaddlePaddle 插件未实现 |
| `deepmt mr delete <op>` | 删除接口未实现，临时方案：直接编辑 SQLite |

## 日志管理

```bash
deepmt data clean-logs --dry-run      # 查看有哪些日志
deepmt data clean-logs --keep-days 7  # 清理 7 天前的日志
```

---

## 典型使用流程

### 0. 浏览算子目录

```bash
deepmt catalog list --framework pytorch --category activation   # 查看所有激活函数
deepmt catalog search conv                                       # 搜索卷积相关算子
deepmt catalog info relu                                         # 查看 relu 框架分布 + MR 情况
```

### 1. 初次检查环境

```bash
deepmt health check
```

### 2. 为算子生成并保存 MR（仅模板来源，无需 LLM）

```bash
deepmt mr generate relu --sources template --precheck --save
```

### 3. 查看知识库状态

```bash
deepmt repo list
deepmt repo info relu
```

### 4. 运行蜕变测试

```bash
deepmt test operator relu --inputs 1.0,-1.0,0.0
```

### 5. 查看测试结果

```bash
deepmt test history relu
deepmt test failures
```

### 6. 触发 LLM 生成并保存（需配置 `config.yaml` 中的 `llm.api_key`）

```bash
deepmt mr generate relu --sources llm,template --precheck --sympy --save
```
