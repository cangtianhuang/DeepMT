# DeepMT CLI 命令大全

DeepMT 提供统一的命令行接口，入口为 `python -m deepmt`。

## 运行方式

```bash
source .venv/bin/activate
PYTHONPATH=/home/lhy/DeepMT python -m deepmt <command> [options]
```

全局选项：

| 选项            | 说明       |
| --------------- | ---------- |
| `-h, --help`    | 显示帮助   |
| `-V, --version` | 显示版本号 |

---

## 命令组总览

| 命令组    | 说明                               | 实现状态                                         |
| --------- | ---------------------------------- | ------------------------------------------------ |
| `mr`      | MR 生成与管理（算子层）            | ✅ 已实现                                         |
| `test`    | 测试执行                           | ✅ 已实现（pytorch 完整，tensorflow/paddle 占位） |
| `repo`    | MR 知识库管理                      | ✅ 已实现                                         |
| `catalog` | 算子目录浏览、跨框架查询、批量导入 | ✅ 已实现                                         |
| `data`    | 数据目录管理（日志清理等）         | ✅ 已实现                                         |
| `health`  | 系统健康检查与进度                 | ✅ 已实现                                         |
| `ui`      | Web 仪表盘服务器                   | ✅ 已实现                                         |

> **注意**：模型层和应用层 MR 生成目前通过 Python API 使用（`ModelMRGenerator`、`ApplicationMRGenerator`），CLI 的 `mr generate --layer model/application` 入口仍在开发中。

---

## `deepmt catalog` — 算子目录浏览与查询

子命令总览：

| 子命令           | 说明                                                                   |
| ---------------- | ---------------------------------------------------------------------- |
| `list`           | 列出框架算子，支持按分类/关键字筛选                                    |
| `search`         | 跨所有框架模糊搜索算子                                                 |
| `info`           | 查询算子的跨框架分布及知识库 MR 数量                                   |
| `latest-version` | 从 PyPI 获取框架最新/历史版本                                          |
| `fetch-doc`      | 获取算子官方文档正文                                                   |
| `update-api`     | 从官方文档更新 API 模块列表缓存                                        |
| `check-updates`  | 对比官方文档 API 与本地目录，报告差异                                  |
| `import-api`     | 批量导入官方文档 API 到算子目录（支持清空重建与 input_specs 自动丰富） |
| `enrich`         | 对单个算子执行 input_specs 丰富（inspect + HTML + 可选 LLM）           |

---

### `deepmt catalog list`

列出指定框架的算子，支持按分类（`--category`）和关键字（`--search`）筛选。
数据来源：`mr_generator/config/operator_catalog/<framework>.yaml`。

```
deepmt catalog list [OPTIONS]
```

| 选项              | 默认值    | 说明                                                                 |
| ----------------- | --------- | -------------------------------------------------------------------- |
| `--framework, -f` | `pytorch` | 目标框架（`pytorch` / `tensorflow` / `paddlepaddle`）                |
| `--category, -c`  | 全部      | 按分类过滤（如 `activation` / `math` / `pooling` / `convolution` …） |
| `--search, -s`    | 无        | 按名称关键字模糊过滤（含别名）                                       |
| `--json`          | `False`   | 以 JSON 格式输出                                                     |
| `--count-only`    | `False`   | 仅输出匹配算子数量                                                   |

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

| 选项     | 默认值  | 说明             |
| -------- | ------- | ---------------- |
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

| 选项     | 默认值  | 说明             |
| -------- | ------- | ---------------- |
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

### `deepmt catalog latest-version`

从 PyPI 快速获取框架最新稳定版本号（无需 LLM，纯 HTTP）。

```
deepmt catalog latest-version [OPTIONS]
```

| 选项              | 默认值    | 说明                                 |
| ----------------- | --------- | ------------------------------------ |
| `--framework, -f` | `pytorch` | 目标框架                             |
| `--all-versions`  | `False`   | 列出所有历史版本（降序，最多 20 条） |
| `--json`          | `False`   | 以 JSON 格式输出                     |

**示例：**

```bash
deepmt catalog latest-version
deepmt catalog latest-version --framework tensorflow
deepmt catalog latest-version --all-versions
deepmt catalog latest-version --json
```

---

### `deepmt catalog fetch-doc <operator>`

从官方文档网页获取并打印算子文档正文（无需 LLM）。

URL 解析优先级（由高到低）：
1. `--url` 显式指定
2. 算子目录 YAML 中的 `doc_url` 字段
3. `build_doc_url(framework, operator)` 按框架模板自动构造

```
deepmt catalog fetch-doc <OPERATOR> [OPTIONS]
```

| 选项              | 默认值                   | 说明                               |
| ----------------- | ------------------------ | ---------------------------------- |
| `--framework, -f` | `pytorch`                | 目标框架                           |
| `--url, -u`       | 自动解析（见上方优先级） | 直接指定文档页面 URL，跳过自动解析 |
| `--max-chars`     | `3000`                   | 最多显示字符数（0 = 不限）         |

**示例：**

```bash
deepmt catalog fetch-doc torch.matmul           # 自动从 YAML 或模板获取 URL
deepmt catalog fetch-doc relu --max-chars 0     # 不限制字符数
deepmt catalog fetch-doc torch.matmul --url https://docs.pytorch.org/docs/stable/generated/torch.matmul.html
```

---

### `deepmt catalog update-api`

从官方文档获取 API 模块列表并缓存到本地（无需 LLM）。

> 获取的是**模块级**列表（如 `torch.nn`），非个体 API。获取个体 API 并与目录对比，请使用 `check-updates`。

```
deepmt catalog update-api [OPTIONS]
```

| 选项                | 默认值    | 说明                            |
| ------------------- | --------- | ------------------------------- |
| `--framework, -f`   | `pytorch` | 目标框架（当前支持 `pytorch`）  |
| `--version, -v`     | `stable`  | 文档版本（`stable` 或如 `2.1`） |
| `--no-cache`        | `False`   | 忽略本地缓存，强制重新拉取      |
| `--json`            | `False`   | 以 JSON 格式输出 API 列表       |
| `--show-cache-path` | `False`   | 显示缓存文件路径及状态          |

**示例：**

```bash
deepmt catalog update-api
deepmt catalog update-api --no-cache
deepmt catalog update-api --json
deepmt catalog update-api --show-cache-path
```

---

### `deepmt catalog check-updates`

对比官方文档 API 与本地算子目录（`pytorch.yaml`），报告新增/变更的 API。

输出三类差异：
1. 目录中的算子，文档中签名已变更
2. 文档中存在，但目录和排除列表均未收录的新 API
3. 目录中的算子，在文档中未找到（可能已改名或删除）

> 首次运行会拉取所有相关模块页面（约 10-30 个，约 1-2 分钟）；之后 24 小时内命中缓存（毫秒级）。

```
deepmt catalog check-updates [OPTIONS]
```

| 选项                | 默认值    | 说明                                                            |
| ------------------- | --------- | --------------------------------------------------------------- |
| `--framework, -f`   | `pytorch` | 目标框架（当前支持 `pytorch`）                                  |
| `--version, -v`     | `stable`  | 文档版本                                                        |
| `--no-cache`        | `False`   | 忽略缓存，强制重新拉取                                          |
| `--show-no-sig`     | `False`   | 显示目录中尚未记录签名的算子                                    |
| `--skip-namespaces` | 无        | 跳过指定命名空间（逗号分隔，如 `torch.distributed,torch.cuda`） |
| `--json`            | `False`   | 以 JSON 格式输出完整 diff 结果                                  |

**示例：**

```bash
deepmt catalog check-updates
deepmt catalog check-updates --no-cache
deepmt catalog check-updates --show-no-sig
deepmt catalog check-updates --skip-namespaces torch.distributions,torch.sparse
deepmt catalog check-updates --json > diff.json
```

---

### `deepmt catalog import-api`

从官方文档批量导入 API 到算子目录 YAML（无需 LLM）。

有两种模式：

| 模式             | 命令                   | 说明                                                        |
| ---------------- | ---------------------- | ----------------------------------------------------------- |
| 合并模式（默认） | `import-api`           | 仅添加目录中不存在的新 API，保留已有条目的分类/版本等元数据 |
| 替换模式         | `import-api --replace` | **清空现有目录**，以文档 API 列表完全重建（元数据将丢失）   |

```
deepmt catalog import-api [OPTIONS]
```

| 选项                             | 默认值    | 说明                                                                  |
| -------------------------------- | --------- | --------------------------------------------------------------------- |
| `--framework, -f`                | `pytorch` | 目标框架（当前支持 `pytorch`）                                        |
| `--version, -v`                  | `stable`  | 文档版本                                                              |
| `--replace`                      | `False`   | 清空现有目录（含 signature/input_specs），完全替换为文档中的 API 列表 |
| `--no-cache`                     | `False`   | 忽略缓存，强制重新拉取                                                |
| `--dry-run`                      | `False`   | 试运行：仅显示将写入的内容，不修改文件                                |
| `--enrich`                       | `False`   | 自动丰富缺少 `input_specs` 的条目（inspect + HTML，默认不调用 LLM）   |
| `--enrich-llm / --no-enrich-llm` | `False`   | `--enrich` 时是否额外启用 LLM 提取约束（需配置 LLM API）              |

丰富策略（`--enrich`）：
1. **inspect（离线）**：参数名、类型注解、签名
2. **HTML 解析（需网络）**：从文档页补充 dtype
3. **LLM（显式开启）**：加 `--enrich-llm` 后提取 value_range、shape 等语义约束；启动时声明总调用次数，每批 8 个算子后请求确认。

生成的 `input_specs` 会自动标记 `input_specs_auto: true`，提示需人工核查。`dtype` 有三种值：`[]`（未知）/ `any`（无约束）/ `[float32, ...]`（明确类型）。

**示例：**

```bash
# 合并模式：仅添加新 API
deepmt catalog import-api

# 替换模式：清空后全量导入（先预览再执行，原 signature/input_specs 会被清除）
deepmt catalog import-api --replace --dry-run
deepmt catalog import-api --replace

# 导入并自动丰富 input_specs（仅 inspect + HTML，不调用 LLM）
deepmt catalog import-api --enrich

# 导入并丰富，启用 LLM（每批 8 次后确认，需配置 config.yaml 中的 llm.api_key）
deepmt catalog import-api --enrich --enrich-llm
```

> **如何对全部算子批量丰富 input_specs（无 LLM）？**
>
> ```bash
> deepmt catalog import-api --enrich
> ```
>
> 约 1 分钟完成（8 线程并发，已有 input_specs 的条目自动跳过）。

> **如何清空现有 PyTorch 算子列表并重新更新？**
>
> ```bash
> # 第 1 步（可选）：先预览会写入什么
> deepmt catalog import-api --replace --dry-run
>
> # 第 2 步：执行清空 + 重建
> deepmt catalog import-api --replace
> ```
>
> `--replace` 会清空 `mr_generator/config/operator_catalog/pytorch.yaml` 中所有现有条目（包括 `signature`、`input_specs` 等字段），用从官方文档抓取的最新 API 列表（经排除列表过滤后）完全替换。原有的 `category`、`since` 等元数据将丢失，需要重新手动标注。

---

### `deepmt catalog enrich <operator>`

对**单个算子**执行 `input_specs` 自动丰富（inspect + HTML + 可选 LLM），并就地写回算子目录 YAML。

适用场景：不想重跑全量 `import-api --enrich`，只需更新某一个算子的 `input_specs`。

```
deepmt catalog enrich <OPERATOR> [OPTIONS]
```

| 选项               | 默认值    | 说明                                        |
| ------------------ | --------- | ------------------------------------------- |
| `--framework, -f`  | `pytorch` | 目标框架                                    |
| `--llm / --no-llm` | `False`   | 是否启用 LLM 提取约束条件（需配置 LLM API） |
| `--dry-run`        | `False`   | 仅打印丰富结果，不写入 YAML                 |

**丰富策略**（按优先级）：

1. **inspect（离线，最快）**：解析函数/方法签名，提取 Tensor 参数名、注解
   - 对 C builtin 方法（如 `torch.Tensor.argmin`）：从 `__doc__` 第一行解析签名，`self` 对应输入张量标记为 `dtype: any`
2. **HTML 解析（需网络）**：当 inspect 结果中存在 `dtype: []`（未知）时，从官方文档页面补充类型信息
3. **LLM（可选）**：提取 `value_range`、`shape` 等语义约束

`dtype` 的三态语义：

| 值               | 含义                                               |
| ---------------- | -------------------------------------------------- |
| `[]`             | 未知，尚未确定                                     |
| `any`            | 无 dtype 限制（如纯 `Tensor` 注解或 builtin 方法） |
| `[float32, ...]` | 受限，仅支持这些 dtype                             |

**示例：**

```bash
# 预览，不写入（推荐先执行）
deepmt catalog enrich torch.Tensor.argmin --dry-run

# 写入 YAML
deepmt catalog enrich torch.Tensor.argmin

# 启用 LLM 提取约束
deepmt catalog enrich torch.matmul --llm

# 查看丰富结果
deepmt catalog info torch.Tensor.argmin
```

---

## `deepmt data` — 数据目录管理

### `deepmt data clean-logs`

清理 `data/logs/` 目录下的日志文件（`.log`）。

```
deepmt data clean-logs [OPTIONS]
```

| 选项          | 默认值      | 说明                                     |
| ------------- | ----------- | ---------------------------------------- |
| `--log-dir`   | `data/logs` | 日志目录路径                             |
| `--before`    | 无          | 删除此日期（YYYY-MM-DD，不含）之前的日志 |
| `--keep-days` | 无          | 保留最近 N 天的日志，更早的删除          |
| `--dry-run`   | `False`     | 仅预览，不实际删除                       |
| `--yes, -y`   | `False`     | 跳过确认提示                             |

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

**当前支持层次**：`operator`（算子层）。`model` / `application` 层的 CLI 入口尚未完整实现，请通过 Python API（`ModelMRGenerator` / `ApplicationMRGenerator`）使用。

```
deepmt mr generate <OPERATOR> [OPTIONS]
```

| 选项                             | 默认值         | 说明                                                  |
| -------------------------------- | -------------- | ----------------------------------------------------- |
| `--layer`                        | `operator`     | 测试层次（`operator` / `model` / `application`）      |
| `--framework`                    | `pytorch`      | 目标框架（`pytorch` / `tensorflow` / `paddlepaddle`） |
| `--sources`                      | `llm,template` | MR 生成来源，逗号分隔（`llm` / `template`）           |
| `--precheck / --no-precheck`     | `True`         | 启用数值预检（5组随机输入快速过滤）                   |
| `--sympy / --no-sympy`           | `False`        | 启用 SymPy 符号证明                                   |
| `--auto-fetch / --no-auto-fetch` | `False`        | 自动从网络获取算子文档                                |
| `--save / --no-save`             | `True`         | 将结果保存至知识库                                    |
| `--version`                      | `1`            | 知识库版本号                                          |

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

### `deepmt mr batch-generate`

批量为算子目录中的算子生成蜕变关系并保存至知识库。算子来源：`mr_templates.yaml` 的 `operator_mr_mapping`，按 `framework` 前缀和 `category` 过滤。

```
deepmt mr batch-generate [OPTIONS]
```

| 选项                         | 默认值     | 说明                                                                   |
| ---------------------------- | ---------- | ---------------------------------------------------------------------- |
| `--framework`                | `pytorch`  | 目标框架                                                               |
| `--category`                 | 无（全部） | 按模板分类过滤（如 `linearity`、`symmetry`、`invariance`、`boundary`） |
| `--limit N`                  | 无限制     | 最多处理 N 个算子                                                      |
| `--skip-existing`            | `False`    | 跳过知识库中已有 MR 的算子（支持 Ctrl+C 后断点续跑）                   |
| `--sources`                  | `template` | MR 生成来源，逗号分隔（`llm` / `template`）                            |
| `--precheck / --no-precheck` | `True`     | 启用数值预检                                                           |
| `--sympy / --no-sympy`       | `False`    | 启用 SymPy 符号证明                                                    |
| `--version`                  | `1`        | 知识库版本号                                                           |
| `--dry-run`                  | `False`    | 仅列出待处理算子，不执行生成                                           |

**示例：**

```bash
# 查看会处理哪些算子（不执行生成）
deepmt mr batch-generate --dry-run

# 批量生成 pytorch 所有有模板的算子
deepmt mr batch-generate --framework pytorch

# 只处理 linearity 分类算子
deepmt mr batch-generate --category linearity

# 跳过已有 MR 的算子（断点续跑）
deepmt mr batch-generate --skip-existing

# 小批量测试前 3 个算子
deepmt mr batch-generate --limit 3 --dry-run
```

每行输出格式：`[idx/total] operator  OK/SKIP/NO_MR/FAIL  生成=N 验证=N 保存=N`  
最终汇总行：完成/跳过/失败/总数

---

### `deepmt mr verify <operator>`

对知识库中已有的 MR 重新执行验证。

```
deepmt mr verify <OPERATOR> [OPTIONS]
```

| 选项                         | 默认值  | 说明                   |
| ---------------------------- | ------- | ---------------------- |
| `--version`                  | 最新    | 版本号                 |
| `--precheck / --no-precheck` | `True`  | 启用数值预检           |
| `--sympy / --no-sympy`       | `False` | 启用 SymPy 符号证明    |
| `--save / --no-save`         | `False` | 将验证结果更新到知识库 |

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

| 选项              | 默认值  | 说明              |
| ----------------- | ------- | ----------------- |
| `--version`       | 最新    | 版本号            |
| `--verified-only` | `False` | 仅显示已验证的 MR |
| `--json`          | `False` | 以 JSON 格式输出  |

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

| 选项     | 默认值  | 说明             |
| -------- | ------- | ---------------- |
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
> 临时替代方案：直接删除 `data/mr_repository/operator/<operator>.yaml` 文件。

---

## `deepmt test` — 测试执行

> **框架支持**：目前仅 `pytorch` 插件已实现。指定 `tensorflow` / `paddlepaddle` 会给出友好提示。

### `deepmt test operator <operator>`

对算子运行蜕变测试。若知识库中已有 MR 则直接加载；否则自动生成（需要 LLM 配置）。输入需手动通过 `--inputs` 指定。

```
deepmt test operator <OPERATOR> [OPTIONS]
```

| 选项                         | 默认值    | 说明                 |
| ---------------------------- | --------- | -------------------- |
| `--framework`                | `pytorch` | 目标框架             |
| `--inputs`                   | `1.0,2.0` | 逗号分隔的输入值     |
| `--generate / --no-generate` | `True`    | 若无 MR 则自动生成   |
| `--json`                     | `False`   | 以 JSON 格式输出结果 |

**示例：**

```bash
deepmt test operator relu
deepmt test operator relu --inputs 1.0,-1.0,0.0
deepmt test operator relu --framework pytorch --json
```

---

### `deepmt test batch`

从 MR 知识库批量执行蜕变测试。自动使用 `RandomGenerator` 生成随机输入，无需手动指定。这是推荐的主要测试入口。

```
deepmt test batch [OPTIONS]
```

| 选项              | 默认值     | 说明                                              |
| ----------------- | ---------- | ------------------------------------------------- |
| `--framework`     | `pytorch`  | 目标框架                                          |
| `--operator`      | （全部）   | 指定单个算子名称（如 `torch.nn.functional.relu`） |
| `--category`      | （不过滤） | 按算子目录分类过滤（如 `activation`）             |
| `--mr-id`         | （全部）   | 指定单条 MR ID                                    |
| `--n-samples`     | `10`       | 每条 MR 的随机测试样本数                          |
| `--verified-only` | `False`    | 仅使用已验证（`verified=True`）的 MR              |
| `--json`          | `False`    | 以 JSON 格式输出结果                              |

**示例：**

```bash
deepmt test batch                                       # 测试知识库中所有算子
deepmt test batch --operator torch.nn.functional.relu   # 测试单个算子
deepmt test batch --n-samples 20 --verified-only        # 仅用已验证 MR，多样本
deepmt test batch --category activation --json          # 激活函数类算子，JSON 输出
```

---

### `deepmt test from-config <config_path>`

从 YAML 配置文件批量运行测试。

```
deepmt test from-config <CONFIG_PATH> [OPTIONS]
```

| 选项     | 默认值  | 说明                 |
| -------- | ------- | -------------------- |
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

| 选项      | 默认值  | 说明             |
| --------- | ------- | ---------------- |
| `--limit` | `20`    | 最多显示条数     |
| `--json`  | `False` | 以 JSON 格式输出 |

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

| 选项      | 默认值  | 说明             |
| --------- | ------- | ---------------- |
| `--limit` | `50`    | 最多显示条数     |
| `--json`  | `False` | 以 JSON 格式输出 |

**示例：**

```bash
deepmt test failures
deepmt test failures --limit 100
deepmt test failures --json
```

---

### `deepmt test cross <operator>`

跨框架一致性测试，对比 PyTorch 与 NumPy 在同一批 MR 上的行为差异。

```
deepmt test cross <OPERATOR> [OPTIONS]
```

| 选项              | 默认值    | 说明                                 |
| ----------------- | --------- | ------------------------------------ |
| `--framework1`    | `pytorch` | 第一框架                             |
| `--framework2`    | `numpy`   | 第二框架（NumPy 参考后端）           |
| `--n-samples`     | `10`      | 每条 MR 的随机样本数                 |
| `--verified-only` | `False`   | 仅使用已验证 MR                      |
| `--save`          | `False`   | 将结果持久化到 `data/cross_results/` |
| `--json`          | `False`   | 以 JSON 格式输出                     |

**示例：**

```bash
deepmt test cross torch.nn.functional.relu --save
deepmt test cross torch.exp --n-samples 30 --json
deepmt test cross torch.tanh --verified-only --save --json
```

---

### `deepmt test experiment`

收集全链路实验数据，映射到论文 RQ1-RQ4，生成结构化报告。

```
deepmt test experiment [OPTIONS]
```

| 选项     | 默认值  | 说明                                        |
| -------- | ------- | ------------------------------------------- |
| `--rq`   | `all`   | 仅收集指定研究问题（`1`/`2`/`3`/`4`/`all`） |
| `--json` | `False` | 以 JSON 格式输出（可重定向到文件）          |

**RQ 映射：**

| RQ  | 数据来源                           | 关键指标                                    |
| --- | ---------------------------------- | ------------------------------------------- |
| RQ1 | MRRepository                       | MR 总数、验证率、分类分布、每算子平均 MR 数 |
| RQ2 | ResultsManager + EvidenceCollector | 通过率、失败分布、证据包数量                |
| RQ3 | CrossFrameworkTester 持久化结果    | 一致率、输出差值、不一致 MR 数              |
| RQ4 | RQ1/RQ2 派生                       | 覆盖算子数、用例密度、自动化范围            |

**示例：**

```bash
deepmt test experiment                             # 全部 RQ 文本报告
deepmt test experiment --rq 2                      # 仅 RQ2（缺陷检测）
deepmt test experiment --json > experiment_data.json  # 导出 JSON 用于论文
```

---

## `deepmt repo` — MR 知识库管理

### `deepmt repo list`

列出知识库中所有有 MR 的算子，含版本、数量摘要。

```
deepmt repo list [OPTIONS]
```

| 选项          | 默认值  | 说明                                                 |
| ------------- | ------- | ---------------------------------------------------- |
| `--framework` | —       | 按框架过滤（如 `pytorch`），仅显示含该框架 MR 的算子 |
| `--json`      | `False` | 以 JSON 格式输出                                     |

**示例：**

```bash
deepmt repo list
deepmt repo list --framework pytorch
deepmt repo list --json
```

---

### `deepmt repo stats`

显示知识库整体统计（总数、已验证、Precheck/SymPy 通过数），以及按算子分布表格。

```
deepmt repo stats [OPTIONS]
```

| 选项     | 默认值  | 说明             |
| -------- | ------- | ---------------- |
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

| 选项          | 默认值   | 说明                          |
| ------------- | -------- | ----------------------------- |
| `--version`   | 所有版本 | 仅查看指定版本                |
| `--framework` | —        | 按框架过滤 MR（如 `pytorch`） |
| `--json`      | `False`  | 以 JSON 格式输出              |

**示例：**

```bash
deepmt repo info relu
deepmt repo info relu --version 1
deepmt repo info relu --framework pytorch
deepmt repo info relu --json
```

---

### `deepmt repo delete <operator>`

删除知识库中的 MR 记录。

```
deepmt repo delete <OPERATOR> [OPTIONS]
```

| 选项           | 默认值  | 说明                          |
| -------------- | ------- | ----------------------------- |
| `--id`         | —       | 只删除该 MR ID（优先级最高）  |
| `--version`    | —       | 只删除该版本的全部 MR         |
| `--all`        | `False` | 删除算子的全部 MR（所有版本） |
| `--yes` / `-y` | `False` | 跳过交互确认提示              |

**示例：**

```bash
deepmt repo delete relu --id <MR_ID>       # 删除单条 MR
deepmt repo delete relu --version 1        # 删除版本 1 的全部 MR
deepmt repo delete relu --all              # 删除算子所有 MR（交互确认）
deepmt repo delete relu --all --yes        # 跳过确认直接删除
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

## `deepmt ui` — Web 仪表盘

启动基于 FastAPI + Bootstrap 5 的只读 Web 仪表盘，可在浏览器中可视化查看实验数据。

**安装依赖（首次）：**

```bash
pip install -e ".[ui]"
```

### `deepmt ui start`

启动仪表盘服务器。

```bash
deepmt ui start [OPTIONS]
```

| 选项         | 默认值      | 说明                               |
| ------------ | ----------- | ---------------------------------- |
| `--port, -p` | `8080`      | 监听端口                           |
| `--host`     | `127.0.0.1` | 监听地址（`0.0.0.0` 可局域网访问） |

**示例：**

```bash
deepmt ui start                        # 默认 http://127.0.0.1:8080
deepmt ui start --port 9090            # 自定义端口
deepmt ui start --host 0.0.0.0 -p 80  # 局域网访问
```

**页面说明：**

| 路径             | 说明                                           |
| ---------------- | ---------------------------------------------- |
| `/`              | 总览页（RQ1-RQ4 KPI 卡片 + 图表）              |
| `/mr`            | MR 知识库（算子列表、分布图、筛选）            |
| `/mr/<operator>` | 单算子 MR 详情（transform_code / oracle_expr） |
| `/tests`         | 测试结果（通过率图、失败用例、证据包）         |
| `/cross`         | 跨框架一致性（会话列表、一致率图表）           |
| `/api/docs`      | OpenAPI 交互文档                               |
| `/api/**`        | JSON 数据接口（供外部脚本调用）                |

---

## 未实现功能速查

以下功能调用后会给出友好错误提示（退出码 2），**不会崩溃**：

| 命令 / 选项                                          | 状态说明                                                         |
| ---------------------------------------------------- | ---------------------------------------------------------------- |
| `deepmt mr generate <op> --layer model`              | 模型层 MR 生成已完成（Phase I），CLI 入口待接入，请用 Python API |
| `deepmt mr generate <op> --layer application`        | 应用层 MR 生成已完成（Phase J），CLI 入口待接入，请用 Python API |
| `deepmt test operator <op> --framework tensorflow`   | TensorFlow 插件未实现                                            |
| `deepmt test operator <op> --framework paddlepaddle` | PaddlePaddle 插件仅基础跨框架适配（Phase H），完整插件未实现     |

**模型层 Python API 快速入口：**

```python
from deepmt.benchmarks.models import ModelBenchmarkRegistry
from deepmt.mr_generator.model import ModelMRGenerator

gen = ModelMRGenerator()
mrs = gen.generate(ModelBenchmarkRegistry().get("SimpleMLP", with_instance=True))
```

**应用层 Python API 快速入口：**

```python
from deepmt.mr_generator.application import ApplicationMRGenerator

gen = ApplicationMRGenerator(use_llm=False)
mrs = gen.generate_from_scenario("TextSentiment")
```

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

### 7. 启动 Web 仪表盘（可视化查看实验数据）

```bash
deepmt ui start
# 浏览器访问 http://127.0.0.1:8080
```
