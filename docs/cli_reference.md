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

| 命令组       | 说明                               | 实现状态                                         |
| ------------ | ---------------------------------- | ------------------------------------------------ |
| `mr`         | MR 生成与管理（算子层）            | ✅ 已实现                                         |
| `test`       | 测试执行                           | ✅ 已实现（pytorch 完整，tensorflow/paddle 占位） |
| `repo`       | MR 知识库管理                      | ✅ 已实现                                         |
| `catalog`    | 算子目录浏览、跨框架查询、批量导入 | ✅ 已实现                                         |
| `data`       | 数据目录管理（日志清理等）         | ✅ 已实现                                         |
| `health`     | 系统健康检查与进度                 | ✅ 已实现                                         |
| `ui`         | Web 仪表盘服务器                   | ✅ 已实现                                         |
| `experiment` | 论文实验基准与自动化数据生产       | ✅ 已实现（Phase L）                              |
| `case`       | 真实缺陷案例管理                   | ✅ 已实现（Phase M）                              |

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

子命令总览：

| 子命令           | 说明                                          |
| ---------------- | --------------------------------------------- |
| `generate`       | 为算子生成 MR（算子层，支持 LLM + 模板）      |
| `batch-generate` | 批量为算子目录生成 MR 并入库                  |
| `verify`         | 对知识库中某算子的 MR 进行数值预检            |
| `list`           | 列出算子 MR 记录                              |
| `stats`          | 统计知识库中 MR 数量与验证状态                |
| `promote`        | 将已验证 MR 迁移到 MR Library（git 追踪）     |
| `model-generate` | 为模型层基准对象生成 MR（结构分析，无需 LLM） |

---

### `deepmt mr generate <operator>`

为算子生成蜕变关系，并可选保存至知识库。

**当前支持层次**：`operator`（算子层，完整实现）；模型层请使用 `deepmt mr model-generate`；应用层请通过 Python API（`ApplicationMRGenerator`）使用。

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

### `deepmt mr promote <operator>`

将用户知识库中已验证（`verified=True`）的 MR 迁移到项目 MR Library（git 可追踪的 YAML 文件）。

```
deepmt mr promote <OPERATOR> [OPTIONS]
```

| 选项      | 默认值     | 说明                                        |
| --------- | ---------- | ------------------------------------------- |
| `--layer` | `operator` | MR 层次（`operator`/`model`/`application`） |

**示例：**

```bash
deepmt mr promote torch.add
deepmt mr promote torch.nn.functional.relu --layer operator
```

---

### `deepmt mr model-generate [model_name]`

为模型层基准对象生成蜕变关系（基于结构分析，无需 LLM）。支持 MLP、CNN、RNN、Transformer 四类基准模型。

```
deepmt mr model-generate [MODEL_NAME] [OPTIONS]
```

| 选项          | 默认值    | 说明                             |
| ------------- | --------- | -------------------------------- |
| `--framework` | `pytorch` | 目标框架（当前仅支持 `pytorch`） |
| `--max-mrs`   | 无限制    | 每个模型最多生成的 MR 数量       |
| `--all`       | `False`   | 为所有基准模型生成 MR            |
| `--json`      | `False`   | 以 JSON 格式输出                 |

**示例：**

```bash
deepmt mr model-generate SimpleMLP              # 为 SimpleMLP 生成 MR
deepmt mr model-generate --all                  # 为所有基准模型生成 MR
deepmt mr model-generate SimpleCNN --max-mrs 5
deepmt mr model-generate SimpleMLP --json
```

---

## `deepmt test` — 测试执行

> **框架支持**：目前仅 `pytorch` 插件已实现。指定 `tensorflow` / `paddlepaddle` 会给出友好提示。

子命令总览：

| 子命令        | 说明                                                     |
| ------------- | -------------------------------------------------------- |
| `operator`    | 对单个算子运行蜕变测试（需手动指定输入）                 |
| `batch`       | 从知识库批量测试（自动生成随机输入，推荐主入口）         |
| `model`       | 模型层蜕变测试（自动生成 MR，支持基准模型）              |
| `mutate`      | 变异测试：对已知错误实现验证 MR 的检测能力               |
| `open`        | 开放测试：对含预设缺陷的插件运行批量测试（受控真实场景） |
| `cross`       | 跨框架一致性测试（PyTorch vs NumPy）                     |
| `report`      | 从数据库生成测试结果报告                                 |
| `dedup`       | 缺陷线索去重：将失败证据包聚类为独立缺陷模式             |
| `evidence`    | 证据包管理子命令组（`list` / `show` / `script`）         |
| `from-config` | 从 YAML 配置文件批量运行测试                             |
| `history`     | 查看测试历史记录                                         |
| `failures`    | 查看所有失败的测试用例                                   |

---

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

### `deepmt test mutate <operator>`

变异测试：对算子注入已知错误实现，验证 MR 的缺陷检测能力。用于受控评估——若系统无法发现已知缺陷，说明 MR 或执行链路存在问题。

```
deepmt test mutate <OPERATOR> [OPTIONS]
```

| 选项              | 默认值    | 说明                                                                         |
| ----------------- | --------- | ---------------------------------------------------------------------------- |
| `--framework`     | `pytorch` | 目标框架                                                                     |
| `--mutant`        | 全部      | 变异类型：`negate`/`add_const`/`scale`/`identity`/`zero`（不指定则运行全部） |
| `--n-samples`     | `10`      | 每条 MR 的测试样本数                                                         |
| `--verified-only` | `False`   | 仅使用已验证的 MR                                                            |
| `--scale`         | `2.0`     | `scale` 变异的缩放系数                                                       |
| `--const`         | `1.0`     | `add_const` 变异的偏置值                                                     |
| `--json`          | `False`   | 以 JSON 格式输出                                                             |

**变异类型说明：**

| 类型        | 注入行为                         |
| ----------- | -------------------------------- |
| `negate`    | 取反输出：`f(x) = -real_f(x)`    |
| `add_const` | 添加偏置：`f(x) = real_f(x) + C` |
| `scale`     | 错误缩放：`f(x) = k * real_f(x)` |
| `identity`  | 恒等函数：`f(x) = x`             |
| `zero`      | 恒零输出：`f(x) = 0`             |

**示例：**

```bash
deepmt test mutate torch.nn.functional.relu
deepmt test mutate torch.nn.functional.relu --mutant negate
deepmt test mutate torch.exp --mutant add_const --const 100 --json
```

---

### `deepmt test open`

开放测试：对含预设缺陷的插件运行批量蜕变测试（受控真实场景）。使用 `FaultyPyTorchPlugin` 代替正常插件，将指定算子替换为含已知缺陷的版本。

```
deepmt test open [OPTIONS]
```

| 选项                 | 默认值    | 说明                                             |
| -------------------- | --------- | ------------------------------------------------ |
| `--framework`        | `pytorch` | 目标框架                                         |
| `--operator`         | 全部      | 指定单个算子（不指定则测试知识库中所有算子）     |
| `--inject-faults`    | 内置目录  | 缺陷注入规格：`all` 或 `op1:mutant1,op2:mutant2` |
| `--list-catalog`     | `False`   | 列出内置缺陷目录后退出                           |
| `--n-samples`        | `10`      | 每条 MR 的随机测试样本数                         |
| `--verified-only`    | `False`   | 仅使用已验证的 MR                                |
| `--collect-evidence` | `False`   | 失败时保存可复现证据包                           |
| `--json`             | `False`   | 以 JSON 格式输出                                 |

**缺陷来源优先级**（从高到低）：
1. `--inject-faults` 命令行参数
2. `DEEPMT_INJECT_FAULTS` 环境变量
3. 若均未设置，使用完整内置缺陷目录

**示例：**

```bash
deepmt test open --list-catalog
deepmt test open --operator torch.nn.functional.relu --inject-faults all
deepmt test open --inject-faults "torch.exp:scale" --n-samples 20 --collect-evidence
DEEPMT_INJECT_FAULTS=all deepmt test open
```

---

### `deepmt test report`

从数据库读取历史测试记录，汇总通过率、失败分布、逐 MR 明细。

```
deepmt test report [OPTIONS]
```

| 选项              | 默认值  | 说明                                       |
| ----------------- | ------- | ------------------------------------------ |
| `--framework`     | 全部    | 按框架过滤（`pytorch`/`tensorflow`/`all`） |
| `--operator`      | 全部    | 按算子名称过滤                             |
| `--failures-only` | `False` | 仅显示失败案例                             |
| `--limit`         | `0`     | 最多显示算子数（0=不限）                   |
| `--json`          | `False` | 以 JSON 格式输出                           |

**示例：**

```bash
deepmt test report
deepmt test report --framework pytorch
deepmt test report --operator torch.nn.functional.relu
deepmt test report --failures-only
deepmt test report --json
```

---

### `deepmt test dedup`

缺陷线索去重：从 `data/results/evidence/` 读取已保存的证据包，按（算子 × MR × 错误类型）签名聚类，将大量重复失败压缩为可人工复核的缺陷线索集。

**前提**：先运行 `deepmt test batch --collect-evidence` 或 `deepmt test open --collect-evidence` 收集证据包。

```
deepmt test dedup [OPTIONS]
```

| 选项          | 默认值  | 说明                   |
| ------------- | ------- | ---------------------- |
| `--operator`  | 全部    | 按算子名称过滤         |
| `--framework` | 全部    | 按框架过滤             |
| `--limit`     | `0`     | 最多显示条数（0=不限） |
| `--json`      | `False` | 以 JSON 格式输出       |

**示例：**

```bash
deepmt test dedup
deepmt test dedup --operator torch.nn.functional.relu
deepmt test dedup --limit 10 --json
```

---

### `deepmt test evidence` — 证据包管理

证据包子命令组，管理 `deepmt test batch --collect-evidence` 生成的可复现失败证据。

#### `deepmt test evidence list`

列出已保存的证据包。

```
deepmt test evidence list [OPTIONS]
```

| 选项          | 默认值  | 说明                   |
| ------------- | ------- | ---------------------- |
| `--operator`  | 全部    | 按算子名称过滤         |
| `--framework` | 全部    | 按框架过滤             |
| `--limit`     | `20`    | 最多显示条数（0=不限） |
| `--json`      | `False` | 以 JSON 格式输出       |

```bash
deepmt test evidence list
deepmt test evidence list --operator torch.nn.functional.relu
deepmt test evidence list --limit 5 --json
```

#### `deepmt test evidence show <evidence_id>`

显示单个证据包的详细信息（时间、算子、框架、MR、输入形状、实测差值等）。

```bash
deepmt test evidence show abc123def456
deepmt test evidence show abc123def456 --json
```

#### `deepmt test evidence script <evidence_id>`

打印指定证据包的可复现 Python 脚本，可直接保存为 `.py` 文件独立运行。

```bash
deepmt test evidence script abc123def456
deepmt test evidence script abc123def456 > repro.py
```

---

### `deepmt test model [model_name]`

模型层蜕变测试：对基准模型自动生成并执行 MR 测试。基于结构分析自动生成模型层蜕变关系，无需 LLM。

```
deepmt test model [MODEL_NAME] [OPTIONS]
```

| 选项           | 默认值    | 说明                             |
| -------------- | --------- | -------------------------------- |
| `--framework`  | `pytorch` | 目标框架（当前仅支持 `pytorch`） |
| `--n-samples`  | `10`      | 每条 MR 的测试样本数             |
| `--max-mrs`    | 无限制    | 每个模型最多使用的 MR 数量       |
| `--batch-size` | `4`       | 每次推理的 batch 大小            |
| `--all`        | `False`   | 测试所有基准模型                 |
| `--list`       | `False`   | 列出可用的基准模型后退出         |
| `--json`       | `False`   | 以 JSON 格式输出结果             |

**示例：**

```bash
deepmt test model --list                          # 列出可用基准模型
deepmt test model SimpleMLP                       # 测试 SimpleMLP
deepmt test model SimpleCNN --n-samples 20
deepmt test model --all                           # 测试所有基准模型
deepmt test model SimpleMLP --json
```

---

## `deepmt repo` — MR 知识库管理（Phase K 治理体系）

知识库支持三层 MR（`operator` / `model` / `application`），所有子命令均通过 `--layer` 参数指定层次（默认 `operator`）。

质量等级（`quality_level`）从低到高：`candidate` → `checked` → `proven` → `curated`，以及已废弃的 `retired`。

子命令总览：

| 子命令   | 说明                                             |
| -------- | ------------------------------------------------ |
| `list`   | 列出指定层的所有主体及 MR 摘要                   |
| `stats`  | 显示统计（含质量等级分布、来源分布）             |
| `info`   | 显示单个主体的 MR 详情（含生命周期、溯源信息）   |
| `delete` | 删除 MR 记录（彻底删除）                         |
| `retire` | 归档退役 MR（保留历史，lifecycle_state=retired） |
| `filter` | 按质量/层次/框架筛选 MR                          |
| `audit`  | 运行全库审计，输出质量报告与异常告警             |

---

### `deepmt repo list`

列出指定层所有主体（算子/模型/应用）及 MR 数量摘要。

```
deepmt repo list [OPTIONS]
```

| 选项          | 默认值     | 说明                                        |
| ------------- | ---------- | ------------------------------------------- |
| `--layer`     | `operator` | MR 层次（`operator`/`model`/`application`） |
| `--framework` | —          | 按框架过滤（如 `pytorch`）                  |
| `--json`      | `False`    | 以 JSON 格式输出                            |

**示例：**

```bash
deepmt repo list
deepmt repo list --layer model
deepmt repo list --framework pytorch
deepmt repo list --json
```

---

### `deepmt repo stats`

显示知识库整体统计（总数、验证率、质量等级分布、来源分布）。

```
deepmt repo stats [OPTIONS]
```

| 选项           | 默认值     | 说明                                        |
| -------------- | ---------- | ------------------------------------------- |
| `--layer`      | `operator` | MR 层次（`operator`/`model`/`application`） |
| `--all-layers` | `False`    | 汇总全部三层统计                            |
| `--json`       | `False`    | 以 JSON 格式输出                            |

**示例：**

```bash
deepmt repo stats
deepmt repo stats --layer model
deepmt repo stats --all-layers
deepmt repo stats --json
```

---

### `deepmt repo info <subject>`

显示主体的详细信息：MR 摘要（类别、oracle 表达式、质量等级、溯源信息）。

```
deepmt repo info <SUBJECT> [OPTIONS]
```

| 选项          | 默认值     | 说明                                        |
| ------------- | ---------- | ------------------------------------------- |
| `--layer`     | `operator` | MR 层次（`operator`/`model`/`application`） |
| `--framework` | —          | 按框架过滤 MR（如 `pytorch`）               |
| `--json`      | `False`    | 以 JSON 格式输出                            |

**示例：**

```bash
deepmt repo info relu
deepmt repo info relu --framework pytorch
deepmt repo info ResNet50 --layer model --json
deepmt repo info ImageClassification --layer application
```

---

### `deepmt repo delete <subject>`

彻底删除知识库中的 MR 记录（不可恢复）。若只是临时停用请用 `retire`。

```
deepmt repo delete <SUBJECT> [OPTIONS]
```

| 选项           | 默认值     | 说明                                        |
| -------------- | ---------- | ------------------------------------------- |
| `--layer`      | `operator` | MR 层次（`operator`/`model`/`application`） |
| `--id`         | —          | 只删除该 MR ID                              |
| `--all`        | `False`    | 删除主体的全部 MR                           |
| `--yes` / `-y` | `False`    | 跳过交互确认提示                            |

**示例：**

```bash
deepmt repo delete relu --id <MR_ID>
deepmt repo delete relu --all
deepmt repo delete relu --all --yes
```

---

### `deepmt repo retire <subject> --id <MR_ID>`

归档退役指定 MR（`lifecycle_state` 设为 `retired`），保留历史记录但不参与测试。

```
deepmt repo retire <SUBJECT> [OPTIONS]
```

| 选项           | 默认值     | 说明                                        |
| -------------- | ---------- | ------------------------------------------- |
| `--id`         | **必填**   | 要退役的 MR ID                              |
| `--layer`      | `operator` | MR 层次（`operator`/`model`/`application`） |
| `--yes` / `-y` | `False`    | 跳过交互确认提示                            |

**示例：**

```bash
deepmt repo retire relu --id <MR_ID>
deepmt repo retire relu --id <MR_ID> --yes
deepmt repo retire ResNet50 --layer model --id <MR_ID>
```

---

### `deepmt repo filter`

按质量等级、层次、框架筛选 MR，输出满足条件的关系列表。

```
deepmt repo filter [OPTIONS]
```

| 选项                | 默认值    | 说明                                                     |
| ------------------- | --------- | -------------------------------------------------------- |
| `--layer`           | —（全部） | 按层次过滤（`operator`/`model`/`application`）           |
| `--min-quality`     | `checked` | 最低质量等级（`candidate`/`checked`/`proven`/`curated`） |
| `--framework`       | —         | 按框架过滤                                               |
| `--exclude-retired` | `True`    | 排除已退役 MR（默认排除）                                |
| `--json`            | `False`   | 以 JSON 格式输出                                         |

**示例：**

```bash
deepmt repo filter --min-quality proven
deepmt repo filter --layer operator --min-quality checked --framework pytorch
deepmt repo filter --min-quality curated --json
deepmt repo filter --layer model --min-quality candidate
```

---

### `deepmt repo audit`

运行全库审计，输出质量统计、来源分布、异常告警和重复 MR 检测报告。

```
deepmt repo audit [OPTIONS]
```

| 选项               | 默认值    | 说明                                         |
| ------------------ | --------- | -------------------------------------------- |
| `--layer`          | —（全部） | 只审计指定层                                 |
| `--pending-review` | `False`   | 输出待复核清单（质量低于 proven 的 MR 汇总） |
| `--json`           | `False`   | 以 JSON 格式输出                             |

**审计报告包含：**
- 全库 MR 总数（按层、按质量等级分布）
- 来源分布（llm / template / manual）
- 异常告警（退役比例过高、空 oracle、缺少溯源信息）
- 重复 MR 组（建议退役的 ID 列表）

**示例：**

```bash
deepmt repo audit                      # 全库审计，文本报告
deepmt repo audit --layer operator     # 仅审计算子层
deepmt repo audit --json               # JSON 输出（可保存备用）
deepmt repo audit --pending-review     # 输出待复核 MR 清单
```

---

## `deepmt health` — 系统健康检查

### `deepmt health check`

逐一尝试导入所有核心模块，输出通过/警告/错误统计，快速诊断环境问题。报告头部附带框架运行时版本矩阵（pytorch / numpy / paddlepaddle / tensorflow）。

```bash
deepmt health check
deepmt health check --deep   # 追加插件契约 + 算子可达性 + 目录对账
```

**输出示例：**

```
总体状态: ✅ healthy
框架运行时版本:
  ✅ pytorch         2.11.0
  ✅ numpy           2.4.4
  ✅ paddlepaddle    3.3.1
  ⛔ tensorflow      uninstalled
```

`--deep` 模式额外执行（Phase O）：
- **插件契约**：反射校验所有登记插件实现 `FrameworkPlugin` 必需原语
- **算子↔插件可达性**：遍历知识库算子在 pytorch/numpy/paddle 的 `_resolve_operator` 是否成功
- **目录对账**：知识库中有 MR 但 `OperatorCatalog` 未收录的算子给 WARN

---

### `deepmt health matrix`

输出"算子 × 框架"可达性矩阵（基于知识库全部算子与 `PLUGIN_REGISTRY` 中可用的插件）。

```bash
deepmt health matrix
deepmt health matrix --json
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

| 命令 / 选项                                          | 状态说明                                                           |
| ---------------------------------------------------- | ------------------------------------------------------------------ |
| `deepmt mr generate <op> --layer model`              | 使用 `deepmt mr model-generate` 代替                               |
| `deepmt mr generate <op> --layer application`        | 应用层 CLI 入口待接入，请用 Python API（`ApplicationMRGenerator`） |
| `deepmt test operator <op> --framework tensorflow`   | TensorFlow 插件未实现                                              |
| `deepmt test operator <op> --framework paddlepaddle` | PaddlePaddle 插件仅基础跨框架适配（Phase H），完整插件未实现       |

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

---

## `deepmt experiment` — 论文实验基准与数据生产（Phase L）

子命令总览：

| 子命令      | 说明                                             |
| ----------- | ------------------------------------------------ |
| `run`       | 创建实验运行清单（记录 seed / benchmark / 环境） |
| `collect`   | 收集 RQ1-RQ4 当前统计数据并打印                  |
| `export`    | 将统计数据导出为 JSON / CSV / Markdown           |
| `list`      | 列出历史实验运行清单                             |
| `show`      | 查看单条运行清单详情                             |
| `benchmark` | 查看固化的论文实验 benchmark 清单                |
| `case`      | Case Study 管理（list / show / export）          |
| `env`       | 查看运行环境与框架版本矩阵                       |

### 快速示例

```bash
# 创建运行清单（记录当前环境与 benchmark）
deepmt experiment run

# 收集 RQ1-RQ4 统计数据
deepmt experiment collect

# 导出论文图表与表格（JSON + CSV + Markdown）
deepmt experiment export --format all

# 同时导出 ASCII 图表
deepmt experiment export --figures --ascii-only

# 查看 benchmark 清单
deepmt experiment benchmark
deepmt experiment benchmark --layer operator

# 查看环境信息
deepmt experiment env

# Case Study 管理
deepmt experiment case list
deepmt experiment case export
```

### `deepmt experiment run`

创建实验运行清单，记录当前 benchmark 对象、随机种子、环境快照，用于实验可复现性追踪。

```
deepmt experiment run [OPTIONS]
```

| 选项       | 默认值  | 说明                               |
| ---------- | ------- | ---------------------------------- |
| `--rq`     | 全部    | 目标 RQ（可多次指定，如 --rq rq1） |
| `--seed`   | `42`    | 随机种子                           |
| `--notes`  | 空      | 附加备注                           |
| `--no-env` | `False` | 跳过环境快照（加速）               |
| `--json`   | `False` | 以 JSON 格式输出                   |

**示例：**

```bash
deepmt experiment run
deepmt experiment run --seed 123 --notes "论文第二次实验"
deepmt experiment run --rq rq1 --rq rq2 --no-env
```

---

### `deepmt experiment collect`

收集 RQ1-RQ4 当前统计数据并打印汇总。

```
deepmt experiment collect [OPTIONS]
```

| 选项       | 默认值  | 说明                               |
| ---------- | ------- | ---------------------------------- |
| `--rq`     | 全部    | 目标 RQ（可多次指定，如 --rq rq1） |
| `--run-id` | 无      | 关联的 RunManifest ID              |
| `--json`   | `False` | 以 JSON 格式输出                   |

**示例：**

```bash
deepmt experiment collect
deepmt experiment collect --rq rq1 --rq rq2
deepmt experiment collect --json
```

---

### `deepmt experiment export`

将统计数据导出为论文可用文件（JSON / CSV / Markdown）。

```
deepmt experiment export [OPTIONS]
```

| 选项           | 默认值                     | 说明                                  |
| -------------- | -------------------------- | ------------------------------------- |
| `--format`     | `all`                      | 格式：json / csv / markdown / all     |
| `--output`     | `data/experiments/exports` | 导出目录                              |
| `--rq`         | 全部                       | 目标 RQ                               |
| `--run-id`     | 无                         | 关联的 RunManifest ID                 |
| `--figures`    | `False`                    | 同时导出图表                          |
| `--ascii-only` | `False`                    | 图表只生成 ASCII（不依赖 matplotlib） |

**示例：**

```bash
deepmt experiment export
deepmt experiment export --format markdown
deepmt experiment export --format all --output data/my_export
deepmt experiment export --figures --ascii-only
```

---

### `deepmt experiment list`

列出历史实验运行清单（由 `deepmt experiment run` 创建）。

```
deepmt experiment list [OPTIONS]
```

| 选项     | 默认值  | 说明             |
| -------- | ------- | ---------------- |
| `--json` | `False` | 以 JSON 格式输出 |

**示例：**

```bash
deepmt experiment list
deepmt experiment list --json
```

---

### `deepmt experiment show <run_id>`

查看单条运行清单详情（seed、RQs、创建时间、环境快照）。

```
deepmt experiment show <RUN_ID> [OPTIONS]
```

| 选项     | 默认值  | 说明             |
| -------- | ------- | ---------------- |
| `--json` | `False` | 以 JSON 格式输出 |

**示例：**

```bash
deepmt experiment show abc123def456
deepmt experiment show abc123def456 --json
```

---

### `deepmt experiment benchmark`

查看固化的论文实验 Benchmark Suite 清单（算子层 / 模型层 / 应用层）。

```
deepmt experiment benchmark [OPTIONS]
```

| 选项      | 默认值  | 说明                                                     |
| --------- | ------- | -------------------------------------------------------- |
| `--layer` | `all`   | 只显示指定层次（`operator`/`model`/`application`/`all`） |
| `--json`  | `False` | 以 JSON 格式输出                                         |

**示例：**

```bash
deepmt experiment benchmark
deepmt experiment benchmark --layer operator
deepmt experiment benchmark --json
```

---

### `deepmt experiment case` — Case Study 管理

Case Study 子命令组，管理论文的案例研究数据。

#### `deepmt experiment case list`

列出所有 case study。

```
deepmt experiment case list [OPTIONS]
```

| 选项       | 默认值  | 说明                                     |
| ---------- | ------- | ---------------------------------------- |
| `--status` | 全部    | 过滤状态（`draft`/`confirmed`/`closed`） |
| `--json`   | `False` | 以 JSON 格式输出                         |

#### `deepmt experiment case show <case_id>`

查看单个 case study 详情（以 Markdown 格式输出）。

```bash
deepmt experiment case show CS001
deepmt experiment case show CS001 --json
```

#### `deepmt experiment case export`

将所有 case study 导出为 Markdown 目录文件。

```
deepmt experiment case export [OPTIONS]
```

| 选项       | 默认值                    | 说明                                     |
| ---------- | ------------------------- | ---------------------------------------- |
| `--output` | `case_studies/catalog.md` | 输出文件路径                             |
| `--status` | 全部                      | 过滤状态（`draft`/`confirmed`/`closed`） |

```bash
deepmt experiment case export
deepmt experiment case export --output data/cases.md --status confirmed
```

---

### `deepmt experiment env`

查看当前运行环境与框架版本矩阵（pinned vs installed 对比）。

```
deepmt experiment env [OPTIONS]
```

| 选项     | 默认值  | 说明             |
| -------- | ------- | ---------------- |
| `--json` | `False` | 以 JSON 格式输出 |

**示例：**

```bash
deepmt experiment env
deepmt experiment env --json
```

---

## `deepmt case` — 真实缺陷案例管理（Phase M）

管理真实缺陷案例的完整生命周期：自动构建、人工确认、状态追踪和导出。

子命令总览：

| 子命令    | 说明                                         |
|-----------|----------------------------------------------|
| `list`    | 列出所有案例及状态                           |
| `show`    | 查看单个案例 Markdown 详情                   |
| `confirm` | 人工更新案例字段（状态、根因、严重程度等）   |
| `build`   | 从缺陷线索或证据包 ID 构建案例包             |
| `export`  | 导出案例目录（Markdown / JSON）              |

---

### `deepmt case list`

列出所有案例，支持按状态过滤。

```
deepmt case list [OPTIONS]
```

| 选项       | 默认值  | 说明                                          |
|------------|---------|-----------------------------------------------|
| `--status` | 全部    | 按状态过滤（`draft` / `confirmed` / `closed`）|
| `--json`   | `False` | 以 JSON 格式输出                              |

**示例：**

```bash
deepmt case list
deepmt case list --status confirmed
deepmt case list --json
```

**输出示例：**

```
案例库（共 2 个）
──────────────────────────────────────────────────────────────────────
  ✅ [009eb89bcb]  gelu           pytorch     mr_quality              severity=low
       gelu 下界性 MR 定义错误：MR 断言 gelu(x) >= -0.15，但真实最小值约 -0.170
  ✅ [e861263744]  exp            pytorch     numerical_precision     severity=low
       exp float32 溢出边界（x≈88）不对称 Inf 现象
──────────────────────────────────────────────────────────────────────
运行 'deepmt case show <case_id>' 查看详情
```

---

### `deepmt case show <case_id>`

以 Markdown 格式查看单个案例详情（根因、触发输入、复现步骤等）。

```
deepmt case show <CASE_ID> [OPTIONS]
```

| 选项     | 默认值  | 说明             |
|----------|---------|------------------|
| `--json` | `False` | 以 JSON 格式输出 |

**示例：**

```bash
deepmt case show 009eb89bcb
deepmt case show e861263744 --json
```

---

### `deepmt case confirm <case_id>`

人工更新案例字段。可单次更新一个或多个字段。

```
deepmt case confirm <CASE_ID> [OPTIONS]
```

| 选项            | 默认值 | 说明                                                        |
|-----------------|--------|-------------------------------------------------------------|
| `--status`      | 无     | 更新状态（`draft` / `confirmed` / `closed`）                |
| `--severity`    | 无     | 更新严重程度（`critical` / `high` / `medium` / `low` / `unknown`） |
| `--root-cause`  | 无     | 填写根因分析文本                                            |
| `--summary`     | 无     | 更新案例摘要                                                |
| `--defect-type` | 无     | 更新缺陷类型（如 `mr_quality` / `numerical_precision`）     |
| `--notes`       | 无     | 追加备注（附加到现有备注后，不覆盖）                        |

**示例：**

```bash
# 确认案例并设置严重程度
deepmt case confirm 009eb89bcb --status confirmed --severity low

# 填写根因分析
deepmt case confirm e861263744 --root-cause "IEEE 754 浮点溢出边界行为，非框架缺陷"

# 更新摘要和追加备注
deepmt case confirm 009eb89bcb --summary "gelu MR 下界定义偏紧" --notes "已修复 MR"

# 关闭案例
deepmt case confirm 009eb89bcb --status closed --notes "已提交 upstream issue #1234"
```

---

### `deepmt case build`

从缺陷线索或证据包构建案例包（`reproduce.py` + `case_summary.md` + `evidence.json` + `metadata.json`）。

```
deepmt case build [OPTIONS]
```

| 选项               | 默认值                           | 说明                                           |
|--------------------|----------------------------------|------------------------------------------------|
| `--from-evidence`  | 无                               | 从指定证据包 ID 构建单个案例                   |
| `--top N`          | `0`（全部）                      | 从去重缺陷线索中构建前 N 个高优先级案例        |
| `--output DIR`     | `deepmt/cases/real_defects/`     | 案例包输出根目录                               |
| `--json`           | `False`                          | 以 JSON 输出构建结果                           |

**示例：**

```bash
# 从特定证据包手动构建
deepmt case build --from-evidence bee67689-15b

# 批量构建前 3 个高优先级案例
deepmt case build --top 3

# 批量构建全部线索，指定输出目录
deepmt case build --top 5 --output /tmp/my_cases
```

**前提**：需要先运行 `deepmt test batch --collect-evidence` 收集证据包。

---

### `deepmt case export`

将案例目录导出为 Markdown 或 JSON 格式。

```
deepmt case export [OPTIONS]
```

| 选项       | 默认值                                       | 说明                                          |
|------------|----------------------------------------------|-----------------------------------------------|
| `--format` | `markdown`                                   | 输出格式（`markdown` / `json`）               |
| `--status` | 全部                                         | 只导出指定状态的案例                          |
| `--output` | `data/experiments/case_studies/catalog.md`   | 输出文件路径（JSON 格式时打印到 stdout）      |

**示例：**

```bash
# 导出全部案例为 Markdown
deepmt case export

# 只导出已确认案例
deepmt case export --status confirmed

# 导出为 JSON 文件
deepmt case export --format json --output /tmp/cases.json

# 导出并指定输出路径
deepmt case export --output docs/case_catalog.md
```
