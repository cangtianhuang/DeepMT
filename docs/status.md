# DeepMT 开发状态

## 当前阶段

**Phase 4**：算子层 MR 生成引擎（单算子）、CrawlAgent、CLI 框架完整，PyTorch 数据层部分实现。  
**下一目标**：打通"数据→生成→测试→报告"完整流程（以 PyTorch 为主），并做好多框架扩展的接口设计。

---

## 已完成模块

### MR 生成（算子层）

| 模块 | 说明 |
|------|------|
| `mr_generator/operator/operator_mr.py` | 主生成器（4 阶段流水线） |
| `mr_generator/operator/operator_llm_mr_generator.py` | LLM 猜想生成 |
| `mr_generator/operator/mr_prechecker.py` | 随机输入快速筛选（5 组，80% 阈值） |
| `mr_generator/operator/sympy_prover.py` | SymPy 符号证明引擎 |
| `mr_generator/operator/sympy_translator.py` | 代码 → SymPy（LLM + AST 双路径，`use_llm` 可控） |
| `mr_generator/operator/ast_parser.py` | Python AST → SymPy |
| `mr_generator/base/mr_templates.py` + YAML | MR 模板池 |
| `mr_generator/base/mr_repository.py` | MR 知识库（SQLite） |
| `mr_generator/base/knowledge_base.py` + YAML | 三层知识库 |
| `mr_generator/base/operator_catalog.py` | 算子目录管理 |

### 工具层

| 模块 | 说明 |
|------|------|
| `tools/llm/client.py` | LLM 客户端（OpenAI/Anthropic） |
| `tools/web_search/search_agent.py` | HTML 解析 + 版本查询（无需 LLM） |
| `tools/web_search/operator_fetcher.py` | 算子信息获取器 |
| `tools/web_search/sphinx_search.py` | Sphinx 文档索引解析 |
| `tools/agent/agent_core.py` | CrawlAgent（ReAct 模式） |
| `tools/agent/task_runner.py` | TaskRunner（语义 API） |

### 已实现 CLI 命令

| 命令 | 说明 |
|------|------|
| `deepmt mr generate` | 单算子生成 MR（LLM + 验证） |
| `deepmt repo list/stats/info` | MR 知识库查询 |
| `deepmt catalog list/search/info/sync` | 算子目录管理 |
| `deepmt catalog latest-version` | 从 PyPI 获取框架最新/历史版本 |
| `deepmt catalog fetch-doc` | 获取算子文档（PyTorch 支持自动 URL） |
| `deepmt catalog update-api-list` | 从官网拉取 PyTorch API 列表并缓存 |
| `deepmt test operator` | 单算子蜕变测试 |
| `deepmt health` | 健康检查 |

### 测试体系

```
tests/
├── unit/
│   ├── test_core.py       6 个用例（config、framework、IR）
│   ├── test_parsers.py    13 个用例（ASTParser + SympyTranslator）
│   ├── test_prover.py     8 个用例（SymPyProver，无 LLM 依赖）
│   └── test_search.py     搜索工具
└── integration/
    ├── test_mr_generation.py   需 LLM API
    └── test_web_tools.py       需网络
```

全部 57 个单元测试通过（无 LLM 依赖）。

---

## 架构设计原则（本阶段约定）

以下设计决策适用于接下来所有开发任务：

### 1. 框架参数化

所有框架相关代码以 `FrameworkType` 作为参数，**PyTorch 先行实现**，接口设计预留给 TensorFlow 和 PaddlePaddle，不写死框架名称。参考现有模式：`_SUPPORTED_FRAMEWORKS = {"pytorch"}` + `not_implemented_error` 提示。

### 2. MR 与 IR 框架无关

- `MetamorphicRelation` 不绑定特定框架，通过 `applicable_frameworks: Optional[List[str]]` 声明适用范围（None = 通用）
- `transform_code` 仅使用 Python 原生算术运算符（PyTorch 张量也支持），不依赖框架特定 API；如使用框架 API 则在 MR 中明确标注为框架相关
- 框架 tensor 的包装/解包由插件负责，不出现在 MR 逻辑中

### 3. 算子双态：function vs module

同一数学运算在 PyTorch 中存在两种调用形态：

| 形态 | 示例 | 适用场景 |
|------|------|----------|
| `function` | `torch.nn.functional.relu(x)` | 算子层测试（当前） |
| `module` | `torch.nn.ReLU()(x)` | 模型层测试（未来） |

**当前阶段**：算子层测试统一使用 `function` 形态（`api_path` 字段），`module_class` 作为元数据保留供模型层使用。Plugin 的 `invoke(operator_ir, inputs)` 根据 `api_style` 字段分发。

### 4. 输入约束严格化

随机测试输入必须满足算子约束，否则会产生大量无意义异常（如对负数求 log）。每个算子在目录 YAML 中声明 `input_specs`，由统一的 `InputGenerator` 读取并生成合法输入。

---

## 近期开发计划

### Phase A：算子数据层完善

#### A1 — 算子目录 YAML 格式扩展 ⚠️ 其他阶段的基础

扩展 `mr_generator/config/operator_catalog/pytorch.yaml` 格式，**向后兼容**，增加：

```yaml
operators:
  - name: relu
    api_path: torch.nn.functional.relu   # 用于测试的完整路径
    api_style: function                   # function | module | method
    module_class: torch.nn.ReLU          # module 风格的等价类（元数据）
    category: activation
    doc_url: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
    input_specs:
      - name: input
        dtype: [float16, float32, float64]
        shape: any              # any | "nd>=1" | "(n,)" | "(n,m)" 等
        value_range: null       # null=无限制 | [min, max] | [0, null]=正数
        required: true
    aliases: [F.relu]
```

同时更新 `mr_generator/base/operator_catalog.py` 解析新字段，`OperatorIR` dataclass 增加 `api_path`、`api_style`、`input_specs` 字段。

文件：`mr_generator/config/operator_catalog/pytorch.yaml`、`ir/schema.py`、`mr_generator/base/operator_catalog.py`

#### A2 — PyTorch API 列表接口框架化

- `fetch_api_list(url, use_cache)` → `fetch_api_list(framework, use_cache)` 统一入口
- `_FRAMEWORK_API_PAGES` 已有结构，补充 `build_doc_url(framework, operator_name)` 函数集中管理 URL 构造逻辑
- paddle/tensorflow 的 `_FRAMEWORK_API_PAGES` 和解析器**留空占位**，抛出 `NotImplementedError`
- `deepmt catalog update-api-list` 命令的 `--framework` 已限制为 `pytorch`，保持不变

文件：`tools/web_search/search_agent.py`、`deepmt/commands/catalog.py`

#### A3 — fetch-doc 使用 YAML 中的 doc_url

- `deepmt catalog fetch-doc <operator>` 不再依赖 URL 格式猜测，直接从 YAML 读取 `doc_url`
- 保留 `--url` 参数用于覆盖
- 相应更新 `tools/web_search/operator_fetcher.py`

文件：`deepmt/commands/catalog.py`、`tools/web_search/operator_fetcher.py`

---

### Phase B：MR 生成层完善

#### B1 — repo delete 命令（小任务，快速完成）

```
deepmt repo delete <operator> [--id MR_ID] [--version V] [--all] [--yes]
```

- 删除单条 MR（by ID）、指定版本全部 MR、算子全部 MR
- 加 `--yes` 跳过确认提示，否则交互确认
- 补充 `MRRepository.delete(operator, version=None, mr_id=None)` 方法

文件：`deepmt/commands/repo.py`、`mr_generator/base/mr_repository.py`

#### B2 — MetamorphicRelation 增加 applicable_frameworks 字段

- `ir/schema.py`：`applicable_frameworks: Optional[List[str]] = None`
- `MRRepository`：增加按 framework 筛选的查询接口
- `deepmt repo list --framework pytorch` 过滤支持
- MR 生成时自动记录生成所用框架

文件：`ir/schema.py`、`mr_generator/base/mr_repository.py`、`deepmt/commands/repo.py`

#### B3 — 批量 MR 生成命令

```
deepmt mr batch-generate [--framework pytorch] [--category activation] [--limit N]
                         [--skip-existing] [--workers N] [--dry-run]
```

- 从算子目录按 framework/category 筛选算子列表
- 逐个调用 `operator_mr.py` 生成流水线，自动保存到 repo
- 进度条显示（已完成/跳过/失败），支持 Ctrl+C 中断后续跑（`--skip-existing`）
- LLM 调用默认串行（`--workers 1`），可选受控并发

文件：`deepmt/commands/mr.py`（新增 `batch-generate` 子命令）

---

### Phase C：批量测试层

#### C1 — InputGenerator（输入约束生成器）

新建 `core/input_generator.py`，读取 `input_specs` 生成随机合法测试输入：

- **dtype**：按声明的类型列表随机选择并生成对应 tensor
- **shape**：
  - `any`：随机 1~4 维，每维 1~32
  - `nd>=N`：至少 N 维
  - 固定形状字符串（如 `"(3,4)"`）：解析后生成
- **value_range**：`[min, max]` 裁剪（支持 `null` 表示无边界），边界值 `0` 单独处理（+epsilon）
- 生成策略：随机值 + 边界值（0、极值）混合，可配置 `n_samples`
- 接口：`InputGenerator.generate(input_specs: List[dict], n_samples: int, framework: str) -> List[Dict[str, Any]]`

文件：`core/input_generator.py`

#### C2 — 插件 invoke 接口规范化

当前 `PyTorchPlugin.ir_to_code()` 返回闭包，不便于批量测试。规范化为：

```python
class BasePlugin:
    def invoke(self, operator_ir: OperatorIR, inputs: Dict[str, Any]) -> Any:
        """执行算子，返回输出。inputs 为 {参数名: 值} 字典。"""
        raise NotImplementedError
```

PyTorch 插件实现：
- `api_style=function`：动态 import `api_path`，以 `inputs` dict 调用
- `api_style=module`：实例化 `module_class`，以 `inputs['input']` 调用
- 保持 `ir_to_code()` 向后兼容（委托给 `invoke`）

文件：`plugins/base_plugin.py`（接口）、`plugins/pytorch_plugin.py`（实现）

#### C3 — 批量测试引擎

扩展 `core/test_runner.py`，支持批量模式：

**执行流程（每条 MR）：**
1. 从 repo 按 `framework` 筛选 MR 列表
2. 从算子目录读取 `input_specs`
3. `InputGenerator.generate(input_specs, n_samples)` → N 组输入
4. `plugin.invoke(operator_ir, inputs)` → `orig`
5. `mr.transform(inputs)` → `transformed_inputs`
6. `plugin.invoke(operator_ir, transformed_inputs)` → `trans`
7. `oracle_evaluator.evaluate(oracle_expr, orig, trans, x)` → pass/fail
8. 聚合：通过率、失败样本、异常统计

**CLI 扩展：**
```
deepmt test batch [--framework pytorch] [--operator relu] [--category activation]
                  [--n-samples 100] [--mr-id ID] [--json]
```

文件：`core/test_runner.py`、`deepmt/commands/test.py`

---

### Phase D：报告层

#### D1 — 报告生成器

新建 `analysis/report_generator.py`：

- 读取 `ResultsManager` 中的测试结果
- 生成纯文本摘要：各算子/MR 通过率、失败案例列表、缺陷分类分布
- 支持 JSON 导出（便于后续可视化或外部处理）

#### D2 — CLI 报告命令

```
deepmt test report [--framework pytorch] [--operator relu] [--format text|json]
                   [--output FILE]
```

文件：`analysis/report_generator.py`、`deepmt/commands/test.py`

---

### 推荐开发顺序

```
A1（格式设计，数据基础）
  → B1（repo delete，独立小任务）
  → A2/A3（数据层补全）
  → B2（MR 框架字段）
  → B3（批量生成）
  → C1（InputGenerator）
  → C2（插件接口规范化）
  → C3（批量测试）
  → D1/D2（报告）
```

A1 是最重要的前置任务：`input_specs` 格式一旦确定，C1、C2、C3 均依赖它。

---

## 长期方向（不详细规划）

1. TensorFlow / PaddlePaddle 插件实现（重复上述流程）
2. 模型层 MR 生成与测试（网络拓扑 + 数据增强策略）
3. 跨框架等价性测试（需算子跨框架映射表）
4. 可视化演示
5. 性能基准对比（与其他测试方法比较）
6. CI/CD 流水线与持续监控
7. 学术论文

---

## 已知限制

1. **SymPy 验证限制**：含浮点的复杂性质无法符号证明，仅依赖 pre-check 数值验证
2. **LLM 依赖**：MR 猜想质量依赖提示工程与 API 密钥，单元测试已通过 `use_llm=False` 隔离
3. **transform_code 可移植性**：跨框架 MR 要求 `transform_code` 不使用框架特定 API，PyTorch 阶段不强制，文档中标注

---

*最后更新：2026-04-04*
