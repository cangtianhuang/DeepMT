# 17_Phase_P_仪表盘三层重设计

> 状态：✅ 完成  完成：2026-04-17

## 1. 阶段目标

以**论文答辩演示**和**项目落地展示**为核心场景，对现有 Web 仪表盘进行系统性重设计，使其完整覆盖项目的三层蜕变测试体系（算子层 / 模型层 / 应用层）、多框架注册信息、真实缺陷案例以及 MR 质量视图。

阶段完成后，仪表盘应满足：

- 一屏展示三层 MR 全景，让评审者即刻理解系统规模；
- 框架信息页直观展示各框架注册算子覆盖、测试支持状态；
- 模型层与应用层有独立展示入口，不再埋藏于算子专属界面；
- 真实缺陷案例（Phase M 产出）在面板中可浏览与追溯；
- MR 质量治理（Phase K 产出）有可视化质量视图；
- 全部数据通过已有 Python API 驱动，**不引入新依赖、不破坏现有路由**。

---

## 2. 当前仪表盘不足分析

### 2.1 页面结构层面

| 现状 | 问题 |
|------|------|
| 导航只有 4 页：总览 / MR 知识库 / 测试结果 / 跨框架 | 缺少模型层、应用层、框架信息、MR质量、真实案例等独立入口 |
| 总览 KPI 对应 RQ1-RQ4，以算子层为主 | 没有三层合并统计；模型层/应用层 MR 和测试数据不可见 |
| MR 知识库只展示算子层 | `api.py:102` `MRRepository(layer="operator")` 写死；模型层/应用层MR仓库数据未接入 |
| 测试结果页只展示算子层通过率 | 没有模型层测试结果（`model_test_runner`）、应用层验证结果 |
| 跨框架一致性展示会话列表 | 不支持矩阵视图（pytorch↔numpy / pytorch↔paddle / paddle↔numpy，Phase M T7已实现） |

### 2.2 数据接入层面

| 现状 | 问题 |
|------|------|
| `dependencies.py` 只注册了 4 个数据源 | 缺少 `ModelBenchmarkRegistry`、`ApplicationBenchmarkRegistry`、`CaseStudyIndex`、`PluginManager`（框架注册信息）的依赖 |
| `/api/mr-quality` 端点已实现（Phase K） | 无对应页面，端点形同虚设 |
| `ExperimentOrganizer.collect_rq1()` 只读 operator 层 | 总览页无法展示三层 MR 全量数据 |
| `CaseStudyIndex`（Phase M）已可列出真实缺陷案例 | 无 API 端点，无页面 |
| `MRRepository` 支持 layer 参数（operator/model/application） | API 中未暴露 layer 切换，前端无法按层筛选 |

### 2.3 框架信息层面

| 现状 | 问题 |
|------|------|
| `mr_repo.html` 框架下拉写死为 PyTorch / NumPy | 未包含 PaddlePaddle（Phase H 已完成插件），框架列表不动态 |
| 没有框架注册页 | 评审者无法了解各框架的算子支持数量、测试覆盖情况、版本信息 |
| 插件能力（`PluginsManager`）无可视化入口 | 框架扩展能力对外不透明 |

### 2.4 论文演示层面

| 现状 | 问题 |
|------|------|
| 总览页没有"方法三层架构图"或层次说明 | 评审者需要先看论文才能理解系统架构，答辩演示效果差 |
| 真实缺陷案例（Phase M `data/cases/real_defects/`）无法在面板查看 | 论文中的案例分析章节无法在演示时直观呈现 |
| MR 质量分级（curated/proven/checked/candidate/retired）无可视化 | 论文 RQ1 质量保障结论缺乏直接佐证展示 |

---

## 3. 改进方案总览

### 3.1 页面规划

新增 **3 个页面**，改造 **4 个已有页面**，导航从 4 项扩展为 7 项：

| 路由 | 标题 | 类型 | 主要内容 |
|------|------|------|---------|
| `GET /` | 系统总览 | 改造 | 三层 MR 统计 KPI + 三层分布 + 系统架构简图 |
| `GET /mr` | MR 知识库 | 改造 | 三层 Tab 切换（算子/模型/应用）+ 动态框架筛选 |
| `GET /mr/{layer}/{subject}` | MR 详情 | 改造 | 统一支持三层的详情卡片 |
| `GET /tests` | 测试结果 | 改造 | 三层 Tab + 模型层/应用层验证结果 |
| `GET /cross` | 跨框架对比 | 改造 | 矩阵热力图 + 会话列表 |
| `GET /frameworks` | 框架信息 | **新增** | 注册框架列表 + 每框架能力卡片 |
| `GET /quality` | MR 质量视图 | **新增** | 三层质量分布 + 异常告警 + 质量等级筛选 |
| `GET /cases` | 真实缺陷案例 | **新增** | Phase M 案例列表 + 详情模态框 |

### 3.2 导航结构

```
[总览]  [MR 知识库]  [测试结果]  [跨框架]  [框架信息]  [MR 质量]  [缺陷案例]
```

---

## 4. 各页面详细设计

### P1：总览页改造（`GET /`）

**现状**：4 个 KPI 卡片（MR 总量、缺陷检测、跨框架一致性、覆盖算子数）+ 算子测试堆叠柱图

**改造目标**：

- **Hero 区**：在副标题下方加入"三层架构标签行"（`算子层 · 模型层 · 应用层`），直观传达系统层次
- **KPI 区**：从 4 个扩展为 **6 个 KPI 卡片**：
  1. 算子层 MR 总量（原 RQ1）
  2. 模型层 MR 总量（新）
  3. 应用层 MR 总量（新）
  4. 缺陷检测通过率（原 RQ2）
  5. 跨框架一致率（原 RQ3）
  6. 已注册框架数（新，取代原"覆盖算子数"作为额外卡片）
- **图表区**：
  - 左：三层 MR 分布环形图（替换原算子 MR 分类分布）
  - 中：各层 MR 来源分布（llm/template/manual）横向柱图
  - 右：测试概况（三层通过率并排条）
- **底部表格**：保留算子层最近测试摘要（可扩展为 tab 切换三层）

**数据来源**：新增 `/api/summary-v2` 端点，覆盖三层 MR 仓库统计。

---

### P2：MR 知识库改造（`GET /mr`）

**现状**：只展示算子层 MR，框架下拉写死为 PyTorch/NumPy

**改造目标**：

- 顶部新增 **三层 Tab**（算子层 / 模型层 / 应用层），默认选中算子层
- 切换 Tab 时，筛选栏的**框架选项动态切换**（调用 `/api/frameworks` 动态填充）
- **算子层**：保持现有功能（筛选 + 统计卡片 + 来源图 + 算子表格），修复框架下拉动态化
- **模型层**：展示已注册模型 benchmark（SimpleMLP/SimpleCNN/SimpleRNN/TinyTransformer），每行显示模型类型/任务类型/MR 数量，点击跳转模型 MR 详情
- **应用层**：展示已注册应用场景（ImageClassification/TextSentiment），每行显示任务类型/领域/MR 数量

**URL 规则**：
- `/mr?layer=operator`（默认）
- `/mr?layer=model`
- `/mr?layer=application`
- `/mr/operator/{operator_name}`（保持现有URL兼容）
- `/mr/model/{model_name}`（新增）
- `/mr/application/{app_name}`（新增）

**数据来源**：扩展 `/api/mr-repository` 增加 `?layer=` 参数，使其复用 `MRRepository(layer=...)` 的三层能力。

---

### P3：测试结果改造（`GET /tests`）

**现状**：只展示算子层通过率和失败用例

**改造目标**：

- 顶部新增 **三层 Tab**（算子层 / 模型层 / 应用层）
- **算子层 Tab**：保持现有功能（KPI 卡 + 通过率横向柱图 + 失败用例表 + 证据包列表）
- **模型层 Tab**：展示各基准模型的测试结果（模型名/MR数量/通过数/失败数/通过率进度条）
- **应用层 Tab**：展示各应用场景的验证结果（语义一致性通过率/置信度可接受率/label_consistent 比例）

**数据来源**：
- 新增 `/api/test-results/model` — 读取 `data/results/` 下模型层测试结果
- 新增 `/api/test-results/application` — 读取应用层验证结果

---

### P4：跨框架对比改造（`GET /cross`）

**现状**：展示会话列表，无矩阵视图

**改造目标**：

- 顶部新增 **矩阵热力图卡片**：3×3 框架对矩阵，单元格颜色编码一致率（绿/黄/红），展示 (pytorch,numpy) / (pytorch,paddle) / (paddle,numpy) 三对结果
- 下方保留现有会话列表表格（点击行展开详情）
- 新增 **框架对筛选器**（下拉选择 fw1 + fw2），过滤会话列表

**数据来源**：在 `/api/cross-framework` 增加矩阵聚合逻辑，返回 `matrix_summary` 字段。

---

### P5：框架信息页（`GET /frameworks`）— 新增

**设计目标**：对外透明展示框架插件能力，回答"系统支持哪些框架、各框架覆盖了什么"

**页面内容**：

- 顶部 KPI：已注册框架数 / 总支持算子数 / 已完整测试框架数
- 框架卡片列表（每框架一张卡片）：
  - 框架名称 + 版本（如 PyTorch 2.x）
  - 插件支持算子数量（`PluginManager` 统计）
  - MR 适用数量（来自 `MRRepository` 的 `applicable_frameworks` 字段统计）
  - 实现状态标签（`全量支持` / `部分实现` / `占位入口`）
  - 当前注册的所有算子名称标签云（可收起）
- 各框架算子覆盖数对比柱图

**数据来源**：新增 `/api/frameworks` 端点，通过 `PluginsManager` 或直接读取插件模块的 `_OP_MAP` / `_SUPPORTED_OPS` 统计。

---

### P6：MR 质量视图（`GET /quality`）— 新增

**设计目标**：可视化展示 Phase K 的 MR 质量治理成果，支撑论文中的质量保障章节

**页面内容**：

- 顶部 KPI：MR 总量 / 已退役数 / 有异常项数
- 三层质量分布堆叠柱图（x 轴为层次，y 轴为 MR 数，颜色区分 curated/proven/checked/candidate/retired）
- MR 来源分布环形图（llm/template/manual，分层展示或合并）
- 异常告警列表（无 oracle / 无 provenance / 重复组等，对应 `RepoAuditor.anomalies()`）
- 质量筛选器（min_quality 下拉 + layer 下拉），展示符合条件的 MR 列表（调用 `/api/mr-quality/filter`）

**数据来源**：直接使用已有的 `/api/mr-quality` 和 `/api/mr-quality/filter` 端点（Phase K 已实现）。

---

### P7：真实缺陷案例（`GET /cases`）— 新增

**设计目标**：将 Phase M 挖掘的真实案例在面板中可浏览，用于答辩演示的案例分析章节

**页面内容**：

- 顶部 KPI：案例总数 / 已确认数 / 高严重度数
- 案例列表表格：案例ID / 算子 / 框架 / 缺陷类型 / 严重程度 / 状态（confirmed/draft/rejected）/ 创建时间
- 点击行展开或跳转详情页 `GET /cases/{case_id}`，展示：
  - 案例摘要（`CaseStudy.summary`）
  - 根因分析（`CaseStudy.root_cause`）
  - 复现脚本代码块（带复制按钮）
  - 受影响版本列表
  - 原始证据包链接

**数据来源**：新增 `/api/cases` 和 `/api/cases/{case_id}` 端点，通过 `CaseStudyIndex().list_all()` 读取 `data/experiments/case_studies/` 目录。

---

## 5. 需要新增的 API 端点

| 端点 | 说明 | 数据来源 | 缓存 |
|------|------|---------|------|
| `GET /api/summary-v2` | 三层 MR 统计（替代/扩充 `/api/summary`） | `MRRepository` × 3 层 | 30s |
| `GET /api/mr-repository?layer=model\|application` | 扩展层参数（已有 operator 逻辑复用） | `MRRepository(layer=...)` | 60s |
| `GET /api/frameworks` | 已注册框架列表与能力信息 | `PluginsManager` / 插件 `_OP_MAP` | 120s |
| `GET /api/test-results/model` | 模型层测试结果汇总 | `ResultsManager` 或 model 结果文件 | 10s |
| `GET /api/test-results/application` | 应用层验证结果汇总 | `ResultsManager` 或 app 结果文件 | 10s |
| `GET /api/cases` | 真实缺陷案例列表 | `CaseStudyIndex().list_all()` | 30s |
| `GET /api/cases/{case_id}` | 单案例详情 | `CaseStudyIndex().get(case_id)` | 无 |

**保留原有端点，保持向后兼容**：`/api/summary`、`/api/mr-repository`（不带 layer 参数时默认 operator）、所有已有端点均不破坏。

---

## 6. 前端改动范围

### 6.1 `base.html`（导航栏）

```
原：总览 | MR知识库 | 测试结果 | 跨框架
新：总览 | MR知识库 | 测试结果 | 跨框架 | 框架信息 | MR质量 | 缺陷案例
```

### 6.2 新增模板文件

| 文件 | 对应页面 |
|------|---------|
| `templates/frameworks.html` | 框架信息页 |
| `templates/quality.html` | MR 质量视图 |
| `templates/cases.html` | 真实缺陷案例列表 |
| `templates/case_detail.html` | 单案例详情 |

### 6.3 改造模板文件

| 文件 | 改造内容 |
|------|---------|
| `templates/overview.html` | 6 KPI + 三层分布图 + 架构层次标签 |
| `templates/mr_repo.html` | 三层 Tab + 动态框架下拉 |
| `templates/mr_detail.html` | 支持模型/应用层详情字段 |
| `templates/test_results.html` | 三层 Tab 切换 |
| `templates/cross_framework.html` | 矩阵热力图 + 框架对筛选器 |

### 6.4 新增路由文件

| 文件 | 路由 |
|------|------|
| `routers/frameworks.py` | `GET /frameworks` |
| `routers/quality.py` | `GET /quality` |
| `routers/cases.py` | `GET /cases`、`GET /cases/{case_id}` |

### 6.5 改造文件

| 文件 | 改造内容 |
|------|---------|
| `routers/api.py` | 新增 5 个端点（summary-v2、frameworks、test-results/model、test-results/application、cases、cases/{id}）；扩展 `mr-repository` layer 参数 |
| `app.py` | 注册 3 个新路由器 |
| `dependencies.py` | 添加 `ModelBenchmarkRegistry`、`ApplicationBenchmarkRegistry`、`CaseStudyIndex` 依赖 |

---

## 7. 任务拆分

### P1：总览页改造

- [x] 新增 `/api/summary-v2` 端点（三层 MR 统计）
- [x] 改造 `overview.html`：6 KPI + 三层 MR 环形图 + 架构层次标签
- [x] 三层 MR 来源分布横向柱图

### P2：MR 知识库三层化

- [x] 扩展 `/api/mr-repository` 的 `layer` 查询参数
- [x] 改造 `mr_repo.html`：三层 Tab + 动态框架下拉（调用 `/api/frameworks`）
- [x] 模型层 / 应用层 subject 列表渲染逻辑
- [x] 改造 `mr_detail.html`：支持模型/应用层字段

### P3：测试结果三层化

- [x] 新增 `/api/test-results/model` 端点
- [x] 新增 `/api/test-results/application` 端点
- [x] 改造 `test_results.html`：三层 Tab 切换

### P4：跨框架矩阵视图

- [x] 在 `/api/cross-framework` 返回值中增加 `matrix_summary` 字段
- [x] 改造 `cross_framework.html`：矩阵热力图 + 框架对筛选

### P5：框架信息页（新增）

- [x] 新增 `/api/frameworks` 端点（读取插件 `_OP_MAP` 等能力信息）
- [x] 新增 `routers/frameworks.py`、`templates/frameworks.html`
- [x] 注册路由至 `app.py`

### P6：MR 质量视图（新增）

- [x] 新增 `routers/quality.py`、`templates/quality.html`
- [x] 注册路由至 `app.py`
- [x] 三层质量堆叠柱图 + 异常告警列表 + 质量筛选器 JS 逻辑

### P7：真实缺陷案例页（新增）

- [x] 新增 `/api/cases` 和 `/api/cases/{case_id}` 端点
- [x] 新增 `routers/cases.py`、`templates/cases.html`、`templates/case_detail.html`
- [x] 注册路由至 `app.py`
- [x] 在 `dependencies.py` 中注册 `CaseStudyIndex` 依赖

### P8：基础设施与导航

- [x] `base.html` 导航栏扩展为 7 项
- [x] `dependencies.py` 新增 `ModelBenchmarkRegistry`、`ApplicationBenchmarkRegistry`、`CaseStudyIndex` 依赖
- [x] 更新 `docs/tech/web_dashboard.md`

---

## 8. 执行优先级

论文答辩演示的核心路径是：**总览 → MR知识库（三层） → 测试结果 → 真实缺陷案例**。

建议按如下顺序执行，优先保证演示路径完整：

| 优先级 | 任务 | 理由 |
|--------|------|------|
| P0（必做） | P7：真实缺陷案例页 | Phase M 核心产出，论文案例章节直接依赖 |
| P0（必做） | P1：总览页改造 | 答辩第一屏，三层架构必须直观可见 |
| P0（必做） | P2：MR 知识库三层化 | RQ1 核心，三层 MR 数据必须可展示 |
| P1（应做） | P5：框架信息页 | 直接回答"支持哪些框架"的评审问题 |
| P1（应做） | P6：MR 质量视图 | 支撑质量保障结论，Phase K 成果可视化 |
| P2（可选） | P3：测试结果三层化 | 模型/应用层测试数据若不充分可先展示算子层 |
| P2（可选） | P4：跨框架矩阵视图 | 增强 RQ3 展示效果 |

---

## 9. 约束与边界

- **只读面板**：本阶段不引入任何写操作（MR 生成、测试执行仍保留在 CLI）
- **无新依赖**：不新增 `pyproject.toml` 中未有的包，静态资源已本地化
- **不破坏现有路由**：所有现有 URL 和 API 端点保持不变，只扩展不替换
- **离线可用**：新页面同样依赖已本地化的 Bootstrap / Chart.js，断网环境可正常展示
- **框架参数化**：新增端点中所有框架名称通过 `FrameworkType` 传入，不得写死字符串
- **测试**：每个新 API 端点需在 `tests/unit/ui/` 下新增最小验证用例（可用 `TestClient` 不依赖真实数据）

---

## 10. 技术文档更新要求

完成本阶段后，需同步更新：

1. `docs/tech/web_dashboard.md` — 更新目录结构、请求流、API 端点列表、页面说明
2. `docs/cli_reference.md` — `deepmt ui` 命令说明无变化，但在示例中可提及新页面
3. `docs/dev/status.md` — 在阶段进度表中添加 Phase P 条目并标记完成状态
