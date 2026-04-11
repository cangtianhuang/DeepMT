# F11 Web 仪表盘设计思路与开发规划

> **状态**：阶段1完成（2026-04-12）；阶段2（数据 API）待实现  
> **适用场景**：论文答辩演示、导师汇报、日常实验数据浏览  
> **范围**：只读仪表盘，不实现表单化 MR 生成（保留在 CLI）

---

## 1. 技术选型

### 1.1 选型依据

论文演示场景对工具链有特殊约束：

| 约束 | 说明 |
|------|------|
| 零依赖启动 | 导师机器上不应出现"先装 Node.js"这类意外 |
| 离线可用 | 答辩环境不一定有网络，CDN 依赖需备份方案 |
| 维护成本低 | 演示代码不应成为未来的技术债 |
| 视觉效果够用 | 不必追求企业级 UI，但图表清晰、布局整洁是底线 |

### 1.2 最终选型

| 层次 | 选型 | 理由 |
|------|------|------|
| Web 框架 | **FastAPI** | 声明式路由、自带 OpenAPI 文档、async 支持、生态成熟 |
| ASGI 服务器 | **uvicorn** | FastAPI 官方搭档，生产可用，`pip install` 即可 |
| 模板引擎 | **Jinja2** | FastAPI 内置集成，无 Node.js 依赖，逻辑嵌入 HTML |
| 图表库 | **Chart.js 4.x（CDN）** | 纯 JS，支持柱图/饼图/折线图，文档齐全，无需打包 |
| CSS 框架 | **Bootstrap 5（CDN）** | 响应式、组件齐全、表格/卡片/导航栏开箱即用 |
| 图标 | **Bootstrap Icons（CDN）** | 与 Bootstrap 5 配套，无需额外安装 |
| 交互增强 | **原生 JS（最小化）** | 过滤/排序/展开不引入额外框架，降低维护复杂度 |

**放弃方案说明**：
- `Streamlit`：布局灵活性差，UI 定制困难，不适合多页面结构化展示
- `Dash/Plotly`：依赖重，演示时加载慢
- `React/Vue SPA`：需 Node.js 构建链，与零依赖目标冲突
- `HTMX`：引入额外学习成本，对只读仪表盘收益有限

### 1.3 新增依赖

在 `pyproject.toml` 增加 `ui` optional group：

```toml
[project.optional-dependencies]
ui = [
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.27.0",
    "jinja2>=3.1.0",
    "python-multipart>=0.0.9",
]
```

安装：`pip install -e ".[ui]"` 或 `pip install -e ".[all,ui]"`

---

## 2. 系统架构

### 2.1 目录结构

```
deepmt/
└── ui/
    ├── __init__.py
    ├── app.py                  # FastAPI 应用实例与路由挂载
    ├── server.py               # uvicorn 启动逻辑（供 CLI 调用）
    ├── dependencies.py         # 共用依赖（数据源实例、路径解析）
    ├── routers/
    │   ├── __init__.py
    │   ├── overview.py         # GET /   →  总览页
    │   ├── mr_repo.py          # GET /mr  →  MR 知识库浏览
    │   ├── test_results.py     # GET /tests → 测试结果面板
    │   ├── cross_framework.py  # GET /cross → 跨框架一致性
    │   └── api.py              # GET /api/** → JSON API 端点
    ├── templates/
    │   ├── base.html           # 公共布局（导航栏、Bootstrap 引入）
    │   ├── overview.html
    │   ├── mr_repo.html
    │   ├── mr_detail.html      # 单算子 MR 详情（弹出/独立页）
    │   ├── test_results.html
    │   └── cross_framework.html
    └── static/                 # （可选）本地化 CDN 资源，确保离线可用
        ├── bootstrap.min.css
        ├── bootstrap.bundle.min.js
        ├── chart.umd.min.js
        └── bootstrap-icons/
```

`deepmt/commands/ui.py` — CLI 命令，挂载到 `deepmt/cli.py`

### 2.2 请求流

```
浏览器请求
  ↓
FastAPI Router（渲染模式 or JSON API 模式）
  ↓
依赖注入层（dependencies.py）
  ↓
数据源层：
  ├── MRRepository          data/mr_repository/operator/*.yaml
  ├── ResultsManager        data/defects.db
  ├── EvidenceCollector     data/evidence/*.json
  ├── CrossFrameworkTester  data/cross_results/*.json
  └── ExperimentOrganizer   ← 聚合上述四者
  ↓
Jinja2 模板渲染 / JSON 响应
  ↓
浏览器（Chart.js 渲染图表）
```

### 2.3 两种响应模式

每个页面路由同时支持 HTML 渲染和 JSON API，通过请求头区分：

- `Accept: text/html` → Jinja2 HTML 页面（浏览器正常访问）
- `GET /api/**` → 纯 JSON 响应（供 Chart.js fetch 使用）

图表数据通过页面内嵌 `<script>` 中的 `fetch("/api/...")` 异步加载，避免首屏 HTML 体积过大。

---

## 3. 页面组织

### 3.1 导航结构

```
顶部导航栏（固定）
├── DeepMT Logo + 版本号
├── 总览     (/overview 或 /)
├── MR 知识库  (/mr)
├── 测试结果  (/tests)
├── 跨框架    (/cross)
└── [外部链接] CLI 命令参考文档
```

### 3.2 总览页（Overview）`GET /`

**目标**：让导师/答辩委员在 10 秒内看懂系统研究贡献。

**布局**：
```
[标题横幅]  DeepMT — 深度学习框架蜕变关系自动生成与分层测试体系

[4 个 KPI 卡片（一行）]
  RQ1: 已生成 MR           RQ2: 缺陷检测率         RQ3: 跨框架一致率        RQ4: 覆盖算子数
  「47 条 / 验证率 89%」    「通过率 94.2%」         「98.1% 一致」            「15 个算子」

[两列布局]
  左列: MR 分类分布（环形饼图）    右列: 算子测试概况（水平柱图，通过/失败）

[最近测试摘要表格]
  算子名 | 框架 | 测试数 | 通过数 | 失败数 | 最后更新时间
```

**数据来源**：`ExperimentOrganizer.collect_all()`，首次加载缓存 30 秒。

### 3.3 MR 知识库页（MR Repository）`GET /mr`

**目标**：展示 MR 自动生成质量（RQ1 支撑）。

**布局**：
```
[筛选栏]  框架: [全部▼]  类别: [全部▼]  来源: [全部▼]  □ 仅已验证  [搜索框]

[3 个统计卡片]  MR 总数: 47    验证率: 89.4%    覆盖算子: 15 个

[两列图表]
  左: 来源分布（饼图）llm / template / manual
  右: 类别分布（水平柱图）linearity / symmetry / monotonicity / ...

[算子表格]
  算子名称 | MR 总数 | 已验证 | 类别 | 来源 | 操作
  torch.relu |  4  |  4  | linearity, monotonicity | llm+template | [详情]

点击 [详情] → 跳转 /mr/<operator_name>
```

**单算子详情页** `GET /mr/{operator}`：
```
[返回按钮]  算子：torch.relu  （framework: pytorch）

[MR 列表（卡片式）]
每张卡片：
  ● 标题：description
  ● 来源标签 / 类别标签 / 验证状态徽章（✅已验证 / ⚠️预检通过 / ❌未通过）
  ● transform_code（代码块）
  ● oracle_expr（公式块）
  ● [展开] analysis 备注
```

### 3.4 测试结果页（Test Results）`GET /tests`

**目标**：展示缺陷检测能力（RQ2 支撑）。

**布局**：
```
[4 个统计卡片]
  总测试数 | 通过率 | 失败数 | 证据包数

[两列图表]
  左: 各算子通过率（水平分段柱图：绿色=通过, 红色=失败）
  右: 证据包 & 去重后独立缺陷数（双柱图或数值展示）

[失败测试用例表格]
  算子 | MR 描述 | Oracle 表达式 | 实际差值 | 时间戳 | 操作

[证据包列表]
  包 ID | 算子 | 变换描述 | 差值 | [查看复现脚本]
  点击后展示完整 Python 复现脚本（代码块）
```

### 3.5 跨框架一致性页（Cross-Framework）`GET /cross`

**目标**：展示多框架验证能力（RQ3 支撑）。

**布局**：
```
[3 个统计卡片]
  对比会话数 | 覆盖算子数 | 整体一致率

[主图：各算子跨框架一致率（分组柱图）]
  X轴: 算子名称；Y轴: 一致率（%）；颜色区分框架对（pytorch vs numpy 等）

[会话列表表格]
  算子 | 框架对 | 一致率 | 最大输出差值 | 不一致 MR 数 | 实验时间

点击行 → 展开详细 MR 级对比结果
```

---

## 4. 数据管理

### 4.1 后端数据层

所有页面的数据通过 `dependencies.py` 中的依赖注入获取，
避免每个路由重复实例化数据源：

```python
# dependencies.py
from functools import lru_cache
from deepmt.mr_generator.base.mr_repository import MRRepository
from deepmt.core.results_manager import ResultsManager
from deepmt.analysis.experiment_organizer import ExperimentOrganizer

@lru_cache(maxsize=1)
def get_mr_repository() -> MRRepository: ...

@lru_cache(maxsize=1)
def get_results_manager() -> ResultsManager: ...

@lru_cache(maxsize=1)
def get_experiment_organizer() -> ExperimentOrganizer: ...
```

**注意**：`lru_cache` 适合演示场景（数据变化不频繁）。生产场景可替换为带 TTL 的缓存。

### 4.2 JSON API 端点设计

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/summary` | GET | RQ1-RQ4 全量摘要（供总览页加载） |
| `/api/mr-repository` | GET | 算子列表 + 统计 |
| `/api/mr-repository/{operator}` | GET | 单算子的 MR 列表 |
| `/api/test-results` | GET | 各算子通过/失败汇总 |
| `/api/test-results/failed` | GET | 最近 N 条失败用例（`?limit=50`） |
| `/api/evidence` | GET | 证据包列表 |
| `/api/evidence/{id}/script` | GET | 单个证据包的复现脚本 |
| `/api/cross-framework` | GET | 跨框架会话列表 |
| `/api/cross-framework/{session_id}` | GET | 单次会话详情 |

所有 API 返回标准结构：
```json
{ "data": {...}, "generated_at": "2026-04-11T10:00:00", "error": null }
```

### 4.3 缓存策略

| 数据 | 缓存时长 | 理由 |
|------|---------|------|
| RQ1-RQ4 摘要 | 30s in-memory | 读文件代价高，演示中数据不实时变化 |
| MR 仓库列表 | 60s | YAML 文件读取较慢 |
| 测试结果汇总 | 10s | SQLite 查询快，允许近实时 |
| 跨框架会话列表 | 60s | JSON 文件列表，变化少 |

实现方式：`functools.lru_cache` + 时间戳比对（`time.time()` 方式），不引入 Redis 等外部依赖。

---

## 5. 用户交互设计

### 5.1 交互原则

1. **加载即可用**：首屏不依赖异步加载，直接 SSR 渲染关键数据
2. **图表异步加载**：图表数据通过 `fetch()` 加载，避免首屏 HTML 过大
3. **搜索过滤客户端处理**：MR 列表/算子表格的过滤在浏览器端 JS 实现，无需后端请求
4. **代码可复制**：所有代码块（transform_code、复现脚本）有一键复制按钮
5. **证据可导出**：每个证据包提供"下载 .py 脚本"按钮

### 5.2 具体交互元素

**MR 知识库页**：
- 框架/类别/来源的下拉筛选 → 即时过滤表格行（客户端 JS，`data-*` 属性匹配）
- 搜索框 → 模糊匹配算子名和 MR 描述
- "仅已验证"复选框 → 隐藏 `verified=false` 的行
- 点击算子名 → 跳转详情页（`/mr/<operator>`）

**测试结果页**：
- 时间范围选择（今天/最近 7 天/全部）→ 过滤表格
- 失败用例的"查看详情"→ 内联展开行，显示 `defect_details` JSON
- 证据包的"查看脚本"→ 模态框展示完整 Python 脚本 + 复制按钮

**跨框架页**：
- 点击会话行 → 展开 MR 级别的对比结果
- 图表悬停提示（Chart.js tooltip）→ 显示具体一致率数值

### 5.3 响应式设计

Bootstrap 5 栅格系统，支持以下断点：
- 桌面（≥1200px）：主要展示目标，3-4列布局
- 平板（≥768px）：导师笔记本，2列布局
- 手机：不强制支持，答辩场景不需要

---

## 6. 前后端对接细节

### 6.1 图表数据流

以总览页的"各算子测试概况"柱图为例：

```html
<!-- templates/overview.html -->
<canvas id="operatorChart"></canvas>
<script>
fetch('/api/test-results')
  .then(r => r.json())
  .then(resp => {
    const data = resp.data;
    new Chart(document.getElementById('operatorChart'), {
      type: 'bar',
      data: {
        labels: data.map(r => r.ir_name),
        datasets: [
          { label: '通过', data: data.map(r => r.passed_tests), backgroundColor: '#198754' },
          { label: '失败', data: data.map(r => r.failed_tests), backgroundColor: '#dc3545' },
        ]
      },
      options: { indexAxis: 'y', plugins: { legend: { position: 'top' } } }
    });
  });
</script>
```

### 6.2 Jinja2 模板与 Python 数据传递

首屏关键数据（KPI 卡片数值）通过模板变量直接嵌入，无需异步请求：

```python
# routers/overview.py
@router.get("/", response_class=HTMLResponse)
async def overview(
    request: Request,
    organizer: ExperimentOrganizer = Depends(get_experiment_organizer),
):
    summary = organizer.collect_all()
    return templates.TemplateResponse("overview.html", {
        "request": request,
        "rq1": summary["rq1"],
        "rq2": summary["rq2"],
        "rq3": summary["rq3"],
        "rq4": summary["rq4"],
    })
```

### 6.3 错误处理

数据源（SQLite、YAML 文件）可能不存在（全新环境）：
- 后端：`try/except` 捕获，返回空数据结构（不报 500）
- 前端：检测空数据时显示"暂无数据，请运行 `deepmt test batch` 生成"提示
- 特别处理 RQ3：明确提示"运行 `deepmt test cross <operator>` 生成跨框架数据"

### 6.4 CLI 集成

新增 `deepmt/commands/ui.py`，注册到 `cli.py`：

```python
# deepmt/commands/ui.py
@click.command("ui")
@click.option("--port", "-p", default=8080, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--reload", is_flag=True, hidden=True)  # 开发模式
def ui_start(port, host, reload):
    """启动 Web 仪表盘。"""
    try:
        import uvicorn
    except ImportError:
        click.echo("请先安装 UI 依赖：pip install -e \".[ui]\"", err=True)
        raise SystemExit(1)
    from deepmt.ui.server import start
    start(host=host, port=port, reload=reload)
```

CLI 入口：`deepmt ui --port 8080`

---

## 7. 离线可用性方案

答辩环境可能无网络，CDN 资源需本地备份：

**方案**：在 `deepmt/ui/static/` 下存储本地化 CDN 文件：
- `bootstrap.min.css`（Bootstrap 5，~30KB gzip）
- `bootstrap.bundle.min.js`（含 Popper，~60KB gzip）
- `chart.umd.min.js`（Chart.js 4，~60KB gzip）
- `bootstrap-icons/` 目录

`base.html` 中通过本地路径引入（`/static/bootstrap.min.css`），不走 CDN。

这些静态文件需声明到 `pyproject.toml`：
```toml
[tool.setuptools.package-data]
"deepmt.ui" = ["templates/**/*.html", "static/**/*"]
```

---

## 8. 开发规划

### 8.1 实施顺序

```
阶段 1：骨架搭建（✅ 已完成，2026-04-12）
  F11-1. ✅ pyproject.toml 新增 ui 依赖组（fastapi/uvicorn/jinja2/python-multipart）
  F11-2. ✅ deepmt/ui/ 目录初始化（app.py、server.py、dependencies.py、templating.py）
  F11-3. ✅ deepmt/commands/ui.py + cli.py 注册（deepmt ui start）
  F11-4. ✅ base.html 公共布局（Bootstrap5 渐变导航栏 + 4页面路由）
         ✅ 全部页面模板（overview/mr_repo/mr_detail/test_results/cross_framework.html）
  F11-5. ✅ 静态资源本地化（bootstrap.min.css/bootstrap.bundle.min.js/
            bootstrap-icons.min.css/fonts/bootstrap-icons.woff2/chart.umd.min.js）

阶段 2：数据 API 层（~1.5h）
  F11-6. /api/summary（RQ1-RQ4）
  F11-7. /api/mr-repository + /api/mr-repository/{operator}
  F11-8. /api/test-results + /api/test-results/failed
  F11-9. /api/evidence + /api/evidence/{id}/script
  F11-10. /api/cross-framework + /api/cross-framework/{session_id}

阶段 3：页面实现（~4h）
  F11-11. 总览页（overview.html + 4 KPI 卡片 + 2 图表）
  F11-12. MR 知识库页（mr_repo.html + 筛选 + 表格）
  F11-13. 单算子详情页（mr_detail.html + MR 卡片列表）
  F11-14. 测试结果页（test_results.html + 柱图 + 证据包）
  F11-15. 跨框架一致性页（cross_framework.html + 分组柱图）

阶段 4：收尾与测试（~1h）
  F11-16. 空数据状态处理（各页面 fallback 提示）
  F11-17. 添加单元测试（API 端点返回正确结构）
  F11-18. 更新 docs/cli_reference.md（新增 deepmt ui 命令）
  F11-19. 更新 docs/status.md
```

### 8.2 完成标准

- [ ] `pip install -e ".[ui]"` 后 `deepmt ui --port 8080` 正常启动
- [ ] 浏览器访问 `http://localhost:8080` 能看到总览页，KPI 卡片有数值
- [ ] MR 知识库页能列出所有已保存算子的 MR，筛选功能可用
- [ ] 测试结果页能展示 `data/defects.db` 中的历史测试数据
- [ ] 跨框架页在有数据时可展示一致率图表，无数据时有友好提示
- [ ] 所有 CDN 资源本地化，断网环境下仪表盘功能正常
- [ ] `/api/**` 端点返回正确 JSON，可供外部脚本调用
- [ ] 新增 API 端点的单元测试（mock 数据源）

### 8.3 不在本期实现的功能

以下功能明确排除，避免范围蔓延：
- 表单化 MR 生成（保留在 CLI）
- 用户认证（只读本地仪表盘不需要）
- 历史对比（趋势图按时间切片）
- MR 编辑与保存
- 实时 WebSocket 推送

---

## 9. 关键设计决策汇总

| 决策点 | 选择 | 放弃的方案 |
|--------|------|----------|
| 模板引擎 | Jinja2（SSR） | React/Vue SPA |
| 图表 | Chart.js CDN | Plotly（体积过大） |
| CSS | Bootstrap 5 | Tailwind（需构建） |
| 交互 | 原生 JS | HTMX（收益有限） |
| 数据流 | SSR + fetch API | 全 SPA |
| 离线支持 | 本地化 static/ | 依赖 CDN |
| 启动方式 | `deepmt ui` CLI | 独立脚本 |
| 缓存 | in-process lru_cache | Redis/Memcached |

---

*撰写日期：2026-04-11*  
*待审阅后开始 F11 实现*
