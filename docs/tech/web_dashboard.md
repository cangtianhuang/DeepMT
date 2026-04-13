# DeepMT Web 仪表盘

> **状态**：已完成（含 F11 初版 + UI 视觉重设计，2026-04-12）  
> **适用场景**：论文答辩演示、导师汇报、日常实验数据浏览  
> **范围**：只读仪表盘；MR 生成等写操作保留在 CLI

---

## 1. 技术选型

| 层次 | 选型 | 理由 |
|------|------|------|
| Web 框架 | **FastAPI** | 声明式路由、自带 OpenAPI 文档、async 支持 |
| ASGI 服务器 | **uvicorn** | FastAPI 官方搭档，`pip install` 即可 |
| 模板引擎 | **Jinja2** | FastAPI 内置集成，无 Node.js 依赖 |
| 图表库 | **Chart.js 4.x（本地化）** | 纯 JS，支持柱图/饼图，文档齐全，无需打包 |
| CSS 框架 | **Bootstrap 5（本地化）** | 响应式、组件齐全，开箱即用 |
| 图标 | **Bootstrap Icons（本地化）** | 与 Bootstrap 5 配套 |
| 交互增强 | **原生 JS** | 无需额外框架，降低维护复杂度 |

**放弃的方案**：Streamlit（布局灵活性差）、Dash/Plotly（依赖重）、React/Vue（需 Node.js 构建链）、HTMX（只读场景收益有限）。

**依赖安装**：`pip install -e ".[ui]"`，或通过 `pip install -e ".[all,ui]"` 一并安装。

---

## 2. 目录结构

```
deepmt/ui/
├── __init__.py
├── app.py              # FastAPI 应用实例与路由挂载
├── server.py           # uvicorn 启动逻辑（供 CLI 调用）
├── dependencies.py     # 共用依赖（数据源实例、路径解析）
├── routers/
│   ├── overview.py     # GET /        总览页
│   ├── mr_repo.py      # GET /mr      MR 知识库
│   ├── test_results.py # GET /tests   测试结果
│   ├── cross_framework.py # GET /cross 跨框架一致性
│   └── api.py          # GET /api/**  JSON API 端点
├── templates/
│   ├── base.html           # 公共布局
│   ├── overview.html
│   ├── mr_repo.html
│   ├── mr_detail.html      # 单算子 MR 详情
│   ├── test_results.html
│   └── cross_framework.html
└── static/             # 本地化静态资源（离线可用）
    ├── bootstrap.min.css
    ├── bootstrap.bundle.min.js
    ├── bootstrap-icons.min.css
    ├── bootstrap-icons.woff2
    └── chart.umd.min.js    # Chart.js v4.4.2
```

CLI 入口：`deepmt/commands/ui.py`，注册命令 `deepmt ui [--port] [--host]`。

---

## 3. 请求流

```
浏览器
  ↓
FastAPI Router
  ↓
dependencies.py（lru_cache 数据源实例）
  ↓
数据源：MRRepository / ResultsManager / EvidenceCollector / CrossFrameworkTester
  ↓
Jinja2 HTML 渲染 / JSON 响应
  ↓
浏览器（Chart.js 渲染图表）
```

图表数据通过页面内嵌 `fetch("/api/...")` 异步加载，首屏 HTML 仅含骨架屏，避免体积过大。

---

## 4. JSON API 端点

| 端点 | 说明 |
|------|------|
| `GET /api/summary` | RQ1-RQ4 全量摘要（供总览页） |
| `GET /api/mr-repository` | 算子列表 + 统计 |
| `GET /api/mr-repository/{operator}` | 单算子 MR 列表 |
| `GET /api/test-results` | 各算子通过/失败汇总 |
| `GET /api/test-results/failed` | 失败用例（`?limit=N`） |
| `GET /api/evidence` | 证据包列表 |
| `GET /api/evidence/{id}/script` | 单证据包复现脚本 |
| `GET /api/cross-framework` | 跨框架会话列表 |
| `GET /api/cross-framework/{session_id}` | 单次会话详情 |

所有 API 返回统一结构：`{ "data": {...}, "generated_at": "...", "error": null }`

---

## 5. 页面说明

### 总览页（`/`）
4 个 KPI 卡片（MR 总量、缺陷检测通过率、跨框架一致率、覆盖算子数）+ MR 分类环形图 + 算子测试堆叠柱图 + 最近测试摘要表格。

### MR 知识库（`/mr`）
筛选栏（框架/类别/来源/验证状态/关键词搜索）+ 统计卡片 + 来源环形图 + 分类横向柱图 + 算子表格。点击"详情"跳转 `/mr/{operator}`，展示该算子下的 MR 卡片列表（含代码块、验证状态、分析备注）。

### 测试结果（`/tests`）
统计卡片 + 各算子通过率横向柱图（含 90%/70% 参考线）+ 失败用例表格（含严重程度分级）+ 证据包列表（模态框展示复现脚本）。

### 跨框架一致性（`/cross`）
统计卡片 + 各算子一致率柱图（含 95%/85% 参考线）+ 会话列表表格（含渐变进度条）。

---

## 6. 视觉设计

**主题**：`Neon Research Lab` — 深宇宙背景 + 霓虹配色。

| 要素 | 方案 |
|------|------|
| 背景 | 纯黑 `#020406` + 点阵网格 + 3 个缓慢漂移的径向渐变 blob |
| 各页强调色 | 总览蓝 `#4d8bff`、知识库紫 `#a855f7`、测试红 `#ff5252`、跨框架青绿 `#00e5b0` |
| KPI 卡片 | 3px 顶部色条 + 右上角径向光晕 + 悬停上浮 + 数字 text-shadow glow |
| 进场动画 | `revealUp`（translateY + scale + blur）+ 骨架屏 shimmer |
| 图表 | 固定颜色数组 + `aspectRatio` 控制高度（无需 wrapper div） + 共享 `TOOLTIP_CFG` |
| 代码块 | 左侧 3px 色条 + 语法高亮 + 悬停复制按钮 + Toast 通知 |

**Chart.js 注意事项**：
- 所有图表使用 `responsive: true` + `aspectRatio: N`，canvas 直接放入 `.card-body`，无需 `position:relative` 父容器。
- 全局配置通过 `Chart.defaults.animation.duration` / `Chart.defaults.animation.easing` 属性赋值（不要替换整个 `animation` 对象）。

---

## 7. 缓存策略

| 数据 | 缓存时长 |
|------|---------|
| RQ1-RQ4 摘要 | 30s in-memory |
| MR 仓库列表 | 60s |
| 测试结果汇总 | 10s |
| 跨框架会话列表 | 60s |

实现：`functools.lru_cache` + 时间戳比对，无外部依赖。

---

## 8. 离线可用性

所有 CDN 资源已本地化至 `deepmt/ui/static/`，`base.html` 通过 `/static/` 路径引用。`pyproject.toml` 中声明：

```toml
[tool.setuptools.package-data]
"deepmt.ui" = ["templates/**/*.html", "static/**/*"]
```

断网环境下仪表盘功能完整可用。

---

## 9. 未实现功能（范围外）

- 表单化 MR 生成（保留在 CLI）
- 用户认证
- 历史趋势对比（时间序列图）
- MR 在线编辑与保存
- WebSocket 实时推送
