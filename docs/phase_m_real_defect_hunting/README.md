# Phase M — 真实缺陷挖掘与案例沉淀（文档索引）

> 归档目录。本目录承载 Phase M 主干（真实缺陷扫描）相关的计划、扫描任务清单、验收报告与案例沉淀。
> 阶段主文档：[`docs/dev/13_Phase_M_真实缺陷挖掘与案例沉淀.md`](../dev/13_Phase_M_真实缺陷挖掘与案例沉淀.md)
> 能力缺口前置：[`docs/dev/15_Phase_M_system_capability_gaps.md`](../dev/15_Phase_M_system_capability_gaps.md)

## 目录结构

| 文件 | 用途 | 关联 M 任务 |
| --- | --- | --- |
| `01_scan_plan.md` | 真实扫描目标版本 / 算子 / 框架对矩阵 / 成本估计 | M1 |
| `05_acceptance_L1.md` | 能力缺口 L1 验收（CLI 到达） | 前置 |
| `06_acceptance_L2.md` | 能力缺口 L2 验收（CLI 闭环） | 前置 |
| `07_acceptance_L3.md` | 能力缺口 L3 验收（受控缺陷闭环） | 前置 |
| `08_scan_todo.md` | **用户手动执行的 deepmt 命令清单**（逐条 checklist） | M2~M4 |
| `09_scan_runlog.md` | 扫描运行日志（用户填写；命中、异常、耗时） | M2 |
| `10_case_index.md` | 真实案例目录（汇总 metadata / 状态 / 论文引用位） | M5~M6 |
| `acceptance_runs/` | 历史验收运行产物（data snapshot） | 前置 |

`01_~04_` 的历史版本（版本矩阵、候选案例草表、CLI-only sweep 说明、原扫描计划）在能力缺口修复过程中被移除。当前目录以 `01_scan_plan.md + 08_scan_todo.md` 重建 M1/M2 文档入口，无需恢复历史草稿。

## 与 `docs/phase_m_defect_hunting_process.md` 的关系

- 本目录：**一次性**的真实扫描执行载体（计划 → 命令 → 日志 → 案例）。
- `phase_m_defect_hunting_process.md`：**可重复**的标准流程说明书（M7），任何后续扫描都套用。
