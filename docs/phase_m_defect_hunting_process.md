# 真实缺陷挖掘流程（M7）

> 可重复的标准作业流程。任何后续真实缺陷扫描都套用本说明；Phase M 的一次性执行记录见 `phase_m_real_defect_hunting/`。

## 0. 前置条件

1. 能力缺口 L1/L2/L3 验收全绿（`docs/dev/15_Phase_M_system_capability_gaps.md`）。
2. `deepmt health check` 输出 HEALTHY。
3. 本次目标框架/版本的插件均已实现必要原语（`make_tensor / _execute_operator / allclose / to_numpy / get_shape`）。
4. 本次目标算子在 `data/knowledge/mr_repository/operator/` 至少有 1 条 MR；若无，先 `deepmt mr generate <op> --sources template --save`。

## 1. 流程五步法

```
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ ① 计划   │ → │ ② 扫描   │ → │ ③ 聚类   │ → │ ④ 沉淀   │ → │ ⑤ 归档   │
  │ (M1)     │   │ (M2)     │   │ (M3)     │   │ (M4+M5)  │   │ (M6)     │
  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

### ① 计划（Plan）

- 产出：一份 `scan_plan.md`，列明目标版本矩阵、算子清单、框架对矩阵、规模预算、收敛条件。
- 模板：参考 `phase_m_real_defect_hunting/01_scan_plan.md`。
- 核心决策：
  - **宽度 vs 深度**：优先宽度（多算子 × 少样本）发现线索，再对命中点加深度。
  - **oracle 可信源**：numpy 作 float64 金标准优先，生产框架对次之。

### ② 扫描（Sweep）

- 入口：`deepmt test cross <op> --matrix --n-samples N --save --save-to-evidence`。
- 默认参数：`N=500, --keep-samples 50`。
- 观察信号（优先级从高到低）：
  1. `exception_f2 > 0` — 参考框架崩溃或不支持，插件层 bug（非真实缺陷）
  2. `inconsistent_cases > 0` — oracle 结论分叉，**一级信号**
  3. `[!] silent-numeric-diff rate ≥ 0.05` — 结论一致但数值显著差，**二级信号**
  4. `output_max_diff` 异常大 — 通常是极值/溢出路径，需人工鉴别

### ③ 聚类（Dedup）

- 命令：`deepmt test dedup --source cross`（必要时加 `--source all`）。
- 聚类签名：`(operator, mr_id, framework_pair, bucket)`；bucket 由 diff_type 决定。
- 输出：`DefectLead` 列表。每个 lead 携带样本指针，不直接入论文。

### ④ 沉淀（Case Build + 人工确认）

- 每个 lead：
  1. `deepmt case build --from-evidence <evidence_pack_id>` 产出案例包；
  2. `python data/cases/real_defects/<case_id>/reproduce.py` 干净 shell 一次复现；
  3. 查阅参考实现 / 文档 / 其他框架表现，确定**哪一侧是正确参考**；
  4. `deepmt case confirm <case_id> --status {confirmed|pending|insufficient}`。
- 判决标准：
  - `confirmed`：差异稳定复现、参考实现明确、排除数值外溢。
  - `pending`：差异可复现但参考不明确，留作待上报。
  - `insufficient`：差异属随机/数值外溢/MR 本身质量问题，不计入论文。

### ⑤ 归档（Index + Report）

- 维护 `case_index.md`（单点总表）。
- 每个 `confirmed` 案例在 `case_summary.md` 中必须包含：现象、条件、影响、复现路径、初步归因、参考链接。
- 若案例计划上报外部社区，套用 `deepmt/templates/bug_report_template.md`。

## 2. 成本与收益标尺

| 规模档位 | 单算子单对耗时（CPU 估） | 建议使用 |
| --- | --- | --- |
| `n=50`   | ~10s | 烟测 / L1 回归 |
| `n=200`  | ~45s | 快速扫描 |
| `n=500`  | ~2~5min | **默认真实挖掘** |
| `n=2000` | ~10~20min | 命中点加深度 |

命中率参考（L1/L2 基线）：一级 9 算子全量 `n=500` 命中 silent-diff ≥ 1 项的概率 ~30%；命中 oracle 违反 < 5%。

## 3. 哪些案例能进论文

| 类型 | 正文 | 附录 | 禁止 |
| --- | --- | --- | --- |
| `confirmed` + 真实框架不一致 | ✅ | ✅ | — |
| `confirmed` + 数值精度边界（且参考文档支持）| ✅ | ✅ | — |
| `pending` 待确认线索 | ⛔ | ✅（标注待确认） | — |
| 受控注入（L3 演示） | ⛔ | ✅（仅作流程演示） | 不得混入真实案例清单 |
| `insufficient` | ⛔ | ⛔ | — |

## 4. 与实验组织器的链接

每个 `confirmed` 案例在 `deepmt/experiments/case_study.py` 登记引用位；`deepmt experiments aggregate` 会把 case_index 汇入论文统计。

## 5. 故障与降级

- Paddle 3.3.1 在高维输入偶有 ABI 警告，不阻塞；若 SIGSEGV 则跳过该算子，在 runlog 记录。
- `deepmt mr generate` 需要 `OPENAI_API_KEY`；离线走 `--sources template` 退化路径（G9）。
- session JSON > 50MB：降 `--keep-samples` 到 20。

## 6. 不在本流程范围

- 不包含外部 issue 提交执行动作（仅产出模板）。
- 不包含 TensorFlow 路径（见 Phase O）。
- 不包含模型层 / 应用层 MR 的真实扫描（本流程专注算子层）。
