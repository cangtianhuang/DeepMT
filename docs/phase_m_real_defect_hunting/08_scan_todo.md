# M2 — 真实扫描 TODO（用户手动执行）

> 逐条 checklist。每一步都是一条具体的 `deepmt` 命令；运行完勾选并把关键输出粘贴到 `09_scan_runlog.md`。
> 所有命令前置：`cd /Users/lihaoyang/Github/DeepMT && source .venv/bin/activate`。
> 前置约束：Phase M 能力缺口 L1/L2/L3 已全绿（`docs/dev/15_Phase_M_system_capability_gaps.md`）。

---

## STEP 0 — 环境自检

- [ ] `deepmt health check`
  - 期望：`HEALTHY`，至少 36 项通过，无 ERROR。
  - 已知盲点：`paddle_plugin` 不在列表里 — **忽略**（留给 Phase O 修）。

- [ ] `python -c "import torch,paddle,numpy; print(torch.__version__, paddle.__version__, numpy.__version__)"`
  - 期望三行版本号全部打印，paddle 不报 ImportError。

- [ ] `deepmt repo list --layer operator`
  - 期望：看到 relu/tanh/exp/abs/sigmoid/gelu/softmax/log_softmax/leaky_relu 九个算子均有 2~3 条 MR。

---

## STEP 1 — 回归基线（已知热点先跑）

目的：确认 L1/L2 修复仍然生效，并对 C1（Paddle log_softmax 截断）做一次显式再发现尝试。

- [ ] `deepmt test cross log_softmax --framework2 paddlepaddle --n-samples 500 --save --save-to-evidence`
  - 重点看：`consistency`、`silent_numeric_diff_rate`、`failed_samples` 是否落盘；是否出现 `[!]` 告警。
  - 把 session 路径粘到 runlog。

- [ ] `deepmt test cross exp --framework2 paddlepaddle --n-samples 500 --save --save-to-evidence`
  - L2.4 观测点：silent-numeric-diff 告警必须触发。

---

## STEP 2 — 一级全量扫描（矩阵模式）

对 9 个一级算子，三个框架对各跑一次。推荐用 `--matrix` 一次跑完一个算子的三对。

按算子依次执行（每条命令约 2~5 分钟）：

- [ ] `deepmt test cross relu       --matrix --n-samples 500 --save --save-to-evidence`
- [ ] `deepmt test cross tanh       --matrix --n-samples 500 --save --save-to-evidence`
- [ ] `deepmt test cross exp        --matrix --n-samples 500 --save --save-to-evidence`
- [ ] `deepmt test cross abs        --matrix --n-samples 500 --save --save-to-evidence`
- [ ] `deepmt test cross sigmoid    --matrix --n-samples 500 --save --save-to-evidence`
- [ ] `deepmt test cross gelu       --matrix --n-samples 500 --save --save-to-evidence`
- [ ] `deepmt test cross softmax    --matrix --n-samples 500 --save --save-to-evidence`
- [ ] `deepmt test cross log_softmax --matrix --n-samples 500 --save --save-to-evidence`
- [ ] `deepmt test cross leaky_relu --matrix --n-samples 500 --save --save-to-evidence`

> 若 `--matrix` 与 `--save-to-evidence` 行为异常（如只对 f1=pytorch 落盘），退化为分对手写：
> `... --framework1 pytorch --framework2 numpy ...` / `... --framework2 paddlepaddle ...` / `... --framework1 paddlepaddle --framework2 numpy ...`

运行中重点关注：
- 终端 `[!]` 告警（silent-numeric-diff）
- `inconsistent_cases > 0` 的算子
- 任何 `exception_f2 > 0` 的异常路径

---

## STEP 3 — 跨框架证据去重

- [ ] `deepmt test dedup --source cross`
  - 期望：按 `(operator, mr_id, framework_pair, diff_type)` 聚类输出 `DefectLead` 列表。
  - 把 lead_id 清单粘贴到 runlog。

- [ ] `deepmt test dedup --source all`（对比单框架 evidence 聚类结果）
  - 若有意料之外的 lead（无注入情况下的 oracle 违反），**重点标注**。

---

## STEP 4 — 案例构建（对每个高优先级 lead）

对 STEP 3 产生的每一个 DefectLead / 关心的 evidence pack ID 依次：

- [ ] `deepmt case build --from-evidence <evidence_pack_id>`
  - 产出 `data/cases/real_defects/<case_id>/`，含 `reproduce.py / evidence.json / metadata.json / case_summary.md`。

- [ ] `python data/cases/real_defects/<case_id>/reproduce.py`
  - 必须在干净 shell 中一次运行通过并打印与 session 一致的 `diff_type` / `numeric_diff`。

- [ ] 对"跨框架数值差"类线索，若 reproduce 证实差异 ≥ 1e-4，视为待确认候选；否则标记 `insufficient_evidence`。

---

## STEP 5 — 人工确认（M5）

对每个候选案例：

- [ ] `deepmt case show <case_id>` — 核对 metadata
- [ ] 查阅参考实现（如 scipy.special / paddle 文档 / torch 文档），判断"是哪一侧是标准"
- [ ] `deepmt case confirm <case_id> --status {confirmed|pending|insufficient}` 标注
- [ ] 将结论与参考链接补充到 `10_case_index.md`

---

## STEP 6 — 对照实验（False Positive 验证）

- [ ] `deepmt test cross relu --matrix --n-samples 500 --save`（已跑过则跳过，直接取上次结果）
  - 确认干净路径（无 faulty 插件注入）不产生 DefectLead；若产生说明**存在真实候选**，进入 STEP 4。

---

## STEP 7 — 收口

- [ ] 把本次扫描命中和耗时填入 `09_scan_runlog.md`
- [ ] 把每个确认案例的一行摘要填入 `10_case_index.md`
- [ ] 更新 `docs/dev/status.md`：Phase M 由 🔄 → ✅
- [ ] 更新 `docs/dev/13_Phase_M_真实缺陷挖掘与案例沉淀.md` §7 验收勾选

## 备注

- **不要**在扫描未全量跑完前提前 commit / push；扫描产物归在 `data/results/{cross_framework,evidence}/`，不入 git。
- 如果某一算子 session 耗时 > 10 分钟，降 `--n-samples` 到 200 再跑；把降级记录在 runlog。
- 如果 paddle 在某算子崩溃，跳过该对，在 runlog 记录异常栈，不阻塞后续算子。
