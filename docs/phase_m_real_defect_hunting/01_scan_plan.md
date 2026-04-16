# M1 — 真实扫描计划表

> 对应 `13_Phase_M §4 M1`。明确本阶段"跑什么版本、哪些对象、多大规模"。

## 1. 目标框架与版本（快照：2026-04-15）

| 框架 | 当前 venv 版本 | 纳入扫描 | 备注 |
| --- | --- | --- | --- |
| PyTorch | 2.11.0 | ✅ 主框架 | 作为 oracle 参考（f1） |
| NumPy | 2.4.4 | ✅ 参考框架 | 天然 float64 金标准 |
| PaddlePaddle | 3.3.1 | ✅ 参考框架 | CPU 版，已在 L1/L2 回归通过 |
| TensorFlow | — | ⛔ 不在本阶段 | 插件未落地，见 Phase O |

不冻结版本号；每次扫描由 `deepmt test cross --save` 的 session JSON 自行记录运行时环境。

## 2. 目标算子清单

### 2.1 一级重点（知识库已有 MR，所有框架对均可跑）

`relu, tanh, exp, abs, sigmoid, gelu, softmax, log_softmax, leaky_relu`

共 9 个，每个在 `data/knowledge/mr_repository/operator/*.yaml` 都有 2~3 条 MR，且三框架插件均已实现。

### 2.2 二级候选（T1/T2 已加入插件映射表，但知识库尚无 MR；扫描前需先跑 `deepmt mr generate --sources template`）

`layer_norm, logsumexp, sqrt, log, reciprocal, cumsum, var, std, sinh, cosh, cos, sin`

二级扫描为选跑项；若一级扫描已命中足量案例，可跳过。

### 2.3 回归监控位（L1/L2 验收天然回归点）

- `log_softmax @ pytorch↔paddlepaddle` — C1（Paddle 截断）的再发现
- `exp @ pytorch↔paddlepaddle` — 历史上报 silent-numeric-diff 触发点

## 3. 框架对矩阵

| 对（f1, f2） | 开启 | 说明 |
| --- | --- | --- |
| (pytorch, numpy) | ✅ | float64 金标准比对 |
| (pytorch, paddlepaddle) | ✅ | 生产框架对 |
| (paddlepaddle, numpy) | ✅ | paddle 作主；需依赖 G7 开放 |
| 其余 | ⛔ | 本阶段不跑 |

CLI 入口：`deepmt test cross --matrix`（T7）。

## 4. 规模与成本

- 默认 `--n-samples 500`；一级 9 算子 × 3 框架对 = 27 次 session，约 9×3×500=13500 次执行对
- CPU 单机预估：**20~40 分钟**/全量一级（参考 L1/L2 验收运行）
- 证据膨胀阈值：`--keep-samples 50`（默认值即可）

## 5. 出口与收敛条件

本阶段不要求命中阈值，但以下任一即可收敛进入 M3~M6：

- 一级扫描产出 ≥ 3 条跨框架不一致 DefectLead；或
- 一级扫描产出 ≥ 1 条**非** silent-numeric-diff 的 oracle 违反（即 `consistency < 100%`）；或
- 一级扫描全绿，但 silent-numeric-diff rate ≥ 5% 的 (算子, 框架对) 数 ≥ 3。

无论哪一条达成，均进入 M4（case build）→ M5（人工确认）→ M6（论文目录）。

## 6. 风险与降级

- **Paddle 与 PyTorch ABI 冲突**：已知 paddle 3.3.1 对 numpy 2.x 有警告但不阻塞；若扫描中 paddle 崩溃，降级到仅 `pytorch↔numpy` 对。
- **内存压力**：`--keep-samples 50` 已限流；若 session JSON > 50MB，进一步降到 20。
- **噪声 silent-diff**：`exp`、`logsumexp` 等天然大值算子的 silent diff 需人工判断是否属数值外溢（非缺陷）。
