# M6 — 真实案例目录

> 汇总 `data/cases/real_defects/` 中所有案例；Phase M 扫描完成后由用户逐行填写/更新。
> 为论文 §5.3.2 / §5.4 / 第 6 章引用提供单点入口。

## 案例总表

| case_id | 算子 | 框架 / 框架对 | kind | severity | status | 现象摘要 | 论文引用位 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 009eb89bcb | gelu | pytorch | mr_quality | low | confirmed | gelu MR 候选质量边界 | §5.3.2 |
| e861263744 | exp | pytorch | numerical_precision | low | confirmed | exp float32 溢出边界 | §5.3.2 |
| fc036b860c | torch.cosh | pytorch (faulty inject) | numerical | medium | L3-acceptance | 受控注入：cosh 偶函数破坏 | 附录 A |
| b382388f68 | torch.asinh | pytorch (faulty inject) | other | medium | L3-acceptance | 受控注入：asinh 单调性破坏 | 附录 A |
| 46529a4bcc | torch.expm1 | pytorch (faulty inject) | other | medium | L3-acceptance | 受控注入：expm1 单调性破坏 | 附录 A |
| 31b99d6745 | gelu | pytorch | unknown | low | draft | （历史草案） | — |
| 0e0cb9946d | exp | pt↔paddle | cross_framework_divergence | medium | draft | silent numeric diff | — |

> L3-acceptance 行是系统闭环演示用的受控注入案例，不作真实缺陷计入论文主文。

## 真实扫描产出（本次 Phase M 新增）

| case_id | 算子 | 框架对 | 一句话现象 | 复现确认 | 论文位 |
| --- | --- | --- | --- | --- | --- |
| _待扫描后填写_ | | | | | |

## 引用规范

- 论文正文引用格式：`[Case <case_id>]`，在附录给出完整 `case_summary.md` 链接。
- 每个引用案例必须满足：`metadata.status == confirmed` 且 `reproduce.py` 可独立运行。
- 未确认案例仅可进入附录"待确认线索"小节，不进入主文。
