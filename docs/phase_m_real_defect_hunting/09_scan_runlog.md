# M2 — 扫描运行日志（待填写）

> 用户执行 `08_scan_todo.md` 过程中在此记录命中、异常、耗时。
> 每条记录一行即可：`[日期] 算子 框架对 n-samples 一致率 silent_rate 命中数 耗时 备注`

## STEP 1 — 回归基线

| 时间 | 命令 | session 路径 | consistency | silent_rate | [!] 告警 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| | log_softmax pt↔paddle n=500 | | | | | |
| | exp pt↔paddle n=500 | | | | | |

## STEP 2 — 一级全量扫描

| 算子 | (f1,f2) | consistency | silent_rate | inconsistent | exception | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| relu | pt↔np | | | | | |
| relu | pt↔paddle | | | | | |
| relu | paddle↔np | | | | | |
| tanh | pt↔np | | | | | |
| ... | | | | | | |

## STEP 3 — dedup 结果

- 命令：`deepmt test dedup --source cross`
- 产出 DefectLead 数：
- lead_id 清单：

## STEP 4/5 — 案例构建与人工确认

| case_id | operator | framework_pair | diff_type | 复现状态 | 确认状态 | 参考依据 |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

## 异常 / 降级记录

（记录任何 paddle 崩溃、超时、不得不降 n-samples 的情况）
