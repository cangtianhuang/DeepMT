# L2 验收报告（CLI 闭环能力）

日期: 2026-04-15

## L2.1 跨框架证据 → `case build --from-evidence` 产出案例包

**命令**
```
deepmt test cross exp --framework1 pytorch --framework2 paddlepaddle --n-samples 200 --save-to-evidence
deepmt case build --from-evidence 6a23b2c5-4d0
```

**结果**: ✅ 通过

- `--save-to-evidence` 将 `CrossSessionResult` 的 150 个 `failed_samples` 导出为 `kind=cross_framework_divergence` 的 EvidencePack
- `deepmt case build --from-evidence 6a23b2c5-4d0` 成功生成 `data/cases/real_defects/0e0cb9946d/`，包含 `reproduce.py / evidence.json / metadata.json / case_summary.md`
- `defect_case_builder._get_reproduce_script` 已支持 cross 分支，调用 `_generate_cross_reproduce_script`

## L2.2 `reproduce.py` 独立运行复现 diff_type

**命令**
```
python data/cases/real_defects/0e0cb9946d/reproduce.py
```

**输出**
```
framework1=pytorch framework2=paddlepaddle
max_abs_diff = 0.5
mean_abs_diff = 0.0313873
diff_type(recorded) = numeric_diff
numeric_diff(recorded) = 0.5
```

**结果**: ✅ 通过

- 脚本干净 Python 环境一次运行成功
- 实测 max_abs_diff 与 session 记录的 numeric_diff=0.5 完全一致，且 diff_type=numeric_diff 与记录一致

## L2.3 `test dedup --source cross` 聚类输出 DefectLead

**命令**
```
deepmt test dedup --source cross --operator exp
```

**结果**: ✅ 通过

- 150 条跨框架证据包 → 3 条独立 DefectLead（3 条 MR × 1 个 framework_pair × 1 个 diff_type）
- lead_id 按 `(operator, mr_id, framework_pair=pytorch__paddlepaddle, bucket)` 签名生成，符合 G8 设计

## L2.4 silent-numeric-diff 告警

**命令**
```
deepmt test cross exp --framework1 pytorch --framework2 paddlepaddle --n-samples 200
```

**输出**
```
MR 数: 3  整体一致率: 100.0%  输出最大差: 1.342e+08
[!] 静默数值差异: 363 样本  (rate=60.5%) — 两框架 oracle 结论一致但输出数值存在显著差异
```

**结果**: ✅ 通过

- 阈值 `SILENT_DIFF_WARN_RATE = 0.05`；363/600 ≫ 5%，触发 `[!]` 告警，终端显眼提示
- 告警同步持久化为 `CrossSessionResult.silent_numeric_diff_count / silent_numeric_diff_rate` 字段

## 小结

L2 四项全部通过。修复记录：
- 新增 `_generate_cross_reproduce_script` 中对 pytorch/paddle/numpy 三框架真实调用代码（原版仅占位注释，导致 reproduce.py 不可直接执行）
- 修正 f-string 占位符引号与变量转义问题
- `defect_case_builder` 增加 `kind==cross_framework_divergence` 分支
