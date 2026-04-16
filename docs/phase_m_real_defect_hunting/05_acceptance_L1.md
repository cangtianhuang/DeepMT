# L1 验收报告（基础能力打通）

日期: 2026-04-15

## L1.1 `deepmt test cross --operator log_softmax` 不报错

**命令**
```
deepmt test cross --operator log_softmax --framework1 pytorch --framework2 paddlepaddle --n-samples 10
```

**结果**: ✅ 通过

- 加载 2 条 MR，consistency=100%
- numeric_diff_max ≈ 9.54e-07（低于默认 1e-5 容差）
- 无异常；修复前 G1（paddle 无 log_softmax lambda）已通过 `paddle_plugin._build_paddle_operators` 扩展修复。

## L1.2 `deepmt test cross --operator gelu --framework2 numpy` 返回数值对比

**命令**
```
deepmt test cross --operator gelu --framework1 pytorch --framework2 numpy --n-samples 10
```

**结果**: ✅ 通过

- 加载 3 条 MR；2 条数值比对 100% 一致；session 文件落盘到 `data/results/cross_framework/`
- 修复前 G2（numpy 无 gelu/erf/log_softmax 等）已通过 `numpy_plugin._np_gelu / _np_erf / _np_log_softmax / _np_logsumexp / _np_layer_norm` 解决。

## L1.3 `deepmt test batch --framework paddlepaddle --operator relu` 正常跑通并落盘证据

**命令**
```
deepmt test batch --framework paddlepaddle --operator relu --n-samples 5 --collect-evidence
```

**结果**: ✅ 通过

- `_SUPPORTED_FRAMEWORKS` 已加入 `paddlepaddle`（原 G3 修复）
- 历史 MR YAML 的 `applicable_frameworks` 仅含 `pytorch`，扩展为 `[pytorch, paddlepaddle]`（涉及 9 个算子：relu/tanh/exp/abs/sigmoid/gelu/softmax/log_softmax/leaky_relu），均为 paddle 插件已实现的等价算子
- 运行输出：加载 3 条 MR，15/15 通过，无错误；证据包机制可用（无失败故无包落盘）

## L1.4 `CrossSessionResult` JSON 保留失败样本

**验证对象**: `data/results/cross_framework/` 下的 session JSON

**结果**: ✅ 通过

- `CrossConsistencyResult` 新增 `failed_samples: List[Dict]` 字段（T2）
- `_record_sample` 在 `_compare_single_mr` 每次样本失配时写入 `{seed, input_summary, values, numeric_diff, diff_type, f2_output_summary}`
- 经抽样 gelu session 验证：50 条 failed_samples 全部包含 shape=[4,4] 的完整浮点数矩阵（含 `values` 字段），满足复现需要
- `keep_samples`（默认 50）CLI 可控，防止过度膨胀

## 小结

L1 四项全部通过。
