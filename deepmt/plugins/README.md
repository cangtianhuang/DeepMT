# deepmt.plugins — 框架适配插件

本目录承载 DeepMT 的框架适配层。插件负责把 DeepMT 的算子测试工作负载
（由 `BatchTestRunner` / `CrossFrameworkTester` 驱动）翻译为具体框架的原生调用，
同时提供跨框架一致性比较所需的数值原语。

## 契约

所有插件继承 `framework_plugin.FrameworkPlugin`（ABC），必须实现以下原语：

| 原语 | 职责 |
| --- | --- |
| `_to_tensor(value)` | 任意 Python 值 → 框架原生张量 |
| `_execute_operator(func, inputs)` | 用位置参数调用算子函数 |
| `to_numpy(tensor)` | 框架张量 → `numpy.ndarray` |
| `get_shape(tensor)` | 读取形状，不得经过 `to_numpy` |
| `make_tensor(shape, dtype, value_range)` | 按确定参数生成随机张量 |
| `allclose(a, b, atol, rtol)` | 返回 `CompareResult`（含差值统计） |
| `eval_expr(expr, orig, trans, x)` | 在框架张量空间内对 oracle 子表达式求值 |
| `element_compare(a, b, op)` | 逐元素不等式比较 `!= < <= > >=` |

可选覆盖：

- `framework_name()` / `framework_version()` / `is_available()` / `supported_operators()`
- `_resolve_operator(name)`（基类已给出默认：`_overrides` + `_root_modules` 属性链）

**严禁**在插件里解析 `input_specs` / MR / YAML 字段。那类逻辑属于
`deepmt.analysis.*`，插件只做"完全确定参数 → 框架原生值"的翻译。

## 已登记插件

| name (FrameworkType) | 状态 | optional | 模块 |
| --- | --- | --- | --- |
| `pytorch` | ✅ 硬依赖 | false | `pytorch_plugin.PyTorchPlugin` |
| `numpy` | ✅ 硬依赖（作 float64 金标准） | false | `numpy_plugin.NumpyPlugin` |
| `paddlepaddle` | ✅ 完整实现 | true | `paddle_plugin.PaddlePlugin` |
| `tensorflow` | 🔶 Phase O MVP（一级 9 算子） | true | `tensorflow_plugin.TensorFlowPlugin` |
| `faulty_pytorch` | ✅ 完整缺陷目录 | — | `faulty_pytorch_plugin.FaultyPyTorchPlugin` |
| `faulty_tensorflow` | 🔶 骨架 | — | `faulty_tensorflow_plugin.FaultyTensorFlowPlugin` |

"optional" 表示框架运行时不存在时不会阻塞 `PluginsManager` 启动，健康检查会以
WARNING 呈现而非 ERROR。

登记表同时存在于两处并由 `deepmt health check --deep` 交叉校验：
- `plugins.yaml` — `PluginsManager` 实际加载使用
- `__init__.PLUGIN_REGISTRY` — 反射、健康检查与测试使用

## 新增插件步骤

1. 新建 `deepmt/plugins/<framework>_plugin.py`，继承 `FrameworkPlugin` 实现全部原语；
2. 使用懒加载模式处理框架依赖（参考 `paddle_plugin._PADDLE_AVAILABLE` / `_require_paddle`）；
3. 在 `plugins.yaml` 声明一行条目，在 `__init__.PLUGIN_REGISTRY` 追加 `PluginEntry`；
4. 运行 `deepmt health check --deep` 应看到契约检查条目；
5. 运行 `pytest tests/unit/test_plugin_contract.py tests/integration/test_plugin_parity.py`；
6. 在 `docs/dev/status.md` 登记版本快照。

## 相关文档

- `docs/dev/16_Phase_O_framework_plugin_closure_and_health.md` — Phase O 阶段规划与验收
- `docs/cli_reference.md#deepmt-health` — health CLI 用法
- `docs/dev/15_Phase_M_system_capability_gaps.md` — 插件层历史能力缺口清单
