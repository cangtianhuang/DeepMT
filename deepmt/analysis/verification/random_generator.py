"""
随机数据生成器：根据 input_specs 自动生成随机张量数据

职责：
  - 解析算子目录中定义的 input_specs 字段（dtype / shape / value_range）
  - 选择生成策略（随机样本 + 可配置概率的边界值注入）
  - 调用被测框架后端的基础接口（make_tensor）创建框架原生张量

框架后端只负责创建具体张量，解析和策略逻辑集中在此处。
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deepmt.plugins.framework_plugin import FrameworkPlugin

# 默认兜底参数（算子目录中未定义 input_specs 时使用）
_DEFAULT_SHAPE: tuple = (4, 4)
_DEFAULT_DTYPE: str = "float32"

# 各 dtype 对应的极值集合（含 ±inf 和 nan）
_BOUNDARY_VALUES: Dict[str, List[float]] = {
    "float32": [float("inf"), float("-inf"), float("nan"), 3.4028235e+38, -3.4028235e+38, 0.0],
    "float64": [float("inf"), float("-inf"), float("nan"), 1.7976931348623157e+308, -1.7976931348623157e+308, 0.0],
    "float16": [float("inf"), float("-inf"), float("nan"), 65504.0, -65504.0, 0.0],
    "int32":   [2147483647, -2147483648, 0],
    "int64":   [9223372036854775807, -9223372036854775808, 0],
    "int8":    [127, -128, 0],
    "uint8":   [255, 0],
}
_DEFAULT_BOUNDARY_VALUES = [float("inf"), float("-inf"), float("nan"), 0.0]


class RandomGenerator:
    """
    根据 input_specs 生成随机张量列表，支持边界值注入。

    Args:
        boundary_injection_prob: 对单个张量注入极值的概率（默认 0.15）。
            设为 0.0 可完全禁用边界值注入（仅生成均匀随机张量）。
        seed: 随机种子；None 表示不固定。

    用法示例：
        gen = RandomGenerator()
        inputs = gen.generate(operator_ir.input_specs or [], backend)
        # inputs: [tensor, ...]  每个元素对应一个必填参数
    """

    def __init__(
        self,
        boundary_injection_prob: float = 0.15,
        seed: Optional[int] = None,
    ):
        self.boundary_injection_prob = boundary_injection_prob
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(
        self,
        input_specs: List[Dict[str, Any]],
        backend: FrameworkPlugin,
    ) -> List[Any]:
        """
        根据 input_specs 生成一组随机框架张量列表。

        Args:
            input_specs: 算子目录中定义的输入参数规范列表。
                格式：[{"name": str, "dtype": list, "shape": str|list,
                        "value_range": null|[lo, hi], "required": bool}, ...]
            backend:     被测框架的计算后端，用于调用 make_tensor 创建框架原生张量

        Returns:
            张量列表，每个元素对应一个必填参数；specs 为空时返回默认单张量列表
        """
        if not input_specs:
            return [backend.make_tensor(_DEFAULT_SHAPE, _DEFAULT_DTYPE)]

        result = [
            self._generate_one(spec, backend)
            for spec in input_specs
            if spec.get("required", True)
        ]
        return result or [backend.make_tensor(_DEFAULT_SHAPE, _DEFAULT_DTYPE)]

    # ── 私有实现 ──────────────────────────────────────────────────────────────

    def _generate_one(self, spec: Dict[str, Any], backend: FrameworkPlugin) -> Any:
        dtype = self._parse_dtype(spec.get("dtype", []))
        shape = self._parse_shape(spec.get("shape", "any"))
        value_range = self._parse_value_range(spec.get("value_range"))

        # 按概率决定是否注入边界值
        if (
            self.boundary_injection_prob > 0
            and "int" not in dtype  # 整数张量不注入浮点极值
            and random.random() < self.boundary_injection_prob
        ):
            return self._inject_boundary(shape, dtype, backend)

        return backend.make_tensor(shape, dtype, value_range)

    def _inject_boundary(self, shape: tuple, dtype: str, backend: FrameworkPlugin) -> Any:
        """创建一个以极值填充的张量，极值类型从该 dtype 的候选集中随机选取。"""
        boundary_vals = _BOUNDARY_VALUES.get(dtype, _DEFAULT_BOUNDARY_VALUES)
        chosen = random.choice(boundary_vals)
        # 构造全部填充 chosen 值的 numpy 数组，再通过 backend 转换
        arr = np.full(shape, chosen, dtype=np.float64)
        try:
            # 通过 make_tensor + value_range=(chosen, chosen) 的方式注入
            # 回退：直接用 numpy 数组创建张量（需 backend 支持）
            return backend.make_tensor(shape, dtype, (chosen, chosen))
        except Exception:
            # 若 backend 不支持极值 range，退回普通随机张量
            return backend.make_tensor(shape, dtype, None)

    def _parse_dtype(self, dtype_spec) -> str:
        """将 dtype 字段（字符串或列表）解析为统一 dtype 字符串。"""
        if not dtype_spec:
            return _DEFAULT_DTYPE
        name = dtype_spec[0] if isinstance(dtype_spec, list) else str(dtype_spec)
        return str(name)

    def _parse_shape(self, shape_spec) -> tuple:
        """
        将 shape 字段解析为具体形状元组。

        支持格式：
          "any"        → (4, 4)（默认形状）
          "nd>=N"      → (4,) * N  （至少 N 维）
          "(n,)"       → (4,)
          "(n, m)"     → (4, 4)
          [4, 8]       → (4, 8)    （已经是列表/元组）
        """
        if not shape_spec or shape_spec == "any":
            return _DEFAULT_SHAPE
        if isinstance(shape_spec, (list, tuple)):
            return tuple(
                int(d) if not isinstance(d, str) or str(d).isdigit() else 4
                for d in shape_spec
            )
        s = str(shape_spec).strip()
        if s.startswith("nd>="):
            n = max(int(s[4:]), 1)
            return (4,) * n
        parts = [p.strip() for p in s.strip("()").split(",") if p.strip()]
        if parts:
            shape = []
            for p in parts:
                try:
                    shape.append(int(p))
                except ValueError:
                    shape.append(4)
            return tuple(shape)
        return _DEFAULT_SHAPE

    def _parse_value_range(
        self, vr
    ) -> Optional[Tuple[Optional[float], Optional[float]]]:
        """将 value_range 字段解析为 (lo, hi) 元组，null 返回 None。"""
        if not vr:
            return None
        lo = float(vr[0]) if vr[0] is not None else None
        hi = float(vr[1]) if vr[1] is not None else None
        return (lo, hi)
