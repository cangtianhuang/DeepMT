"""
输入生成器：根据 input_specs 自动生成随机测试输入

职责：
  - 解析算子目录中定义的 input_specs 字段（dtype / shape / value_range）
  - 选择生成策略（当前：随机样本；后续可扩展边界值注入等）
  - 调用框架插件的基础接口（make_tensor）创建框架原生张量

框架插件只负责创建具体张量，解析和策略逻辑集中在此处。
"""

from typing import Any, Dict, List, Optional, Tuple

from deepmt.plugins.framework_plugin import FrameworkPlugin

# 默认兜底参数（算子目录中未定义 input_specs 时使用）
_DEFAULT_SHAPE: tuple = (4, 4)
_DEFAULT_DTYPE: str = "float32"


class InputGenerator:
    """
    根据 input_specs 生成随机输入列表，驱动 MR 数值预检与测试执行。

    用法示例：
        gen = InputGenerator()
        inputs = gen.generate(operator_ir.input_specs or [], plugin)
        # inputs: [tensor, ...]  每个元素对应一个必填参数
    """

    def generate(
        self,
        input_specs: List[Dict[str, Any]],
        plugin: FrameworkPlugin,
    ) -> List[Any]:
        """
        根据 input_specs 生成一组随机框架张量列表。

        Args:
            input_specs: 算子目录中定义的输入参数规范列表。
                格式：[{"name": str, "dtype": list, "shape": str|list,
                        "value_range": null|[lo, hi], "required": bool}, ...]
            plugin:      框架插件实例，用于调用 make_tensor 创建框架原生张量

        Returns:
            张量列表，每个元素对应一个必填参数；specs 为空时返回默认单张量列表
        """
        if not input_specs:
            return [plugin.make_tensor(_DEFAULT_SHAPE, _DEFAULT_DTYPE)]

        result = [
            self._generate_one(spec, plugin)
            for spec in input_specs
            if spec.get("required", True)
        ]
        return result or [plugin.make_tensor(_DEFAULT_SHAPE, _DEFAULT_DTYPE)]

    # ── 私有实现 ──────────────────────────────────────────────────────────────

    def _generate_one(self, spec: Dict[str, Any], plugin: FrameworkPlugin) -> Any:
        dtype = self._parse_dtype(spec.get("dtype", []))
        shape = self._parse_shape(spec.get("shape", "any"))
        value_range = self._parse_value_range(spec.get("value_range"))
        return plugin.make_tensor(shape, dtype, value_range)

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
