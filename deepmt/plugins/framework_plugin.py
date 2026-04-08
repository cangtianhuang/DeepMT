"""
框架插件抽象基类

所有框架适配插件（PyTorch、TensorFlow、PaddlePaddle）均继承此基类。

子类必须实现：
    _to_tensor(value)             — 将任意值转换为框架张量
    _execute_operator(func, inputs) — 调用算子函数并返回输出

子类可声明的类属性：
    _root_modules: list   — 框架根模块列表，供算子名称解析使用
                            例：[torch] for PyTorch, [tf] for TensorFlow
    _overrides: dict      — 算子名称的特殊映射，覆盖自动解析
                            例：{"torch.Tensor.add": torch.add}
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List

from deepmt.ir.schema import MetamorphicRelation, OperatorIR


class FrameworkPlugin(ABC):
    """框架适配插件抽象基类"""

    _root_modules: list = []
    _overrides: dict = {}

    # ── 抽象方法（子类必须实现） ──────────────────────────────────────────────

    @abstractmethod
    def _to_tensor(self, value: Any) -> Any:
        """将任意值转换为框架张量"""
        ...

    @abstractmethod
    def _execute_operator(self, func: Callable, inputs: list) -> Any:
        """以 inputs 为位置参数调用算子函数，返回输出"""
        ...

    # ── 具体方法（子类通常无需覆盖） ─────────────────────────────────────────

    def _resolve_operator(self, name: str) -> Callable:
        """
        将算子名称（如 "torch.nn.functional.relu"）解析为可调用对象。

        解析顺序：
          1. 检查 _overrides 字典（优先，用于特殊情况）
          2. 遍历 _root_modules，按属性链逐级查找

        若解析失败，抛出 ValueError 并提示将算子加入 _overrides。
        """
        if name in self._overrides:
            return self._overrides[name]

        parts = name.split(".")
        for root in self._root_modules:
            if root.__name__ != parts[0]:
                continue
            try:
                obj = root
                for attr in parts[1:]:
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue

        raise ValueError(
            f"Cannot resolve operator '{name}' via {[m.__name__ for m in self._root_modules]}. "
            f"Add it to {type(self).__name__}._overrides if the name differs from the import path."
        )

    def _normalize_inputs(self, inputs: Any) -> List[Any]:
        """
        将输入归一化为张量列表。

        - 若 inputs 不是 tuple/list，包装为单元素序列
        - 逐元素调用 _to_tensor 转换
        """
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        return [self._to_tensor(v) for v in inputs]

    def ir_to_code(self, ir_object: OperatorIR, mr: MetamorphicRelation) -> Callable:
        """
        将 IR 和 MR 组合为可执行的闭包。

        执行流程：
          1. 解析算子函数
          2. 归一化原始输入并执行算子，得到 orig_output
          3. 应用 MR 变换得到变换后输入
          4. 归一化变换后输入并执行算子，得到 trans_output
          5. 返回 (orig_output, trans_output)
        """
        operator_func = self._resolve_operator(ir_object.name)

        def run():
            orig_inputs = self._normalize_inputs(ir_object.inputs)
            orig_output = self._execute_operator(operator_func, orig_inputs)

            transformed_raw = mr.transform(*ir_object.inputs)
            trans_inputs = self._normalize_inputs(transformed_raw)
            trans_output = self._execute_operator(operator_func, trans_inputs)

            return orig_output, trans_output

        return run

    def execute(self, run_func: Callable) -> tuple:
        """执行 ir_to_code 返回的闭包"""
        return run_func()
