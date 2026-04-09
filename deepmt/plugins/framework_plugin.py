"""
框架插件抽象基类

所有框架适配插件（PyTorch、TensorFlow、PaddlePaddle）均继承此基类。

子类必须实现：
    _to_tensor(value)             — 将任意值转换为框架张量
    _execute_operator(func, inputs) — 调用算子函数并返回输出
    generate_random_inputs(input_specs) — 根据 input_specs 生成随机张量列表

子类可声明的类属性：
    _root_modules: list   — 框架根模块列表，供算子名称解析使用
                            例：[torch] for PyTorch, [tf] for TensorFlow
    _overrides: dict      — 算子名称的特殊映射，覆盖自动解析
                            例：{"torch.Tensor.add": torch.add}
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from deepmt.ir.schema import MetamorphicRelation, OperatorIR


@dataclass
class CompareResult:
    """
    张量比较的详细结果。

    Attributes:
        passed:               是否在给定容差内相等
        max_abs_diff:         最大绝对差值 max|a - b|
        max_rel_diff:         最大相对差值 max(|a - b| / max(|b|, eps))
        mismatched_elements:  超出容差的元素数；形状不匹配时为 0
        total_elements:       张量总元素数；形状不匹配时为 0
    """

    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    mismatched_elements: int
    total_elements: int

    @property
    def mismatched_ratio(self) -> float:
        """超出容差的元素占比；total_elements 为 0 时返回 0.0"""
        return self.mismatched_elements / self.total_elements if self.total_elements > 0 else 0.0


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

    @abstractmethod
    def to_numpy(self, tensor: Any) -> Any:
        """将框架张量转换为 numpy ndarray"""
        ...

    @abstractmethod
    def get_shape(self, tensor: Any) -> tuple:
        """返回张量形状（不应调用 to_numpy，应使用框架原生接口）"""
        ...

    @abstractmethod
    def make_tensor(
        self,
        shape: tuple,
        dtype: str,
        value_range: "tuple[float | None, float | None] | None" = None,
    ) -> Any:
        """
        创建一个具有给定形状、数据类型和值域约束的随机张量。

        这是框架适配器提供的基础接口——仅负责将完全确定的参数转换为框架原生张量，
        不包含任何格式解析或策略选择逻辑（那些属于上层 InputGenerator）。

        Args:
            shape:       张量形状，如 (4, 4)
            dtype:       数据类型字符串，如 "float32"、"int64"、"bool"
            value_range: (lo, hi) 值域约束；None 表示无限制；lo/hi 单独为 None 表示单侧无界

        Returns:
            框架原生张量
        """
        ...

    @abstractmethod
    def allclose(self, a: Any, b: Any, atol: float, rtol: float = 0.0) -> CompareResult:
        """
        比较两个张量，返回详细的差值统计。支持广播语义。

        Args:
            a, b:  待比较的张量（或可转换为张量的值）；形状广播兼容即可
            atol:  绝对容差
            rtol:  相对容差（默认 0.0）；判定条件：|a-b| <= atol + rtol*|b|

        Returns:
            CompareResult，含通过状态、最大绝对/相对差值、不匹配元素统计
        """
        ...

    @abstractmethod
    def eval_expr(self, expr: str, orig: Any, trans: Any, x: Any) -> Any:
        """
        委托计算接口：在框架张量空间内执行表达式，返回框架原生张量。

        expr 中可使用变量 orig / trans / x 及框架原生数学函数（abs、exp 等）。
        标量结果需包装为与 orig 同 dtype 的零维张量。
        不包含任何 MR / oracle 业务语义，仅作为计算代理。
        """
        ...

    @abstractmethod
    def element_compare(self, a: Any, b: Any, op: str) -> CompareResult:
        """
        逐元素不等式比较原语，返回完整统计。

        op: "!=" | "<" | "<=" | ">" | ">="
        支持框架原生广播语义（a/b 形状广播兼容即可）。
        """
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
            raw_inputs = ir_object.inputs or []
            orig_inputs = self._normalize_inputs(raw_inputs)
            orig_output = self._execute_operator(operator_func, orig_inputs)

            transformed_raw = mr.transform(*raw_inputs)
            trans_inputs = self._normalize_inputs(transformed_raw)
            trans_output = self._execute_operator(operator_func, trans_inputs)

            return orig_output, trans_output

        return run

    def execute(self, run_func: Callable) -> tuple:
        """执行 ir_to_code 返回的闭包"""
        return run_func()
