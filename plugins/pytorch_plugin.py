"""
PyTorch框架适配插件
"""

import torch
from typing import Any, Callable

from ir.schema import OperatorIR, MetamorphicRelation
from core.logger import get_logger


class PyTorchPlugin:
    """PyTorch框架适配插件"""

    def __init__(self):
        self.logger = get_logger()
        # 算子名称到PyTorch函数的映射
        self.operator_map = {
            "Add": torch.add,
            "Multiply": torch.mul,
            "Subtract": torch.sub,
            "Divide": torch.div,
            "MatMul": torch.matmul,
            "Pow": torch.pow,
            "Sum": torch.sum,
            "Mean": torch.mean,
        }

    def ir_to_code(self, ir_object: OperatorIR, mr: MetamorphicRelation) -> Callable:
        """
        将IR和MR映射为Python可执行函数

        Args:
            ir_object: 算子IR对象
            mr: 蜕变关系对象

        Returns:
            可执行的函数，返回(原始输出, 变换后输出)
        """
        operator_name = ir_object.name
        operator_func = self.operator_map.get(operator_name)

        if operator_func is None:
            raise ValueError(f"Unsupported operator: {operator_name}")

        def run():
            """执行算子并应用MR变换"""
            try:
                # 转换输入为tensor
                inputs = [self._to_tensor(inp) for inp in ir_object.inputs]

                # 原始执行
                orig_output = self._execute_operator(operator_func, inputs)

                # 应用MR变换
                transformed_inputs = mr.transform(*ir_object.inputs)

                # 确保transformed_inputs是元组或列表
                if not isinstance(transformed_inputs, (tuple, list)):
                    transformed_inputs = (transformed_inputs,)

                # 转换变换后的输入为tensor
                trans_inputs = [self._to_tensor(inp) for inp in transformed_inputs]

                # 执行变换后的算子
                mr_output = self._execute_operator(operator_func, trans_inputs)

                return orig_output, mr_output

            except Exception as e:
                self.logger.error(f"Error in run function: {e}")
                raise

        return run

    def _to_tensor(self, value: Any) -> torch.Tensor:
        """将值转换为PyTorch tensor"""
        if isinstance(value, torch.Tensor):
            return value
        elif isinstance(value, (list, tuple)):
            return torch.tensor(value, dtype=torch.float32)
        elif isinstance(value, (int, float)):
            return torch.tensor(value, dtype=torch.float32)
        else:
            return torch.tensor(value)

    def _execute_operator(self, operator_func: Callable, inputs: list) -> torch.Tensor:
        """执行算子函数"""
        if len(inputs) == 1:
            return operator_func(inputs[0])
        elif len(inputs) == 2:
            return operator_func(inputs[0], inputs[1])
        else:
            # 对于多输入算子，使用第一个输入作为主要参数
            return operator_func(inputs[0], *inputs[1:])

    def execute(self, run_func: Callable) -> tuple:
        """
        执行传入的函数并返回结果

        Args:
            run_func: 可执行函数

        Returns:
            (原始输出, 变换后输出) 元组
        """
        try:
            return run_func()
        except Exception as e:
            self.logger.error(f"Error executing function: {e}")
            raise
