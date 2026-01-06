"""
IR转换器：将用户输入自动转换为IR对象
对用户隐藏IR的创建细节
"""

from typing import Any, Dict, List, Optional, Union

from core.framework import FrameworkType
from core.logger import get_logger
from ir.schema import ApplicationIR, ModelIR, OperatorIR


class IRConverter:
    """IR转换器：将用户友好的输入转换为IR对象"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    @staticmethod
    def from_operator_name(
        name: str,
        inputs: List[Any],
        properties: Optional[Dict[str, Any]] = None,
        outputs: Optional[List[Any]] = None,
    ) -> OperatorIR:
        """
        从算子名称和输入创建OperatorIR

        Args:
            name: 算子名称（如 "Add", "Multiply"）
            inputs: 输入值列表
            properties: 算子属性（可选，会自动推断）
            outputs: 输出值列表（可选）

        Returns:
            OperatorIR对象
        """
        logger = get_logger(IRConverter.__name__)
        logger.debug(f"Creating OperatorIR from name: {name}")

        # 算子相关属性应该由后续的步骤推断，这里不再自动推断
        if properties is None:
            properties = {}

        return OperatorIR(
            name=name, inputs=inputs, outputs=outputs or [], properties=properties
        )

    @staticmethod
    def from_user_input(
        user_input: Dict[str, Any],
    ) -> Union[OperatorIR, ModelIR, ApplicationIR]:
        """
        从用户输入字典创建IR对象

        Args:
            user_input: 用户输入字典，格式如：
                {
                    "type": "operator",  # 或 "model", "application"
                    "name": "Add",
                    "inputs": [1.0, 2.0],
                    "properties": {...}  # 可选
                }

        Returns:
            IR对象（OperatorIR, ModelIR, 或 ApplicationIR）
        """
        ir_type = user_input.get("type", "operator")

        if ir_type == "operator":
            return IRConverter.from_operator_name(
                name=user_input["name"],
                inputs=user_input["inputs"],
                properties=user_input.get("properties"),
                outputs=user_input.get("outputs"),
            )
        elif ir_type == "model":
            return IRConverter.from_model_input(user_input)
        elif ir_type == "application":
            return IRConverter.from_application_input(user_input)
        else:
            raise ValueError(f"Unknown IR type: {ir_type}")

    @staticmethod
    def from_model_input(user_input: Dict[str, Any]) -> ModelIR:
        """从用户输入创建ModelIR"""
        return ModelIR(
            name=user_input.get("name", "UnknownModel"),
            layers=user_input.get("layers", []),
            connections=user_input.get("connections", []),
        )

    @staticmethod
    def from_application_input(user_input: Dict[str, Any]) -> ApplicationIR:
        """从用户输入创建ApplicationIR"""
        return ApplicationIR(
            name=user_input.get("name", "UnknownApp"),
            purpose=user_input.get("purpose", ""),
            input_format=user_input.get("input_format", ""),
            output_format=user_input.get("output_format", ""),
        )

    @staticmethod
    def from_framework_code(
        code: str, framework: FrameworkType = "pytorch"
    ) -> OperatorIR:
        """
        从框架代码创建IR（简化实现，完整版需要AST解析）

        Args:
            code: 框架代码字符串
            framework: 框架名称（"pytorch", "tensorflow", "paddlepaddle"）

        Returns:
            OperatorIR对象

        Raises:
            NotImplementedError: 功能尚未完全实现
        """
        logger = get_logger(IRConverter.__name__)
        logger.warning("from_framework_code is a simplified implementation")

        # TODO: 实现完整的AST解析
        # 当前只做简单示例
        if "torch.add" in code or "add" in code.lower():
            # 简单提取，实际需要AST解析
            return IRConverter.from_operator_name(
                name="Add",
                inputs=[1.0, 2.0],  # 需要从代码中提取
                properties={"commutative": True},
            )
        else:
            raise NotImplementedError("Framework code parsing not fully implemented")
