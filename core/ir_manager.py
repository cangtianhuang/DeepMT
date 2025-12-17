"""
IR管理器：负责IR的加载、保存、转换
"""

import json
import yaml
from pathlib import Path
from typing import Any, Optional, Union

from ir.schema import OperatorIR, ModelIR, ApplicationIR
from core.logger import get_logger


class IRManager:
    """统一中间表示管理器"""

    def __init__(self):
        self.logger = get_logger()
        self.ir_types = {
            "OperatorIR": OperatorIR,
            "ModelIR": ModelIR,
            "ApplicationIR": ApplicationIR,
        }

    def load_ir(self, path: Union[str, Path]) -> Any:
        """
        从JSON/YAML文件加载IR对象

        Args:
            path: 文件路径

        Returns:
            IR对象（OperatorIR, ModelIR, 或 ApplicationIR）
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"IR file not found: {path}")

        self.logger.info(f"Loading IR from {path}")

        # 读取文件
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        # 根据类型创建IR对象
        ir_type = data.get("type", "OperatorIR")
        if ir_type not in self.ir_types:
            raise ValueError(f"Unknown IR type: {ir_type}")

        ir_class = self.ir_types[ir_type]
        ir_data = data.get("data", data)

        try:
            ir_object = ir_class(**ir_data)
            self.logger.info(
                f"Successfully loaded {ir_type}: {ir_object.name if hasattr(ir_object, 'name') else 'unknown'}"
            )
            return ir_object
        except Exception as e:
            self.logger.error(f"Failed to create IR object: {e}")
            raise

    def save_ir(self, ir_object: Any, path: Union[str, Path], format: str = "json"):
        """
        保存IR对象到文件

        Args:
            ir_object: IR对象
            path: 保存路径
            format: 文件格式 ("json" 或 "yaml")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving IR to {path}")

        # 获取IR类型
        ir_type = type(ir_object).__name__

        # 转换为字典
        if isinstance(ir_object, (OperatorIR, ModelIR, ApplicationIR)):
            data = {"type": ir_type, "data": ir_object.__dict__}
        else:
            data = {"type": ir_type, "data": ir_object}

        # 保存文件
        with open(path, "w", encoding="utf-8") as f:
            if format == "yaml":
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Successfully saved {ir_type} to {path}")

    def create_ir_from_framework(self, model_code: str, framework: str = "pytorch"):
        """
        从框架代码生成IR（基础实现，后续可扩展）

        Args:
            model_code: 框架代码字符串
            framework: 框架名称

        Returns:
            IR对象
        """
        self.logger.warning(
            f"create_ir_from_framework is not fully implemented for {framework}"
        )
        # TODO: 实现框架代码解析逻辑
        # 这需要AST解析或使用框架特定的工具
        raise NotImplementedError("Framework code parsing not yet implemented")

    def validate_ir(self, ir_object: Any) -> bool:
        """
        验证IR对象的有效性

        Args:
            ir_object: IR对象

        Returns:
            是否有效
        """
        if not isinstance(ir_object, (OperatorIR, ModelIR, ApplicationIR)):
            self.logger.error(f"Invalid IR type: {type(ir_object)}")
            return False

        # 基本验证
        if hasattr(ir_object, "name") and not ir_object.name:
            self.logger.error("IR object missing name")
            return False

        self.logger.debug(f"IR validation passed: {type(ir_object).__name__}")
        return True
