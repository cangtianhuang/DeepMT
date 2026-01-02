"""
框架管理模块：定义支持的深度学习框架类型
"""

from typing import List, Literal

# 支持的框架类型（用于类型提示）
FrameworkType = Literal["pytorch", "tensorflow", "paddlepaddle"]

# 支持的框架列表
SUPPORTED_FRAMEWORKS: List[str] = ["pytorch", "tensorflow", "paddlepaddle"]

# 框架别名映射（供需要时查询）
FRAMEWORK_ALIASES: dict[str, str] = {
    "pytorch": "pytorch",
    "torch": "pytorch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "paddlepaddle": "paddlepaddle",
    "paddle": "paddlepaddle",
}
