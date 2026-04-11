"""
MR 生成器模块。

当前仅算子层（OperatorMRGenerator）完整实现。
模型层与应用层为占位 stub，尚未实现，不对外导出。
"""

from deepmt.mr_generator.operator import OperatorMRGenerator

__all__ = ["OperatorMRGenerator"]
