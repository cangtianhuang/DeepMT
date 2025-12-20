"""
算子知识库：向后兼容接口
实际实现已迁移至 mr_generator/base/knowledge_base.py
"""

# 向后兼容：从新位置导入
from mr_generator.base.knowledge_base import KnowledgeBase

__all__ = ["KnowledgeBase"]
