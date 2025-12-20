"""
MR生成器基础模块
包含所有层（算子、模型、应用）的公共逻辑
"""

from mr_generator.base.mr_templates import MRTemplatePool
from mr_generator.base.knowledge_base import KnowledgeBase
from mr_generator.base.mr_repository import MRRepository

__all__ = ["MRTemplatePool", "KnowledgeBase", "MRRepository"]
