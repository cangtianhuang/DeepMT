"""
算子层MR生成器：基于知识库生成算子的蜕变关系
"""

from typing import List, Optional

from ir.schema import OperatorIR, MetamorphicRelation
from mr_generator.knowledge_base import KnowledgeBase
from core.logger import get_logger


class OperatorMRGenerator:
    """算子层MR生成器"""

    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        """
        初始化算子MR生成器

        Args:
            knowledge_base: 知识库实例，如果为None则创建默认实例
        """
        self.kb = knowledge_base if knowledge_base else KnowledgeBase()
        self.logger = get_logger()

    def generate(self, operator_ir: OperatorIR) -> List[MetamorphicRelation]:
        """
        为算子IR生成蜕变关系

        Args:
            operator_ir: 算子IR对象

        Returns:
            MR对象列表
        """
        self.logger.info(f"Generating MRs for operator: {operator_ir.name}")

        # 从知识库获取MR生成函数
        mr_functions = self.kb.get_mrs_for_operator(operator_ir.name)

        mrs = []
        for mr_func in mr_functions:
            try:
                # 调用MR生成函数，传入算子输入
                mr_obj = mr_func(operator_ir.inputs)

                # 验证MR对象
                if isinstance(mr_obj, MetamorphicRelation):
                    mrs.append(mr_obj)
                    self.logger.debug(f"Generated MR: {mr_obj.description}")
                else:
                    self.logger.warning(
                        f"MR function returned invalid object: {type(mr_obj)}"
                    )

            except Exception as e:
                self.logger.error(f"Error generating MR with {mr_func.__name__}: {e}")

        self.logger.info(f"Generated {len(mrs)} MRs for {operator_ir.name}")
        return mrs
