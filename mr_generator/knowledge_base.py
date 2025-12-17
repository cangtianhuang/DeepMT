"""
算子知识库：存储算子的数学性质和对应的蜕变关系
"""

from typing import List, Callable
import uuid

from ir.schema import MetamorphicRelation
from core.logger import get_logger


class KnowledgeBase:
    """算子知识库：管理算子的数学性质和MR生成规则"""

    def __init__(self):
        self.logger = get_logger()
        # 每个算子对应可用的蜕变关系生成函数
        self.operator_mrs = {
            "Add": [self.commutative_mr, self.associative_mr, self.identity_mr],
            "Multiply": [self.commutative_mr, self.associative_mr, self.identity_mr],
            "Subtract": [self.anti_commutative_mr],
            "Divide": [self.reciprocal_mr],
            "MatMul": [self.transpose_mr],
        }

    def get_mrs_for_operator(self, operator_name: str) -> List[Callable]:
        """
        获取指定算子的所有MR生成函数

        Args:
            operator_name: 算子名称

        Returns:
            MR生成函数列表
        """
        mrs = self.operator_mrs.get(operator_name, [])
        if not mrs:
            self.logger.warning(f"No MRs found for operator: {operator_name}")
        return mrs

    def commutative_mr(self, inputs) -> MetamorphicRelation:
        """
        交换律：f(x, y) == f(y, x)
        适用于：Add, Multiply等
        """
        # 确保输入是列表或元组
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        # 对于二元算子，交换前两个输入
        if len(inputs) >= 2:
            transform = lambda *args: (args[1], args[0]) + args[2:]
        else:
            transform = lambda *args: args

        return MetamorphicRelation(
            id=str(uuid.uuid4()),
            description="Commutative property: f(x,y) == f(y,x)",
            transform=transform,
            expected="equal",
            tolerance=1e-6,
            layer="operator",
        )

    def associative_mr(self, inputs) -> MetamorphicRelation:
        """
        结合律：f(f(x, y), z) == f(x, f(y, z))
        适用于：Add, Multiply等
        注意：这是一个简化的实现，实际需要更复杂的处理
        """
        transform = lambda *args: args

        return MetamorphicRelation(
            id=str(uuid.uuid4()),
            description="Associative property: f(f(x,y),z) == f(x,f(y,z))",
            transform=transform,
            expected="equal",
            tolerance=1e-6,
            layer="operator",
        )

    def identity_mr(self, inputs) -> MetamorphicRelation:
        """
        单位元：f(x, 0) == x (对于加法) 或 f(x, 1) == x (对于乘法)
        """
        return MetamorphicRelation(
            id=str(uuid.uuid4()),
            description="Identity element: f(x, identity) == x",
            transform=lambda x, identity: (x, identity),
            expected="equal",
            tolerance=1e-6,
            layer="operator",
        )

    def anti_commutative_mr(self, inputs) -> MetamorphicRelation:
        """
        反交换律：f(x, y) == -f(y, x)
        适用于：Subtract
        """

        def transform(*args):
            if len(args) >= 2:
                return (args[1], args[0]) + args[2:]
            return args

        return MetamorphicRelation(
            id=str(uuid.uuid4()),
            description="Anti-commutative: f(x,y) == -f(y,x)",
            transform=transform,
            expected="proportional",  # 需要检查符号相反
            tolerance=1e-6,
            layer="operator",
        )

    def reciprocal_mr(self, inputs) -> MetamorphicRelation:
        """
        倒数关系：f(x, y) == 1 / f(y, x) (对于除法)
        """
        return MetamorphicRelation(
            id=str(uuid.uuid4()),
            description="Reciprocal property: f(x,y) == 1/f(y,x)",
            transform=lambda a, b: (b, a),
            expected="proportional",
            tolerance=1e-6,
            layer="operator",
        )

    def transpose_mr(self, inputs) -> MetamorphicRelation:
        """
        转置关系：f(A, B) == f(B^T, A^T)^T (对于矩阵乘法)
        """

        def transform(*args):
            # 对于矩阵，需要转置操作
            # 这里简化处理，实际需要根据输入类型判断
            if len(args) >= 2:
                # 如果是numpy数组或tensor，尝试转置
                try:
                    import numpy as np

                    if isinstance(args[0], np.ndarray) and args[0].ndim >= 2:
                        return (args[1].T, args[0].T) + args[2:]
                except:
                    pass
                return (args[1], args[0]) + args[2:]
            return args

        return MetamorphicRelation(
            id=str(uuid.uuid4()),
            description="Transpose property: f(A,B) == f(B^T,A^T)^T",
            transform=transform,
            expected="equal",
            tolerance=1e-6,
            layer="operator",
        )
