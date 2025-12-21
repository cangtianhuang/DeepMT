"""
SymPy证明引擎：使用SymPy进行形式化证明
使用 code_to_sympy 动态将 Python 代码（可能是复杂的 Python 代码）转化为 sympy 表达
然后使用 simplify(LHS - RHS) == 0 进行证明
"""

import sympy as sp
from typing import Callable, Any, List, Optional

from ir.schema import MetamorphicRelation
from core.logger import get_logger
from mr_generator.operator.code_translator import CodeToSymPyTranslator


class SymPyProver:
    """SymPy形式化证明引擎：使用动态代码转换"""

    def __init__(self, code_translator: Optional[CodeToSymPyTranslator] = None):
        """
        初始化SymPy证明引擎

        Args:
            code_translator: 代码转换器（如果为None则创建默认实例）
        """
        self.logger = get_logger()
        self.code_translator = code_translator or CodeToSymPyTranslator()

    def code_to_sympy(
        self,
        code: Optional[str] = None,
        func: Optional[Callable] = None,
        doc: Optional[str] = None,
        signature: Optional[str] = None,
        num_inputs: Optional[int] = None,
    ) -> Optional[sp.Expr]:
        """
        动态将 Python 代码（可能是复杂的 Python 代码）转化为 sympy 表达

        Args:
            code: Python代码字符串
            func: Python函数对象
            doc: 函数文档字符串
            signature: 函数签名字符串
            num_inputs: 输入数量（用于创建符号变量）

        Returns:
            SymPy表达式，如果转换失败则返回None
        """
        # 使用 CodeToSymPyTranslator 进行动态转换
        sympy_expr = self.code_translator.translate(
            code=code,
            func=func,
            doc=doc,
            signature=signature,
        )

        if sympy_expr is None:
            self.logger.warning("Failed to convert code to SymPy expression")

        return sympy_expr

    def prove_mr(
        self,
        mr: MetamorphicRelation,
        operator_func: Optional[Callable] = None,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        operator_name: Optional[str] = None,
        num_inputs: Optional[int] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        使用SymPy证明MR（动态代码转换）

        Args:
            mr: 蜕变关系
            operator_func: 算子函数对象
            operator_code: 算子代码字符串
            operator_doc: 算子文档字符串
            operator_name: 算子名称（可选，用于日志）
            num_inputs: 输入数量

        Returns:
            (是否证明成功, 错误信息)
        """
        try:
            # 动态创建SymPy表达式
            sympy_expr = self.code_to_sympy(
                code=operator_code,
                func=operator_func,
                doc=operator_doc,
                num_inputs=num_inputs,
            )
            if sympy_expr is None:
                error_msg = (
                    f"Cannot convert operator '{operator_name}' to SymPy expression"
                    if operator_name
                    else "Cannot convert operator to SymPy expression"
                )
                if operator_name:
                    self.logger.warning(error_msg)
                return (False, error_msg)

            # 推断输入数量（如果未提供）
            if num_inputs is None:
                # 尝试从sympy表达式推断
                free_symbols = sympy_expr.free_symbols
                num_inputs = len(free_symbols) if free_symbols else 1

            # 创建符号变量
            symbols = [sp.Symbol(f"x{i}") for i in range(num_inputs)]

            # 构建LHS（原始表达式）
            lhs = sympy_expr

            # 构建RHS（变换后的表达式）
            # 应用MR变换到符号变量
            transformed_symbols = mr.transform(*symbols)
            if not isinstance(transformed_symbols, (tuple, list)):
                transformed_symbols = (transformed_symbols,)

            # 重新创建表达式，使用变换后的符号
            # 构建替换字典
            subs_dict = {}
            for i in range(min(len(symbols), len(transformed_symbols))):
                subs_dict[symbols[i]] = transformed_symbols[i]

            if subs_dict:
                rhs = sympy_expr.subs(subs_dict)
            else:
                rhs = sympy_expr

            # 根据期望关系类型进行证明
            diff = None
            is_proven = False

            if mr.expected == "equal":
                # 证明 LHS == RHS，即 simplify(LHS - RHS) == 0
                diff = sp.simplify(lhs - rhs)
                is_proven = diff == 0

            elif mr.expected == "proportional":
                # 证明 LHS == k * RHS 或 LHS == -RHS
                # 检查 LHS + RHS == 0 (负比例)
                diff1 = sp.simplify(lhs + rhs)
                # 检查 LHS / RHS 是常数（正比例）
                try:
                    ratio = sp.simplify(lhs / rhs)
                    # 如果ratio是常数且不等于0，则成比例
                    is_proven = (diff1 == 0) or (ratio.is_constant())
                    diff = diff1 if diff1 != 0 else ratio
                except:
                    is_proven = diff1 == 0
                    diff = diff1

            else:
                # 其他关系类型，默认使用相等检查
                diff = sp.simplify(lhs - rhs)
                is_proven = diff == 0

            if is_proven:
                self.logger.debug(f"MR proven: {mr.description}")
                return True, None
            else:
                error_msg = (
                    f"SymPy proof failed: {diff if diff is not None else 'unknown'}"
                )
                return False, error_msg

        except Exception as e:
            error_msg = f"SymPy proof error: {str(e)}"
            self.logger.debug(f"SymPy proof exception: {error_msg}")
            return False, error_msg

    def prove_mrs(
        self,
        mrs: List[MetamorphicRelation],
        operator_func: Optional[Callable] = None,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        operator_name: Optional[str] = None,
        num_inputs: Optional[int] = None,
    ) -> List[MetamorphicRelation]:
        """
        证明MR列表，返回经过证明的MR（动态代码转换）

        Args:
            mrs: MR列表
            operator_func: 算子函数对象
            operator_code: 算子代码字符串
            operator_doc: 算子文档字符串
            operator_name: 算子名称（可选，用于日志）
            num_inputs: 输入数量

        Returns:
            经过证明的MR列表
        """
        proven_mrs = []

        self.logger.info(f"Proving {len(mrs)} MRs using SymPy...")

        for i, mr in enumerate(mrs):
            is_proven, error_msg = self.prove_mr(
                mr=mr,
                operator_func=operator_func,
                operator_code=operator_code,
                operator_doc=operator_doc,
                operator_name=operator_name,
                num_inputs=num_inputs,
            )

            if is_proven:
                proven_mrs.append(mr)
                self.logger.debug(f"MR {i+1} proven: {mr.description}")
            else:
                self.logger.debug(
                    f"MR {i+1} proof failed: {mr.description}. " f"Reason: {error_msg}"
                )

        self.logger.info(
            f"SymPy proof completed: {len(proven_mrs)}/{len(mrs)} MRs proven"
        )

        return proven_mrs
