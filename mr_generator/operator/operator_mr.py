"""
算子层MR生成器：多源融合自动生成引擎

生成流程：
1. 信息准备：自动获取算子信息
2. MR猜想生成：LLM猜想 + 模板池
3. 快速筛选：Pre-check
4. 形式化验证：SymPy证明
"""

from typing import Callable, Dict, List, Literal, Optional, Set

from core.framework import FrameworkType
from core.logger import get_logger
from ir.schema import MetamorphicRelation, OperatorIR
from mr_generator.base.knowledge_base import KnowledgeBase
from mr_generator.base.mr_templates import MRTemplatePool
from mr_generator.operator.code_translator import CodeToSymPyTranslator
from mr_generator.operator.mr_precheck import MRPreChecker
from mr_generator.operator.operator_llm_mr_generator import OperatorLLMMRGenerator
from mr_generator.operator.sympy_prover import SymPyProver
from tools.llm.client import LLMClient
from tools.web_search.operator_fetcher import OperatorInfoFetcher

# MR生成来源类型
MRSource = Literal["llm", "template"]


class OperatorMRGenerator:
    """
    算子层MR生成器：多源融合自动生成引擎


    执行流程：
    1. 信息准备阶段
       - 自动获取算子信息（网络搜索文档/代码）
       - 提取算子代码（来自operator_code参数或网络）

    2. 多源MR生成阶段
       - LLM猜想 → 生成候选MR（主要来源）
       - 模板池猜测 → 生成候选MR（辅助来源）
       → 合并去重所有生成的MR

    3. 快速筛选阶段（Pre-check，可选）
       - 要求：需提供operator_func参数
       - 用随机输入执行算子验证候选MR

    4. SymPy形式化证明阶段（可选）
       - 要求：需提供代码或SymPy表达式
       - 对候选MR进行形式化验证

    输出：经过验证的MR列表
    """

    def __init__(self):
        """初始化算子MR生成器"""
        self.logger = get_logger()

        # 初始化核心组件
        self.llm_client = LLMClient()
        self.llm_generator = OperatorLLMMRGenerator(llm_client=self.llm_client)
        self.template_pool = MRTemplatePool()
        self.code_translator = CodeToSymPyTranslator(llm_client=self.llm_client)
        self.info_fetcher = OperatorInfoFetcher()
        self.prechecker = MRPreChecker()
        self.sympy_prover = SymPyProver(code_translator=self.code_translator)

        # 向后兼容：保留知识库（内部使用）
        self._kb = KnowledgeBase()

        self.logger.info("OperatorMRGenerator initialized")

    def fetch_operator_info(
        self,
        operator_name: str,
        framework: FrameworkType = "pytorch",
    ) -> Dict[str, str]:
        """
        仅获取算子信息（不生成MR）

        Args:
            operator_name: 算子名称
            framework: 框架名称

        Returns:
            包含 'doc', 'code', 'signature', 'examples' 的字典
        """
        self.logger.info(f"Fetching operator info for '{operator_name}'...")
        return self.info_fetcher.fetch_operator_info(
            operator_name=operator_name, framework=framework
        )

    def generate(
        self,
        operator_ir: OperatorIR,
        operator_func: Optional[Callable] = None,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        auto_fetch_info: bool = True,
        framework: FrameworkType = "pytorch",
        # 控制MR生成来源
        sources: Optional[List[MRSource]] = None,
        # 控制后处理步骤
        use_precheck: bool = True,
        use_sympy_proof: bool = True,
    ) -> List[MetamorphicRelation]:
        """
        为算子IR生成蜕变关系

        Args:
            operator_ir: 算子IR对象
            operator_func: 算子函数对象（可选，用于快速筛选）
            operator_code: 算子源代码（可选，用于SymPy证明）
            operator_doc: 算子文档（可选，用于LLM猜想）
            auto_fetch_info: 是否自动从网络获取算子信息（默认True）
            framework: 框架名称（默认pytorch）
            sources: MR生成来源列表，可选值：
                - "llm": LLM猜想（主要来源）
                - "template": 模板池（辅助来源）
                - None: 使用所有来源（默认）
            use_precheck: 是否进行快速筛选（默认True，需要operator_func）
            use_sympy_proof: 是否进行SymPy证明（默认True，需要代码）

        Returns:
            MR对象列表
        """
        self.logger.info(f"Generating MRs for operator: {operator_ir.name}")

        # 默认使用所有来源
        if sources is None:
            sources = ["llm", "template"]

        # ========== 阶段1：信息准备 ==========
        operator_code, operator_doc = self._prepare_info(
            operator_ir=operator_ir,
            operator_func=operator_func,
            operator_code=operator_code,
            operator_doc=operator_doc,
            auto_fetch_info=auto_fetch_info,
            framework=framework,
        )

        num_inputs = len(operator_ir.inputs)
        has_code = operator_code is not None

        # 存储SymPy表达式（用于后续证明）
        sympy_expr = None

        # 如果有代码，预先转换为SymPy表达式（用于证明阶段）
        if has_code and use_sympy_proof:
            try:
                sympy_expr = self.code_translator.translate(
                    code=operator_code, func=operator_func, doc=operator_doc
                )
                if sympy_expr is not None:
                    self.logger.info("Successfully converted code to SymPy expression")
            except Exception as e:
                self.logger.warning(f"Failed to convert code to SymPy: {e}")

        # ========== 阶段2：MR猜想生成 ==========
        candidate_mrs: List[MetamorphicRelation] = []

        # --- 来源1：LLM猜想（主要来源） ---
        if "llm" in sources:
            self.logger.info("Source 1: LLM-based MR generation...")
            try:
                llm_mrs = self.llm_generator.generate_mr_candidates(
                    operator_name=operator_ir.name,
                    operator_code=operator_code,
                    operator_doc=operator_doc,
                    num_inputs=num_inputs,
                    top_k=5,
                )
                candidate_mrs.extend(llm_mrs)
                self.logger.info(f"  → Generated {len(llm_mrs)} candidate MRs")
            except Exception as e:
                self.logger.warning(f"  → LLM generation failed: {e}")

        # --- 来源2：模板池猜测（辅助来源） ---
        if "template" in sources:
            self.logger.info("Source 2: Template pool generation...")
            try:
                template_mrs = self.template_pool.generate_mr_candidates(
                    operator_name=operator_ir.name,
                    operator_inputs=operator_ir.inputs,
                )
                candidate_mrs.extend(template_mrs)
                self.logger.info(
                    f"  → Generated {len(template_mrs)} candidate MRs from templates"
                )
            except Exception as e:
                self.logger.warning(f"  → Template pool generation failed: {e}")

        # --- 向后兼容：如果没有任何MR，使用旧的知识库方法 ---
        if not candidate_mrs:
            self.logger.info("Fallback: Using legacy knowledge base method...")
            mr_functions = self._kb.get_mrs_for_operator(operator_ir.name)
            for mr_func in mr_functions:
                try:
                    mr_obj = mr_func(operator_ir.inputs)
                    if isinstance(mr_obj, MetamorphicRelation):
                        candidate_mrs.append(mr_obj)
                except Exception as e:
                    self.logger.error(f"Error generating MR from knowledge base: {e}")

        # 统计
        self.logger.info(
            f"After MR generation: {len(candidate_mrs)} candidate MRs to verify"
        )

        if not candidate_mrs:
            self.logger.warning(f"No MRs generated for {operator_ir.name}")
            return []

        # ========== 阶段3：快速筛选 ==========
        if use_precheck:
            if operator_func:
                self.logger.info("Pre-checking candidate MRs...")
                candidate_mrs = self.prechecker.filter_mrs(
                    operator_func=operator_func,
                    mr_candidates=candidate_mrs,
                    original_inputs=operator_ir.inputs,
                )
                self.logger.info(
                    f"  → After pre-check: {len(candidate_mrs)} candidates remain"
                )
            else:
                self.logger.warning("Pre-check skipped: operator_func not provided.")

        # ========== 阶段4：SymPy形式化证明 ==========
        verified_mrs: List[MetamorphicRelation] = []

        if use_sympy_proof:
            if has_code or sympy_expr is not None:
                self.logger.info("Proving candidate MRs using SymPy...")
                proven_mrs = self.sympy_prover.prove_mrs(
                    mrs=candidate_mrs,
                    operator_func=operator_func,
                    operator_code=operator_code,
                    operator_doc=operator_doc,
                    operator_name=operator_ir.name,
                    num_inputs=num_inputs,
                    sympy_expr=sympy_expr,
                )
                for mr in proven_mrs:
                    mr.verified = True
                self.logger.info(f"  → Proven {len(proven_mrs)} MRs")
                verified_mrs.extend(proven_mrs)

                # 未证明的MR也加入输出（标记为未验证）
                proven_ids = {mr.id for mr in proven_mrs}
                for mr in candidate_mrs:
                    if mr.id not in proven_ids:
                        mr.verified = False
                        verified_mrs.append(mr)
            else:
                self.logger.warning(
                    "SymPy proof skipped: no code or SymPy expression available."
                )
                verified_mrs.extend(candidate_mrs)
        else:
            # 不进行证明，直接加入输出
            verified_mrs.extend(candidate_mrs)

        # ========== 去重 ==========
        final_mrs = self._deduplicate_mrs(verified_mrs)

        self.logger.info(
            f"Final output: {len(final_mrs)} MRs "
            f"({sum(1 for mr in final_mrs if mr.verified)} verified)"
        )
        return final_mrs

    def _prepare_info(
        self,
        operator_ir: OperatorIR,
        operator_func: Optional[Callable],
        operator_code: Optional[str],
        operator_doc: Optional[str],
        auto_fetch_info: bool,
        framework: FrameworkType,
    ) -> tuple:
        """准备算子信息（代码和文档）"""
        # 尝试从函数对象提取代码
        if operator_func and not operator_code:
            try:
                import inspect

                operator_code = inspect.getsource(operator_func)
                self.logger.info("Extracted code from function object using inspect")
            except Exception as e:
                self.logger.debug(f"Cannot extract code from function: {e}")

        # 自动从网络获取信息
        if auto_fetch_info:
            self.logger.info(f"Auto-fetching operator info for '{operator_ir.name}'...")
            try:
                fetched_info = self.info_fetcher.fetch_operator_info(
                    operator_name=operator_ir.name, framework=framework
                )

                # 合并文档
                if fetched_info.get("doc"):
                    fetched_doc = fetched_info["doc"]
                    if operator_doc:
                        operator_doc = f"{operator_doc}\n\n---\n\n[网络搜索获取的文档]\n{fetched_doc}"
                    else:
                        operator_doc = fetched_doc
                    self.logger.info(f"  → Fetched doc ({len(fetched_doc)} chars)")

                # 如果本地没有代码，使用网络获取的代码
                if not operator_code and fetched_info.get("code"):
                    fetched_code = fetched_info["code"]
                    operator_code = fetched_code
                    self.logger.info(f"  → Fetched code ({len(fetched_code)} chars)")

            except Exception as e:
                self.logger.warning(f"  → Failed to fetch operator info: {e}")

        return operator_code, operator_doc

    def _deduplicate_mrs(
        self, mrs: List[MetamorphicRelation]
    ) -> List[MetamorphicRelation]:
        """去重MR列表（基于描述）"""
        seen: Set[str] = set()
        unique_mrs: List[MetamorphicRelation] = []

        for mr in mrs:
            key = mr.description
            if key not in seen:
                seen.add(key)
                unique_mrs.append(mr)

        if len(mrs) != len(unique_mrs):
            self.logger.info(f"Deduplicated: {len(mrs)} → {len(unique_mrs)} MRs")

        return unique_mrs
