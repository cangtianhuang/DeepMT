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
from core.logger import get_logger, log_error, log_structured
from ir.schema import MetamorphicRelation, OperatorIR
from mr_generator.base.mr_templates import MRTemplatePool
from mr_generator.operator.mr_prechecker import MRPreChecker
from mr_generator.operator.operator_llm_mr_generator import OperatorLLMMRGenerator
from mr_generator.operator.sympy_prover import SymPyProver
from mr_generator.operator.sympy_translator import SympyTranslator
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
        self.logger = get_logger(self.__class__.__name__)

        self.llm_generator = OperatorLLMMRGenerator()
        self.template_pool = MRTemplatePool()
        self.code_translator = SympyTranslator()
        self.info_fetcher = OperatorInfoFetcher()
        self.prechecker = MRPreChecker()
        self.sympy_prover = SymPyProver(code_translator=self.code_translator)

        # 向后兼容：保留知识库（内部使用）
        # self._kb = KnowledgeBase()

        log_structured(
            self.logger,
            "INIT",
            "OperatorMRGenerator initialized",
            level="DEBUG",
        )

    def fetch_operator_info(
        self,
        operator_name: str,
        framework: FrameworkType = "pytorch",
    ) -> Dict[str, str]:
        """获取算子信息"""
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
        sources: Optional[List[MRSource]] = None,
        use_precheck: bool = True,
        use_sympy_proof: bool = True,
    ) -> List[MetamorphicRelation]:
        """
        为算子生成蜕变关系

        Args:
            operator_ir: 算子IR对象
            operator_func: 算子函数对象（可选）
            operator_code: 算子源代码（可选）
            operator_doc: 算子文档（可选）
            auto_fetch_info: 是否自动从网络获取算子信息（默认True）
            framework: 框架名称（默认pytorch）
            sources: MR生成来源列表，可选值：
                - "llm": LLM猜想
                - "template": 模板池
                - None: 使用所有来源（默认）
            use_precheck: 是否进行快速筛选（默认True，需要operator_func）
            use_sympy_proof: 是否进行SymPy证明（默认True）

        Returns:
            MR对象列表
        """
        # 默认使用所有来源
        if sources is None:
            sources = ["llm", "template"]

        operator_name = operator_ir.name
        log_structured(
            self.logger,
            "GEN",
            f"Generating MRs for '{operator_name}' | sources from {sources}",
        )

        # ========== 阶段1：信息准备 ==========
        operator_code, operator_doc = self._prepare_info(
            operator_ir=operator_ir,
            operator_func=operator_func,
            operator_code=operator_code,
            operator_doc=operator_doc,
            auto_fetch_info=auto_fetch_info,
            framework=framework,
        )
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
                    log_structured(
                        self.logger,
                        "GEN",
                        "Successfully converted code to SymPy expression",
                    )
            except Exception as e:
                log_structured(
                    self.logger,
                    "WARN",
                    f"Failed to convert code to SymPy: {e}",
                    level="WARNING",
                )

        # ========== 阶段2：MR猜想生成 ==========
        candidate_mrs: List[MetamorphicRelation] = []

        # --- 来源1：LLM猜想 ---
        if "llm" in sources:
            log_structured(
                self.logger,
                "LLM",
                "Generating MR candidates...",
            )
            try:
                llm_mrs = self.llm_generator.generate_mr_candidates(
                    operator_name=operator_name,
                    operator_func=operator_func,
                    operator_code=operator_code,
                    operator_doc=operator_doc,
                    top_k=5,
                )
                candidate_mrs.extend(llm_mrs)
                log_structured(
                    self.logger,
                    "GEN",
                    f"Generated {len(llm_mrs)} candidates from LLM",
                )
            except Exception as e:
                log_error(
                    self.logger,
                    f"LLM generation failed for '{operator_name}'",
                    exception=e,
                )

        # --- 来源2：模板池猜测 ---
        if "template" in sources:
            log_structured(self.logger, "GEN", "Template pool matching...")
            try:
                template_mrs = self.template_pool.generate_mr_candidates(
                    operator_name=operator_name,
                    operator_func=operator_func,
                )
                candidate_mrs.extend(template_mrs)
                log_structured(
                    self.logger,
                    "GEN",
                    f"Generated {len(template_mrs)} candidates from templates",
                )
            except Exception as e:
                log_error(
                    self.logger,
                    f"Template pool generation failed for '{operator_name}'",
                    exception=e,
                )

        # --- 如果没有任何MR，使用旧的知识库方法 ---
        # if not candidate_mrs:
        #     self.logger.info("Fallback: Using legacy knowledge base method...")
        #     mr_functions = self._kb.get_mrs_for_operator(operator_ir.name)
        #     for mr_func in mr_functions:
        #         try:
        #             mr_obj = mr_func(operator_ir.inputs)
        #             if isinstance(mr_obj, MetamorphicRelation):
        #                 candidate_mrs.append(mr_obj)
        #         except Exception as e:
        #             self.logger.error(f"Error generating MR from knowledge base: {e}")

        # 统计
        if candidate_mrs:
            log_structured(
                self.logger,
                "GEN",
                f"Total {len(candidate_mrs)} candidates to verify",
            )
        else:
            log_structured(
                self.logger,
                "WARN",
                f"No MRs generated for '{operator_name}'",
                level="WARNING",
            )
            return []

        # ========== 阶段3：快速筛选 ==========
        if use_precheck:
            if operator_func:
                log_structured(self.logger, "CHECK", "Pre-checking candidates...")
                candidate_mrs = self.prechecker.filter_mrs(
                    operator_func=operator_func,
                    mr_candidates=candidate_mrs,
                    original_inputs=operator_ir.inputs or [],
                    framework=framework,
                )
                log_structured(
                    self.logger, "CHECK", f"{len(candidate_mrs)} candidates passed"
                )
            else:
                log_structured(
                    self.logger,
                    "CHECK",
                    "Skipped (no operator_func)",
                    level="DEBUG",
                )

        # ========== 阶段4：SymPy形式化证明 ==========
        verified_mrs: List[MetamorphicRelation] = []

        if use_sympy_proof:
            if has_code or sympy_expr is not None:
                # 获取输入数量
                num_inputs = len(operator_ir.inputs) if operator_ir.inputs else None

                log_structured(self.logger, "CHECK", "Proving with SymPy...")
                proven_mrs = self.sympy_prover.prove_mrs(
                    mrs=candidate_mrs,
                    operator_func=operator_func,
                    operator_code=operator_code,
                    operator_doc=operator_doc,
                    operator_name=operator_name,
                    num_inputs=num_inputs,
                    sympy_expr=sympy_expr,
                )
                for mr in proven_mrs:
                    mr.verified = True
                log_structured(self.logger, "CHECK", f"{len(proven_mrs)} MRs proven")
                verified_mrs.extend(proven_mrs)

                # 未证明的MR也加入输出（标记为未验证）
                proven_ids = {mr.id for mr in proven_mrs}
                for mr in candidate_mrs:
                    if mr.id not in proven_ids:
                        mr.verified = False
                        verified_mrs.append(mr)
            else:
                log_structured(
                    self.logger,
                    "CHECK",
                    "SymPy proof skipped (no code)",
                    level="DEBUG",
                )
                verified_mrs.extend(candidate_mrs)
        else:
            # 不进行证明，直接加入输出
            verified_mrs.extend(candidate_mrs)

        # ========== 去重 ==========
        final_mrs = self._deduplicate_mrs(verified_mrs)

        verified_count = sum(1 for mr in final_mrs if mr.verified)
        if verified_count == len(final_mrs):
            log_structured(
                self.logger, "SUCCESS", f"{len(final_mrs)} MRs (all verified)"
            )
        elif verified_count > 0:
            log_structured(
                self.logger,
                "GEN",
                f"{len(final_mrs)} MRs ({verified_count} verified)",
            )
        else:
            log_structured(self.logger, "GEN", f"{len(final_mrs)} MRs (unverified)")

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
        operator_name = operator_ir.name

        # 尝试从函数对象提取代码
        if operator_func and not operator_code:
            try:
                import inspect

                operator_code = inspect.getsource(operator_func)
                log_structured(
                    self.logger,
                    "GEN",
                    "Extracted code from function object using inspect",
                )
            except Exception as e:
                self.logger.debug(f"Cannot extract code from function: {e}")

        # 自动从网络获取信息
        if auto_fetch_info:
            log_structured(self.logger, "SEARCH", f"Fetching '{operator_name}' docs...")
            try:
                fetched_info = self.info_fetcher.fetch_operator_info(
                    operator_name=operator_name, framework=framework
                )

                # 合并文档
                if fetched_info.get("doc"):
                    fetched_doc = fetched_info["doc"]
                    if operator_doc:
                        operator_doc = f"{operator_doc}\n\n---\n\n[网络搜索获取的文档]\n{fetched_doc}"
                    else:
                        operator_doc = fetched_doc
                    source_urls = fetched_info.get("source_urls", [])
                    log_structured(
                        self.logger,
                        "SEARCH",
                        f"Fetched {len(fetched_doc)} chars from {len(source_urls)} sources",
                    )
                    # Debug: 打印获取到的文档链接
                    if source_urls:
                        self.logger.debug("  Sources:")
                        for idx, url in enumerate(source_urls):
                            self.logger.debug(f"    [{idx + 1}] {url}")

                # 如果本地没有代码，使用网络获取的代码
                if not operator_code and fetched_info.get("code"):
                    fetched_code = fetched_info["code"]
                    operator_code = fetched_code
                    log_structured(
                        self.logger,
                        "SEARCH",
                        f"Fetched code ({len(fetched_code)} chars)",
                        level="DEBUG",
                    )

            except Exception as e:
                log_error(
                    self.logger,
                    f"Failed to fetch info for '{operator_name}'",
                    exception=e,
                )

        return operator_code, operator_doc

    def _deduplicate_mrs(
        self, mrs: List[MetamorphicRelation]
    ) -> List[MetamorphicRelation]:
        """去重MR列表"""
        seen: Set[str] = set()
        unique_mrs: List[MetamorphicRelation] = []

        for mr in mrs:
            key = mr.description
            if key not in seen:
                seen.add(key)
                unique_mrs.append(mr)

        if len(mrs) != len(unique_mrs):
            log_structured(
                self.logger,
                "DEBUG",
                f"Deduplicated: {len(mrs)} → {len(unique_mrs)} MRs",
                level="DEBUG",
            )

        return unique_mrs
