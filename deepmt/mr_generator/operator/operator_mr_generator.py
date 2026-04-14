"""
算子层MR生成器：多源融合自动生成引擎

生成流程：
1. 信息准备：自动获取算子信息
2. MR猜想生成：LLM猜想 + 模板池
3. 快速筛选：Pre-check
4. 形式化验证：SymPy证明
"""

from typing import Callable, Dict, List, Literal, Optional, Set

from deepmt.core.plugins_manager import FrameworkType
from deepmt.core.logger import logger
from deepmt.ir.schema import MetamorphicRelation, OperatorIR
from deepmt.mr_generator.base.mr_templates import MRTemplatePool
from deepmt.analysis.verification.mr_prechecker import MRPreChecker
from deepmt.mr_generator.operator.operator_llm_mr_generator import OperatorLLMMRGenerator
from deepmt.mr_generator.operator.sympy_prover import SymPyProver
from deepmt.mr_generator.operator.sympy_translator import SympyTranslator
from deepmt.tools.web_search.operator_fetcher import OperatorInfoFetcher

# MR生成来源类型
MRSource = Literal["llm", "template"]


class OperatorMRGenerator:
    """
    算子层MR生成器：多源融合自动生成引擎

    内部流程（四个阶段）：
      1. 信息准备 — 从函数对象/网络获取算子代码与文档
      2. MR猜想生成 — LLM猜想（主要来源）+ 模板池匹配（辅助来源）
      3. 快速筛选（Pre-check，可选）— 用随机输入数值验证候选MR
      4. SymPy形式化证明（可选）— 对候选MR进行符号化验证

    核心公开方法：
      generate()               完整流程：生成候选 → 验证 → 返回结果
      generate_only()          仅执行阶段1-2：生成候选，不做任何验证
      verify_mrs(mrs, ...)     仅执行阶段3-4：对已有MR列表进行验证
      verify_from_repository() 从知识库加载MR后调用 verify_mrs()
      fetch_operator_info()    单独调用网络搜索获取算子信息

    generate() ≈ generate_only() + verify_mrs()，三个方法共享相同的私有实现。
    """

    def __init__(self):
        """初始化算子MR生成器"""

        self.llm_generator = OperatorLLMMRGenerator()
        self.template_pool = MRTemplatePool()
        self.code_translator = SympyTranslator()
        self.info_fetcher = OperatorInfoFetcher()
        self.prechecker = MRPreChecker()
        self.sympy_prover = SymPyProver(code_translator=self.code_translator)

        logger.debug("🚀 [INIT] " + "OperatorMRGenerator initialized")

    def fetch_operator_info(
        self,
        operator_name: str,
        framework: FrameworkType,
    ) -> Dict[str, str]:
        """获取算子信息"""
        return self.info_fetcher.fetch_operator_info(
            operator_name=operator_name, framework=framework
        )

    def generate(
        self,
        operator_ir: OperatorIR,
        framework: FrameworkType,
        operator_func: Optional[Callable] = None,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        auto_fetch_info: bool = True,
        sources: Optional[List[MRSource]] = None,
        use_precheck: bool = True,
        use_sympy_proof: bool = True,
    ) -> List[MetamorphicRelation]:
        """
        为算子生成蜕变关系

        Args:
            operator_ir: 算子IR对象
            framework: 框架名称（"pytorch" / "tensorflow" / "paddlepaddle"）
            operator_func: 算子函数对象（可选）
            operator_code: 算子源代码（可选）
            operator_doc: 算子文档（可选）
            auto_fetch_info: 是否自动从网络获取算子信息（默认True）
            sources: MR生成来源列表，可选值：
                - "llm": LLM猜想
                - "template": 模板池
                - None: 使用所有来源（默认）
            use_precheck: 是否进行快速筛选（默认True，需要operator_func）
            use_sympy_proof: 是否进行SymPy证明（默认True）

        Returns:
            MR对象列表
        """
        # ========== 阶段1：信息准备 ==========
        operator_code, operator_doc = self._prepare_info(
            operator_ir=operator_ir,
            operator_func=operator_func,
            operator_code=operator_code,
            operator_doc=operator_doc,
            auto_fetch_info=auto_fetch_info,
            framework=framework,
        )
        # ========== 阶段2：MR猜想生成 ==========
        candidate_mrs = self._generate_candidates(
            operator_ir=operator_ir,
            operator_func=operator_func,
            operator_code=operator_code,
            operator_doc=operator_doc,
            sources=sources,
        )

        if not candidate_mrs:
            return []

        # ========== 阶段3：快速筛选 ==========
        if use_precheck and operator_func:
            candidate_mrs = self._apply_precheck(
                operator_func=operator_func,
                mr_candidates=candidate_mrs,
                operator_ir=operator_ir,
                framework=framework,
            )

        # ========== 阶段4：SymPy形式化证明 ==========
        verified_mrs = self._apply_sympy_proof(
            candidate_mrs=candidate_mrs,
            operator_func=operator_func,
            operator_code=operator_code,
            operator_doc=operator_doc,
            operator_name=operator_ir.name,
            num_inputs=len(operator_ir.input_specs) if operator_ir.input_specs else None,
            use_sympy_proof=use_sympy_proof,
        )

        # ========== 去重 ==========
        final_mrs = self._deduplicate_mrs(verified_mrs)

        for mr in final_mrs:
            mr.verified = (mr.checked is True) and (mr.proven is True)

        verified_count = sum(1 for mr in final_mrs if mr.verified)
        if verified_count == len(final_mrs):
            logger.success("✨ [SUCCESS] " + f"{len(final_mrs)} MRs (all verified)")
        elif verified_count > 0:
            logger.info("⚡ [GEN] " + f"{len(final_mrs)} MRs ({verified_count} verified)")
        else:
            logger.info("⚡ [GEN] " + f"{len(final_mrs)} MRs (unverified)")

        return final_mrs

    def generate_only(
        self,
        operator_ir: OperatorIR,
        framework: FrameworkType,
        operator_func: Optional[Callable] = None,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        auto_fetch_info: bool = True,
        sources: Optional[List[MRSource]] = None,
    ) -> List[MetamorphicRelation]:
        """
        仅生成MR候选，不进行任何验证（阶段1-2）。

        等价于 generate(..., use_precheck=False, use_sympy_proof=False)，
        但更轻量——跳过所有验证相关的初始化逻辑。

        返回：
            未验证的MR列表（mr.verified 均为 False）
        """
        operator_code, operator_doc = self._prepare_info(
            operator_ir=operator_ir,
            operator_func=operator_func,
            operator_code=operator_code,
            operator_doc=operator_doc,
            auto_fetch_info=auto_fetch_info,
            framework=framework,
        )

        candidate_mrs = self._generate_candidates(
            operator_ir=operator_ir,
            operator_func=operator_func,
            operator_code=operator_code,
            operator_doc=operator_doc,
            sources=sources,
        )

        return self._deduplicate_mrs(candidate_mrs)

    def verify_mrs(
        self,
        mrs: List[MetamorphicRelation],
        operator_ir: OperatorIR,
        framework: FrameworkType,
        operator_func: Optional[Callable] = None,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        use_precheck: bool = True,
        use_sympy_proof: bool = True,
    ) -> List[MetamorphicRelation]:
        """
        对已生成的MR进行验证

        执行阶段：
        1. 快速筛选 (Pre-check, 可选)
        2. SymPy形式化证明 (可选)

        参数：
            mrs: 待验证的MR列表
            use_precheck: 是否进行快速筛选（需提供 operator_func）
            use_sympy_proof: 是否进行SymPy证明

        返回：
            经过验证的MR列表
        """
        operator_name = operator_ir.name

        if not mrs:
            logger.debug("✅ [CHECK] " + f"No MRs to verify for '{operator_name}', skipping")
            return []

        logger.info("✅ [CHECK] " + f"Verifying {len(mrs)} MRs for '{operator_name}'")

        operator_code, operator_doc = self._prepare_info(
            operator_ir=operator_ir,
            operator_func=operator_func,
            operator_code=operator_code,
            operator_doc=operator_doc,
            auto_fetch_info=False,
            framework=framework,
        )

        # ========== 阶段1：快速筛选 ==========
        if use_precheck and operator_func:
            mrs = self._apply_precheck(
                operator_func=operator_func,
                mr_candidates=mrs,
                operator_ir=operator_ir,
                framework=framework,
            )

        # ========== 阶段2：SymPy形式化证明 ==========
        mrs = self._apply_sympy_proof(
            candidate_mrs=mrs,
            operator_func=operator_func,
            operator_code=operator_code,
            operator_doc=operator_doc,
            operator_name=operator_name,
            num_inputs=len(operator_ir.input_specs) if operator_ir.input_specs else None,
            use_sympy_proof=use_sympy_proof,
        )

        final_mrs = self._deduplicate_mrs(mrs)

        for mr in final_mrs:
            mr.verified = (mr.checked is True) and (mr.proven is True)

        verified_count = sum(1 for mr in final_mrs if mr.verified)
        logger.success("✨ [SUCCESS] " + f"Verification completed: {verified_count}/{len(final_mrs)} MRs verified")

        return final_mrs

    @staticmethod
    def verify_from_repository(
        operator_name: str,
        operator_ir: OperatorIR,
        framework: FrameworkType,
        operator_func: Optional[Callable] = None,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        use_precheck: bool = True,
        use_sympy_proof: bool = True,
    ) -> List[MetamorphicRelation]:
        """从 MR 仓库加载 MR 并进行验证。"""
        from deepmt.mr_generator.base.mr_repository import MRRepository

        generator = OperatorMRGenerator()
        repo = MRRepository()
        mrs = repo.load(operator_name=operator_name)

        if not mrs:
            logger.warning(f"⚠️ [WARN] No MRs found for operator '{operator_name}'")
            return []

        verified_mrs = generator.verify_mrs(
            mrs=mrs,
            operator_ir=operator_ir,
            framework=framework,
            operator_func=operator_func,
            operator_code=operator_code,
            operator_doc=operator_doc,
            use_precheck=use_precheck,
            use_sympy_proof=use_sympy_proof,
        )

        return verified_mrs

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
                logger.info("⚡ [GEN] " + "Extracted code from function object using inspect")
            except Exception as e:
                logger.warning(
                    f"⚠️ [WARN] Cannot get source for '{operator_name}' "
                    f"({type(operator_func).__name__}): {e} — SymPy proof will be skipped"
                )

        # 自动从网络获取信息
        if auto_fetch_info:
            logger.info("🔍 [SEARCH] " + f"Fetching '{operator_name}' docs...")
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
                    logger.info("🔍 [SEARCH] " + f"Fetched {len(source_urls)} sources | {len(fetched_doc)} chars")
                    # Debug: 打印获取到的文档链接
                    if source_urls:
                        logger.debug("  Sources:")
                        for idx, url in enumerate(source_urls):
                            logger.debug(f"    [{idx + 1}] {url}")

                # 如果本地没有代码，使用网络获取的代码
                if not operator_code and fetched_info.get("code"):
                    fetched_code = fetched_info["code"]
                    operator_code = fetched_code
                    logger.debug("🔍 [SEARCH] " + f"Fetched code ({len(fetched_code)} chars)")

            except Exception as e:
                logger.opt(exception=e).error("❌ " + f"Failed to fetch info for '{operator_name}'")

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
            logger.debug("🐛 [DEBUG] " + f"Deduplicated: {len(mrs)} → {len(unique_mrs)} MRs")

        return unique_mrs

    # ==================== 私有辅助方法 ====================

    def _generate_candidates(
        self,
        operator_ir: OperatorIR,
        operator_func: Optional[Callable],
        operator_code: Optional[str],
        operator_doc: Optional[str],
        sources: Optional[List[MRSource]],
    ) -> List[MetamorphicRelation]:
        """
        生成MR候选列表（信息准备 + MR猜想生成）

        Args:
            operator_ir: 算子IR
            operator_func: 算子函数
            operator_code: 算子代码
            operator_doc: 算子文档
            sources: MR生成来源

        Returns:
            MR候选列表
        """
        operator_name = operator_ir.name

        # 默认使用所有来源
        if sources is None:
            sources = ["llm", "template"]

        logger.info("⚡ [GEN] " + f"Generating MRs for '{operator_name}' | sources from {sources}")

        candidate_mrs: List[MetamorphicRelation] = []

        # 无算子信息时给出明显提示
        if not operator_code and not operator_doc:
            logger.warning(f"⚠️ [WARN] No code or doc available for '{operator_name}' — quality may be degraded")

        # --- 来源1：LLM猜想 ---
        if "llm" in sources:
            try:
                llm_mrs = self.llm_generator.generate_mr_candidates(
                    operator_name=operator_name,
                    operator_func=operator_func,
                    operator_code=operator_code,
                    operator_doc=operator_doc,
                    top_k=5,
                )
                candidate_mrs.extend(llm_mrs)
                logger.info("⚡ [GEN] " + f"Generated {len(llm_mrs)} candidates from LLM for '{operator_name}'")
            except Exception as e:
                logger.opt(exception=e).error("❌ " + f"LLM generation failed for '{operator_name}'")

        # --- 来源2：模板池猜测 ---
        if "template" in sources:
            logger.info("⚡ [GEN] " + "Template pool matching ...")
            try:
                template_mrs = self.template_pool.generate_mr_candidates(
                    operator_name=operator_name,
                    operator_func=operator_func,
                )
                candidate_mrs.extend(template_mrs)
                logger.info("⚡ [GEN] " + f"Generated {len(template_mrs)} candidates from templates for '{operator_name}'")
            except Exception as e:
                logger.opt(exception=e).error("❌ " + f"Template pool generation failed for '{operator_name}'")

        # 统计
        if candidate_mrs:
            logger.info("⚡ [GEN] " + f"Total {len(candidate_mrs)} candidates generated")
        else:
            logger.warning("⚠️ [WARN] " + f"No MRs generated for '{operator_name}'")

        return candidate_mrs

    def _apply_precheck(
        self,
        operator_func: Callable,
        mr_candidates: List[MetamorphicRelation],
        operator_ir: OperatorIR,
        framework: FrameworkType,
    ) -> List[MetamorphicRelation]:
        """
        应用快速筛选

        Args:
            operator_func: 算子函数
            mr_candidates: MR候选列表
            operator_ir:   算子 IR（通过 input_specs 驱动随机输入生成）
            framework:     框架类型

        Returns:
            通过快速筛选的MR列表
        """
        logger.info("✅ [CHECK] " + "Pre-checking candidates...")
        filtered = self.prechecker.filter_mrs(
            operator_func=operator_func,
            mr_candidates=mr_candidates,
            operator_ir=operator_ir,
            framework=framework,
        )
        for mr in filtered:
            mr.checked = True
        logger.info("✅ [CHECK] " + f"{len(filtered)}/{len(mr_candidates)} candidates passed pre-check")
        return filtered

    def _apply_sympy_proof(
        self,
        candidate_mrs: List[MetamorphicRelation],
        operator_func: Optional[Callable],
        operator_code: Optional[str],
        operator_doc: Optional[str],
        operator_name: str,
        num_inputs: Optional[int],
        use_sympy_proof: bool,
    ) -> List[MetamorphicRelation]:
        """
        应用SymPy形式化证明

        Args:
            candidate_mrs: MR候选列表
            operator_func: 算子函数
            operator_code: 算子代码
            operator_doc: 算子文档
            operator_name: 算子名称
            num_inputs: 输入数量
            use_sympy_proof: 是否使用SymPy证明

        Returns:
            经过验证的MR列表
        """
        verified_mrs: List[MetamorphicRelation] = []

        if use_sympy_proof and (operator_code or operator_func):
            # 转换为SymPy表达式
            sympy_expr = None
            try:
                sympy_expr = self.code_translator.translate(
                    code=operator_code, func=operator_func, doc=operator_doc
                )
                if sympy_expr is not None:
                    logger.info("✅ [CHECK] " + "Successfully converted code to SymPy expression")
            except Exception as e:
                logger.warning("⚠️ [WARN] " + f"Failed to convert code to SymPy: {e}")

            if sympy_expr is not None:
                logger.info("✅ [CHECK] " + "Proving with SymPy...")
                proven_mrs = self.sympy_prover.prove_mrs(
                    mrs=candidate_mrs,
                    operator_func=operator_func,
                    operator_code=operator_code,
                    operator_doc=operator_doc,
                    operator_name=operator_name,
                    num_inputs=num_inputs,
                    sympy_expr=sympy_expr,
                )
                proven_ids = {mr.id for mr in proven_mrs}
                for mr in proven_mrs:
                    mr.proven = True
                logger.info("✅ [CHECK] " + f"{len(proven_mrs)} MRs proven")
                verified_mrs.extend(proven_mrs)

                for mr in candidate_mrs:
                    if mr.id not in proven_ids:
                        mr.proven = False
                        verified_mrs.append(mr)
            else:
                logger.warning(
                    f"⚠️ [WARN] SymPy proof skipped for '{operator_name}': "
                    "code_to_sympy returned None (C extension or unsupported code format)"
                )
                verified_mrs.extend(candidate_mrs)
        else:
            # 不进行证明，直接加入输出
            if use_sympy_proof and not operator_code and not operator_func:
                logger.warning(
                    f"⚠️ [WARN] SymPy proof skipped for '{operator_name}': "
                    "no source code and no operator function available."
                )
            verified_mrs.extend(candidate_mrs)

        return verified_mrs
