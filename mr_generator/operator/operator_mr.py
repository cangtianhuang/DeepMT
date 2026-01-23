"""
算子层MR生成器：多源融合自动生成引擎

生成流程：
1. 信息准备：自动获取算子信息
2. MR猜想生成：LLM猜想 + 模板池
3. 快速筛选：Pre-check
4. 形式化验证：SymPy证明
"""

from datetime import datetime
from typing import Callable, Dict, List, Literal, Optional, Set

from core.framework import FrameworkType
from core.logger import get_logger, log_error, log_structured
from ir.schema import MetamorphicRelation, OperatorIR
from mr_generator.base.mr_repository import MRRepository
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
                original_inputs=operator_ir.inputs or [],
                framework=framework,
            )

        # ========== 阶段4：SymPy形式化证明 ==========
        verified_mrs = self._apply_sympy_proof(
            candidate_mrs=candidate_mrs,
            operator_func=operator_func,
            operator_code=operator_code,
            operator_doc=operator_doc,
            operator_name=operator_ir.name,
            num_inputs=len(operator_ir.inputs) if operator_ir.inputs else None,
            use_sympy_proof=use_sympy_proof,
            has_code=has_code,
        )

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

    def generate_only(
        self,
        operator_ir: OperatorIR,
        operator_func: Optional[Callable] = None,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        auto_fetch_info: bool = True,
        framework: FrameworkType = "pytorch",
        sources: Optional[List[MRSource]] = None,
    ) -> List[MetamorphicRelation]:
        """
        仅生成MR候选（不进行验证）

        执行阶段：
        1. 信息准备
        2. MR猜想生成 (LLM + 模板池)

        参数：
            save_to_repo: 是否保存到MR知识库
            repo: MR知识库实例（如果为None则创建默认实例）

        返回：
            未验证的MR列表
        """
        operator_name = operator_ir.name
        log_structured(
            self.logger,
            "GEN",
            f"Generating MR candidates (no validation) for '{operator_name}'",
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

        # ========== 阶段2：MR猜想生成 ==========
        candidate_mrs = self._generate_candidates(
            operator_ir=operator_ir,
            operator_func=operator_func,
            operator_code=operator_code,
            operator_doc=operator_doc,
            sources=sources,
        )

        # ========== 去重 ==========
        final_mrs = self._deduplicate_mrs(candidate_mrs)

        # 标记为未验证
        for mr in final_mrs:
            mr.verified = False

        log_structured(
            self.logger,
            "GEN",
            f"Generated {len(final_mrs)} MR candidates (unverified)",
        )

        return final_mrs

    def verify_mrs(
        self,
        mrs: List[MetamorphicRelation],
        operator_ir: OperatorIR,
        operator_func: Optional[Callable] = None,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        use_precheck: bool = True,
        use_sympy_proof: bool = True,
        framework: FrameworkType = "pytorch",
        save_validation: bool = True,
        repo: Optional["MRRepository"] = None,
    ) -> List[MetamorphicRelation]:
        """
        对已生成的MR进行验证

        执行阶段：
        1. 快速筛选 (Pre-check, 可选)
        2. SymPy形式化证明 (可选)
        3. 保存验证记录

        参数：
            mrs: 待验证的MR列表
            use_precheck: 是否进行快速筛选
            use_sympy_proof: 是否进行SymPy证明
            save_validation: 是否保存验证记录
            repo: MR知识库实例

        返回：
            经过验证的MR列表
        """

        operator_name = operator_ir.name
        log_structured(
            self.logger,
            "CHECK",
            f"Verifying {len(mrs)} MRs for '{operator_name}'",
        )

        # 准备信息
        operator_code, operator_doc = self._prepare_info(
            operator_ir=operator_ir,
            operator_func=operator_func,
            operator_code=operator_code,
            operator_doc=operator_doc,
            auto_fetch_info=False,  # 不自动获取，使用提供的
            framework=framework,
        )
        has_code = operator_code is not None

        # 存储验证记录
        validation_records = []

        # ========== 阶段1：快速筛选 ==========
        if use_precheck and operator_func:
            mrs_after_precheck = []
            for mr in mrs:
                is_valid, error_msg = self.prechecker.check_mr(
                    operator_func=operator_func,
                    mr=mr,
                    original_inputs=operator_ir.inputs or [],
                    framework=framework,
                )

                # 记录验证结果
                validation_records.append(
                    {
                        "mr_id": mr.id,
                        "validation_type": "precheck",
                        "result": "passed" if is_valid else "failed",
                        "error_message": "" if is_valid else error_msg,
                        "details": {"passed": is_valid, "message": error_msg},
                        "validated_at": datetime.now().isoformat(),
                    }
                )

                if is_valid:
                    mrs_after_precheck.append(mr)
                    log_structured(
                        self.logger, "CHECK", f"MR passed pre-check: {mr.description}"
                    )
                else:
                    log_structured(
                        self.logger,
                        "CHECK",
                        f"MR failed pre-check: {mr.description}. Reason: {error_msg}",
                        level="DEBUG",
                    )

            mrs = mrs_after_precheck
            log_structured(
                self.logger,
                "CHECK",
                f"Pre-check: {len(mrs_after_precheck)}/{len(mrs_after_precheck) + len(validation_records) - len(mrs_after_precheck)} MRs passed",
            )

        # ========== 阶段2：SymPy形式化证明 ==========
        if use_sympy_proof and (has_code or operator_code):
            num_inputs = len(operator_ir.inputs) if operator_ir.inputs else None

            # 转换为SymPy表达式
            sympy_expr = None
            try:
                sympy_expr = self.code_translator.translate(
                    code=operator_code, func=operator_func, doc=operator_doc
                )
                if sympy_expr is not None:
                    log_structured(
                        self.logger,
                        "CHECK",
                        "Successfully converted code to SymPy expression",
                    )
            except Exception as e:
                log_structured(
                    self.logger,
                    "WARN",
                    f"Failed to convert code to SymPy: {e}",
                    level="WARNING",
                )

            if sympy_expr is not None:
                # 验证每个MR
                proven_mrs = []
                for mr in mrs:
                    is_proven, error_msg = self.sympy_prover.prove_mr_with_expr(
                        mr=mr,
                        sympy_expr=sympy_expr,
                        num_inputs=num_inputs if num_inputs else 1,
                        operator_name=operator_name,
                    )

                    # 记录验证结果
                    validation_records.append(
                        {
                            "mr_id": mr.id,
                            "validation_type": "sympy",
                            "result": "passed" if is_proven else "failed",
                            "error_message": "" if is_proven else error_msg,
                            "details": {"proven": is_proven, "message": error_msg},
                            "validated_at": datetime.now().isoformat(),
                        }
                    )

                    if is_proven:
                        mr.verified = True
                        proven_mrs.append(mr)
                        log_structured(
                            self.logger,
                            "CHECK",
                            f"MR proven by SymPy: {mr.description}",
                        )
                    else:
                        log_structured(
                            self.logger,
                            "CHECK",
                            f"MR proof failed: {mr.description}. Reason: {error_msg}",
                            level="DEBUG",
                        )

                # 更新MR列表
                proven_ids = {mr.id for mr in proven_mrs}
                final_mrs = []
                for mr in mrs:
                    if mr.id in proven_ids:
                        final_mrs.append(mr)
                    else:
                        # 未证明的MR也保留，但标记为未验证
                        mr.verified = False
                        final_mrs.append(mr)

                mrs = final_mrs
                log_structured(
                    self.logger,
                    "CHECK",
                    f"SymPy: {len(proven_mrs)}/{len(final_mrs)} MRs proven",
                )

        # ========== 阶段3：保存验证记录 ==========
        if save_validation and repo is None:
            repo = MRRepository()

        if save_validation and repo:
            # 获取每个MR的record_id
            for mr in mrs:
                record_id = repo.get_record_id_by_mr_id(
                    mr_id=mr.id,
                    operator_name=operator_name,
                )
                if record_id:
                    # 更新验证状态
                    for record in validation_records:
                        if record["mr_id"] == mr.id:
                            repo.update_validation_status(
                                record_id=record_id,
                                mr_id=mr.id,
                                validation_type=record["validation_type"],
                                result=record["validation_result"],
                                error_message=record["error_message"],
                                details=record["details"],
                            )

        # ========== 去重 ==========
        final_mrs = self._deduplicate_mrs(mrs)

        verified_count = sum(1 for mr in final_mrs if mr.verified)
        log_structured(
            self.logger,
            "SUCCESS",
            f"Verification completed: {verified_count}/{len(final_mrs)} MRs verified",
        )

        return final_mrs

    @staticmethod
    def verify_from_repository(
        operator_name: str,
        operator_ir: OperatorIR,
        operator_func: Optional[Callable] = None,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        use_precheck: bool = True,
        use_sympy_proof: bool = True,
        framework: FrameworkType = "pytorch",
        version: Optional[int] = None,
    ) -> List[MetamorphicRelation]:
        """
        从MR知识库加载MR并进行验证

        参数：
            operator_name: 算子名称
            version: MR版本号（默认最新版本）
            其他参数同 verify_mrs()

        返回：
            经过验证的MR列表
        """
        from mr_generator.base.mr_repository import MRRepository

        # 创建生成器实例
        generator = OperatorMRGenerator()

        # 加载MR
        repo = MRRepository()

        mrs = repo.load(operator_name=operator_name, version=version)

        if not mrs:
            log_structured(
                get_logger("OperatorMRGenerator"),
                "WARN",
                f"No MRs found for operator '{operator_name}' (version {version or 'latest'})",
                level="WARNING",
            )
            return []

        # 验证MR
        verified_mrs = generator.verify_mrs(
            mrs=mrs,
            operator_ir=operator_ir,
            operator_func=operator_func,
            operator_code=operator_code,
            operator_doc=operator_doc,
            use_precheck=use_precheck,
            use_sympy_proof=use_sympy_proof,
            framework=framework,
            save_validation=True,
            repo=repo,
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
                        f"Fetched {len(source_urls)} sources | {len(fetched_doc)} chars",
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

        log_structured(
            self.logger,
            "GEN",
            f"Generating MRs for '{operator_name}' | sources from {sources}",
        )

        # 信息准备已在调用方完成

        # MR猜想生成
        candidate_mrs: List[MetamorphicRelation] = []

        # --- 来源1：LLM猜想 ---
        if "llm" in sources:
            log_structured(
                self.logger,
                "GEN",
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
                    f"Generated {len(llm_mrs)} candidates from LLM for '{operator_name}'",
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
                    f"Generated {len(template_mrs)} candidates from templates for '{operator_name}'",
                )
            except Exception as e:
                log_error(
                    self.logger,
                    f"Template pool generation failed for '{operator_name}'",
                    exception=e,
                )

        # 统计
        if candidate_mrs:
            log_structured(
                self.logger,
                "GEN",
                f"Total {len(candidate_mrs)} candidates generated",
            )
        else:
            log_structured(
                self.logger,
                "WARN",
                f"No MRs generated for '{operator_name}'",
                level="WARNING",
            )

        return candidate_mrs

    def _apply_precheck(
        self,
        operator_func: Callable,
        mr_candidates: List[MetamorphicRelation],
        original_inputs: List,
        framework: FrameworkType,
    ) -> List[MetamorphicRelation]:
        """
        应用快速筛选

        Args:
            operator_func: 算子函数
            mr_candidates: MR候选列表
            original_inputs: 原始输入
            framework: 框架类型

        Returns:
            通过快速筛选的MR列表
        """
        log_structured(self.logger, "CHECK", "Pre-checking candidates...")
        filtered = self.prechecker.filter_mrs(
            operator_func=operator_func,
            mr_candidates=mr_candidates,
            original_inputs=original_inputs,
            framework=framework,
        )
        log_structured(
            self.logger,
            "CHECK",
            f"{len(filtered)}/{len(mr_candidates)} candidates passed pre-check",
        )
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
        has_code: bool,
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
            has_code: 是否有代码

        Returns:
            经过验证的MR列表
        """
        verified_mrs: List[MetamorphicRelation] = []

        if use_sympy_proof and (has_code or operator_code):
            # 转换为SymPy表达式
            sympy_expr = None
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

            if sympy_expr is not None:
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

        return verified_mrs
