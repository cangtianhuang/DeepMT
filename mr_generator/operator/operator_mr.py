"""
算子层MR生成器：多源融合自动生成引擎
支持自动推导、LLM猜想、模板池生成，包含快速筛选和SymPy证明
"""

from typing import Any, Callable, List, Optional, Set

from core.framework import FrameworkType
from core.logger import get_logger
from ir.schema import MetamorphicRelation, OperatorIR
from mr_generator.base.knowledge_base import KnowledgeBase
from mr_generator.base.mr_templates import MRTemplatePool
from mr_generator.operator.code_translator import CodeToSymPyTranslator
from mr_generator.operator.mr_deriver import MRDeriver
from mr_generator.operator.mr_precheck import MRPreChecker
from mr_generator.operator.operator_llm_mr_generator import OperatorLLMMRGenerator
from mr_generator.operator.sympy_prover import SymPyProver
from tools.llm.client import LLMClient
from tools.web_search.operator_fetcher import OperatorInfoFetcher


class OperatorMRGenerator:
    """
    算子层MR生成器：多源融合自动生成引擎

    执行路径：
    ┌─────────────────────────────────────────────────────────────┐
    │ 阶段0：信息准备                                               │
    │   - 自动信息获取（网络搜索算子文档/代码）                        │
    │   - 代码提取（从 operator_code 参数或网络获取）                 │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 阶段1：多源MR生成（并行执行，多源融合）                          │
    │                                                             │
    │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
    │   │ 自动推导        │  │ LLM猜想         │  │ 模板池猜测   │ │
    │   │ (有代码时)      │  │ (始终执行)      │  │ (始终执行)   │ │
    │   │ → 已验证MR      │  │ → 候选MR        │  │ → 候选MR     │ │
    │   └────────┬────────┘  └────────┬────────┘  └──────┬──────┘ │
    │            │                    │                  │        │
    │            └────────────────────┴──────────────────┘        │
    │                              ↓                              │
    │                        合并 & 去重                           │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 阶段2：快速筛选（Pre-check）                                   │
    │   - 需要 operator_func 参数                                  │
    │   - 用随机输入执行算子，验证MR是否满足                          │
    │   - 仅对候选MR执行，已验证MR跳过                                │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 阶段3：SymPy形式化证明                                        │
    │   - 需要代码或SymPy表达式                                     │
    │   - 仅对候选MR执行，已验证MR跳过                                │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 输出：合并后的MR列表（已验证 + 经过筛选/证明的候选）             │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        use_llm: bool = True,
        use_template_pool: bool = True,
        use_precheck: bool = True,
        use_sympy_proof: bool = True,
        llm_api_key: Optional[str] = None,
        use_auto_derivation: bool = True,
    ):
        """
        初始化算子MR生成器

        Args:
            knowledge_base: 知识库实例（向后兼容，保留旧接口）
            use_llm: 是否使用LLM生成MR（默认True，始终执行）
            use_template_pool: 是否使用模板池生成MR（默认True）
            use_precheck: 是否使用快速筛选（默认True，需要operator_func）
            use_sympy_proof: 是否使用SymPy证明（默认True，需要代码）
            llm_api_key: LLM API密钥（可选）
            use_auto_derivation: 是否使用自动推导（默认True，需要代码）
        """
        # 向后兼容：保留旧的知识库
        self.kb = knowledge_base if knowledge_base else KnowledgeBase()

        # 配置选项
        self.use_llm = use_llm
        self.use_template_pool = use_template_pool
        self.use_precheck = use_precheck
        self.use_sympy_proof = use_sympy_proof
        self.use_auto_derivation = use_auto_derivation

        self.logger = get_logger()

        # 初始化组件：模板池
        if use_template_pool:
            self.template_pool = MRTemplatePool()
        else:
            self.template_pool = None

        # 初始化组件：快速筛选器
        if use_precheck:
            self.prechecker = MRPreChecker()
        else:
            self.prechecker = None

        # 初始化组件：SymPy证明器
        if use_sympy_proof:
            try:
                self.sympy_prover = SymPyProver()
            except ImportError:
                self.logger.warning("SymPy not available, disabling SymPy proof")
                self.use_sympy_proof = False
                self.sympy_prover = None
        else:
            self.sympy_prover = None

        # 初始化组件：LLM客户端（共享使用）
        try:
            self.llm_client = LLMClient(api_key=llm_api_key)
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            raise

        # 初始化组件：LLM MR生成器
        try:
            self.llm_generator = OperatorLLMMRGenerator(llm_client=self.llm_client)
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM MR generator: {e}")
            raise

        # 初始化组件：代码到SymPy转换器和MR推导器
        if use_auto_derivation:
            self.code_translator = CodeToSymPyTranslator(llm_client=self.llm_client)
            self.mr_deriver = MRDeriver(
                template_pool=self.template_pool,
                use_z3=False,  # 可选：启用Z3
                llm_client=None,  # 自动推导中不再使用LLM，统一由阶段1的LLM猜想处理
            )
        else:
            self.code_translator = None
            self.mr_deriver = None

        # 初始化组件：算子信息获取器
        self.info_fetcher = OperatorInfoFetcher()

        self.logger.info(
            f"OperatorMRGenerator initialized: "
            f"LLM={use_llm}, TemplatePool={use_template_pool}, "
            f"PreCheck={use_precheck}, SymPyProof={use_sympy_proof}, "
            f"AutoDerivation={use_auto_derivation}"
        )

    def generate(
        self,
        operator_ir: OperatorIR,
        operator_func: Optional[Callable] = None,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        auto_fetch_info: bool = True,
        framework: FrameworkType = "pytorch",
    ) -> List[MetamorphicRelation]:
        """
        为算子IR生成蜕变关系（多源融合自动生成引擎）

        Args:
            operator_ir: 算子IR对象
            operator_func: 算子函数对象（可选）
                - 主要用途：快速筛选（Pre-check）需要执行函数验证MR
                - 次要用途：通过inspect提取源码（仅对纯Python函数有效）
                - 注意：对于PyTorch内置算子（C++实现），无法提取源码
            operator_code: 算子源代码（可选）
                - 用于自动推导（代码 → SymPy → MR）
                - 用于SymPy形式化证明
                - 可以手动提供，或通过网络搜索获取
            operator_doc: 算子文档（可选）
                - 用于LLM猜想生成MR
                - 可以手动提供，或通过网络搜索获取
            auto_fetch_info: 是否自动从网络获取算子信息（默认True）
            framework: 框架名称（默认pytorch，支持pytorch/tensorflow/paddlepaddle）

        Returns:
            MR对象列表（已验证 + 经过筛选/证明的候选）
        """
        self.logger.info(f"Generating MRs for operator: {operator_ir.name}")

        # ========== 阶段0：信息准备 ==========
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

        # ========== 阶段1：多源MR生成（并行执行） ==========
        verified_mrs: List[MetamorphicRelation] = []  # 已验证的MR
        candidate_mrs: List[MetamorphicRelation] = []  # 待验证的候选MR

        # --- 来源1：自动推导（有代码时执行，生成已验证MR）---
        if (
            self.use_auto_derivation
            and has_code
            and self.code_translator
            and self.mr_deriver
        ):
            self.logger.info("Source 1: Automatic derivation from code...")
            try:
                # 转换为SymPy表达式
                sympy_expr = self.code_translator.translate(
                    code=operator_code, func=operator_func, doc=operator_doc
                )

                if sympy_expr is not None:
                    # 基于符号表达式自动推导MR（这些MR已通过符号验证）
                    derived_mrs = self.mr_deriver.derive_mrs(
                        sympy_expr=sympy_expr,
                        num_inputs=num_inputs,
                        operator_name=operator_ir.name,
                    )
                    # 标记为已验证
                    for mr in derived_mrs:
                        mr.verified = True
                    verified_mrs.extend(derived_mrs)
                    self.logger.info(f"  → Derived {len(derived_mrs)} verified MRs")
                else:
                    self.logger.warning(
                        "  → Failed to convert code to SymPy expression"
                    )
            except Exception as e:
                self.logger.warning(f"  → Auto-derivation failed: {e}")

        # --- 来源2：LLM猜想（始终执行，生成候选MR）---
        if self.use_llm and self.llm_generator:
            self.logger.info("Source 2: LLM-based MR generation...")
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

        # --- 来源3：模板池猜测（始终执行，生成候选MR）---
        if self.use_template_pool and self.template_pool:
            self.logger.info("Source 3: Template pool generation...")
            try:
                template_mrs = self.template_pool.generate_mr_candidates(
                    operator_name=operator_ir.name,
                    operator_inputs=operator_ir.inputs,
                )
                candidate_mrs.extend(template_mrs)
                self.logger.info(f"  → Generated {len(template_mrs)} candidate MRs")
            except Exception as e:
                self.logger.warning(f"  → Template pool generation failed: {e}")

        # --- 向后兼容：如果没有任何MR，使用旧的知识库方法 ---
        if not verified_mrs and not candidate_mrs:
            self.logger.info("Fallback: Using legacy knowledge base method...")
            mr_functions = self.kb.get_mrs_for_operator(operator_ir.name)
            for mr_func in mr_functions:
                try:
                    mr_obj = mr_func(operator_ir.inputs)
                    if isinstance(mr_obj, MetamorphicRelation):
                        candidate_mrs.append(mr_obj)
                except Exception as e:
                    self.logger.error(f"Error generating MR from knowledge base: {e}")

        # 统计
        self.logger.info(
            f"After source generation: "
            f"{len(verified_mrs)} verified MRs, {len(candidate_mrs)} candidate MRs"
        )

        if not verified_mrs and not candidate_mrs:
            self.logger.warning(f"No MRs generated for {operator_ir.name}")
            return []

        # ========== 阶段2：快速筛选（仅对候选MR执行）==========
        if candidate_mrs and self.use_precheck and self.prechecker:
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
                self.logger.warning(
                    "Pre-check skipped: operator_func not provided. "
                    "Pre-check requires a callable function to execute and validate MRs."
                )

        # ========== 阶段3：SymPy形式化证明（仅对候选MR执行）==========
        if candidate_mrs and self.use_sympy_proof and self.sympy_prover:
            if has_code or sympy_expr is not None:
                self.logger.info("Proving candidate MRs using SymPy...")
                proven_mrs = self.sympy_prover.prove_mrs(
                    mrs=candidate_mrs,
                    operator_func=operator_func,
                    operator_code=operator_code,
                    operator_doc=operator_doc,
                    operator_name=operator_ir.name,
                    num_inputs=num_inputs,
                )
                # 标记为已验证
                for mr in proven_mrs:
                    mr.verified = True
                self.logger.info(f"  → Proven {len(proven_mrs)} MRs")
                # 将证明后的MR加入已验证列表
                verified_mrs.extend(proven_mrs)
            else:
                self.logger.warning(
                    "SymPy proof skipped: no code or SymPy expression available. "
                    "Adding unproven candidates to output."
                )
                # 无法证明，将候选MR直接加入输出
                verified_mrs.extend(candidate_mrs)
        elif candidate_mrs:
            # SymPy证明未启用，将候选MR直接加入输出
            verified_mrs.extend(candidate_mrs)

        # ========== 去重 ==========
        final_mrs = self._deduplicate_mrs(verified_mrs)

        self.logger.info(f"Final output: {len(final_mrs)} MRs")
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
        """
        准备算子信息（代码和文档）

        Returns:
            (operator_code, operator_doc) 元组
        """
        # 尝试从函数对象提取代码（仅对纯Python函数有效）
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
        """
        去重MR列表（基于描述）
        """
        seen: Set[str] = set()
        unique_mrs: List[MetamorphicRelation] = []

        for mr in mrs:
            # 使用描述作为唯一标识
            key = mr.description
            if key not in seen:
                seen.add(key)
                unique_mrs.append(mr)

        if len(mrs) != len(unique_mrs):
            self.logger.info(f"Deduplicated: {len(mrs)} → {len(unique_mrs)} MRs")

        return unique_mrs
