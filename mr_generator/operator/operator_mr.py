"""
算子层MR生成器：多源融合自动生成引擎
支持路径A（LLM）和路径B（模板池），包含快速筛选和SymPy证明
"""

from typing import List, Optional, Callable, Any

from ir.schema import OperatorIR, MetamorphicRelation
from mr_generator.base.knowledge_base import KnowledgeBase
from mr_generator.base.mr_templates import MRTemplatePool
from mr_generator.operator.mr_precheck import MRPreChecker
from mr_generator.operator.sympy_prover import SymPyProver
from mr_generator.operator.operator_llm_mr_generator import OperatorLLMMRGenerator
from mr_generator.operator.code_translator import CodeToSymPyTranslator
from mr_generator.operator.mr_deriver import MRDeriver
from tools.web_search.operator_fetcher import OperatorInfoFetcher
from core.logger import get_logger


class OperatorMRGenerator:
    """
    算子层MR生成器：多源融合自动生成引擎

    流程：
    1. 猜想阶段：路径A（LLM）或路径B（模板池）
    2. 筛选阶段：快速测试过滤
    3. 证明阶段：SymPy形式化证明
    4. 输出：经过证明的MR列表
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        use_llm: bool = False,
        use_template_pool: bool = True,
        use_precheck: bool = True,
        use_sympy_proof: bool = True,
        llm_api_key: Optional[str] = None,
        use_auto_derivation: bool = True,  # 新增：是否使用自动推导
    ):
        """
        初始化算子MR生成器

        Args:
            knowledge_base: 知识库实例（向后兼容，保留旧接口）
            use_llm: 是否使用LLM生成MR（路径A）
            use_template_pool: 是否使用模板池生成MR（路径B）
            use_precheck: 是否使用快速筛选
            use_sympy_proof: 是否使用SymPy证明
            llm_api_key: LLM API密钥（可选）
        """
        # 向后兼容：保留旧的知识库
        self.kb = knowledge_base if knowledge_base else KnowledgeBase()

        # 新组件
        self.use_llm = use_llm
        self.use_template_pool = use_template_pool
        self.use_precheck = use_precheck
        self.use_sympy_proof = use_sympy_proof

        # 初始化组件
        if use_template_pool:
            self.template_pool = MRTemplatePool()
        else:
            self.template_pool = None

        if use_precheck:
            self.prechecker = MRPreChecker()
        else:
            self.prechecker = None

        if use_sympy_proof:
            try:
                self.sympy_prover = SymPyProver()
            except ImportError:
                self.logger.warning("SymPy not available, disabling SymPy proof")
                self.use_sympy_proof = False
                self.sympy_prover = None
        else:
            self.sympy_prover = None

        # LLM现在是必需的（用于代码翻译和MR生成）
        if not use_llm:
            self.logger.warning(
                "LLM is now required for MR generation. Enabling LLM..."
            )
            use_llm = True

        # 创建LLM客户端（共享使用）
        try:
            from tools.llm.client import LLMClient

            self.llm_client = LLMClient(api_key=llm_api_key)
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            raise

        # 创建LLM MR生成器
        try:
            self.llm_generator = OperatorLLMMRGenerator(llm_client=self.llm_client)
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM MR generator: {e}")
            raise

        # 新增：代码到SymPy转换器和MR推导器
        self.use_auto_derivation = use_auto_derivation
        if use_auto_derivation:
            self.code_translator = CodeToSymPyTranslator(llm_client=self.llm_client)
            self.mr_deriver = MRDeriver(
                template_pool=self.template_pool,
                use_z3=False,  # 可选：启用Z3
                llm_client=self.llm_client,
            )
        else:
            self.code_translator = None
            self.mr_deriver = None

        # 新增：算子信息获取器（用于自动搜索算子信息）
        self.info_fetcher = OperatorInfoFetcher()

        self.logger = get_logger()

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
        auto_fetch_info: bool = True,  # 新增：是否自动获取算子信息
        framework: str = "pytorch",  # 新增：框架名称
    ) -> List[MetamorphicRelation]:
        """
        为算子IR生成蜕变关系（自动生成引擎）

        Args:
            operator_ir: 算子IR对象
            operator_func: 算子函数（可选，用于快速筛选）
            operator_code: 算子代码字符串（可选，用于LLM生成）
            operator_doc: 算子文档（可选，用于LLM生成）
            auto_fetch_info: 是否自动从网络获取算子信息（默认True）
            framework: 框架名称（默认pytorch）

        Returns:
            MR对象列表（经过筛选和证明）
        """
        self.logger.info(f"Generating MRs for operator: {operator_ir.name}")

        # 如果启用自动获取且缺少代码/文档，则从网络搜索
        if auto_fetch_info and (not operator_code or not operator_doc):
            self.logger.info(f"Auto-fetching operator info for '{operator_ir.name}'")
            operator_info = self.info_fetcher.fetch_operator_info(
                operator_name=operator_ir.name, framework=framework
            )

            # 使用获取的信息填充缺失的部分
            if not operator_code and operator_info.get("code"):
                operator_code = operator_info["code"]
                self.logger.info("Fetched operator code from web")

            if not operator_doc and operator_info.get("doc"):
                operator_doc = operator_info["doc"]
                self.logger.info("Fetched operator doc from web")

            # 如果没有获取到代码，尝试使用示例代码
            if not operator_code and operator_info.get("examples"):
                operator_code = operator_info["examples"][0]
                self.logger.info("Using example code from web search")

        # 如果仍然没有代码，尝试从函数对象获取
        if not operator_code and operator_func:
            try:
                import inspect

                operator_code = inspect.getsource(operator_func)
                self.logger.info("Extracted code from function object")
            except Exception as e:
                self.logger.debug(f"Failed to extract code from function: {e}")

        num_inputs = len(operator_ir.inputs)
        mr_candidates = []

        # ========== 阶段0：自动推导（新功能）==========
        # 如果提供了代码或函数，尝试自动推导
        if self.use_auto_derivation and self.code_translator and self.mr_deriver:
            if operator_code or operator_func:
                self.logger.info("Attempting automatic MR derivation from code...")
                try:
                    # 转换为SymPy表达式
                    sympy_expr = self.code_translator.translate(
                        code=operator_code, func=operator_func, doc=operator_doc
                    )

                    if sympy_expr is not None:
                        # 基于符号表达式自动推导MR
                        derived_mrs = self.mr_deriver.derive_mrs(
                            sympy_expr=sympy_expr,
                            num_inputs=num_inputs,
                            operator_name=operator_ir.name,
                        )
                        mr_candidates.extend(derived_mrs)
                        self.logger.info(
                            f"Auto-derived {len(derived_mrs)} MRs from code"
                        )
                    else:
                        self.logger.warning(
                            "Failed to convert code to SymPy expression"
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Auto-derivation failed: {e}, falling back to other methods"
                    )

        # ========== 阶段1：猜想生成 ==========

        # 路径A：LLM生成（如果有知识）
        if self.use_llm and self.llm_generator:
            self.logger.info("Using LLM to generate MR candidates (Path A)")
            llm_mrs = self.llm_generator.generate_mr_candidates(
                operator_name=operator_ir.name,
                operator_code=operator_code,
                operator_doc=operator_doc,
                num_inputs=num_inputs,
                top_k=5,
            )
            mr_candidates.extend(llm_mrs)

        # 路径B：模板池生成（无知识）
        if self.use_template_pool and self.template_pool:
            self.logger.info("Using template pool to generate MR candidates (Path B)")
            template_mrs = self.template_pool.generate_mr_candidates(
                operator_name=operator_ir.name, operator_inputs=operator_ir.inputs
            )
            mr_candidates.extend(template_mrs)

        # 向后兼容：如果都没有启用，使用旧的知识库方法
        if not mr_candidates:
            self.logger.info("Using legacy knowledge base method")
            mr_functions = self.kb.get_mrs_for_operator(operator_ir.name)
            for mr_func in mr_functions:
                try:
                    mr_obj = mr_func(operator_ir.inputs)
                    if isinstance(mr_obj, MetamorphicRelation):
                        mr_candidates.append(mr_obj)
                except Exception as e:
                    self.logger.error(f"Error generating MR: {e}")

        if not mr_candidates:
            self.logger.warning(f"No MR candidates generated for {operator_ir.name}")
            return []

        self.logger.info(f"Generated {len(mr_candidates)} MR candidates")

        # ========== 阶段2：快速筛选（Pre-check）==========

        if self.use_precheck and self.prechecker and operator_func:
            self.logger.info("Pre-checking MR candidates...")
            mr_candidates = self.prechecker.filter_mrs(
                operator_func=operator_func,
                mr_candidates=mr_candidates,
                original_inputs=operator_ir.inputs,
            )
            self.logger.info(f"After pre-check: {len(mr_candidates)} MRs remain")

        # ========== 阶段3：SymPy形式化证明 ==========

        if self.use_sympy_proof and self.sympy_prover:
            self.logger.info("Proving MRs using SymPy...")
            proven_mrs = self.sympy_prover.prove_mrs(
                mrs=mr_candidates,
                operator_func=operator_func,
                operator_code=operator_code,
                operator_doc=operator_doc,
                operator_name=operator_ir.name,
                num_inputs=num_inputs,
            )
            self.logger.info(f"After SymPy proof: {len(proven_mrs)} MRs proven")
            return proven_mrs

        # 如果没有启用SymPy证明，返回筛选后的MR
        self.logger.info(f"Returning {len(mr_candidates)} MRs (without SymPy proof)")
        return mr_candidates
