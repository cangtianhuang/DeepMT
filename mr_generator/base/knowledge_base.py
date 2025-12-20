"""
知识库：存储算子、模型、应用三个级别的知识
支持从配置文件读取，便于后续拓展开发
"""

import yaml
import uuid
from pathlib import Path
from typing import List, Dict, Callable, Any, Optional

from ir.schema import MetamorphicRelation
from core.logger import get_logger


class KnowledgeBase:
    """
    三层知识库：管理算子、模型、应用三个级别的知识

    层次划分：
    - operator: 算子层知识（数学性质和MR生成规则）
    - model: 模型层知识（网络结构相关的MR）
    - application: 应用层知识（语义层面的MR）
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化知识库

        Args:
            config_path: 配置文件路径（如果为None，则使用默认路径）
        """
        self.logger = get_logger()
        self.operator_knowledge: Dict[str, List[Dict[str, Any]]] = {}
        self.model_knowledge: Dict[str, List[Dict[str, Any]]] = {}
        self.application_knowledge: Dict[str, List[Dict[str, Any]]] = {}

        if config_path is None:
            # 默认配置文件路径：mr_generator/config/knowledge_base.yaml
            base_dir = Path(__file__).parent.parent
            default_path = base_dir / "config" / "knowledge_base.yaml"
            self.config_path = default_path
        else:
            self.config_path = Path(config_path)
        self._load_config()

    def _load_config(self):
        """从配置文件加载知识库"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # 加载算子层知识
            self.operator_knowledge = config.get("operator", {})

            # 加载模型层知识
            self.model_knowledge = config.get("model", {})

            # 加载应用层知识
            self.application_knowledge = config.get("application", {})

            self.logger.info(
                f"Loaded knowledge base from {self.config_path}: "
                f"{len(self.operator_knowledge)} operators, "
                f"{len(self.model_knowledge)} models, "
                f"{len(self.application_knowledge)} applications"
            )

        except FileNotFoundError:
            self.logger.warning(
                f"Knowledge base config file not found: {self.config_path}"
            )
            # 使用空知识库，不抛出异常
        except Exception as e:
            self.logger.error(f"Failed to load knowledge base config: {e}")
            # 使用空知识库，不抛出异常

    def get_mrs_for_operator(self, operator_name: str) -> List[Callable]:
        """
        获取指定算子的所有MR生成函数（向后兼容接口）

        Args:
            operator_name: 算子名称

        Returns:
            MR生成函数列表
        """
        mrs_config = self.operator_knowledge.get(operator_name, [])
        mrs = []

        for mr_config in mrs_config:
            try:
                transform_code = mr_config.get("transform_code", "")
                transform_func = (
                    eval(transform_code) if transform_code else lambda *args: args
                )

                def create_mr_func(transform_func, description, expected):
                    def mr_func(inputs):
                        return MetamorphicRelation(
                            id=str(uuid.uuid4()),
                            description=description,
                            transform=transform_func,
                            expected=expected,
                            tolerance=1e-6,
                            layer="operator",
                        )

                    return mr_func

                mr_func = create_mr_func(
                    transform_func,
                    mr_config.get("description", ""),
                    mr_config.get("expected", "equal"),
                )
                mrs.append(mr_func)
            except Exception as e:
                self.logger.warning(f"Failed to create MR function from config: {e}")

        if not mrs:
            self.logger.debug(f"No MRs found for operator: {operator_name}")

        return mrs

    def get_mrs_for_model(self, model_name: str) -> List[Callable]:
        """
        获取指定模型的所有MR生成函数

        Args:
            model_name: 模型名称

        Returns:
            MR生成函数列表
        """
        mrs_config = self.model_knowledge.get(model_name, [])
        mrs = []

        for mr_config in mrs_config:
            try:
                transform_code = mr_config.get("transform_code", "")
                transform_func = (
                    eval(transform_code) if transform_code else lambda *args: args
                )

                def create_mr_func(transform_func, description, expected):
                    def mr_func(inputs):
                        return MetamorphicRelation(
                            id=str(uuid.uuid4()),
                            description=description,
                            transform=transform_func,
                            expected=expected,
                            tolerance=1e-6,
                            layer="model",
                        )

                    return mr_func

                mr_func = create_mr_func(
                    transform_func,
                    mr_config.get("description", ""),
                    mr_config.get("expected", "invariant"),
                )
                mrs.append(mr_func)
            except Exception as e:
                self.logger.warning(f"Failed to create MR function from config: {e}")

        if not mrs:
            self.logger.debug(f"No MRs found for model: {model_name}")

        return mrs

    def get_mrs_for_application(self, application_name: str) -> List[Callable]:
        """
        获取指定应用的所有MR生成函数

        Args:
            application_name: 应用名称

        Returns:
            MR生成函数列表
        """
        mrs_config = self.application_knowledge.get(application_name, [])
        mrs = []

        for mr_config in mrs_config:
            try:
                transform_code = mr_config.get("transform_code", "")
                transform_func = (
                    eval(transform_code) if transform_code else lambda *args: args
                )

                def create_mr_func(transform_func, description, expected):
                    def mr_func(inputs):
                        return MetamorphicRelation(
                            id=str(uuid.uuid4()),
                            description=description,
                            transform=transform_func,
                            expected=expected,
                            tolerance=1e-6,
                            layer="application",
                        )

                    return mr_func

                mr_func = create_mr_func(
                    transform_func,
                    mr_config.get("description", ""),
                    mr_config.get("expected", "invariant"),
                )
                mrs.append(mr_func)
            except Exception as e:
                self.logger.warning(f"Failed to create MR function from config: {e}")

        if not mrs:
            self.logger.debug(f"No MRs found for application: {application_name}")

        return mrs

    def add_operator_knowledge(self, operator_name: str, mr_config: Dict[str, Any]):
        """
        动态添加算子知识（运行时注册）

        Args:
            operator_name: 算子名称
            mr_config: MR配置字典
        """
        if operator_name not in self.operator_knowledge:
            self.operator_knowledge[operator_name] = []
        self.operator_knowledge[operator_name].append(mr_config)
        self.logger.debug(f"Added knowledge for operator: {operator_name}")

    def add_model_knowledge(self, model_name: str, mr_config: Dict[str, Any]):
        """
        动态添加模型知识（运行时注册）

        Args:
            model_name: 模型名称
            mr_config: MR配置字典
        """
        if model_name not in self.model_knowledge:
            self.model_knowledge[model_name] = []
        self.model_knowledge[model_name].append(mr_config)
        self.logger.debug(f"Added knowledge for model: {model_name}")

    def add_application_knowledge(
        self, application_name: str, mr_config: Dict[str, Any]
    ):
        """
        动态添加应用知识（运行时注册）

        Args:
            application_name: 应用名称
            mr_config: MR配置字典
        """
        if application_name not in self.application_knowledge:
            self.application_knowledge[application_name] = []
        self.application_knowledge[application_name].append(mr_config)
        self.logger.debug(f"Added knowledge for application: {application_name}")
