"""算子信息获取器：通过 CrawlAgent 从网络获取算子文档"""

from typing import Any, Dict, Optional

from core.config_loader import get_config_value
from core.framework import FrameworkType
from core.logger import get_logger, log_structured


class OperatorInfoFetcher:
    """
    算子信息获取器

    使用 CrawlAgent（tools.agent.TaskRunner）获取算子文档。
    需要在 config.yaml 中启用 agent：

        agent:
          enabled: true

    若 agent 未启用或执行出错，返回空结果，不抛异常。
    """

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.enabled: bool = get_config_value("agent.enabled", False)
        self._runner = None  # 懒加载

    @property
    def runner(self):
        """懒加载 TaskRunner，避免启动时初始化 LLM"""
        if self._runner is None:
            from tools.agent.task_runner import TaskRunner
            self._runner = TaskRunner()
        return self._runner

    def fetch_operator_info(
        self,
        operator_name: str,
        framework: FrameworkType = "pytorch",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        获取算子信息。

        Args:
            operator_name: 算子名称（如 "relu"、"torch.nn.ReLU"）
            framework:     框架名称（pytorch / tensorflow / paddlepaddle）
            use_cache:     是否使用缓存

        Returns:
            {"name": str, "doc": str, "source_urls": list}
            若未启用或出错则返回空值但保持结构
        """
        empty = {"name": operator_name, "doc": "", "source_urls": []}

        if not self.enabled:
            log_structured(
                self.logger,
                "FETCH",
                "Agent disabled (agent.enabled=false), skipping doc fetch",
                operator=operator_name,
                level="DEBUG",
            )
            return empty

        log_structured(
            self.logger,
            "FETCH",
            f"Fetching doc for '{operator_name}' via CrawlAgent",
            framework=framework,
        )

        try:
            result = self.runner.run_task(
                "get_operator_doc",
                inputs={"operator_name": operator_name, "framework": framework},
                use_cache=use_cache,
            )
            doc = result.get("doc", "")
            source_url = result.get("source_url", "")
            return {
                "name": result.get("operator_name", operator_name),
                "doc": doc,
                "source_urls": [source_url] if source_url else [],
            }
        except Exception as e:
            log_structured(
                self.logger,
                "FETCH",
                f"CrawlAgent fetch failed for '{operator_name}': {e}",
                level="WARNING",
            )
            return empty

    def get_operator_doc(
        self,
        operator_name: str,
        framework: FrameworkType = "pytorch",
    ) -> Optional[str]:
        """
        获取算子文档字符串。

        Returns:
            文档字符串；若未启用或获取失败返回 None
        """
        doc = self.fetch_operator_info(operator_name, framework).get("doc", "")
        return doc if doc else None
