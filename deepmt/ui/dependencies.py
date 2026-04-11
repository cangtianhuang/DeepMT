"""
共用依赖注入层

向 FastAPI 路由提供数据源单例，供 FastAPI Depends() 调用。
使用 lru_cache 实现进程内单例（演示场景数据变化不频繁）。

用法示例：
    from fastapi import Depends
    from deepmt.ui.dependencies import get_mr_repository

    @router.get("/mr")
    async def mr_page(repo=Depends(get_mr_repository)):
        ...
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def get_mr_repository():
    """MR 用户工作区仓库（YAML per operator）"""
    from deepmt.mr_generator.base.mr_repository import MRRepository
    return MRRepository()


@lru_cache(maxsize=1)
def get_results_manager():
    """测试结果持久化（SQLite）"""
    from deepmt.core.results_manager import ResultsManager
    return ResultsManager()


@lru_cache(maxsize=1)
def get_evidence_collector():
    """证据包采集器"""
    from deepmt.analysis.evidence_collector import EvidenceCollector
    return EvidenceCollector()


@lru_cache(maxsize=1)
def get_cross_framework_tester():
    """跨框架一致性测试器"""
    from deepmt.analysis.cross_framework_tester import CrossFrameworkTester
    return CrossFrameworkTester()


@lru_cache(maxsize=1)
def get_experiment_organizer():
    """论文实验数据组织器（RQ1-RQ4 聚合）"""
    from deepmt.analysis.experiment_organizer import ExperimentOrganizer
    return ExperimentOrganizer()
