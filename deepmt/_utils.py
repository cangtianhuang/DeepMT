"""CLI 内部共用工具函数"""

import sys
import click


def not_implemented_error(feature: str, hint: str = ""):
    """以统一格式打印"功能未实现"错误并退出。"""
    msg = f"功能未实现: {feature}"
    if hint:
        msg += f"\n提示: {hint}"
    click.echo(click.style(msg, fg="yellow"), err=True)
    sys.exit(2)


def get_repo(db_path: str = "data/mr_knowledge_base.db"):
    """获取 MRRepository 实例（复用配置中的路径）。"""
    from mr_generator.base.mr_repository import MRRepository
    return MRRepository(db_path=db_path)


def get_results_manager(db_path: str = "data/defects.db"):
    """获取 ResultsManager 实例。"""
    from core.results_manager import ResultsManager
    return ResultsManager(db_path=db_path)
