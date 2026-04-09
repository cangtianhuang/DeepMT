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


def get_repo(repo_dir: str = "data/mr_repository/operator"):
    """获取 MRRepository 实例（用户工作区）。"""
    from deepmt.mr_generator.base.mr_repository import MRRepository
    return MRRepository(repo_dir=repo_dir)


def get_library(layer: str = "operator", library_dir: str = "data/mr_library"):
    """获取 MRLibrary 实例（项目库）。"""
    from deepmt.mr_generator.base.mr_library import MRLibrary
    return MRLibrary(layer=layer, library_dir=library_dir)


def get_results_manager(db_path: str = "data/defects.db"):
    """获取 ResultsManager 实例。"""
    from deepmt.core.results_manager import ResultsManager
    return ResultsManager(db_path=db_path)
