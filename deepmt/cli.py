"""
DeepMT 顶层 CLI 入口

使用方式:
    python -m deepmt <command> [options]

命令组:
    mr       MR 生成与管理
    test     测试执行
    repo     MR 知识库管理
    catalog  算子目录浏览与查询
    data     数据目录管理
    health   系统健康检查
    ui       Web 仪表盘服务器管理
"""

import click

from deepmt import __version__
from deepmt.commands.mr import mr
from deepmt.commands.test import test
from deepmt.commands.repo import repo
from deepmt.commands.health import health
from deepmt.commands.catalog import catalog
from deepmt.commands.data import data
from deepmt.commands.ui import ui
from deepmt.commands.experiment import experiment
from deepmt.commands.case import case

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], max_content_width=100)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, "-V", "--version", prog_name="deepmt")
def cli():
    """DeepMT — 深度学习框架蜕变关系自动生成与分层测试系统。

    \b
    命令组:
      mr          MR 生成与管理（算子层）
      test        测试执行
      repo        MR 知识库管理
      catalog     算子目录浏览与跨框架查询
      data        数据目录管理（日志清理等）
      health      系统健康检查
      ui          Web 仪表盘（deepmt ui start）
      experiment  论文实验基准与数据生产（Phase L）
      case        真实缺陷案例管理（Phase M）

    \b
    快速开始:
      deepmt catalog list --framework pytorch                          # 列出 PyTorch 所有算子
      deepmt catalog search relu                                       # 跨框架搜索 relu
      deepmt catalog info relu                                         # 查询 relu 的框架分布和 MR 数量
      deepmt catalog latest-version --framework pytorch                # 查询 PyTorch 最新版本
      deepmt catalog fetch-doc torch.matmul                            # 获取 torch.matmul 文档
      deepmt catalog update-api-list --framework pytorch               # 更新 PyTorch API 模块列表
      deepmt catalog sync --framework pytorch                          # Agent 自动更新算子目录
      deepmt mr generate relu --save                                   # 为 relu 生成并保存 MR
      deepmt test operator relu                                        # 运行蜕变测试
      deepmt data clean-logs --keep-days 7                             # 清理 7 天前的日志
    """


cli.add_command(mr)
cli.add_command(test)
cli.add_command(repo)
cli.add_command(health)
cli.add_command(catalog)
cli.add_command(data)
cli.add_command(ui)
cli.add_command(experiment)
cli.add_command(case)
