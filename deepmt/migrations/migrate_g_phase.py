"""Phase G 数据库迁移脚本：为 test_results 追加运行追踪字段，新建 run_manifests 表。

迁移内容：
  1. test_results 表新增列：run_id / framework_version / random_seed
  2. 新建 run_manifests 表

安全性保证：
  - 所有 ALTER TABLE 操作均先检查列是否已存在（catch OperationalError）
  - CREATE TABLE 使用 IF NOT EXISTS
  - 不修改任何现有行的数据
  - 旧数据的新列值自动为 NULL，旧查询接口不受影响

使用方式：
    source .venv/bin/activate
    PYTHONPATH=$(pwd) python -m deepmt.migrations.migrate_g_phase [--db-path PATH]
"""

import argparse
import sqlite3
from pathlib import Path


_DEFAULT_DB = "data/results/defects.db"


def run(db_path: str = _DEFAULT_DB, dry_run: bool = False) -> None:
    """执行 Phase G 迁移。

    Args:
        db_path: SQLite 数据库路径
        dry_run: True 时只打印计划，不实际执行
    """
    path = Path(db_path)
    if not path.exists():
        print(f"[migrate_g_phase] 数据库不存在，跳过迁移：{path}")
        return

    print(f"[migrate_g_phase] 开始迁移：{path}")

    new_columns = [
        ("run_id",            "TEXT"),
        ("framework_version", "TEXT"),
        ("random_seed",       "INTEGER"),
    ]

    create_run_manifests = """
    CREATE TABLE IF NOT EXISTS run_manifests (
        run_id            TEXT PRIMARY KEY,
        subject_name      TEXT NOT NULL,
        subject_type      TEXT NOT NULL DEFAULT 'operator',
        framework         TEXT NOT NULL,
        framework_version TEXT,
        random_seed       INTEGER,
        n_samples         INTEGER,
        mr_ids            TEXT,
        env_summary       TEXT,
        notes             TEXT,
        timestamp         TEXT NOT NULL
    )
    """

    if dry_run:
        print("[DRY RUN] 将执行以下操作：")
        for col, typedef in new_columns:
            print(f"  ALTER TABLE test_results ADD COLUMN {col} {typedef}")
        print(f"  {create_run_manifests.strip()[:60]}...")
        return

    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    for col, typedef in new_columns:
        try:
            cursor.execute(f"ALTER TABLE test_results ADD COLUMN {col} {typedef}")
            conn.commit()
            print(f"[migrate_g_phase] + test_results.{col} 列已添加")
        except sqlite3.OperationalError:
            print(f"[migrate_g_phase] ~ test_results.{col} 列已存在，跳过")

    cursor.execute(create_run_manifests)
    conn.commit()
    print("[migrate_g_phase] + run_manifests 表已就绪")

    conn.close()
    print("[migrate_g_phase] 迁移完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase G 数据库迁移")
    parser.add_argument(
        "--db-path",
        default=_DEFAULT_DB,
        help=f"SQLite 数据库路径（默认：{_DEFAULT_DB}）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划，不实际执行",
    )
    args = parser.parse_args()
    run(db_path=args.db_path, dry_run=args.dry_run)
