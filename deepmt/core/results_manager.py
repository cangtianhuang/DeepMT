"""
结果管理器：持久化 MR 测试结果

职责（单一）：
  - 接收已评估好的 OracleResult，写入 SQLite
  - 维护 test_results 与 defect_summary 两张表
  - 提供查询接口（get_summary、get_failed_tests）

比较/评估逻辑已移至 deepmt/analysis/mr_verifier.py（MRVerifier）。
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deepmt.core.logger import logger
from deepmt.ir.schema import MetamorphicRelation, OracleResult


class ResultsManager:
    """结果管理器：持久化 MR 测试结果"""

    def __init__(self, db_path: str = "data/results/defects.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """初始化数据库表结构，并做向前兼容迁移"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS test_results (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ir_name           TEXT NOT NULL,
                ir_type           TEXT NOT NULL,
                framework         TEXT NOT NULL,
                mr_id             TEXT NOT NULL,
                mr_description    TEXT,
                oracle_expr       TEXT,
                status            TEXT NOT NULL,
                actual_diff       REAL,
                tolerance         REAL,
                defect_details    TEXT,
                timestamp         TEXT NOT NULL,
                run_id            TEXT,
                framework_version TEXT,
                random_seed       INTEGER
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS defect_summary (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ir_name       TEXT NOT NULL,
                framework     TEXT NOT NULL,
                total_tests   INTEGER,
                passed_tests  INTEGER,
                failed_tests  INTEGER,
                last_updated  TEXT NOT NULL
            )
            """
        )

        conn.commit()

        # 新列迁移（对已有旧库兼容追加）
        for col, definition in [
            ("oracle_expr",       "TEXT"),
            ("actual_diff",       "REAL"),
            ("tolerance",         "REAL"),
            ("run_id",            "TEXT"),
            ("framework_version", "TEXT"),
            ("random_seed",       "INTEGER"),
        ]:
            try:
                cursor.execute(
                    f"ALTER TABLE test_results ADD COLUMN {col} {definition}"
                )
                conn.commit()
            except sqlite3.OperationalError:
                pass  # 列已存在

        conn.close()

    def store_result(
        self,
        ir_object: Any,
        results: List[Tuple[MetamorphicRelation, OracleResult]],
        framework: str,
        run_id: Optional[str] = None,
        framework_version: Optional[str] = None,
        random_seed: Optional[int] = None,
    ):
        """
        持久化一批 MR 测试结果。

        Args:
            ir_object:         IR 对象（OperatorIR / ModelIR / ApplicationIR）
            results:           [(MR, OracleResult), ...] 列表
            framework:         框架名称字符串（如 "pytorch"）
            run_id:            关联的实验运行 ID（可选）
            framework_version: 框架版本字符串（可选）
            random_seed:       随机种子（可选）
        """
        ir_name = ir_object.name if hasattr(ir_object, "name") else "unknown"
        ir_type = type(ir_object).__name__
        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for mr, oracle_result in results:
            cursor.execute(
                """
                INSERT INTO test_results
                (ir_name, ir_type, framework, mr_id, mr_description,
                 oracle_expr, status, actual_diff, tolerance, defect_details, timestamp,
                 run_id, framework_version, random_seed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ir_name,
                    ir_type,
                    framework,
                    mr.id,
                    mr.description,
                    oracle_result.expr,
                    "PASS" if oracle_result.passed else "FAIL",
                    oracle_result.actual_diff if oracle_result.actual_diff != float("inf") else None,
                    oracle_result.tolerance,
                    oracle_result.detail or None,
                    now,
                    run_id,
                    framework_version,
                    random_seed,
                ),
            )

        conn.commit()
        conn.close()

        self._update_summary(ir_name, framework, results)

    def _update_summary(
        self,
        ir_name: str,
        framework: str,
        results: List[Tuple[MetamorphicRelation, OracleResult]],
    ):
        """更新 defect_summary 聚合统计"""
        total = len(results)
        passed = sum(1 for _, r in results if r.passed)
        failed = total - passed

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO defect_summary
            (ir_name, framework, total_tests, passed_tests, failed_tests, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ir_name, framework, total, passed, failed, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    def get_summary(self, ir_name: Optional[str] = None) -> List[Dict]:
        """获取测试结果摘要"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if ir_name:
            cursor.execute(
                "SELECT * FROM defect_summary WHERE ir_name = ?", (ir_name,)
            )
        else:
            cursor.execute("SELECT * FROM defect_summary")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return [dict(zip(columns, row)) for row in rows]

    def get_failed_tests(self, limit: int = 100) -> List[Dict]:
        """获取失败的测试用例"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM test_results
            WHERE status = 'FAIL'
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return [dict(zip(columns, row)) for row in rows]
