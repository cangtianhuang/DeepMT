"""
结果管理与比对模块
负责存储测试结果、比对MR期望与实际输出、检测缺陷
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from analysis.defect_classifier import DefectClassifier
from ir.schema import ApplicationIR, MetamorphicRelation, ModelIR, OperatorIR


class ResultsManager:
    """结果管理器：存储、比对、分析测试结果"""

    def __init__(self, db_path: str = "data/defects.db", tolerance: float = 1e-6):
        """
        初始化结果管理器

        Args:
            db_path: SQLite数据库路径
            tolerance: 默认数值容差
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.tolerance = tolerance
        self.defect_classifier = DefectClassifier()
        self._init_database()

    def _init_database(self):
        """初始化SQLite数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建测试结果表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ir_name TEXT NOT NULL,
                ir_type TEXT NOT NULL,
                framework TEXT NOT NULL,
                mr_id TEXT NOT NULL,
                mr_description TEXT,
                status TEXT NOT NULL,
                original_output TEXT,
                transformed_output TEXT,
                defect_type TEXT,
                defect_details TEXT,
                timestamp TEXT NOT NULL
            )
        """
        )

        # 创建缺陷统计表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS defect_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ir_name TEXT NOT NULL,
                framework TEXT NOT NULL,
                total_tests INTEGER,
                passed_tests INTEGER,
                failed_tests INTEGER,
                defect_types TEXT,
                last_updated TEXT NOT NULL
            )
        """
        )

        conn.commit()
        conn.close()

    def compare_and_store(
        self, ir_object: Any, results: List[Tuple[MetamorphicRelation, Tuple[Any, Any]]]
    ):
        """
        比对MR期望与实际输出，并存储结果

        Args:
            ir_object: IR对象（OperatorIR, ModelIR, 或 ApplicationIR）
            results: MR和输出结果的列表，每个元素为 (MR对象, (原始输出, 变换后输出))
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for mr, (orig_output, trans_output) in results:
            # 比对结果
            status, defect_type, defect_details = self._compare_outputs(
                mr, orig_output, trans_output
            )

            # 序列化输出（支持numpy数组和tensor）
            orig_str = self._serialize_output(orig_output)
            trans_str = self._serialize_output(trans_output)

            # 获取IR信息
            ir_name = ir_object.name if hasattr(ir_object, "name") else "unknown"
            ir_type = type(ir_object).__name__

            # 存储到数据库
            cursor.execute(
                """
                INSERT INTO test_results 
                (ir_name, ir_type, framework, mr_id, mr_description, status,
                 original_output, transformed_output, defect_type, defect_details, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    ir_name,
                    ir_type,
                    "pytorch",  # TODO: 从上下文获取框架名称
                    mr.id,
                    mr.description,
                    status,
                    orig_str,
                    trans_str,
                    defect_type,
                    defect_details,
                    datetime.now().isoformat(),
                ),
            )

        conn.commit()
        conn.close()

        # 更新缺陷统计
        self._update_summary(ir_object, results)

    def _compare_outputs(
        self, mr: MetamorphicRelation, orig_output: Any, trans_output: Any
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        比对原始输出和变换后输出，检查MR是否满足

        Returns:
            (status, defect_type, defect_details)
            status: "PASS" 或 "FAIL"
            defect_type: 缺陷类型（如果失败）
            defect_details: 缺陷详情（如果失败）
        """
        tolerance = mr.tolerance if mr.tolerance is not None else self.tolerance

        try:
            # 使用缺陷分类器进行比对
            is_match, defect_info = self.defect_classifier.compare(
                orig_output, trans_output, mr, tolerance
            )

            if is_match:
                return "PASS", None, None
            else:
                defect_type = defect_info.get("type", "UNKNOWN")
                defect_details = defect_info.get("details", "")
                return "FAIL", defect_type, defect_details

        except Exception as e:
            return "ERROR", "EXCEPTION", str(e)

    def _serialize_output(self, output: Any) -> str:
        """序列化输出为字符串（支持numpy数组、tensor等）"""
        try:
            # 如果是numpy数组
            if isinstance(output, np.ndarray):
                return json.dumps(
                    {
                        "type": "ndarray",
                        "shape": list(output.shape),
                        "dtype": str(output.dtype),
                        "data": output.tolist(),
                    }
                )

            # 如果是PyTorch tensor
            if hasattr(output, "numpy"):
                return json.dumps(
                    {
                        "type": "tensor",
                        "shape": list(output.shape),
                        "dtype": str(output.dtype),
                        "data": output.detach().cpu().numpy().tolist(),
                    }
                )

            # 如果是标量
            if isinstance(output, (int, float, bool)):
                return json.dumps({"type": "scalar", "value": output})

            # 其他类型转为字符串
            return json.dumps({"type": "other", "value": str(output)})

        except Exception as e:
            return json.dumps({"type": "error", "error": str(e)})

    def _update_summary(self, ir_object: Any, results: List[Tuple]):
        """更新缺陷统计摘要"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        ir_name = ir_object.name if hasattr(ir_object, "name") else "unknown"
        framework = "pytorch"  # TODO: 从上下文获取

        # 统计结果
        total = len(results)
        passed = sum(
            1
            for _, (orig, trans) in results
            if self._compare_outputs(_, orig, trans)[0] == "PASS"
        )
        failed = total - passed

        # 收集缺陷类型
        defect_types = {}
        for mr, (orig, trans) in results:
            status, defect_type, _ = self._compare_outputs(mr, orig, trans)
            if status == "FAIL" and defect_type:
                defect_types[defect_type] = defect_types.get(defect_type, 0) + 1

        # 更新或插入统计
        cursor.execute(
            """
            INSERT OR REPLACE INTO defect_summary
            (ir_name, framework, total_tests, passed_tests, failed_tests, 
             defect_types, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                ir_name,
                framework,
                total,
                passed,
                failed,
                json.dumps(defect_types),
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

    def get_summary(self, ir_name: Optional[str] = None) -> List[Dict]:
        """获取测试结果摘要"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if ir_name:
            cursor.execute(
                """
                SELECT * FROM defect_summary WHERE ir_name = ?
            """,
                (ir_name,),
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
