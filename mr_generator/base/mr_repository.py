"""
MR知识库：持久化存储和管理蜕变关系
实现MR生成与测试的分离
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logger import get_logger, log_structured
from ir.schema import MetamorphicRelation


class MRRepository:
    """
    MR知识库：存储和管理蜕变关系

    实现MR生成与测试的分离：
    - MR生成阶段：生成MR并保存到知识库
    - MR测试阶段：从知识库加载MR进行测试
    """

    def __init__(self, db_path: str = "data/mr_knowledge_base.db"):
        """
        初始化MR知识库

        Args:
            db_path: SQLite数据库路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.__class__.__name__)
        self._init_database()

    def _init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS mr_knowledge_base (
                id TEXT PRIMARY KEY,
                operator_name TEXT NOT NULL,
                mr_id TEXT NOT NULL,
                mr_description TEXT NOT NULL,
                mr_type TEXT NOT NULL,
                mr_data TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                verified BOOLEAN DEFAULT 0,
                precheck_passed BOOLEAN,
                sympy_proven BOOLEAN,
                last_validated_at TEXT,
                validation_summary TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_operator_name
            ON mr_knowledge_base(operator_name)
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS mr_validation_records (
                id TEXT PRIMARY KEY,
                record_id TEXT NOT NULL,
                mr_id TEXT NOT NULL,
                operator_name TEXT NOT NULL,
                validation_type TEXT NOT NULL,
                validation_result TEXT NOT NULL,
                error_message TEXT,
                validated_at TEXT NOT NULL,
                validation_details TEXT,
                FOREIGN KEY (record_id) REFERENCES mr_knowledge_base(id)
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mr_id_validation
            ON mr_validation_records(mr_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_record_id_validation
            ON mr_validation_records(record_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_operator_name_validation
            ON mr_validation_records(operator_name)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_validated_at
            ON mr_validation_records(validated_at)
        """
        )

        conn.commit()
        conn.close()
        log_structured(
            self.logger, "REPO", f"MR knowledge base initialized at: {self.db_path}"
        )

    def save(
        self, operator_name: str, mrs: List[MetamorphicRelation], version: int = 1
    ) -> int:
        """
        保存MR列表到知识库

        Args:
            operator_name: 算子名称
            mrs: MR对象列表
            version: MR版本号

        Returns:
            保存的MR数量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        saved_count = 0
        timestamp = datetime.now().isoformat()

        for mr in mrs:
            try:
                mr_data = self._serialize_mr(mr)

                record_id = str(uuid.uuid4())

                cursor.execute(
                    """
                    INSERT INTO mr_knowledge_base
                    (id, operator_name, mr_id, mr_description, mr_type, mr_data, version, created_at, updated_at, verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record_id,
                        operator_name,
                        mr.id,
                        mr.description,
                        mr.oracle_expr,
                        mr_data,
                        version,
                        timestamp,
                        timestamp,
                        mr.verified,
                    ),
                )

                saved_count += 1

            except Exception as e:
                self.logger.error(f"Error saving MR {mr.id}: {e}")

        conn.commit()
        conn.close()

        log_structured(
            self.logger,
            "REPO",
            f"Saved {saved_count} MRs for operator: {operator_name}",
        )
        return saved_count

    def load(
        self, operator_name: str, version: Optional[int] = None
    ) -> List[MetamorphicRelation]:
        """
        从知识库加载MR列表

        Args:
            operator_name: 算子名称
            version: MR版本号（可选，None表示最新版本）

        Returns:
            MR对象列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if version is None:
            # 获取最新版本
            cursor.execute(
                """
                SELECT MAX(version) FROM mr_knowledge_base
                WHERE operator_name = ?
            """,
                (operator_name,),
            )
            result = cursor.fetchone()
            version = result[0] if result[0] else 1

        # 加载MR
        cursor.execute(
            """
            SELECT mr_data, mr_id, mr_description, mr_type, verified
            FROM mr_knowledge_base
            WHERE operator_name = ? AND version = ?
        """,
            (operator_name, version),
        )

        rows = cursor.fetchall()
        conn.close()

        mrs = []
        for row in rows:
            try:
                mr_data_str, mr_id, description, mr_type, verified = row
                mr = self._deserialize_mr(mr_data_str, mr_id, description, mr_type)
                if verified is not None:
                    mr.verified = bool(verified)
                mrs.append(mr)
            except Exception as e:
                self.logger.error(f"Error loading MR: {e}")

        log_structured(
            self.logger,
            "REPO",
            f"Loaded {len(mrs)} MRs for operator: {operator_name} (version {version})",
        )
        return mrs

    def exists(self, operator_name: str, version: Optional[int] = None) -> bool:
        """
        检查算子是否有保存的MR

        Args:
            operator_name: 算子名称
            version: MR版本号（可选）

        Returns:
            是否存在
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if version is None:
            cursor.execute(
                """
                SELECT COUNT(*) FROM mr_knowledge_base
                WHERE operator_name = ?
            """,
                (operator_name,),
            )
        else:
            cursor.execute(
                """
                SELECT COUNT(*) FROM mr_knowledge_base
                WHERE operator_name = ? AND version = ?
            """,
                (operator_name, version),
            )

        count = cursor.fetchone()[0]
        conn.close()

        return count > 0

    def list_operators(self) -> List[str]:
        """列出所有有MR的算子名称"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT DISTINCT operator_name FROM mr_knowledge_base
            ORDER BY operator_name
        """
        )

        operators = [row[0] for row in cursor.fetchall()]
        conn.close()

        return operators

    def get_versions(self, operator_name: str) -> List[int]:
        """获取算子的所有MR版本"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT DISTINCT version FROM mr_knowledge_base
            WHERE operator_name = ?
            ORDER BY version DESC
        """,
            (operator_name,),
        )

        versions = [row[0] for row in cursor.fetchall()]
        conn.close()

        return versions

    def _serialize_mr(self, mr: MetamorphicRelation) -> str:
        """序列化MR对象为JSON字符串"""
        data = {
            "id": mr.id,
            "description": mr.description,
            "oracle_expr": mr.oracle_expr,
            "transform_code": mr.transform_code,
            "category": mr.category,
            "tolerance": mr.tolerance,
            "analysis": mr.analysis,
            "layer": mr.layer,
            "verified": mr.verified,
            "transform_type": "lambda",
        }
        return json.dumps(data)

    def _deserialize_mr(
        self, mr_data_str: str, mr_id: str, description: str, mr_type: str
    ) -> MetamorphicRelation:
        """反序列化MR对象"""
        data = json.loads(mr_data_str)

        transform = self._rebuild_transform(mr_type, description)

        return MetamorphicRelation(
            id=mr_id,
            description=description,
            transform=transform,
            transform_code=data.get("transform_code", ""),
            oracle_expr=data.get("oracle_expr", mr_type),
            category=data.get("category", "general"),
            tolerance=data.get("tolerance"),
            analysis=data.get("analysis", ""),
            layer=data.get("layer", "operator"),
            verified=data.get("verified", False),
        )

    def _rebuild_transform(self, expected: str, description: str):
        """重建transform函数"""
        if "commutative" in description.lower():
            return lambda *args: (
                (args[1], args[0]) + args[2:] if len(args) >= 2 else args
            )
        elif "associative" in description.lower():
            return lambda *args: args
        elif "anti-commutative" in description.lower():
            return lambda *args: (
                (args[1], args[0]) + args[2:] if len(args) >= 2 else args
            )
        else:
            return lambda *args: args

    def save_with_validation(
        self,
        operator_name: str,
        mrs: List[MetamorphicRelation],
        validation_records: List[Dict],
        version: int = 1,
    ) -> int:
        """
        保存MR及其验证记录

        Args:
            operator_name: 算子名称
            mrs: MR对象列表
            validation_records: 验证记录列表，每个记录包含:
                {
                    'mr_id': str,
                    'validation_type': 'precheck' | 'sympy',
                    'result': 'passed' | 'failed',
                    'error_message': str,
                    'details': dict,
                    'validated_at': str
                }
            version: MR版本号

        Returns:
            保存的MR数量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        saved_count = 0
        timestamp = datetime.now().isoformat()

        # 创建验证记录的索引映射
        validation_map = {}
        for record in validation_records:
            mr_id = record.get("mr_id")
            if mr_id:
                if mr_id not in validation_map:
                    validation_map[mr_id] = []
                validation_map[mr_id].append(record)

        for mr in mrs:
            try:
                mr_data = self._serialize_mr(mr)

                record_id = str(uuid.uuid4())

                # 从验证记录中汇总验证状态
                validations = validation_map.get(mr.id, [])
                precheck_passed = None
                sympy_proven = None
                last_validated_at = None
                verified = mr.verified

                for v in validations:
                    v_type = v.get("validation_type")
                    v_result = v.get("validation_result")
                    if v_type == "precheck":
                        precheck_passed = v_result == "passed"
                    elif v_type == "sympy":
                        sympy_proven = v_result == "passed"

                    if v.get("validated_at"):
                        if (
                            last_validated_at is None
                            or v["validated_at"] > last_validated_at
                        ):
                            last_validated_at = v["validated_at"]

                cursor.execute(
                    """
                    INSERT INTO mr_knowledge_base
                    (id, operator_name, mr_id, mr_description, mr_type, mr_data, version, created_at, updated_at, verified, precheck_passed, sympy_proven, last_validated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record_id,
                        operator_name,
                        mr.id,
                        mr.description,
                        mr.oracle_expr,
                        mr_data,
                        version,
                        timestamp,
                        timestamp,
                        verified,
                        precheck_passed,
                        sympy_proven,
                        last_validated_at,
                    ),
                )

                for v in validations:
                    validation_id = str(uuid.uuid4())
                    cursor.execute(
                        """
                        INSERT INTO mr_validation_records
                        (id, record_id, mr_id, operator_name, validation_type, validation_result, error_message, validated_at, validation_details)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            validation_id,
                            record_id,
                            mr.id,
                            operator_name,
                            v.get("validation_type"),
                            v.get("validation_result"),
                            v.get("error_message", ""),
                            v.get("validated_at", timestamp),
                            json.dumps(v.get("details", {})),
                        ),
                    )

                saved_count += 1

            except Exception as e:
                self.logger.error(f"Error saving MR {mr.id} with validation: {e}")

        conn.commit()
        conn.close()

        log_structured(
            self.logger,
            "REPO",
            f"Saved {saved_count} MRs for '{operator_name}' | version: {version}",
        )
        return saved_count

    def update_validation_status(
        self,
        record_id: str,
        mr_id: str,
        validation_type: str,
        result: str,
        error_message: str = "",
        details: Optional[Dict] = None,
    ) -> bool:
        """
        更新MR的验证状态

        Args:
            record_id: mr_knowledge_base表中的id
            mr_id: MR的id
            validation_type: 'precheck' 或 'sympy'
            result: 'passed' 或 'failed'
            error_message: 错误信息
            details: 详细信息(字典)

        Returns:
            是否更新成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 插入验证记录
            validation_id = str(uuid.uuid4())
            validated_at = datetime.now().isoformat()

            cursor.execute(
                """
                INSERT INTO mr_validation_records
                (id, record_id, mr_id, operator_name, validation_type, validation_result, error_message, validated_at, validation_details)
                SELECT ?, ?, ?, operator_name, ?, ?, ?, ?, ?
                FROM mr_knowledge_base WHERE id = ?
            """,
                (
                    validation_id,
                    record_id,
                    mr_id,
                    validation_type,
                    result,
                    error_message,
                    validated_at,
                    json.dumps(details or {}),
                    record_id,
                ),
            )

            # 更新 mr_knowledge_base 表的验证状态
            if validation_type == "precheck":
                cursor.execute(
                    """
                    UPDATE mr_knowledge_base
                    SET precheck_passed = ?, last_validated_at = ?, updated_at = ?
                    WHERE id = ?
                """,
                    (result == "passed", validated_at, validated_at, record_id),
                )
            elif validation_type == "sympy":
                cursor.execute(
                    """
                    UPDATE mr_knowledge_base
                    SET sympy_proven = ?, last_validated_at = ?, updated_at = ?
                    WHERE id = ?
                """,
                    (result == "passed", validated_at, validated_at, record_id),
                )

            # 更新 verified 字段（如果有任一验证通过）
            cursor.execute(
                """
                UPDATE mr_knowledge_base
                SET verified = CASE
                    WHEN (precheck_passed = 1 OR sympy_proven = 1) THEN 1
                    ELSE 0
                END
                WHERE id = ?
            """,
                (record_id,),
            )

            conn.commit()
            log_structured(
                self.logger,
                "REPO",
                f"Updated validation status for MR '{mr_id}' | {validation_type} = {result}",
            )
            return True

        except Exception as e:
            self.logger.error(f"Error updating validation status for MR {mr_id}: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_validation_history(
        self,
        mr_id: str,
    ) -> List[Dict]:
        """
        获取MR的验证历史

        Args:
            mr_id: MR的id

        Returns:
            验证记录列表，按时间倒序排列
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT validation_type, validation_result, error_message, validated_at, validation_details
                FROM mr_validation_records
                WHERE mr_id = ?
                ORDER BY validated_at DESC
            """,
                (mr_id,),
            )

            records = []
            for row in cursor.fetchall():
                records.append(
                    {
                        "validation_type": row[0],
                        "validation_result": row[1],
                        "error_message": row[2],
                        "validated_at": row[3],
                        "validation_details": json.loads(row[4]) if row[4] else {},
                    }
                )

            return records

        except Exception as e:
            self.logger.error(f"Error getting validation history for MR {mr_id}: {e}")
            return []
        finally:
            conn.close()

    def get_mr_with_validation_status(
        self,
        operator_name: str,
        version: Optional[int] = None,
        verified_only: bool = False,
    ) -> List[MetamorphicRelation]:
        """
        获取MR及其验证状态

        Args:
            operator_name: 算子名称
            version: 版本号（None表示最新版本）
            verified_only: 是否只返回已验证的MR

        Returns:
            MR列表，每个MR包含验证状态信息
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 获取版本号
            if version is None:
                cursor.execute(
                    """
                    SELECT MAX(version) FROM mr_knowledge_base
                    WHERE operator_name = ?
                """,
                    (operator_name,),
                )
                result = cursor.fetchone()
                version = result[0] if result and result[0] else 1

            # 构建查询
            query = """
                SELECT mr_data, mr_id, mr_description, mr_type, verified, precheck_passed, sympy_proven, last_validated_at
                FROM mr_knowledge_base
                WHERE operator_name = ? AND version = ?
            """

            if verified_only:
                query += " AND verified = 1"

            cursor.execute(query, (operator_name, version))

            mrs = []
            for row in cursor.fetchall():
                try:
                    (
                        mr_data_str,
                        mr_id,
                        description,
                        expected,
                        verified,
                        precheck_passed,
                        sympy_proven,
                        last_validated_at,
                    ) = row
                    mr = self._deserialize_mr(mr_data_str, mr_id, description, expected)

                    mr.verified = (
                        bool(verified) if verified is not None else mr.verified
                    )

                    validation_info = {
                        "precheck_passed": precheck_passed,
                        "sympy_proven": sympy_proven,
                        "last_validated_at": last_validated_at,
                    }
                    if mr.analysis:
                        mr.analysis = f"{mr.analysis}\n\nValidation: {json.dumps(validation_info)}"
                    else:
                        mr.analysis = json.dumps(validation_info)

                    mrs.append(mr)

                except Exception as e:
                    self.logger.error(f"Error loading MR with validation status: {e}")

            log_structured(
                self.logger,
                "REPO",
                f"Loaded {len(mrs)} MRs for '{operator_name}' | version: {version}",
            )
            return mrs

        except Exception as e:
            self.logger.error(f"Error getting MRs with validation status: {e}")
            return []
        finally:
            conn.close()

    def get_statistics(self, operator_name: Optional[str] = None) -> Dict:
        """
        获取MR统计信息

        Args:
            operator_name: 算子名称（None表示统计所有）

        Returns:
            统计信息:
            {
                'total_mrs': int,
                'verified_mrs': int,
                'unverified_mrs': int,
                'precheck_passed': int,
                'sympy_proven': int,
                'by_operator': {
                    'operator_name': {
                        'total': int,
                        'verified': int,
                        ...
                    }
                }
            }
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 初始化统计信息
        stats = {
            "total_mrs": 0,
            "verified_mrs": 0,
            "unverified_mrs": 0,
            "precheck_passed": 0,
            "sympy_proven": 0,
            "by_operator": {},
        }

        try:
            if operator_name:
                # 特定算子统计
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN verified = 1 THEN 1 ELSE 0 END) as verified,
                        SUM(CASE WHEN precheck_passed = 1 THEN 1 ELSE 0 END) as precheck,
                        SUM(CASE WHEN sympy_proven = 1 THEN 1 ELSE 0 END) as sympy
                    FROM mr_knowledge_base
                    WHERE operator_name = ?
                """,
                    (operator_name,),
                )
                row = cursor.fetchone()
                stats["total_mrs"] = row[0]
                stats["verified_mrs"] = row[1] or 0
                stats["unverified_mrs"] = stats["total_mrs"] - stats["verified_mrs"]
                stats["precheck_passed"] = row[2] or 0
                stats["sympy_proven"] = row[3] or 0

                stats["by_operator"][operator_name] = {
                    "total": stats["total_mrs"],
                    "verified": stats["verified_mrs"],
                    "unverified": stats["unverified_mrs"],
                    "precheck_passed": stats["precheck_passed"],
                    "sympy_proven": stats["sympy_proven"],
                }
            else:
                # 整体统计
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN verified = 1 THEN 1 ELSE 0 END) as verified,
                        SUM(CASE WHEN precheck_passed = 1 THEN 1 ELSE 0 END) as precheck,
                        SUM(CASE WHEN sympy_proven = 1 THEN 1 ELSE 0 END) as sympy
                    FROM mr_knowledge_base
                """,
                )
                row = cursor.fetchone()
                stats["total_mrs"] = row[0]
                stats["verified_mrs"] = row[1] or 0
                stats["unverified_mrs"] = stats["total_mrs"] - stats["verified_mrs"]
                stats["precheck_passed"] = row[2] or 0
                stats["sympy_proven"] = row[3] or 0

                # 按算子统计
                cursor.execute(
                    """
                    SELECT operator_name,
                        COUNT(*) as total,
                        SUM(CASE WHEN verified = 1 THEN 1 ELSE 0 END) as verified,
                        SUM(CASE WHEN precheck_passed = 1 THEN 1 ELSE 0 END) as precheck,
                        SUM(CASE WHEN sympy_proven = 1 THEN 1 ELSE 0 END) as sympy
                    FROM mr_knowledge_base
                    GROUP BY operator_name
                    ORDER BY operator_name
                """
                )
                for row in cursor.fetchall():
                    op_name = row[0]
                    stats["by_operator"][op_name] = {
                        "total": row[1],
                        "verified": row[2] or 0,
                        "unverified": row[1] - (row[2] or 0),
                        "precheck_passed": row[3] or 0,
                        "sympy_proven": row[4] or 0,
                    }

            return stats

        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return stats
        finally:
            conn.close()

    def get_record_id_by_mr_id(
        self, mr_id: str, operator_name: str, version: Optional[int] = None
    ) -> Optional[str]:
        """
        根据MR ID获取记录ID

        Args:
            mr_id: MR的id
            operator_name: 算子名称
            version: 版本号（None表示最新版本）

        Returns:
            记录ID，如果不存在则返回None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            query = (
                "SELECT id FROM mr_knowledge_base WHERE mr_id = ? AND operator_name = ?"
            )
            params: List = [str(mr_id), str(operator_name)]

            if version is not None:
                query += " AND version = ?"
                params.append(version)
            else:
                query += " ORDER BY version DESC LIMIT 1"

            cursor.execute(query, params)
            result = cursor.fetchone()

            return result[0] if result else None

        except Exception as e:
            self.logger.error(f"Error getting record ID for MR {mr_id}: {e}")
            return None
        finally:
            conn.close()
