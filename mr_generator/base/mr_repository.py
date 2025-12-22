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

from core.logger import get_logger
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
        self.logger = get_logger()
        self._init_database()

    def _init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # MR知识库表
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
                updated_at TEXT NOT NULL
            )
        """
        )

        # MR索引表（按算子名称快速查询）
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_operator_name 
            ON mr_knowledge_base(operator_name)
        """
        )

        conn.commit()
        conn.close()
        self.logger.info(f"MR knowledge base initialized at {self.db_path}")

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
                # 序列化MR对象
                mr_data = self._serialize_mr(mr)

                # 生成唯一ID
                record_id = str(uuid.uuid4())

                cursor.execute(
                    """
                    INSERT INTO mr_knowledge_base
                    (id, operator_name, mr_id, mr_description, mr_type, mr_data, version, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record_id,
                        operator_name,
                        mr.id,
                        mr.description,
                        mr.expected,
                        mr_data,
                        version,
                        timestamp,
                        timestamp,
                    ),
                )

                saved_count += 1

            except Exception as e:
                self.logger.error(f"Error saving MR {mr.id}: {e}")

        conn.commit()
        conn.close()

        self.logger.info(f"Saved {saved_count} MRs for operator: {operator_name}")
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
            SELECT mr_data, mr_id, mr_description, mr_type
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
                mr_data_str, mr_id, description, expected = row
                mr = self._deserialize_mr(mr_data_str, mr_id, description, expected)
                mrs.append(mr)
            except Exception as e:
                self.logger.error(f"Error loading MR: {e}")

        self.logger.info(
            f"Loaded {len(mrs)} MRs for operator: {operator_name} (version {version})"
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
        # 注意：transform函数无法直接序列化，需要特殊处理
        data = {
            "id": mr.id,
            "description": mr.description,
            "expected": mr.expected,
            "tolerance": mr.tolerance,
            "layer": mr.layer,
            # transform函数无法序列化，需要在加载时重建
            "transform_type": "lambda",  # 标记为lambda函数
        }
        return json.dumps(data)

    def _deserialize_mr(
        self, mr_data_str: str, mr_id: str, description: str, expected: str
    ) -> MetamorphicRelation:
        """反序列化MR对象"""
        data = json.loads(mr_data_str)

        # 重建transform函数（简化实现）
        # 实际应该根据MR类型重建对应的transform
        transform = self._rebuild_transform(expected, description)

        return MetamorphicRelation(
            id=mr_id,
            description=description,
            transform=transform,
            expected=expected,
            tolerance=data.get("tolerance"),
            layer=data.get("layer", "operator"),
        )

    def _rebuild_transform(self, expected: str, description: str):
        """重建transform函数（简化实现）"""
        # 根据描述和类型重建transform
        # 这是一个简化实现，实际应该更智能

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
            # 默认返回原样
            return lambda *args: args
