"""
测试报告生成器：从 ResultsManager 读取测试记录，生成可读摘要。

职责：
  - 聚合 test_results 表中的原始记录，生成结构化摘要
  - 输出文本报告（终端展示）或 JSON 数据（程序消费）
  - 支持按算子、框架、状态过滤
  - 为 Phase D 的 RQ1-RQ4 提供数据导出接口

与 ResultsManager 的关系：
  ResultsManager 负责写入和基本查询；ReportGenerator 负责聚合与格式化输出。
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepmt.core.logger import logger
from deepmt.core.results_manager import ResultsManager


class ReportGenerator:
    """
    测试报告生成器。

    用法示例：
        gen = ReportGenerator()
        report = gen.generate(framework="pytorch")
        print(gen.format_text(report))
        # 或
        import json
        print(json.dumps(report, ensure_ascii=False, indent=2))
    """

    def __init__(self, results_manager: Optional[ResultsManager] = None):
        self.rm = results_manager or ResultsManager()

    # ── 主接口 ──────────────────────────────────────────────────────────────

    def generate(
        self,
        framework: Optional[str] = None,
        operator: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 0,
    ) -> Dict[str, Any]:
        """
        生成测试结果报告数据。

        Args:
            framework: 按框架过滤（如 "pytorch"）；None 表示全部
            operator:  按算子名称过滤；None 表示全部
            status:    按状态过滤（"PASS" / "FAIL"）；None 表示全部
            limit:     最多返回的算子数（0 = 不限）

        Returns:
            结构化报告字典，可直接 JSON 序列化
        """
        raw = self._query_test_results(framework=framework, operator=operator, status=status)
        per_op = self._aggregate_by_operator(raw)
        if limit > 0:
            per_op = dict(list(per_op.items())[:limit])

        total_operators = len(per_op)
        total_cases = sum(v["total"] for v in per_op.values())
        total_passed = sum(v["passed"] for v in per_op.values())
        total_failed = sum(v["failed"] for v in per_op.values())

        return {
            "generated_at": datetime.now().isoformat(),
            "filters": {
                "framework": framework,
                "operator": operator,
                "status": status,
            },
            "summary": {
                "total_operators": total_operators,
                "total_cases": total_cases,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "pass_rate": round(total_passed / total_cases, 4) if total_cases > 0 else 0.0,
            },
            "operators": list(per_op.values()),
        }

    def get_failures(
        self,
        framework: Optional[str] = None,
        operator: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """
        获取失败案例列表，带原始记录详情。

        Returns:
            按时间倒序排列的失败记录列表
        """
        raw = self._query_test_results(
            framework=framework, operator=operator, status="FAIL"
        )
        raw.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        if limit > 0:
            raw = raw[:limit]
        return raw

    def get_mr_breakdown(
        self,
        operator: str,
        framework: Optional[str] = None,
    ) -> List[Dict]:
        """
        获取指定算子下各 MR 的通过率分解。

        Returns:
            每个 MR 对应一个统计字典的列表
        """
        raw = self._query_test_results(framework=framework, operator=operator)
        return self._aggregate_by_mr(raw)

    def format_text(self, report: Dict) -> str:
        """将 generate() 的输出格式化为终端可读文本。"""
        lines = []
        now = report.get("generated_at", "")[:19]
        lines.append(f"\nDeepMT 测试报告  [{now}]")
        lines.append("═" * 70)

        s = report["summary"]
        lines.append(
            f"算子数: {s['total_operators']}  |  "
            f"总用例: {s['total_cases']}  |  "
            f"通过: {s['total_passed']}  |  "
            f"失败: {s['total_failed']}  |  "
            f"通过率: {s['pass_rate']:.1%}"
        )
        lines.append("─" * 70)

        for op in report["operators"]:
            status_mark = "✓" if op["failed"] == 0 else "✗"
            lines.append(
                f"  {status_mark} {op['operator']:<50}"
                f"  pass={op['passed']}/{op['total']}"
                f"  fail={op['failed']}"
            )
            for mr in op.get("mrs", []):
                mr_mark = "  ✓" if mr["failed"] == 0 else "  ✗"
                desc = mr.get("description", mr.get("mr_id", "?"))[:48]
                lines.append(
                    f"      {mr_mark} {desc:<48}"
                    f"  {mr['passed']}/{mr['total']}"
                )

        lines.append("═" * 70)
        return "\n".join(lines)

    def format_failure_text(self, failures: List[Dict]) -> str:
        """将 get_failures() 的输出格式化为终端可读文本。"""
        if not failures:
            return "未发现失败案例。"

        lines = [f"\n失败案例（共 {len(failures)} 条）", "─" * 70]
        for r in failures:
            lines.append(
                f"  [{r.get('timestamp','')[:19]}]"
                f"  {r.get('ir_name','?')}"
                f"  MR: {r.get('mr_description','?')[:40]}"
            )
            if r.get("defect_details"):
                lines.append(f"    详情: {r['defect_details'][:80]}")
        lines.append("─" * 70)
        return "\n".join(lines)

    # ── 内部查询 ──────────────────────────────────────────────────────────────

    def _query_test_results(
        self,
        framework: Optional[str] = None,
        operator: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict]:
        """直接查询 ResultsManager 的 SQLite 数据库。"""
        conn = sqlite3.connect(self.rm.db_path)
        cursor = conn.cursor()

        conditions = []
        params = []
        if framework:
            conditions.append("framework = ?")
            params.append(framework)
        if operator:
            conditions.append("ir_name = ?")
            params.append(operator)
        if status:
            conditions.append("status = ?")
            params.append(status.upper())

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cursor.execute(
            f"SELECT * FROM test_results {where} ORDER BY timestamp DESC",
            params,
        )
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return [dict(zip(columns, row)) for row in rows]

    def _aggregate_by_operator(self, records: List[Dict]) -> Dict[str, Dict]:
        """按算子聚合记录，生成通过/失败统计。"""
        per_op: Dict[str, Dict] = {}
        for r in records:
            op = r["ir_name"]
            fw = r["framework"]
            key = f"{op}::{fw}"
            if key not in per_op:
                per_op[key] = {
                    "operator": op,
                    "framework": fw,
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "mrs": {},
                }
            per_op[key]["total"] += 1
            if r["status"] == "PASS":
                per_op[key]["passed"] += 1
            else:
                per_op[key]["failed"] += 1

            # MR 级别聚合
            mr_key = r.get("mr_id", "")
            if mr_key not in per_op[key]["mrs"]:
                per_op[key]["mrs"][mr_key] = {
                    "mr_id": mr_key,
                    "description": r.get("mr_description", ""),
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                }
            per_op[key]["mrs"][mr_key]["total"] += 1
            if r["status"] == "PASS":
                per_op[key]["mrs"][mr_key]["passed"] += 1
            else:
                per_op[key]["mrs"][mr_key]["failed"] += 1

        # 将 mrs dict 转为 list
        for key in per_op:
            per_op[key]["mrs"] = list(per_op[key]["mrs"].values())

        return per_op

    def _aggregate_by_mr(self, records: List[Dict]) -> List[Dict]:
        """按 MR 聚合记录，生成各 MR 的通过率。"""
        per_mr: Dict[str, Dict] = {}
        for r in records:
            mr_key = r.get("mr_id", "")
            if mr_key not in per_mr:
                per_mr[mr_key] = {
                    "mr_id": mr_key,
                    "description": r.get("mr_description", ""),
                    "oracle_expr": r.get("oracle_expr", ""),
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                }
            per_mr[mr_key]["total"] += 1
            if r["status"] == "PASS":
                per_mr[mr_key]["passed"] += 1
            else:
                per_mr[mr_key]["failed"] += 1

        result = list(per_mr.values())
        for m in result:
            m["pass_rate"] = round(m["passed"] / m["total"], 4) if m["total"] > 0 else 0.0
        return result
