"""被测主体注册表：统一管理三层被测对象的注册与查找。

SubjectRegistry 是 Phase G 引入的轻量注册机制，解决多层次被测对象的发现与分发问题：
  - 算子层：OperatorIR 通过 catalog 批量注册
  - 模型层（Phase I）：ModelIR 通过模型目录注册
  - 应用层（Phase J）：ApplicationIR 通过任务目录注册

注册后，执行引擎、报告生成器、CLI 命令均通过本注册表统一查找对象，
不再直接硬编码到具体 IR 类型。
"""

from typing import Dict, Iterator, List, Optional

from deepmt.core.logger import logger
from deepmt.ir.schema import SubjectType, TestSubject


class SubjectRegistry:
    """被测主体注册表。

    轻量字典封装，支持按名称注册、查找，以及按层次类型批量列举。

    示例::

        registry = SubjectRegistry()
        registry.register(operator_ir)
        subject = registry.lookup("torch.add")
        ops = registry.list_by_type("operator")
    """

    def __init__(self) -> None:
        # 主索引：name → TestSubject（子类实例）
        self._store: Dict[str, TestSubject] = {}

    # ── 写入 ─────────────────────────────────────────────────────────────────

    def register(self, subject: TestSubject, overwrite: bool = False) -> None:
        """注册一个被测主体。

        Args:
            subject:   TestSubject（或其子类）实例
            overwrite: True 时允许覆盖同名主体；默认 False（重复注册触发警告）
        """
        if not isinstance(subject, TestSubject):
            raise TypeError(
                f"subject 必须是 TestSubject 的子类实例，实际类型：{type(subject).__name__}"
            )
        if subject.name in self._store and not overwrite:
            logger.warning(
                f"[SubjectRegistry] '{subject.name}' 已注册，跳过（传 overwrite=True 可强制覆盖）"
            )
            return
        self._store[subject.name] = subject
        logger.debug(f"[SubjectRegistry] 注册: {subject.subject_type}/{subject.name}")

    def register_many(
        self, subjects: List[TestSubject], overwrite: bool = False
    ) -> int:
        """批量注册，返回实际写入数量。"""
        count = 0
        for s in subjects:
            before = len(self._store)
            self.register(s, overwrite=overwrite)
            if len(self._store) > before or overwrite:
                count += 1
        return count

    def unregister(self, name: str) -> bool:
        """注销指定主体，返回是否找到并删除。"""
        if name in self._store:
            del self._store[name]
            logger.debug(f"[SubjectRegistry] 注销: {name}")
            return True
        return False

    # ── 查找 ─────────────────────────────────────────────────────────────────

    def lookup(self, name: str) -> Optional[TestSubject]:
        """按名称查找，未找到返回 None。"""
        return self._store.get(name)

    def get(self, name: str) -> TestSubject:
        """按名称查找，未找到抛出 KeyError。"""
        if name not in self._store:
            raise KeyError(f"[SubjectRegistry] 未找到主体: '{name}'")
        return self._store[name]

    # ── 枚举 ─────────────────────────────────────────────────────────────────

    def list_by_type(self, subject_type: SubjectType) -> List[TestSubject]:
        """返回指定层次类型的全部主体列表。"""
        return [s for s in self._store.values() if s.subject_type == subject_type]

    def all_subjects(self) -> List[TestSubject]:
        """返回所有已注册主体列表。"""
        return list(self._store.values())

    def names(self) -> List[str]:
        """返回所有已注册主体名称列表。"""
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __iter__(self) -> Iterator[TestSubject]:
        return iter(self._store.values())
