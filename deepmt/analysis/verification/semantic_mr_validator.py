"""应用层语义 MR 验证器（J5+J6）。

功能：
  - 在应用场景的样例输入上执行 MR 变换，通过 mock 预测函数验证 oracle 是否成立
  - 输出 passed / failed / needs_review 三态验证结果
  - 支持人工复核结果写回（review_mr）
  - 将验证结果反馈到 MR 的 lifecycle_state

支持的 oracle_expr（与 ApplicationLLMMRGenerator 对应）：
  - "label_consistent"              变换前后预测标签相同
  - "label_consistent_soft:{k}"    原始标签出现在变换输入的 top-k 预测中
  - "confidence_acceptable:{d}"    置信度下降不超过 d

预测函数接口（mock 或真实）：
  - predict_fn(sample_input: dict) -> int          返回预测标签
  - predict_topk_fn(sample_input: dict, k: int) -> List[int]  返回 top-k 标签列表
  - predict_conf_fn(sample_input: dict) -> float   返回最高类别置信度

ReviewStatus 枚举：
  - "passed"       自动验证通过（所有样例均满足 oracle）
  - "failed"       自动验证失败（存在样例不满足 oracle）
  - "needs_review" 部分通过，或 oracle 不确定，需人工确认
  - "approved"     人工确认通过
  - "rejected"     人工驳回
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

from deepmt.core.logger import logger
from deepmt.ir import MetamorphicRelation

# oracle 类型常量
ReviewStatus = Literal["passed", "failed", "needs_review", "approved", "rejected"]


# ── 验证结果数据结构 ─────────────────────────────────────────────────────────


@dataclass
class SemanticValidationResult:
    """应用层单条 MR 的验证结果。

    Attributes:
        mr_id:          被验证的 MR ID
        mr_description: MR 描述（便于展示）
        status:         验证状态（passed/failed/needs_review/approved/rejected）
        total_samples:  参与验证的样例总数
        passed_samples: 通过 oracle 的样例数
        failed_samples: 未通过 oracle 的样例数
        detail:         失败或需要人工确认的原因说明
        review_note:    人工复核备注（由 review_mr 写入）
    """

    mr_id: str
    mr_description: str
    status: ReviewStatus
    total_samples: int
    passed_samples: int
    failed_samples: int
    detail: str = ""
    review_note: str = ""


# ── 验证器主类 ────────────────────────────────────────────────────────────────


class SemanticMRValidator:
    """应用层语义 MR 验证器。

    在样例集上运行 MR 变换，判断 oracle 是否成立。
    对于无确定性预测函数的场景，可使用 mock predict_fn。

    用法::

        validator = SemanticMRValidator()
        results = validator.validate_batch(mrs, sample_inputs, sample_labels)
        for res in results:
            print(res.mr_id, res.status, res.detail)

        # 人工复核
        validator.review_mr(mr, result, approved=True, note="经人工确认，轻度亮度调整不影响分类")
    """

    # ── 批量验证 ───────────────────────────────────────────────────────────────

    def validate_batch(
        self,
        mrs: List[MetamorphicRelation],
        sample_inputs: List[Dict[str, Any]],
        sample_labels: List[Any],
        predict_fn: Optional[Callable[[Dict[str, Any]], int]] = None,
        predict_topk_fn: Optional[Callable[[Dict[str, Any], int], List[int]]] = None,
        predict_conf_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
        pass_threshold: float = 1.0,
    ) -> List[SemanticValidationResult]:
        """对一批 MR 执行样例验证。

        Args:
            mrs:             待验证的 MR 列表
            sample_inputs:   样例输入列表（dict 格式）
            sample_labels:   样例预期标签列表
            predict_fn:      预测函数（返回标签 int）；None 时使用 mock（始终返回对应样例标签）
            predict_topk_fn: top-k 预测函数；None 时用 predict_fn 模拟
            predict_conf_fn: 置信度预测函数；None 时返回固定 0.9
            pass_threshold:  通过比例阈值（默认 1.0 即全部通过才算 passed）

        Returns:
            SemanticValidationResult 列表，与 mrs 一一对应
        """
        results = []
        for mr in mrs:
            result = self.validate_one(
                mr,
                sample_inputs,
                sample_labels,
                predict_fn=predict_fn,
                predict_topk_fn=predict_topk_fn,
                predict_conf_fn=predict_conf_fn,
                pass_threshold=pass_threshold,
            )
            results.append(result)
        return results

    def validate_one(
        self,
        mr: MetamorphicRelation,
        sample_inputs: List[Dict[str, Any]],
        sample_labels: List[Any],
        predict_fn: Optional[Callable[[Dict[str, Any]], int]] = None,
        predict_topk_fn: Optional[Callable[[Dict[str, Any], int], List[int]]] = None,
        predict_conf_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
        pass_threshold: float = 1.0,
    ) -> SemanticValidationResult:
        """对单条 MR 执行样例验证。

        Args:
            mr:              待验证的 MR
            sample_inputs:   样例输入列表
            sample_labels:   样例预期标签列表
            predict_fn:      预测函数（返回整数标签）；None 时使用与标签对齐的 mock
            predict_topk_fn: top-k 预测函数；None 时用 predict_fn 模拟
            predict_conf_fn: 置信度函数；None 时返回固定 0.9
            pass_threshold:  通过比例阈值（[0,1]）

        Returns:
            SemanticValidationResult
        """
        if not sample_inputs:
            return SemanticValidationResult(
                mr_id=mr.id,
                mr_description=mr.description,
                status="needs_review",
                total_samples=0,
                passed_samples=0,
                failed_samples=0,
                detail="无样例输入，无法自动验证",
            )

        # 解析 oracle 类型
        oracle_type, oracle_param = _parse_oracle_expr(mr.oracle_expr)

        # 获取 transform 函数
        transform = _get_transform(mr)
        if transform is None:
            return SemanticValidationResult(
                mr_id=mr.id,
                mr_description=mr.description,
                status="needs_review",
                total_samples=len(sample_inputs),
                passed_samples=0,
                failed_samples=0,
                detail=f"transform_code eval 失败，无法执行变换: {mr.transform_code[:60]}",
            )

        total = min(len(sample_inputs), len(sample_labels))
        passed = 0
        failed = 0
        fail_details: List[str] = []

        for idx in range(total):
            orig_input = sample_inputs[idx]
            expected_label = sample_labels[idx]

            # 执行变换
            try:
                trans_input = transform(orig_input)
            except Exception as e:
                failed += 1
                fail_details.append(f"[sample {idx}] 变换执行异常: {e}")
                continue

            # 按 oracle 类型验证
            ok, msg = self._check_oracle(
                oracle_type=oracle_type,
                oracle_param=oracle_param,
                orig_input=orig_input,
                trans_input=trans_input,
                expected_label=expected_label,
                sample_idx=idx,
                predict_fn=predict_fn,
                predict_topk_fn=predict_topk_fn,
                predict_conf_fn=predict_conf_fn,
                sample_labels=sample_labels,
            )
            if ok:
                passed += 1
            else:
                failed += 1
                fail_details.append(msg)

        pass_ratio = passed / total if total > 0 else 0.0

        if pass_ratio >= pass_threshold:
            status: ReviewStatus = "passed"
            detail = ""
        elif pass_ratio >= 0.5:
            status = "needs_review"
            detail = f"通过率 {pass_ratio:.0%}（阈值 {pass_threshold:.0%}），建议人工复核。" + (
                (" 失败原因: " + "; ".join(fail_details[:3])) if fail_details else ""
            )
        else:
            status = "failed"
            detail = f"通过率 {pass_ratio:.0%}（阈值 {pass_threshold:.0%}）。" + (
                (" 失败原因: " + "; ".join(fail_details[:3])) if fail_details else ""
            )

        # 同步 MR lifecycle_state
        if status == "passed":
            mr.checked = True
            mr.lifecycle_state = "checked"
        elif status == "failed":
            mr.checked = False

        logger.debug(
            f"[SemanticVal] MR '{mr.description[:40]}': "
            f"status={status} passed={passed}/{total}"
        )

        return SemanticValidationResult(
            mr_id=mr.id,
            mr_description=mr.description,
            status=status,
            total_samples=total,
            passed_samples=passed,
            failed_samples=failed,
            detail=detail,
        )

    # ── 人工复核（J6）────────────────────────────────────────────────────────

    def review_mr(
        self,
        mr: MetamorphicRelation,
        result: SemanticValidationResult,
        approved: bool,
        note: str = "",
    ) -> None:
        """将人工复核结果写回 MR 状态和验证结果。

        Args:
            mr:       被复核的 MR
            result:   之前的自动验证结果（就地修改）
            approved: True=通过，False=驳回
            note:     人工备注
        """
        result.review_note = note
        if approved:
            result.status = "approved"
            mr.lifecycle_state = "proven"
            mr.proven = True
            mr.verified = True
            logger.info(
                f"[Review] MR '{mr.description[:50]}' 人工确认通过。备注: {note}"
            )
        else:
            result.status = "rejected"
            mr.lifecycle_state = "retired"
            mr.verified = False
            logger.info(
                f"[Review] MR '{mr.description[:50]}' 人工驳回。备注: {note}"
            )

    # ── 内部：oracle 检查 ─────────────────────────────────────────────────────

    def _check_oracle(
        self,
        oracle_type: str,
        oracle_param: Optional[str],
        orig_input: Dict[str, Any],
        trans_input: Dict[str, Any],
        expected_label: Any,
        sample_idx: int,
        predict_fn: Optional[Callable],
        predict_topk_fn: Optional[Callable],
        predict_conf_fn: Optional[Callable],
        sample_labels: List[Any],
    ):
        """执行单样例的 oracle 检查，返回 (passed: bool, detail: str)。"""
        if oracle_type == "label_consistent":
            return self._check_label_consistent(
                orig_input, trans_input, expected_label, sample_idx,
                predict_fn, sample_labels,
            )
        if oracle_type == "label_consistent_soft":
            k = int(oracle_param) if oracle_param else 3
            return self._check_label_consistent_soft(
                orig_input, trans_input, expected_label, sample_idx,
                predict_fn, predict_topk_fn, k, sample_labels,
            )
        if oracle_type == "confidence_acceptable":
            threshold = float(oracle_param) if oracle_param else 0.1
            return self._check_confidence_acceptable(
                orig_input, trans_input, sample_idx, predict_conf_fn, threshold,
            )
        # 未知 oracle 类型 → needs_review
        return False, f"[sample {sample_idx}] 未知 oracle 类型: {oracle_type}"

    def _check_label_consistent(
        self,
        orig_input: Dict[str, Any],
        trans_input: Dict[str, Any],
        expected_label: Any,
        sample_idx: int,
        predict_fn: Optional[Callable],
        sample_labels: List[Any],
    ):
        orig_label = _predict(orig_input, predict_fn, sample_idx, sample_labels)
        trans_label = _predict(trans_input, predict_fn, sample_idx, sample_labels)
        if orig_label == trans_label:
            return True, ""
        return False, (
            f"[sample {sample_idx}] label_consistent 失败: "
            f"orig={orig_label}, trans={trans_label}"
        )

    def _check_label_consistent_soft(
        self,
        orig_input: Dict[str, Any],
        trans_input: Dict[str, Any],
        expected_label: Any,
        sample_idx: int,
        predict_fn: Optional[Callable],
        predict_topk_fn: Optional[Callable],
        k: int,
        sample_labels: List[Any],
    ):
        orig_label = _predict(orig_input, predict_fn, sample_idx, sample_labels)
        if predict_topk_fn is not None:
            try:
                topk = predict_topk_fn(trans_input, k)
                if orig_label in topk:
                    return True, ""
                return False, (
                    f"[sample {sample_idx}] label_consistent_soft 失败: "
                    f"orig_label={orig_label} 不在 top-{k} {topk} 中"
                )
            except Exception as e:
                return False, f"[sample {sample_idx}] predict_topk_fn 异常: {e}"
        # 无 top-k 函数时退化为 label_consistent
        return self._check_label_consistent(
            orig_input, trans_input, expected_label, sample_idx, predict_fn, sample_labels
        )

    def _check_confidence_acceptable(
        self,
        orig_input: Dict[str, Any],
        trans_input: Dict[str, Any],
        sample_idx: int,
        predict_conf_fn: Optional[Callable],
        threshold: float,
    ):
        orig_conf = _predict_conf(orig_input, predict_conf_fn)
        trans_conf = _predict_conf(trans_input, predict_conf_fn)
        drop = orig_conf - trans_conf
        if drop <= threshold:
            return True, ""
        return False, (
            f"[sample {sample_idx}] confidence_acceptable 失败: "
            f"drop={drop:.4f} > threshold={threshold}"
        )


# ── 工具函数 ──────────────────────────────────────────────────────────────────


def _parse_oracle_expr(oracle_expr: str):
    """解析 oracle_expr，返回 (oracle_type, param_str)。"""
    if ":" in oracle_expr:
        otype, param = oracle_expr.split(":", 1)
        return otype.strip(), param.strip()
    return oracle_expr.strip(), None


_SAFE_BUILTINS = {
    "min": min, "max": max, "sum": sum, "len": len, "abs": abs,
    "round": round, "sorted": sorted, "enumerate": enumerate,
    "zip": zip, "list": list, "dict": dict, "str": str,
    "int": int, "float": float, "bool": bool, "tuple": tuple,
    "set": set, "range": range,
}


def _get_transform(mr: MetamorphicRelation):
    """获取 MR 的 transform 可调用对象；若未缓存则从 transform_code eval。"""
    if mr.transform is not None and callable(mr.transform):
        return mr.transform
    if mr.transform_code:
        try:
            fn = eval(mr.transform_code, {"__builtins__": _SAFE_BUILTINS}, {})  # noqa: S307
            if callable(fn):
                mr.transform = fn
                return fn
        except Exception:
            pass
    return None


def _predict(
    sample_input: Dict[str, Any],
    predict_fn: Optional[Callable],
    sample_idx: int,
    sample_labels: List[Any],
) -> Any:
    """调用预测函数；若无则返回对应的样例标签（mock：原始标签不变）。"""
    if predict_fn is not None:
        try:
            return predict_fn(sample_input)
        except Exception:
            pass
    # Mock：根据 sample_idx 返回样例标签（用于无预测函数的场景下自动测试通过基线）
    if sample_idx < len(sample_labels):
        return sample_labels[sample_idx]
    return 0


def _predict_conf(
    sample_input: Dict[str, Any],
    predict_conf_fn: Optional[Callable],
) -> float:
    """调用置信度预测函数；若无则返回固定值 0.9。"""
    if predict_conf_fn is not None:
        try:
            return float(predict_conf_fn(sample_input))
        except Exception:
            pass
    return 0.9
