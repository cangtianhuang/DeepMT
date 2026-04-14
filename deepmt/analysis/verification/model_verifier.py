"""模型层 Oracle 验证器：验证模型输出是否满足 MR 约束。

支持的 oracle_expr 格式（由 ModelMRGenerator 生成）：
  - "prediction_consistent"      argmax(orig) == argmax(trans)（逐样本）
  - "topk_consistent:{k}"        top-k 预测集合相同（如 "topk_consistent:3"）
  - "output_close:{atol}"        allclose(orig, trans, atol)（如 "output_close:1e-5"）
  - "output_order_invariant"     orig.flip(0) ≈ trans（用于 batch flip 变换验证）

与算子层 MRVerifier 的关系：
  - 算子层验证器（mr_verifier.py）处理算子输出张量的符号/数值关系
  - 模型层验证器处理模型完整前向输出（通常是 logits 或概率分布）
  - 两者接口独立，共享 OracleResult 数据结构

注意：本模块不依赖框架插件（FrameworkPlugin），直接在 PyTorch 张量上操作。
后续若需要支持多框架，可通过 backend 参数扩展。
"""

from typing import Any, Optional

import numpy as np

from deepmt.core.logger import logger
from deepmt.ir.schema import MetamorphicRelation, OracleResult


# ── Oracle 解析辅助 ───────────────────────────────────────────────────────────


def _parse_oracle_expr(oracle_expr: str):
    """解析 oracle_expr 字符串，返回 (oracle_type, params_dict)。"""
    if ":" in oracle_expr:
        oracle_type, param_str = oracle_expr.split(":", 1)
        return oracle_type.strip(), param_str.strip()
    return oracle_expr.strip(), None


# ── Tensor→numpy 辅助 ─────────────────────────────────────────────────────────


def _to_numpy(tensor: Any) -> np.ndarray:
    """将框架张量（或 numpy）转为 numpy ndarray。"""
    if isinstance(tensor, np.ndarray):
        return tensor
    # PyTorch
    try:
        return tensor.detach().cpu().numpy()
    except AttributeError:
        pass
    # PaddlePaddle
    try:
        return tensor.numpy()
    except AttributeError:
        pass
    return np.asarray(tensor)


# ── 验证器主类 ────────────────────────────────────────────────────────────────


class ModelVerifier:
    """模型层 Oracle 验证器。

    根据 MR 的 oracle_expr 对模型输出进行验证。

    用法::

        verifier = ModelVerifier()
        result = verifier.verify(orig_output, trans_output, mr)
        print(result.passed, result.detail)
    """

    def verify(
        self,
        orig: Any,
        trans: Any,
        mr: MetamorphicRelation,
        transform_input: Optional[Any] = None,
    ) -> OracleResult:
        """验证一对模型输出是否满足 MR 约束。

        Args:
            orig:             原始输入对应的模型输出（框架张量）
            trans:            变换后输入对应的模型输出（框架张量）
            mr:               蜕变关系（含 oracle_expr、tolerance）
            transform_input:  原始输入张量（用于 batch flip 验证，可选）

        Returns:
            OracleResult
        """
        oracle_type, param_str = _parse_oracle_expr(mr.oracle_expr)

        try:
            if oracle_type == "prediction_consistent":
                return self._check_prediction_consistent(orig, trans, mr.oracle_expr)

            if oracle_type == "topk_consistent":
                k = int(param_str) if param_str else 3
                return self._check_topk_consistent(orig, trans, k, mr.oracle_expr)

            if oracle_type == "output_close":
                atol = float(param_str) if param_str else mr.tolerance
                return self._check_output_close(orig, trans, atol, mr.oracle_expr)

            if oracle_type == "output_order_invariant":
                return self._check_order_invariant(orig, trans, mr.oracle_expr)

            # 未知类型：退化为数值接近验证
            logger.debug(
                f"[ModelVerifier] 未知 oracle_type={oracle_type!r}，退化为 output_close"
            )
            return self._check_output_close(orig, trans, mr.tolerance, mr.oracle_expr)

        except Exception as e:
            logger.debug(f"[ModelVerifier] oracle 验证异常: {e}")
            return OracleResult(
                passed=False,
                expr=mr.oracle_expr,
                actual_diff=float("inf"),
                tolerance=mr.tolerance,
                detail=f"ORACLE_ERROR: {e}",
            )

    # ── 各 oracle 类型实现 ────────────────────────────────────────────────────

    def _check_prediction_consistent(
        self, orig: Any, trans: Any, expr: str
    ) -> OracleResult:
        """验证 argmax 逐样本一致性。"""
        orig_np = _to_numpy(orig)
        trans_np = _to_numpy(trans)

        # 形状检查：至少 2D (batch, num_classes)
        if orig_np.ndim < 2 or trans_np.ndim < 2:
            return OracleResult(
                passed=False,
                expr=expr,
                actual_diff=float("inf"),
                tolerance=0.0,
                detail=(
                    f"SHAPE_ERROR: 期望至少 2D 输出，"
                    f"实际 orig={orig_np.shape}, trans={trans_np.shape}"
                ),
            )

        orig_pred = np.argmax(orig_np, axis=-1)   # (batch,)
        trans_pred = np.argmax(trans_np, axis=-1)

        if orig_pred.shape != trans_pred.shape:
            return OracleResult(
                passed=False,
                expr=expr,
                actual_diff=float("inf"),
                tolerance=0.0,
                detail=f"SHAPE_MISMATCH: {orig_pred.shape} vs {trans_pred.shape}",
            )

        mismatched = int(np.sum(orig_pred != trans_pred))
        total = int(orig_pred.size)
        passed = mismatched == 0

        return OracleResult(
            passed=passed,
            expr=expr,
            actual_diff=float(mismatched),
            tolerance=0.0,
            detail="" if passed else (
                f"PREDICTION_MISMATCH: {mismatched}/{total} 样本预测不一致 "
                f"({mismatched/total:.1%})"
            ),
            mismatched_elements=mismatched,
            total_elements=total,
        )

    def _check_topk_consistent(
        self, orig: Any, trans: Any, k: int, expr: str
    ) -> OracleResult:
        """验证 top-k 预测集合一致性（集合相同即可，顺序无关）。"""
        orig_np = _to_numpy(orig)
        trans_np = _to_numpy(trans)

        if orig_np.ndim < 2 or trans_np.ndim < 2:
            return OracleResult(
                passed=False,
                expr=expr,
                actual_diff=float("inf"),
                tolerance=0.0,
                detail=f"SHAPE_ERROR: 期望至少 2D 输出",
            )

        num_classes = orig_np.shape[-1]
        k_actual = min(k, num_classes)

        # top-k 索引集合（不要求顺序）
        orig_topk = np.argsort(orig_np, axis=-1)[..., -k_actual:]   # (batch, k)
        trans_topk = np.argsort(trans_np, axis=-1)[..., -k_actual:]

        mismatched = 0
        total = orig_np.shape[0]
        for i in range(total):
            if set(orig_topk[i].tolist()) != set(trans_topk[i].tolist()):
                mismatched += 1

        passed = mismatched == 0
        return OracleResult(
            passed=passed,
            expr=expr,
            actual_diff=float(mismatched),
            tolerance=0.0,
            detail="" if passed else (
                f"TOPK_MISMATCH (k={k_actual}): {mismatched}/{total} 样本 top-k 不一致"
            ),
            mismatched_elements=mismatched,
            total_elements=total,
        )

    def _check_output_close(
        self, orig: Any, trans: Any, atol: float, expr: str
    ) -> OracleResult:
        """验证输出数值接近（allclose）。"""
        orig_np = _to_numpy(orig)
        trans_np = _to_numpy(trans)

        if orig_np.shape != trans_np.shape:
            return OracleResult(
                passed=False,
                expr=expr,
                actual_diff=float("inf"),
                tolerance=atol,
                detail=f"SHAPE_MISMATCH: {orig_np.shape} vs {trans_np.shape}",
            )

        abs_diff = np.abs(orig_np - trans_np)
        max_diff = float(np.max(abs_diff))
        mismatched = int(np.sum(abs_diff > atol))
        total = int(abs_diff.size)
        passed = mismatched == 0

        return OracleResult(
            passed=passed,
            expr=expr,
            actual_diff=max_diff,
            tolerance=atol,
            detail="" if passed else (
                f"NUMERICAL_DEVIATION: max_abs={max_diff:.6g}, "
                f"mismatched={mismatched}/{total} ({mismatched/total:.1%})"
            ),
            mismatched_elements=mismatched,
            total_elements=total,
        )

    def _check_order_invariant(
        self, orig: Any, trans: Any, expr: str
    ) -> OracleResult:
        """验证 batch 顺序不变性：orig 翻转后应与 trans 数值接近。

        适用于 batch_flip / batch_shuffle 类变换：
          transform = lambda x: x.flip(0)
          期望：model(x.flip(0)) ≈ model(x).flip(0)
        """
        orig_np = _to_numpy(orig)
        trans_np = _to_numpy(trans)

        if orig_np.shape != trans_np.shape:
            return OracleResult(
                passed=False,
                expr=expr,
                actual_diff=float("inf"),
                tolerance=1e-5,
                detail=f"SHAPE_MISMATCH: {orig_np.shape} vs {trans_np.shape}",
            )

        # orig 沿 batch 维翻转，应与 trans 一致
        orig_flipped = orig_np[::-1]
        abs_diff = np.abs(orig_flipped - trans_np)
        max_diff = float(np.max(abs_diff))
        atol = 1e-5
        mismatched = int(np.sum(abs_diff > atol))
        total = int(abs_diff.size)
        passed = mismatched == 0

        return OracleResult(
            passed=passed,
            expr=expr,
            actual_diff=max_diff,
            tolerance=atol,
            detail="" if passed else (
                f"ORDER_INVARIANT_VIOLATION: orig.flip(0) vs trans, "
                f"max_abs={max_diff:.6g}, mismatched={mismatched}/{total}"
            ),
            mismatched_elements=mismatched,
            total_elements=total,
        )
