"""
Phase O · O7 — health check --deep 回归

断言：
  - run_deep_checks() 不抛异常；
  - 整体状态 ∈ {HEALTHY, WARNING}（不允许 ERROR）；
  - 插件契约结果对已安装插件无 ERROR；
  - compute_reachability_matrix() 对一级算子返回至少一个框架可达。
"""

from deepmt.core.health_checker import HealthChecker, HealthStatus

LEVEL_1_OPERATORS = {
    "relu", "tanh", "exp", "abs", "sigmoid", "gelu",
    "softmax", "log_softmax", "leaky_relu",
}


def test_deep_check_no_error():
    checker = HealthChecker()
    report = checker.run_deep_checks()
    assert report.overall_status in (HealthStatus.HEALTHY, HealthStatus.WARNING)
    error_results = [r for r in report.results if r.status == HealthStatus.ERROR]
    assert not error_results, f"deep check 返回 ERROR: {[r.name for r in error_results]}"


def test_contract_check_no_error_for_installed():
    checker = HealthChecker()
    results = checker._check_plugin_contract()
    for r in results:
        assert r.status != HealthStatus.ERROR, f"契约 ERROR: {r.name} — {r.message}"


def test_reachability_matrix_covers_level1():
    checker = HealthChecker()
    matrix = checker.compute_reachability_matrix()
    covered = set(matrix.keys()) & LEVEL_1_OPERATORS
    assert covered, "一级算子全部缺失，知识库异常？"
    for op in covered:
        assert any(matrix[op].values()), f"算子 {op} 在所有框架下都不可达"
