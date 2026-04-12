"""
Unit tests for deepmt.ui.routers.api

All tests use FastAPI TestClient with mocked data sources so they
run without any LLM API, network access, or real database files.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_mr(
    id="mr-001",
    description="test MR",
    transform_code="lambda x: -x",
    oracle_expr="orig == -trans",
    category="linearity",
    source="template",
    verified=True,
    checked=True,
    proven=False,
    tolerance=1e-6,
    layer="operator",
    applicable_frameworks=None,
    analysis="",
):
    m = MagicMock()
    m.id = id
    m.description = description
    m.transform_code = transform_code
    m.oracle_expr = oracle_expr
    m.category = category
    m.source = source
    m.verified = verified
    m.checked = checked
    m.proven = proven
    m.tolerance = tolerance
    m.layer = layer
    m.applicable_frameworks = applicable_frameworks
    m.analysis = analysis
    return m


def _make_evidence(
    evidence_id="ev-001",
    operator="torch.relu",
    framework="pytorch",
    framework_version="2.0.0",
    mr_id="mr-001",
    mr_description="test MR",
    actual_diff=0.01,
    tolerance=1e-6,
    detail={},
    timestamp="2026-01-01T00:00:00",
    reproduce_script="print('hello')",
):
    p = MagicMock()
    p.evidence_id = evidence_id
    p.operator = operator
    p.framework = framework
    p.framework_version = framework_version
    p.mr_id = mr_id
    p.mr_description = mr_description
    p.actual_diff = actual_diff
    p.tolerance = tolerance
    p.detail = detail
    p.timestamp = timestamp
    p.reproduce_script = reproduce_script
    return p


def _make_session(
    session_id="sess-001",
    operator="torch.relu",
    framework1="pytorch",
    framework2="numpy",
    n_samples=50,
    mr_count=3,
    overall_consistency_rate=0.98,
    output_max_diff=1e-7,
    inconsistent_mr_count=0,
    timestamp="2026-01-01T00:00:00",
):
    s = MagicMock()
    s.session_id = session_id
    s.operator = operator
    s.framework1 = framework1
    s.framework2 = framework2
    s.n_samples = n_samples
    s.mr_count = mr_count
    s.overall_consistency_rate = overall_consistency_rate
    s.output_max_diff = output_max_diff
    s.inconsistent_mr_count = inconsistent_mr_count
    s.timestamp = timestamp
    s.to_dict.return_value = {
        "session_id": session_id,
        "operator": operator,
        "framework1": framework1,
        "framework2": framework2,
        "overall_consistency_rate": overall_consistency_rate,
        "mr_results": [],
    }
    return s


# ── Fixture: TestClient ────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """FastAPI TestClient with a freshly created app (cache cleared)."""
    # Clear module-level cache before each test
    from deepmt.ui.routers import api as api_module
    api_module._CACHE.clear()

    from deepmt.ui.app import create_app
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── /api/health ────────────────────────────────────────────────────────────────

class TestApiHealth:
    def test_returns_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_structure(self, client):
        body = client.get("/api/health").json()
        assert body["error"] is None
        assert "generated_at" in body
        assert body["data"]["status"] == "ok"
        assert "version" in body["data"]


# ── /api/summary ───────────────────────────────────────────────────────────────

class TestApiSummary:
    def test_returns_200(self, client):
        summary_data = {"rq1": {"total_mr_count": 10}, "rq2": {}, "rq3": {}, "rq4": {}}
        with patch(
            "deepmt.ui.routers.api.ExperimentOrganizer",
            return_value=MagicMock(collect_all=MagicMock(return_value=summary_data)),
        ):
            r = client.get("/api/summary")
        assert r.status_code == 200

    def test_returns_data(self, client):
        summary_data = {"rq1": {"total_mr_count": 5}, "rq2": {}, "rq3": {}, "rq4": {}}
        with patch(
            "deepmt.ui.routers.api.ExperimentOrganizer",
            return_value=MagicMock(collect_all=MagicMock(return_value=summary_data)),
        ):
            from deepmt.ui.routers import api as api_module
            api_module._CACHE.clear()
            body = client.get("/api/summary").json()
        assert body["data"]["rq1"]["total_mr_count"] == 5

    def test_error_propagated_in_data(self, client):
        """If ExperimentOrganizer raises, the error string appears in data."""
        with patch(
            "deepmt.ui.routers.api.ExperimentOrganizer",
            side_effect=RuntimeError("db missing"),
        ):
            from deepmt.ui.routers import api as api_module
            api_module._CACHE.clear()
            body = client.get("/api/summary").json()
        assert body["error"] is None   # outer envelope OK
        assert "error" in body["data"]  # inner error field set


# ── /api/mr-repository ─────────────────────────────────────────────────────────

class TestApiMrRepository:
    def _mock_repo(self, operators=None, mrs_per_op=None):
        """Return a mock MRRepository."""
        if operators is None:
            operators = ["torch.relu"]
        if mrs_per_op is None:
            mrs_per_op = {op: [_make_mr()] for op in operators}

        repo = MagicMock()
        repo.list_operators.return_value = operators
        repo.load.side_effect = lambda op: mrs_per_op.get(op, [])
        return repo

    def test_returns_200(self, client):
        with patch("deepmt.ui.routers.api.MRRepository", return_value=self._mock_repo()):
            r = client.get("/api/mr-repository")
        assert r.status_code == 200

    def test_summary_fields(self, client):
        with patch("deepmt.ui.routers.api.MRRepository", return_value=self._mock_repo()):
            from deepmt.ui.routers import api as api_module
            api_module._CACHE.clear()
            body = client.get("/api/mr-repository").json()
        d = body["data"]
        assert d["total_mr_count"] == 1
        assert d["operators_with_mr"] == 1
        assert d["verified_mr_count"] == 1
        assert "operators" in d
        assert len(d["operators"]) == 1

    def test_empty_repository(self, client):
        repo = MagicMock()
        repo.list_operators.return_value = []
        with patch("deepmt.ui.routers.api.MRRepository", return_value=repo):
            from deepmt.ui.routers import api as api_module
            api_module._CACHE.clear()
            body = client.get("/api/mr-repository").json()
        assert body["data"]["total_mr_count"] == 0
        assert body["data"]["operators"] == []


class TestApiMrDetail:
    def test_returns_200(self, client):
        repo = MagicMock()
        repo.load.return_value = [_make_mr()]
        with patch("deepmt.ui.routers.api.MRRepository", return_value=repo):
            r = client.get("/api/mr-repository/torch.relu")
        assert r.status_code == 200

    def test_mr_fields_present(self, client):
        repo = MagicMock()
        mr = _make_mr(id="test-id", description="my MR")
        repo.load.return_value = [mr]
        with patch("deepmt.ui.routers.api.MRRepository", return_value=repo):
            body = client.get("/api/mr-repository/torch.relu").json()
        assert body["error"] is None
        assert len(body["data"]) == 1
        item = body["data"][0]
        assert item["id"] == "test-id"
        assert item["description"] == "my MR"
        assert "transform_code" in item
        assert "oracle_expr" in item
        assert "verified" in item

    def test_dotted_operator_name(self, client):
        """Operator names with dots must be routable via {operator_name:path}."""
        repo = MagicMock()
        repo.load.return_value = []
        with patch("deepmt.ui.routers.api.MRRepository", return_value=repo):
            r = client.get("/api/mr-repository/torch.nn.functional.relu")
        assert r.status_code == 200


# ── /api/test-results ──────────────────────────────────────────────────────────

class TestApiTestResults:
    def _mock_rm(self, rows=None):
        rm = MagicMock()
        rm.get_summary.return_value = rows or []
        return rm

    def test_returns_200(self, client):
        with patch("deepmt.ui.routers.api.ResultsManager", return_value=self._mock_rm()):
            r = client.get("/api/test-results")
        assert r.status_code == 200

    def test_deduplication_keeps_latest(self, client):
        """When same (ir_name, framework) appears twice, latest last_updated wins."""
        rows = [
            {"ir_name": "torch.relu", "framework": "pytorch",
             "passed_tests": 5, "failed_tests": 0, "total_tests": 5,
             "last_updated": "2026-01-01 10:00:00"},
            {"ir_name": "torch.relu", "framework": "pytorch",
             "passed_tests": 8, "failed_tests": 2, "total_tests": 10,
             "last_updated": "2026-01-02 10:00:00"},  # newer
        ]
        with patch(
            "deepmt.ui.routers.api.ResultsManager",
            return_value=self._mock_rm(rows=rows)
        ):
            from deepmt.ui.routers import api as api_module
            api_module._CACHE.clear()
            body = client.get("/api/test-results").json()
        data = body["data"]
        assert len(data) == 1
        assert data[0]["passed_tests"] == 8   # newer row wins

    def test_sorted_by_failed_descending(self, client):
        rows = [
            {"ir_name": "op_a", "framework": "pytorch",
             "passed_tests": 9, "failed_tests": 1, "total_tests": 10,
             "last_updated": "2026-01-01"},
            {"ir_name": "op_b", "framework": "pytorch",
             "passed_tests": 5, "failed_tests": 5, "total_tests": 10,
             "last_updated": "2026-01-01"},
        ]
        with patch(
            "deepmt.ui.routers.api.ResultsManager",
            return_value=self._mock_rm(rows=rows)
        ):
            from deepmt.ui.routers import api as api_module
            api_module._CACHE.clear()
            body = client.get("/api/test-results").json()
        assert body["data"][0]["ir_name"] == "op_b"  # more failures first


class TestApiTestFailed:
    def test_returns_200(self, client):
        rm = MagicMock()
        rm.get_failed_tests.return_value = []
        with patch("deepmt.ui.routers.api.ResultsManager", return_value=rm):
            r = client.get("/api/test-results/failed")
        assert r.status_code == 200

    def test_limit_param_passed(self, client):
        rm = MagicMock()
        rm.get_failed_tests.return_value = []
        with patch("deepmt.ui.routers.api.ResultsManager", return_value=rm):
            client.get("/api/test-results/failed?limit=10")
        rm.get_failed_tests.assert_called_once_with(limit=10)


# ── /api/evidence ──────────────────────────────────────────────────────────────

class TestApiEvidence:
    def test_returns_200(self, client):
        ec = MagicMock()
        ec.list_all.return_value = []
        with patch("deepmt.ui.routers.api.EvidenceCollector", return_value=ec):
            r = client.get("/api/evidence")
        assert r.status_code == 200

    def test_fields_serialized(self, client):
        pack = _make_evidence(evidence_id="ev-abc", operator="torch.relu")
        ec = MagicMock()
        ec.list_all.return_value = [pack]
        with patch("deepmt.ui.routers.api.EvidenceCollector", return_value=ec):
            from deepmt.ui.routers import api as api_module
            api_module._CACHE.clear()
            body = client.get("/api/evidence").json()
        assert body["error"] is None
        item = body["data"][0]
        assert item["id"] == "ev-abc"
        assert item["operator"] == "torch.relu"
        assert "reproduce_script" not in item  # script excluded from list


class TestApiEvidenceScript:
    def test_returns_script(self, client):
        pack = _make_evidence(evidence_id="ev-001", reproduce_script="x = 1")
        ec = MagicMock()
        ec.load.return_value = pack
        with patch("deepmt.ui.routers.api.EvidenceCollector", return_value=ec):
            body = client.get("/api/evidence/ev-001/script").json()
        assert body["error"] is None
        assert body["data"]["script"] == "x = 1"

    def test_missing_evidence_returns_error(self, client):
        ec = MagicMock()
        ec.load.return_value = None
        with patch("deepmt.ui.routers.api.EvidenceCollector", return_value=ec):
            body = client.get("/api/evidence/no-such-id/script").json()
        assert body["error"] is not None
        assert body["data"] is None


# ── /api/cross-framework ───────────────────────────────────────────────────────

class TestApiCrossFramework:
    def test_returns_200(self, client):
        cf = MagicMock()
        cf.load_all.return_value = []
        with patch("deepmt.ui.routers.api.CrossFrameworkTester", return_value=cf):
            r = client.get("/api/cross-framework")
        assert r.status_code == 200

    def test_session_fields(self, client):
        sess = _make_session()
        cf = MagicMock()
        cf.load_all.return_value = [sess]
        with patch("deepmt.ui.routers.api.CrossFrameworkTester", return_value=cf):
            from deepmt.ui.routers import api as api_module
            api_module._CACHE.clear()
            body = client.get("/api/cross-framework").json()
        item = body["data"][0]
        assert item["session_id"] == "sess-001"
        assert item["operator"] == "torch.relu"
        assert "overall_consistency_rate" in item

    def test_nan_output_max_diff_becomes_none(self, client):
        sess = _make_session(output_max_diff=float("nan"))
        cf = MagicMock()
        cf.load_all.return_value = [sess]
        with patch("deepmt.ui.routers.api.CrossFrameworkTester", return_value=cf):
            from deepmt.ui.routers import api as api_module
            api_module._CACHE.clear()
            body = client.get("/api/cross-framework").json()
        assert body["data"][0]["output_max_diff"] is None


class TestApiCrossSession:
    def test_returns_detail(self, client):
        sess = _make_session(session_id="sess-xyz")
        cf = MagicMock()
        cf.load_all.return_value = [sess]
        with patch("deepmt.ui.routers.api.CrossFrameworkTester", return_value=cf):
            body = client.get("/api/cross-framework/sess-xyz").json()
        assert body["error"] is None
        assert body["data"]["session_id"] == "sess-xyz"

    def test_missing_session_returns_error(self, client):
        cf = MagicMock()
        cf.load_all.return_value = []
        with patch("deepmt.ui.routers.api.CrossFrameworkTester", return_value=cf):
            body = client.get("/api/cross-framework/no-such-session").json()
        assert body["error"] is not None


# ── TTL cache ──────────────────────────────────────────────────────────────────

class TestCache:
    def test_cache_is_hit_on_second_call(self):
        """Second call within TTL should not call factory again."""
        call_count = {"n": 0}

        def factory():
            call_count["n"] += 1
            return {"result": call_count["n"]}

        from deepmt.ui.routers.api import _cached, _CACHE
        _CACHE.clear()
        r1 = _cached("test_key", 60, factory)
        r2 = _cached("test_key", 60, factory)
        assert r1 == r2
        assert call_count["n"] == 1

    def test_cache_expires(self):
        """After TTL, factory is called again."""
        call_count = {"n": 0}

        def factory():
            call_count["n"] += 1
            return {"result": call_count["n"]}

        from deepmt.ui.routers.api import _cached, _CACHE
        _CACHE.clear()
        _cached("expire_key", 0, factory)   # ttl=0 → always expired
        _cached("expire_key", 0, factory)
        assert call_count["n"] == 2
