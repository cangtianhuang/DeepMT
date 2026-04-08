"""
OperatorEnricher 单元测试（无 LLM / 无网络依赖）

测试 inspect 层和内部工具方法；HTML 解析和 LLM 层通过 mock 覆盖。
"""

import pytest
from unittest.mock import MagicMock, patch

from deepmt.mr_generator.base.operator_enricher import (
    OperatorEnricher,
    _InputSpec,
    _annotation_dtype,
    _annotation_is_tensor,
    _FLOAT_DTYPES,
    _INT_DTYPES,
)
from deepmt.mr_generator.base.operator_catalog import OperatorEntry


# ── _InputSpec ──────────────────────────────────────────────────────────────

class TestInputSpec:
    def test_to_dict_basic(self):
        spec = _InputSpec(name="input", dtype=["float32", "float64"])
        d = spec.to_dict()
        assert d["name"] == "input"
        assert d["dtype"] == ["float32", "float64"]
        assert d["shape"] == "any"
        assert d["value_range"] is None
        assert d["required"] is True

    def test_to_dict_with_value_range(self):
        spec = _InputSpec(name="input", dtype=["float32"], value_range=[0, None])
        d = spec.to_dict()
        assert d["value_range"] == [0, None]

    def test_to_dict_optional(self):
        spec = _InputSpec(name="bias", dtype=[], required=False)
        d = spec.to_dict()
        assert d["required"] is False


# ── OperatorEntry 新字段 ──────────────────────────────────────────────────────

class TestOperatorEntryNewFields:
    def test_new_fields_defaults(self):
        entry = OperatorEntry({"name": "torch.nn.ReLU"})
        assert entry.api_path == ""
        assert entry.api_style == ""
        assert entry.module_class == ""
        assert entry.input_specs == []
        assert entry.input_specs_auto is False

    def test_new_fields_from_data(self):
        data = {
            "name": "torch.nn.functional.relu",
            "api_type": "function",
            "api_path": "torch.nn.functional.relu",
            "api_style": "function",
            "module_class": "torch.nn.ReLU",
            "input_specs": [
                {"name": "input", "dtype": ["float32"], "shape": "any",
                 "value_range": None, "required": True}
            ],
            "input_specs_auto": True,
        }
        entry = OperatorEntry(data)
        assert entry.api_path == "torch.nn.functional.relu"
        assert entry.api_style == "function"
        assert entry.module_class == "torch.nn.ReLU"
        assert len(entry.input_specs) == 1
        assert entry.input_specs[0]["name"] == "input"
        assert entry.input_specs_auto is True

    def test_to_dict_includes_new_fields(self):
        data = {
            "name": "torch.nn.functional.relu",
            "api_type": "function",
            "api_path": "torch.nn.functional.relu",
            "input_specs": [
                {"name": "input", "dtype": ["float32"], "shape": "any",
                 "value_range": None, "required": True}
            ],
            "input_specs_auto": True,
        }
        entry = OperatorEntry(data)
        d = entry.to_dict()
        assert "api_path" in d
        assert "input_specs" in d
        assert "input_specs_auto" in d
        assert d["input_specs_auto"] is True

    def test_to_dict_omits_empty_new_fields(self):
        entry = OperatorEntry({"name": "torch.nn.ReLU"})
        d = entry.to_dict()
        assert "api_path" not in d
        assert "input_specs" not in d
        assert "input_specs_auto" not in d


# ── OperatorEnricher: inspect 层 ────────────────────────────────────────────

class TestOperatorEnricherInspect:
    def setup_method(self):
        self.enricher = OperatorEnricher()

    def test_inspect_function(self):
        """torch.nn.functional.relu 可通过 inspect 提取到 input 参数"""
        result = self.enricher._from_inspect(
            "torch.nn.functional.relu", "function"
        )
        assert "input_specs" in result
        specs = result["input_specs"]
        assert len(specs) >= 1
        input_spec = next((s for s in specs if s["name"] == "input"), None)
        assert input_spec is not None
        assert input_spec["required"] is True
        assert input_spec["shape"] == "any"

    def test_inspect_class_uses_forward(self):
        """torch.nn.ReLU 类通过 forward 提取参数"""
        result = self.enricher._from_inspect("torch.nn.ReLU", "class")
        assert "input_specs" in result
        specs = result["input_specs"]
        assert any(s["name"] == "input" for s in specs)

    def test_inspect_invalid_name(self):
        """无效名称不抛异常，返回空字典"""
        result = self.enricher._from_inspect("torch.nonexistent.op", "function")
        assert result == {}

    def test_inspect_signature_returned(self):
        """inspect 同时返回 signature 字符串"""
        result = self.enricher._from_inspect(
            "torch.nn.functional.relu", "function"
        )
        assert "signature" in result
        assert "input" in result["signature"]


# ── OperatorEnricher: HTML 解析层 ────────────────────────────────────────────

class TestOperatorEnricherHTML:
    def setup_method(self):
        self.enricher = OperatorEnricher()

    def test_enrich_from_html_fills_dtype(self):
        """HTML 参数描述中的 Tensor 类型可补充 dtype"""
        html = """
        <html><body>
        <dl class="field-list simple">
          <dd><ul>
            <li><p><strong>input</strong> (<em>Tensor</em>) – Input tensor.</p></li>
          </ul></dd>
        </dl>
        </body></html>
        """
        updates = {
            "input_specs": [
                {"name": "input", "dtype": [], "shape": "any",
                 "value_range": None, "required": True}
            ]
        }
        self.enricher._enrich_from_html(html, updates, framework="pytorch")
        # 纯 "Tensor" 类型 → dtype=any（无具体限制，有别于空列表的"未知"）
        assert updates["input_specs"][0]["dtype"] == "any"

    def test_enrich_from_html_no_overwrite(self):
        """HTML 解析不覆盖已有 dtype"""
        html = """
        <html><body>
        <li><p><strong>input</strong> (<em>Tensor</em>) – ...</p></li>
        </body></html>
        """
        updates = {
            "input_specs": [
                {"name": "input", "dtype": ["float32"], "shape": "any",
                 "value_range": None, "required": True}
            ]
        }
        self.enricher._enrich_from_html(html, updates, framework="pytorch")
        # 已有 dtype，不应覆盖
        assert updates["input_specs"][0]["dtype"] == ["float32"]

    def test_dtype_from_html_type_int(self):
        assert self.enricher._dtype_from_html_type("IntTensor") == _INT_DTYPES

    def test_dtype_from_html_type_bool(self):
        assert self.enricher._dtype_from_html_type("BoolTensor") == ["bool"]

    def test_dtype_from_html_type_non_tensor(self):
        assert self.enricher._dtype_from_html_type("int") == []


# ── OperatorEnricher: LLM 合并逻辑 ──────────────────────────────────────────

class TestOperatorEnricherLLMMerge:
    def setup_method(self):
        self.enricher = OperatorEnricher()

    def test_merge_llm_specs_fills_missing(self):
        """LLM specs 可补充空字段"""
        updates = {
            "input_specs": [
                {"name": "input", "dtype": [], "shape": "any",
                 "value_range": None, "required": True}
            ]
        }
        llm_specs = [
            {"name": "input", "dtype": ["float32", "float64"],
             "shape": "nd>=1", "value_range": [0, None], "required": True}
        ]
        self.enricher._merge_llm_specs(updates, llm_specs)
        spec = updates["input_specs"][0]
        assert spec["dtype"] == ["float32", "float64"]
        assert spec["value_range"] == [0, None]
        assert spec["shape"] == "nd>=1"

    def test_merge_llm_specs_no_overwrite_existing(self):
        """LLM specs 不覆盖已填写的字段"""
        updates = {
            "input_specs": [
                {"name": "input", "dtype": ["float16"], "shape": "any",
                 "value_range": None, "required": True}
            ]
        }
        llm_specs = [
            {"name": "input", "dtype": ["float32", "float64"],
             "shape": "nd>=2", "value_range": None, "required": True}
        ]
        self.enricher._merge_llm_specs(updates, llm_specs)
        # dtype 已有，不覆盖
        assert updates["input_specs"][0]["dtype"] == ["float16"]

    def test_merge_llm_specs_adds_new_param(self):
        """LLM 发现的新参数追加到 input_specs"""
        updates: dict = {"input_specs": []}
        llm_specs = [
            {"name": "weight", "dtype": ["float32"], "shape": "(out, in)",
             "value_range": None, "required": True}
        ]
        self.enricher._merge_llm_specs(updates, llm_specs)
        assert len(updates["input_specs"]) == 1
        assert updates["input_specs"][0]["name"] == "weight"


# ── enrich() 集成测试（mock 网络/LLM）──────────────────────────────────────

class TestOperatorEnricherIntegration:
    def setup_method(self):
        self.enricher = OperatorEnricher()

    def test_enrich_sets_auto_flag(self):
        """enrich() 成功提取到 input_specs 后自动设置 input_specs_auto"""
        updates = self.enricher.enrich(
            name="torch.nn.functional.relu",
            framework="pytorch",
            api_type="function",
            doc_url="",
            use_llm=False,
        )
        assert updates.get("input_specs_auto") is True

    def test_enrich_no_result_no_auto_flag(self):
        """无法提取任何 input_specs 时不设置 input_specs_auto"""
        updates = self.enricher.enrich(
            name="torch.nonexistent_module.op",
            framework="pytorch",
            api_type="function",
            doc_url="",
            use_llm=False,
        )
        assert "input_specs_auto" not in updates

    @patch.object(OperatorEnricher, "_fetch_doc_text", return_value=None)
    @patch.object(OperatorEnricher, "_fetch_html", return_value=None)
    def test_enrich_skips_network_when_no_doc_url(self, mock_html, mock_text):
        """doc_url 为空时不发起网络请求"""
        self.enricher.enrich(
            name="torch.nn.functional.relu",
            framework="pytorch",
            api_type="function",
            doc_url="",
            use_llm=True,
            llm_client=MagicMock(),
        )
        mock_html.assert_not_called()
        mock_text.assert_not_called()
