"""测试 WebSearchTool 的功能"""

import unittest
from tools.web_search.search_tool import WebSearchTool


class TestWebSearchTool(unittest.TestCase):
    """测试 WebSearchTool 类"""

    def setUp(self):
        """初始化测试环境"""
        self.tool = WebSearchTool()
        self.operator_name = "relu"
        self.framework = "pytorch"

    def test_docs_search(self):
        """测试文档搜索功能"""
        results = self.tool.search_operator(
            operator_name=self.operator_name,
            framework=self.framework,
            sources={"docs": True, "github": False, "web_search": False},
        )

        # 验证返回结果
        self.assertIsInstance(results, list)
        if len(results) > 0:
            # 验证结果结构
            result = results[0]
            self.assertTrue(hasattr(result, 'title'))
            self.assertTrue(hasattr(result, 'url'))
            self.assertTrue(hasattr(result, 'source'))
            self.assertTrue(hasattr(result, 'snippet'))
            self.assertEqual(result.source, 'docs')

    def test_github_search(self):
        """测试 GitHub 搜索功能"""
        results = self.tool.search_operator(
            operator_name=self.operator_name,
            framework=self.framework,
            sources={"docs": False, "github": True, "web_search": False},
        )

        # 验证返回结果
        self.assertIsInstance(results, list)
        # GitHub 搜索可能需要 token，所以不强制要求有结果

    def test_web_search(self):
        """测试网络搜索功能"""
        results = self.tool.search_operator(
            operator_name=self.operator_name,
            framework=self.framework,
            sources={"docs": False, "github": False, "web_search": True},
        )

        # 验证返回结果
        self.assertIsInstance(results, list)
        # 网络搜索可能需要 API key，所以不强制要求有结果

    def test_all_sources(self):
        """测试所有搜索源"""
        results = self.tool.search_operator(
            operator_name=self.operator_name,
            framework=self.framework,
            sources={"docs": True, "github": True, "web_search": True},
        )

        # 验证返回结果
        self.assertIsInstance(results, list)

        # 统计各来源的结果数量
        source_counts = {}
        for result in results:
            source_counts[result.source] = source_counts.get(result.source, 0) + 1

        # 至少应该有文档搜索的结果
        self.assertGreater(len(results), 0, "应该至少有一些搜索结果")

    def test_empty_operator_name(self):
        """测试空算子名称"""
        with self.assertRaises(ValueError):
            self.tool.search_operator(
                operator_name="",
                framework=self.framework,
            )

    def test_invalid_framework(self):
        """测试无效的框架名称"""
        # 应该能处理无效框架，可能返回空结果或使用默认值
        results = self.tool.search_operator(
            operator_name=self.operator_name,
            framework="invalid_framework",
        )
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
