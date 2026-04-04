"""浏览工具：获取网页内容"""

from typing import Optional

import requests
from bs4 import BeautifulSoup

from deepmt.tools.agent.tool_registry import tool

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
_DEFAULT_TIMEOUT = 15


@tool(
    name="fetch_page",
    description=(
        "获取指定 URL 的网页，返回清洗后的纯文本内容（去除 HTML 标签和脚本）。"
        "适合读取文档页面、GitHub release 页、API 列表等。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "目标网页 URL",
            },
            "selector": {
                "type": "string",
                "description": (
                    "可选的 CSS 选择器，指定只提取页面中特定区域的文本。"
                    "留空则提取整页主要内容。"
                ),
            },
        },
        "required": ["url"],
    },
)
def fetch_page(url: str, selector: str = "") -> str:
    """
    获取网页纯文本内容。

    Args:
        url: 目标 URL
        selector: 可选 CSS 选择器，限定提取区域

    Returns:
        网页纯文本（截断至 8000 字符以适应 LLM 上下文）
    """
    html = _get_html(url)
    if html is None:
        return f"[错误] 无法获取页面: {url}"

    soup = BeautifulSoup(html, "html.parser")

    # 移除无关标签
    for tag in soup(["script", "style", "nav", "footer", "head", "noscript"]):
        tag.decompose()

    if selector:
        container = soup.select_one(selector)
        if container is None:
            return f"[警告] 未找到选择器 '{selector}' 匹配的元素，返回全页内容\n\n" + _extract_text(soup)
        return _extract_text(container)

    # 尝试常见主内容区域
    for candidate in ["main", "article", "#pytorch-article", "#content", ".content"]:
        container = soup.select_one(candidate)
        if container:
            return _extract_text(container)

    return _extract_text(soup)


@tool(
    name="fetch_page_html",
    description=(
        "获取指定 URL 的原始 HTML 源码。"
        "适合需要精确分析页面结构（如查找链接、表格）时使用，"
        "通常与 extract_links / extract_text 配合使用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "目标网页 URL",
            },
        },
        "required": ["url"],
    },
)
def fetch_page_html(url: str) -> str:
    """
    获取网页原始 HTML（截断至 20000 字符）。

    Args:
        url: 目标 URL

    Returns:
        HTML 源码字符串
    """
    html = _get_html(url)
    if html is None:
        return f"[错误] 无法获取页面: {url}"
    return html[:20000]


# --------------------------------------------------------------------------- #
#  内部辅助函数（不注册为工具）                                                  #
# --------------------------------------------------------------------------- #

def _get_html(url: str, timeout: int = _DEFAULT_TIMEOUT) -> Optional[str]:
    """发送 GET 请求，返回响应文本；失败返回 None"""
    try:
        resp = requests.get(url, headers=_DEFAULT_HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text
    except Exception:
        return None


def _extract_text(soup_element) -> str:
    """从 BeautifulSoup 元素提取纯文本，压缩空白行，截断至 8000 字符"""
    lines = []
    for line in soup_element.get_text(separator="\n").splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)

    text = "\n".join(lines)
    if len(text) > 8000:
        text = text[:8000] + f"\n... [内容已截断，共 {len(text)} 字符]"
    return text
