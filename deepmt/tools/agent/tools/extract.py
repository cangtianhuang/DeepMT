"""提取工具：从 HTML 或文本中提取结构化信息"""

import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from deepmt.tools.agent.tool_registry import tool


@tool(
    name="extract_links",
    description=(
        "从 HTML 源码中提取超链接，返回 JSON 列表（每项含 text 和 url 字段）。"
        "可用 filter_pattern 按正则过滤 URL，用 base_url 补全相对路径。"
        "适合从文档索引页或 GitHub release 页面提取所有链接。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "html": {
                "type": "string",
                "description": "待解析的 HTML 源码",
            },
            "base_url": {
                "type": "string",
                "description": "用于补全相对 URL 的基础地址，如 https://pytorch.org",
            },
            "filter_pattern": {
                "type": "string",
                "description": (
                    "可选正则表达式，只返回 URL 匹配该模式的链接。"
                    "例如 r'/releases/tag/' 只保留 release 页链接。"
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "最多返回多少条链接，默认 50",
            },
        },
        "required": ["html"],
    },
)
def extract_links(
    html: str,
    base_url: str = "",
    filter_pattern: str = "",
    max_results: int = 50,
) -> str:
    """
    从 HTML 中提取链接。

    Returns:
        JSON 字符串，格式: [{"text": "...", "url": "..."}]
    """
    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict[str, str]] = []
    seen_urls: set = set()

    pattern = re.compile(filter_pattern) if filter_pattern else None

    for a in soup.find_all("a", href=True):
        href = str(a["href"]).strip()
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue

        # 补全相对路径
        if base_url and not href.startswith(("http://", "https://")):
            href = urljoin(base_url, href)

        if pattern and not pattern.search(href):
            continue

        if href in seen_urls:
            continue
        seen_urls.add(href)

        text = a.get_text(strip=True)
        results.append({"text": text, "url": href})

        if len(results) >= max_results:
            break

    return json.dumps(results, ensure_ascii=False)


@tool(
    name="extract_text",
    description=(
        "从 HTML 源码中按 CSS 选择器提取指定区域的纯文本。"
        "selector 留空则提取 <body> 全部文本。"
        "适合精确提取页面中特定区域（如版本号标题、API 列表）。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "html": {
                "type": "string",
                "description": "待解析的 HTML 源码",
            },
            "selector": {
                "type": "string",
                "description": (
                    "CSS 选择器，如 'h1.version', '.release-title', '#content'。"
                    "留空则提取全部 body 文本。"
                ),
            },
            "max_chars": {
                "type": "integer",
                "description": "返回文本最大字符数，默认 4000",
            },
        },
        "required": ["html"],
    },
)
def extract_text(html: str, selector: str = "", max_chars: int = 4000) -> str:
    """
    从 HTML 按 CSS 选择器提取文本。

    Returns:
        纯文本字符串
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    if selector:
        elements = soup.select(selector)
        if not elements:
            return f"[未找到] 选择器 '{selector}' 没有匹配的元素"
        text = "\n".join(el.get_text(separator="\n", strip=True) for el in elements)
    else:
        body = soup.find("body") or soup
        text = body.get_text(separator="\n", strip=True)

    # 压缩连续空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [已截断，共 {len(text)} 字符]"
    return text


@tool(
    name="find_version_tags",
    description=(
        "在 HTML 或纯文本中搜索版本号（如 v2.5.1、2.5.1、2.5），"
        "同时识别 'Latest'、'stable' 等标记，返回 JSON 列表。"
        "适合从 GitHub release 页或框架官网提取最新版本号。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "待搜索的文本内容（HTML 或纯文本均可）",
            },
            "max_results": {
                "type": "integer",
                "description": "最多返回多少个版本号，默认 20",
            },
        },
        "required": ["text"],
    },
)
def find_version_tags(text: str, max_results: int = 20) -> str:
    """
    在文本中找到版本号。

    Returns:
        JSON 字符串，格式:
        [{"version": "2.5.1", "is_latest": true, "context": "...surrounding text..."}]
    """
    # 如果是 HTML，先提取文本
    if "<html" in text.lower() or "<body" in text.lower() or "<div" in text.lower():
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")

    # 版本号正则：vX.Y.Z、X.Y.Z、X.Y（主版本至少2段）
    version_pattern = re.compile(
        r"\bv?(\d+\.\d+(?:\.\d+)?(?:\.\d+)?(?:[.-](?:rc|alpha|beta|post|dev)\d*)?)\b",
        re.IGNORECASE,
    )

    # "latest" / "stable" 等关键词附近的行
    latest_keywords = re.compile(r"\b(latest|stable|current|recommended)\b", re.IGNORECASE)

    results: List[Dict[str, Any]] = []
    seen_versions: set = set()

    lines = text.splitlines()
    for i, line in enumerate(lines):
        for match in version_pattern.finditer(line):
            ver = match.group(1)
            if ver in seen_versions:
                continue
            seen_versions.add(ver)

            # 上下文：当前行 ± 1 行（仅用于展示）
            ctx_lines = lines[max(0, i - 1) : i + 2]
            context = " | ".join(l.strip() for l in ctx_lines if l.strip())[:200]

            # is_latest 只检查当前行，避免相邻行的 "Latest" 关键词污染其他版本
            is_latest = bool(latest_keywords.search(line))

            results.append(
                {
                    "version": ver,
                    "is_latest": is_latest,
                    "context": context,
                }
            )
            if len(results) >= max_results:
                break
        if len(results) >= max_results:
            break

    # 将标记为 latest 的排在前面
    results.sort(key=lambda x: (not x["is_latest"], x["version"]), reverse=False)
    return json.dumps(results, ensure_ascii=False)


@tool(
    name="search_in_text",
    description=(
        "在纯文本中按正则表达式搜索，返回所有匹配的行（含行号）。"
        "适合在大段文本中快速定位关键词，如算子名称、URL、版本号等。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "待搜索的文本内容",
            },
            "pattern": {
                "type": "string",
                "description": "Python 正则表达式，如 r'torch\\.nn\\.\\w+'",
            },
            "max_results": {
                "type": "integer",
                "description": "最多返回多少行，默认 100",
            },
            "context_lines": {
                "type": "integer",
                "description": "每个匹配项上下各显示多少行，默认 0（只显示匹配行）",
            },
        },
        "required": ["text", "pattern"],
    },
)
def search_in_text(
    text: str,
    pattern: str,
    max_results: int = 100,
    context_lines: int = 0,
) -> str:
    """
    正则搜索文本，返回匹配行。

    Returns:
        匹配结果文本，格式: "行号: 内容"，多个结果以换行分隔
    """
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"[正则错误] {e}"

    lines = text.splitlines()
    output_lines: List[str] = []
    count = 0

    for i, line in enumerate(lines):
        if compiled.search(line):
            if context_lines > 0:
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                for j in range(start, end):
                    prefix = ">>>" if j == i else "   "
                    output_lines.append(f"{prefix} {j + 1}: {lines[j]}")
                output_lines.append("---")
            else:
                output_lines.append(f"{i + 1}: {line}")

            count += 1
            if count >= max_results:
                output_lines.append(f"... [已达上限 {max_results} 条，结果截断]")
                break

    if not output_lines:
        return f"[未找到] 模式 '{pattern}' 在文本中没有匹配"
    return "\n".join(output_lines)
