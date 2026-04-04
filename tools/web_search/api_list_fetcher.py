"""
API 列表获取器：从官方文档 HTML 解析框架个体 API 条目

从模块索引页获取所有模块页面 URL，再逐页提取每个 API 的名称、类型、签名。
支持缓存（每个模块页独立缓存，TTL 与 sphinx_search 共享），并发拉取。
"""

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from core.logger import get_logger
from tools.web_search.sphinx_search import CACHE_DIR, load_json_cache, save_json_cache

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ── 框架 API 索引页（stable） ────────────────────────────────────────────────
_FRAMEWORK_API_INDEX: Dict[str, str] = {
    "pytorch": "https://docs.pytorch.org/docs/stable/pytorch-api.html",
    # tensorflow:    预留，暂不支持
    # paddlepaddle:  预留，暂不支持
}

# 框架文档基础 URL（stable），用于构造版本化 URL
_FRAMEWORK_DOC_BASE: Dict[str, str] = {
    "pytorch": "https://docs.pytorch.org/docs/stable/",
}

# 版本化 URL 模板（future use）
_FRAMEWORK_DOC_VERSION_TPL: Dict[str, str] = {
    "pytorch": "https://docs.pytorch.org/docs/{version}/",
}

# 全量 API 缓存文件名前缀
_ALL_APIS_CACHE_PREFIX = "all_apis"

# 并发拉取模块页面的线程数
_FETCH_WORKERS = 8


class APIListFetcher:
    """从官方文档 HTML 提取框架 API 个体条目（类/函数级别）"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def fetch_all_apis(
        self,
        framework: str,
        version: str = "stable",
        use_cache: bool = True,
        skip_namespaces: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        从官方文档提取该框架全部 API 条目（类/函数级别）。

        Args:
            framework:        框架名称（目前仅 "pytorch" 可用）
            version:          "stable"（默认）或具体版本号如 "2.1"
                              非 stable 版本尚未实现，会抛 NotImplementedError
            use_cache:        是否使用本地文件缓存（TTL = CACHE_EXPIRY_SECONDS）
            skip_namespaces:  跳过这些前缀的模块页面（加速，用于已知不相关的命名空间）

        Returns:
            API 条目列表，每项包含：
                name      (str)  全路径，如 "torch.nn.ReLU"
                type      (str)  "class" | "function" | "method" | "attribute" | "other"
                signature (str)  参数签名，如 "(inplace=False)"
                url       (str)  文档页 URL（锚点到该条目）
        """
        self._check_supported(framework, version)

        # 全量缓存
        cache_path = self._all_apis_cache_path(framework, version)
        if use_cache:
            cached = load_json_cache(cache_path)
            if cached is not None:
                self.logger.debug(f"[{framework}] Loaded {len(cached)} APIs from cache")
                return cached

        # 1. 获取模块页面列表
        module_list = self.fetch_module_list(framework, version)
        self.logger.info(f"[{framework}] Found {len(module_list)} module pages")

        # 2. 过滤掉不需要的命名空间
        if skip_namespaces:
            filtered = []
            for m in module_list:
                if any(m["name"].startswith(ns) for ns in skip_namespaces):
                    self.logger.debug(f"  Skipping namespace: {m['name']}")
                else:
                    filtered.append(m)
            self.logger.info(
                f"[{framework}] After namespace filter: {len(filtered)} pages to fetch"
            )
        else:
            filtered = module_list

        # 3. 并发拉取每个模块页面，提取 API
        all_apis = self._fetch_pages_concurrent(filtered, use_cache=use_cache)

        # 4. 去重（同一 API 可能在多页出现，取第一次）
        seen: set = set()
        deduped: List[Dict] = []
        for api in all_apis:
            if api["name"] not in seen:
                seen.add(api["name"])
                deduped.append(api)

        self.logger.info(f"[{framework}] Total unique APIs: {len(deduped)}")
        save_json_cache(cache_path, deduped, indent=2)
        return deduped

    def fetch_module_list(self, framework: str, version: str = "stable") -> List[Dict]:
        """
        从 API 索引页获取模块页面列表。

        Returns:
            [ {"name": "torch.nn", "url": "https://..."}, ... ]
        """
        self._check_supported(framework, version)
        index_url = self._get_index_url(framework, version)
        try:
            resp = requests.get(index_url, headers=_HEADERS, timeout=30)
            resp.raise_for_status()
            return self._parse_module_index(resp.text, base_url=index_url)
        except Exception as e:
            self.logger.warning(f"Failed to fetch module list for {framework}: {e}")
            return []

    def fetch_module_page_apis(self, url: str, use_cache: bool = True) -> List[Dict]:
        """
        从单个模块页面提取所有 API 条目（带独立缓存）。

        Returns:
            API 条目列表（同 fetch_all_apis 返回格式）
        """
        cache_path = CACHE_DIR / f"module_page_{hashlib.md5(url.encode()).hexdigest()[:16]}.json"
        if use_cache:
            cached = load_json_cache(cache_path)
            if cached is not None:
                return cached
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
            resp.raise_for_status()
            apis = self._parse_module_page(resp.text, base_url=url)
            save_json_cache(cache_path, apis)
            return apis
        except Exception as e:
            self.logger.debug(f"Failed to fetch module page {url}: {e}")
            return []

    # ── HTML 解析 ──────────────────────────────────────────────────────────────

    def parse_module_index_html(self, html: str, base_url: str = "") -> List[Dict]:
        """从 API 索引页 HTML 提取模块页面链接（公开版本，供测试/外部使用）"""
        return self._parse_module_index(html, base_url)

    def parse_module_page_html(self, html: str, base_url: str = "") -> List[Dict]:
        """从模块页面 HTML 提取 API 条目（公开版本，供测试/外部使用）"""
        return self._parse_module_page(html, base_url)

    def _parse_module_index(self, html: str, base_url: str = "") -> List[Dict]:
        """解析索引页，提取所有 torch.* 模块页面链接"""
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find(id="pytorch-article") or soup.find("main") or soup.body
        if not article:
            return []

        results = []
        seen_urls: set = set()
        for a_tag in article.find_all("a", class_="reference"):
            href = str(a_tag.get("href") or "")
            # 跳过：无 href、纯页内锚点（以 # 开头）、站外链接（非文档域）
            if not href or href.startswith("#"):
                continue

            name = a_tag.get_text(strip=True)
            if not name:
                continue

            # 只保留 torch.* 开头的条目
            if not name.startswith("torch"):
                continue

            resolved = href if href.startswith("http") else urljoin(base_url, href)

            # 去掉同一页不同锚点的重复（取第一个 URL）
            url_without_anchor = resolved.split("#")[0]
            if url_without_anchor in seen_urls:
                continue
            seen_urls.add(url_without_anchor)

            results.append({"name": name, "url": url_without_anchor})

        return results

    def _parse_module_page(self, html: str, base_url: str = "") -> List[Dict]:
        """
        从 PyTorch 文档页面提取所有 API 条目，支持两种页面结构：

        **模块概览页**（如 torch.nn.html、torch.html）：
            页面内不内联 API 签名，而是通过链接指向个体 generated 页面，如：
              <a class="reference internal" href="generated/torch.nn.ReLU.html">ReLU</a>
            → 从 URL 提取 API 名称，从命名约定推断类型（首字母大写 = class）。
            签名在此阶段不提取（为空），可后续通过 fetch-doc 按需获取。

        **个体 generated 页**（如 generated/torch.matmul.html）：
            页面内联 <dt class="sig sig-object py" id="torch.matmul"> 元素，
            可直接提取 id（API 名）、父 dl 类（类型）、sig-param（签名）。
        """
        soup = BeautifulSoup(html, "html.parser")

        # URL 中含有 "generated/" 说明是个体 API 页面，直接走策略二
        is_generated_page = "generated/" in base_url

        if not is_generated_page:
            # ── 策略一：模块概览页（如 torch.nn.html）────────────────────────────
            # 文章区内的 <a class="reference"> 链接指向 generated/*.html，
            # 从 URL 提取 API 完整名称，从命名约定推断类型（首字母大写 = class）
            article = soup.find(id="pytorch-article") or soup.find("main") or soup.body
            results: List[Dict] = []
            if article:
                seen_urls: set = set()
                for a in article.find_all("a", class_="reference"):
                    href = str(a.get("href") or "")
                    m = re.search(r"generated/([^/#]+)\.html", href)
                    if not m:
                        continue
                    api_name = m.group(1)
                    if not api_name.startswith("torch"):
                        continue
                    resolved = href if href.startswith("http") else urljoin(base_url, href)
                    url_no_anchor = resolved.split("#")[0]
                    if url_no_anchor in seen_urls:
                        continue
                    seen_urls.add(url_no_anchor)
                    last_part = api_name.rsplit(".", 1)[-1]
                    api_type = "class" if last_part[:1].isupper() else "function"
                    results.append({
                        "name": api_name,
                        "type": api_type,
                        "signature": "",   # 模块概览页不含签名；通过 fetch-doc 按需获取
                        "url": url_no_anchor,
                    })
            return results

        # ── 策略二：个体 generated 页面（如 generated/torch.matmul.html）─────────
        # 直接解析内联 <dt class="sig sig-object py" id="torch.*"> 获取名称和签名
        results = []
        for dt in soup.find_all("dt", class_="sig"):
            api_id = dt.get("id", "")
            if not api_id or not api_id.startswith("torch"):
                continue

            parent_dl = dt.parent
            dl_classes = set(parent_dl.get("class", [])) if parent_dl else set()
            if "class" in dl_classes:
                api_type = "class"
            elif "function" in dl_classes:
                api_type = "function"
            elif "method" in dl_classes:
                api_type = "method"
            elif "attribute" in dl_classes or "property" in dl_classes:
                api_type = "attribute"
            elif "exception" in dl_classes:
                api_type = "exception"
            else:
                api_type = "other"

            sig = self._extract_signature(dt)
            link = dt.find("a", class_="headerlink")
            doc_url = urljoin(base_url, str(link["href"])) if (link and link.get("href")) else base_url

            results.append({
                "name": api_id,
                "type": api_type,
                "signature": sig,
                "url": doc_url,
            })

        return results

    def _extract_signature(self, dt) -> str:
        """从 <dt> 节点提取参数签名字符串，如 '(input, other, *, out=None)'"""
        # 参数在 <em class="sig-param"> 中
        params = dt.find_all("em", class_="sig-param")
        if not params:
            # 检查是否根本没有括号（属性/常量）
            if not dt.find("span", class_="sig-paren"):
                return ""
            return "()"
        param_strs = [p.get_text(strip=True) for p in params]
        return f"({', '.join(param_strs)})"

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _fetch_pages_concurrent(
        self, module_list: List[Dict], use_cache: bool = True
    ) -> List[Dict]:
        """并发拉取模块页面并聚合 API"""
        all_apis: List[Dict] = []

        with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as executor:
            future_to_module = {
                executor.submit(self.fetch_module_page_apis, m["url"], use_cache): m
                for m in module_list
            }
            for future in as_completed(future_to_module):
                module = future_to_module[future]
                try:
                    apis = future.result()
                    all_apis.extend(apis)
                    self.logger.debug(f"  {module['name']}: {len(apis)} APIs")
                except Exception as e:
                    self.logger.warning(f"  {module['name']}: error - {e}")

        return all_apis

    def _check_supported(self, framework: str, version: str) -> None:
        """校验 framework 和 version 是否支持，不支持则抛 NotImplementedError"""
        if framework not in _FRAMEWORK_API_INDEX:
            raise NotImplementedError(
                f"API list fetching for framework '{framework}' is not yet implemented. "
                f"Currently supported: {list(_FRAMEWORK_API_INDEX.keys())}"
            )
        if version != "stable":
            raise NotImplementedError(
                f"Versioned API list (version='{version}') is not yet implemented. "
                f"Different documentation versions have different HTML structures. "
                f"Currently only version='stable' is supported."
            )

    def _get_index_url(self, framework: str, version: str) -> str:
        """获取框架 API 索引页 URL"""
        if version == "stable":
            return _FRAMEWORK_API_INDEX[framework]
        # 版本化 URL（future use，此处不会到达，因为 _check_supported 已拦截）
        tpl = _FRAMEWORK_DOC_VERSION_TPL.get(framework, "")
        return tpl.format(version=version) + "pytorch-api.html"

    def _all_apis_cache_path(self, framework: str, version: str) -> Path:
        """返回全量 API 列表的缓存文件路径"""
        key = f"{_ALL_APIS_CACHE_PREFIX}_{framework}_{version}"
        return CACHE_DIR / f"{key}_{hashlib.md5(key.encode()).hexdigest()[:12]}.json"
