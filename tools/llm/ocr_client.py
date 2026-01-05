"""
OCR客户端：使用百度千帆OCR API识别图片中的公式和文本
"""

import re
from typing import Any, Dict, Optional, Set
from urllib.parse import urlparse

import requests

from core.config_loader import get_config_value
from core.logger import get_logger

# OCR API 支持的图片格式
SUPPORTED_IMAGE_FORMATS: Set[str] = {"pdf", "jpeg", "jpg", "png", "tiff", "tif", "bmp"}

# 不支持的格式（直接跳过，不警告）
UNSUPPORTED_IMAGE_FORMATS: Set[str] = {
    "svg",
    "webp",
    "gif",
    "ico",
    "avif",
    "heic",
    "heif",
}


class OCRClient:
    """
    OCR 客户端：使用百度千帆 OCR API 识别图片

    主要用于识别网页中的公式图片
    """

    def __init__(self) -> None:
        """初始化 OCR 客户端"""
        self.logger = get_logger()
        # 从配置加载器获取配置值
        self.api_key = get_config_value("web_search.baidu_api_key", "")
        self.enabled = get_config_value("web_search.ocr", False)

        # 从 LLM 配置获取基础 URL
        base_url = get_config_value("llm.url", "https://qianfan.baidubce.com/v2/")
        self.ocr_api_url = base_url.rstrip("/") + "/ocr/paddleocr"

        if not self.api_key:
            self.logger.warning(
                "Baidu API key not found. OCR functionality will be disabled."
            )
            self.enabled = False

    def _is_supported_format(self, image_url: str) -> bool:
        """
        检查图片URL是否为支持的格式

        Args:
            image_url: 图片URL

        Returns:
            是否为支持的格式
        """
        # 跳过 data URI
        if image_url.startswith("data:"):
            return False

        # 提取文件扩展名
        parsed = urlparse(image_url)
        path = parsed.path.lower()
        netloc = parsed.netloc.lower()

        # 跳过追踪像素和分析脚本（常见的非图片 URL）
        tracking_domains = {
            "facebook.com",
            "www.facebook.com",
            "google.com",
            "www.google.com",
            "googletagmanager.com",
            "www.googletagmanager.com",
            "google-analytics.com",
            "www.google-analytics.com",
            "doubleclick.net",
            "analytics.",  # 匹配任何包含 analytics 的域名
        }
        for domain in tracking_domains:
            if domain in netloc:
                return False

        # 跳过常见的追踪路径
        tracking_paths = {"/tr", "/pixel", "/beacon", "/track", "/analytics"}
        for tracking_path in tracking_paths:
            if path.startswith(tracking_path) or tracking_path + "?" in path:
                return False

        # 从路径中提取扩展名
        ext_match = re.search(r"\.([a-z0-9]+)(?:\?|#|$)", path)
        if ext_match:
            ext = ext_match.group(1)
            # 如果是不支持的格式，直接返回 False
            if ext in UNSUPPORTED_IMAGE_FORMATS:
                return False
            # 如果是支持的格式，返回 True
            if ext in SUPPORTED_IMAGE_FORMATS:
                return True
            # 其他扩展名（如 php、asp 等）跳过
            return False

        # 如果没有扩展名，默认跳过（避免调用无效的 API）
        # 真正的图片 URL 通常都有明确的扩展名
        return False

    def recognize_formula(
        self, image_url: str, use_layout_detection: bool = False
    ) -> Optional[str]:
        """
        识别图片中的公式

        Args:
            image_url: 图片URL
            use_layout_detection: 是否使用版面分析（False时专门识别公式）

        Returns:
            识别出的公式文本（LaTeX格式），如果失败则返回None
        """
        if not self.enabled:
            return None

        return self._call_ocr_api(
            image_url=image_url,
            prompt_label="formula" if not use_layout_detection else None,
            use_layout_detection=use_layout_detection,
        )

    def recognize_text(
        self, image_url: str, use_layout_detection: bool = True
    ) -> Optional[str]:
        """
        识别图片中的文本

        Args:
            image_url: 图片URL
            use_layout_detection: 是否使用版面分析

        Returns:
            识别出的文本内容，如果失败则返回None
        """
        if not self.enabled:
            return None

        if result := self._call_ocr_api(
            image_url=image_url,
            prompt_label="ocr" if not use_layout_detection else None,
            use_layout_detection=use_layout_detection,
        ):
            self.logger.info(f"OCR API result for {image_url}: {result}")
            return result

        self.logger.warning(f"OCR API failed for {image_url}")
        return None

    def _call_ocr_api(
        self,
        image_url: str,
        prompt_label: Optional[str] = None,
        use_layout_detection: bool = True,
    ) -> Optional[str]:
        """
        调用百度千帆OCR API

        Args:
            image_url: 图片URL
            prompt_label: prompt类型（ocr/formula/table/chart），仅在use_layout_detection=False时有效
            use_layout_detection: 是否使用版面分析

        Returns:
            识别结果文本，如果失败则返回None
        """
        if not self.api_key:
            self.logger.warning("Baidu API key not configured")
            return None

        # 检查图片格式是否支持
        if not self._is_supported_format(image_url):
            self.logger.debug(f"Skipping unsupported image format: {image_url}")
            return None

        try:
            # 构建请求体
            request_body: Dict[str, Any] = {
                "model": "paddleocr-vl-0.9b",
                "file": image_url,
                "useLayoutDetection": use_layout_detection,
                "useDocUnwarping": False,  # 默认关闭扭曲矫正
                "useDocOrientationClassify": False,  # 默认关闭方向矫正
                "useChartRecognition": False,  # 默认关闭图表识别
                "layoutNms": True,  # 启用NMS后处理
                "visualize": False,  # 不返回可视化图像
            }

            # 如果关闭版面分析，必须指定prompt_label
            if not use_layout_detection and prompt_label:
                request_body["promptLabel"] = prompt_label

            # 发送请求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                self.ocr_api_url, headers=headers, json=request_body, timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # 解析结果
            result = data.get("result", {})
            layout_parsing_results = result.get("layoutParsingResults", [])

            if not layout_parsing_results:
                return None

            # 提取文本内容
            texts = []
            for layout_result in layout_parsing_results:
                # 优先使用markdown格式的文本
                markdown = layout_result.get("markdown", {})
                if markdown and markdown.get("text"):
                    texts.append(markdown["text"])

                # 如果没有markdown，尝试从prunedResult中提取
                pruned_result = layout_result.get("prunedResult", {})
                parsing_res_list = pruned_result.get("parsing_res_list", [])
                for item in parsing_res_list:
                    block_content = item.get("block_content", "")
                    if block_content:
                        texts.append(block_content)

            # 合并所有文本
            if texts:
                return "\n".join(texts).strip()

            return None

        except Exception as e:
            self.logger.warning(f"OCR API call failed: {e}")
            return None
