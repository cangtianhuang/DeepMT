"""
OCR客户端：使用百度千帆OCR API识别图片中的公式和文本
"""

from typing import Any, Dict, Optional

import requests

from core.config_loader import get_config_value
from core.logger import get_logger


class OCRClient:
    """
    OCR客户端：使用百度千帆OCR API识别图片

    主要用于识别网页中的公式图片
    """

    _instance: Optional["OCRClient"] = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化OCR客户端"""
        if OCRClient._initialized:
            return

        self.logger = get_logger()
        # 从配置加载器获取配置值（不保存完整配置）
        self.api_key = get_config_value("web_search.baidu_api_key", "")
        self.enabled = get_config_value("web_search.ocr", False)

        # 从LLM配置获取基础URL，如果没有则使用默认值
        base_url = get_config_value("llm.url", "https://qianfan.baidubce.com/v2/")
        # 确保base_url以/结尾，然后拼接OCR端点
        self.ocr_api_url = base_url.rstrip("/") + "/ocr/paddleocr"

        if not self.api_key:
            self.logger.warning(
                "Baidu API key not found. OCR functionality will be disabled."
            )
            self.enabled = False

        OCRClient._initialized = True

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

        return self._call_ocr_api(
            image_url=image_url,
            prompt_label="ocr" if not use_layout_detection else None,
            use_layout_detection=use_layout_detection,
        )

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

            if response.status_code == 200:
                data = response.json()

                # 检查错误
                if "error" in data:
                    error_info = data["error"]
                    self.logger.warning(
                        f"OCR API error: {error_info.get('message', 'Unknown error')}"
                    )
                    return None

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

            else:
                self.logger.warning(
                    f"OCR API request failed with status {response.status_code}: {response.text}"
                )
                return None

        except Exception as e:
            self.logger.warning(f"OCR API call failed: {e}")
            return None
