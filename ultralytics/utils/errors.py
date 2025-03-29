# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import emojis


class HUBModelError(Exception):
    """
    自定义异常类，用于处理与 Ultralytics YOLO 中模型获取相关的错误。

    当请求的模型未找到或无法检索时，会引发此异常。
    错误信息会经过处理，包含表情符号，以提高用户体验。

    属性:
        message (str): 异常引发时显示的错误信息。

    注意:
        错误信息会自动通过 'ultralytics.utils' 包中的 'emojis' 函数进行处理。
    """

    def __init__(self, message="Model not found. Please check model URL and try again."):
        """当模型未找到时创建异常。"""
        super().__init__(emojis(message))
