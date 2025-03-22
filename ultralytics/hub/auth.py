# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import requests

from ultralytics.hub.utils import HUB_API_ROOT, HUB_WEB_ROOT, PREFIX, request_with_credentials
from ultralytics.utils import IS_COLAB, LOGGER, SETTINGS, emojis

API_KEY_URL = f"{HUB_WEB_ROOT}/settings?tab=api+keys"


class Auth:
    """
    管理认证过程，包括API密钥处理、基于cookie的认证和头部生成。

    该类支持不同的认证方法：
    1. 直接使用API密钥。
    2. 使用浏览器cookie认证（专门用于Google Colab）。
    3. 提示用户输入API密钥。

    属性：
        id_token (str or bool): 用于身份验证的令牌，初始值为False。
        api_key (str or bool): 用于认证的API密钥，初始值为False。
        model_key (bool): 模型密钥的占位符，初始值为False。
    """

    id_token = api_key = model_key = False

    def __init__(self, api_key="", verbose=False):
        """
        初始化Auth类并进行用户认证。

        处理API密钥验证、Google Colab认证和新的密钥请求。在成功认证后，更新SETTINGS。

        参数：
            api_key (str): API密钥或组合的key_id格式。
            verbose (bool): 启用详细日志。
        """
        # 如果API密钥包含组合key_model格式，分割并保留API密钥部分
        api_key = api_key.split("_")[0]

        # 设置API密钥属性为传入值或SETTINGS中的API密钥（如果未传入）
        self.api_key = api_key or SETTINGS.get("api_key", "")

        # 如果提供了API密钥
        if self.api_key:
            # 如果提供的API密钥与SETTINGS中的API密钥匹配
            if self.api_key == SETTINGS.get("api_key"):
                # 记录用户已经登录
                if verbose:
                    LOGGER.info(f"{PREFIX}已认证 ✅")
                return
            else:
                # 尝试使用提供的API密钥进行认证
                success = self.authenticate()
        # 如果未提供API密钥且环境为Google Colab笔记本
        elif IS_COLAB:
            # 尝试使用浏览器cookie进行认证
            success = self.auth_with_cookies()
        else:
            # 请求API密钥
            success = self.request_api_key()

        # 在成功认证后更新SETTINGS中的API密钥
        if success:
            SETTINGS.update({"api_key": self.api_key})
            # 记录新认证成功
            if verbose:
                LOGGER.info(f"{PREFIX}新认证成功 ✅")
        elif verbose:
            LOGGER.info(f"{PREFIX}从{API_KEY_URL}获取API密钥，然后运行'yolo login API_KEY'")

    def request_api_key(self, max_attempts=3):
        """
        提示用户输入API密钥。

        返回模型ID。
        """
        import getpass

        for attempts in range(max_attempts):
            LOGGER.info(f"{PREFIX}登录，尝试 {attempts + 1} 次，共 {max_attempts} 次")
            input_key = getpass.getpass(f"请输入来自 {API_KEY_URL} 的API密钥 ")
            self.api_key = input_key.split("_")[0]  # 如果包含模型ID，则去除
            if self.authenticate():
                return True
        raise ConnectionError(emojis(f"{PREFIX}认证失败 ❌"))

    def authenticate(self) -> bool:
        """
        尝试使用id_token或API密钥进行认证。

        返回：
            (bool): 如果认证成功返回True，失败则返回False。
        """
        try:
            if header := self.get_auth_header():
                r = requests.post(f"{HUB_API_ROOT}/v1/auth", headers=header)
                if not r.json().get("success", False):
                    raise ConnectionError("无法认证。")
                return True
            raise ConnectionError("用户未在本地认证。")
        except ConnectionError:
            self.id_token = self.api_key = False  # 重置为无效
            LOGGER.warning(f"{PREFIX}无效的API密钥 ⚠️")
            return False

    def auth_with_cookies(self) -> bool:
        """
        尝试通过cookie获取认证并设置id_token。用户必须登录到HUB并在支持的浏览器中运行。

        返回：
            (bool): 如果认证成功返回True，失败则返回False。
        """
        if not IS_COLAB:
            return False  # 当前仅在Colab中有效
        try:
            authn = request_with_credentials(f"{HUB_API_ROOT}/v1/auth/auto")
            if authn.get("success", False):
                self.id_token = authn.get("data", {}).get("idToken", None)
                self.authenticate()
                return True
            raise ConnectionError("无法获取浏览器认证信息。")
        except ConnectionError:
            self.id_token = False  # 重置为无效
            return False

    def get_auth_header(self):
        """
        获取用于进行API请求的认证头。

        返回：
            (dict): 如果id_token或API密钥已设置，则返回认证头；否则返回None。
        """
        if self.id_token:
            return {"authorization": f"Bearer {self.id_token}"}
        elif self.api_key:
            return {"x-api-key": self.api_key}
        # 否则返回None
