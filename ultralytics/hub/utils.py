# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import platform
import random
import threading
import time
from pathlib import Path

import requests

from ultralytics.utils import (
    ARGV,
    ENVIRONMENT,
    IS_COLAB,
    IS_GIT_DIR,
    IS_PIP_PACKAGE,
    LOGGER,
    ONLINE,
    RANK,
    SETTINGS,
    TESTS_RUNNING,
    TQDM,
    TryExcept,
    __version__,
    colorstr,
    get_git_origin_url,
)
from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

HUB_API_ROOT = os.environ.get("ULTRALYTICS_HUB_API", "https://api.ultralytics.com")
HUB_WEB_ROOT = os.environ.get("ULTRALYTICS_HUB_WEB", "https://hub.ultralytics.com")

PREFIX = colorstr("Ultralytics HUB: ")
HELP_MSG = "如果问题持续存在，请访问 https://github.com/ultralytics/hub/issues 寻求帮助。"


def request_with_credentials(url: str) -> any:
    """
    在 Google Colab 环境中，附带 cookies 发起 AJAX 请求。

    参数:
        url (str): 要请求的 URL。

    返回:
        (any): AJAX 请求的响应数据。

    异常:
        OSError: 如果函数没有在 Google Colab 环境中运行，则抛出此异常。
    """
    if not IS_COLAB:
        raise OSError("request_with_credentials() 必须在 Colab 环境中运行")
    from google.colab import output  # noqa
    from IPython import display  # noqa

    display.display(
        display.Javascript(
            f"""
            window._hub_tmp = new Promise((resolve, reject) => {{
                const timeout = setTimeout(() => reject("认证现有浏览器会话失败"), 5000)
                fetch("{url}", {{
                    method: 'POST',
                    credentials: 'include'
                }})
                    .then((response) => resolve(response.json()))
                    .then((json) => {{
                    clearTimeout(timeout);
                    }}).catch((err) => {{
                    clearTimeout(timeout);
                    reject(err);
                }});
            }});
            """
        )
    )
    return output.eval_js("_hub_tmp")


def requests_with_progress(method, url, **kwargs):
    """
    使用指定的方法和 URL 发起 HTTP 请求，并可选地显示进度条。

    参数:
        method (str): 要使用的 HTTP 方法（例如 'GET', 'POST'）。
        url (str): 要请求的 URL。
        **kwargs (any): 传递给底层 `requests.request` 函数的其他关键字参数。

    返回:
        (requests.Response): HTTP 请求的响应对象。

    注意:
        - 如果 'progress' 设置为 True，则响应的下载进度将显示在已知内容长度的进度条上。
        - 如果 'progress' 是一个数字，则进度条将假设内容长度 = progress。
    """
    progress = kwargs.pop("progress", False)
    if not progress:
        return requests.request(method, url, **kwargs)
    response = requests.request(method, url, stream=True, **kwargs)
    total = int(response.headers.get("content-length", 0) if isinstance(progress, bool) else progress)  # 总大小
    try:
        pbar = TQDM(total=total, unit="B", unit_scale=True, unit_divisor=1024)
        for data in response.iter_content(chunk_size=1024):
            pbar.update(len(data))
        pbar.close()
    except requests.exceptions.ChunkedEncodingError:  # 避免 'Connection broken: IncompleteRead' 警告
        response.close()
    return response


def smart_request(method, url, retry=3, timeout=30, thread=True, code=-1, verbose=True, progress=False, **kwargs):
    """
    使用 'requests' 库发起 HTTP 请求，并在指定的超时之前使用指数回退重试。

    参数:
        method (str): 请求使用的 HTTP 方法。选择 'post' 或 'get'。
        url (str): 要请求的 URL。
        retry (int, 可选): 尝试重试的次数，默认值为 3。
        timeout (int, 可选): 超时时间（秒），超过该时间后将放弃重试。默认值为 30。
        thread (bool, 可选): 是否在单独的守护线程中执行请求。默认值为 True。
        code (int, 可选): 请求的标识符，用于日志记录。默认值为 -1。
        verbose (bool, 可选): 是否打印到控制台的标志。默认值为 True。
        progress (bool, 可选): 是否在请求期间显示进度条。默认值为 False。
        **kwargs (any): 传递给方法中 requests 函数的其他关键字参数。

    返回:
        (requests.Response): HTTP 响应对象。如果请求是在单独的线程中执行的，则返回 None。
    """
    retry_codes = (408, 500)  # 仅在遇到这些状态码时进行重试

    @TryExcept(verbose=verbose)
    def func(func_method, func_url, **func_kwargs):
        """发起带有重试和超时的 HTTP 请求，支持可选的进度条跟踪。"""
        r = None  # 响应
        t0 = time.time()  # 初始化计时器
        for i in range(retry + 1):
            if (time.time() - t0) > timeout:
                break
            r = requests_with_progress(func_method, func_url, **func_kwargs)  # 例如 get(url, data, json, files)
            if r.status_code < 300:  # 2xx 范围的返回码一般被认为是“成功”的
                break
            try:
                m = r.json().get("message", "没有 JSON 消息。")
            except AttributeError:
                m = "无法读取 JSON。"
            if i == 0:
                if r.status_code in retry_codes:
                    m += f" 正在重试 {retry} 次，超时 {timeout} 秒。" if retry else ""
                elif r.status_code == 429:  # 达到速率限制
                    h = r.headers  # 响应头
                    m = (
                        f"达到速率限制（{h['X-RateLimit-Remaining']}/{h['X-RateLimit-Limit']}）。"
                        f" 请在 {h['Retry-After']} 秒后重试。"
                    )
                if verbose:
                    LOGGER.warning(f"{PREFIX}{m} {HELP_MSG} ({r.status_code} #{code})")
                if r.status_code not in retry_codes:
                    return r
            time.sleep(2**i)  # 指数退避
        return r

    args = method, url
    kwargs["progress"] = progress
    if thread:
        threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()
    else:
        return func(*args, **kwargs)


class Events:
    """
    一个用于收集匿名事件分析的类。当settings中的sync=True时启用事件分析，当sync=False时禁用事件分析。
    运行 'yolo settings' 来查看和更新设置。

    属性：
        url (str): 发送匿名事件的URL。
        rate_limit (float): 发送事件的速率限制（秒）。
        metadata (dict): 一个包含环境元数据的字典。
        enabled (bool): 一个标志，根据特定条件启用或禁用事件。
    """

    url = "https://www.google-analytics.com/mp/collect?measurement_id=G-X8NCJYTQXM&api_secret=QLQrATrNSwGRFRLE-cbHJw"

    def __init__(self):
        """初始化Events对象，设置事件、速率限制和元数据的默认值。"""
        self.events = []  # 事件列表
        self.rate_limit = 30.0  # 速率限制（秒）
        self.t = 0.0  # 速率限制定时器（秒）
        self.metadata = {
            "cli": Path(ARGV[0]).name == "yolo",
            "install": "git" if IS_GIT_DIR else "pip" if IS_PIP_PACKAGE else "other",
            "python": ".".join(platform.python_version_tuple()[:2]),  # 例如：3.10
            "version": __version__,
            "env": ENVIRONMENT,
            "session_id": round(random.random() * 1e15),
            "engagement_time_msec": 1000,
        }
        self.enabled = (
            SETTINGS["sync"]
            and RANK in {-1, 0}
            and not TESTS_RUNNING
            and ONLINE
            and (IS_PIP_PACKAGE or get_git_origin_url() == "https://github.com/ultralytics/ultralytics.git")
        )

    def __call__(self, cfg):
        """
        尝试将新事件添加到事件列表，并在达到速率限制时发送事件。

        参数：
            cfg (IterableSimpleNamespace): 包含模式和任务信息的配置对象。
        """
        if not self.enabled:
            # 事件禁用，不执行任何操作
            return

        # 尝试添加到事件列表
        if len(self.events) < 25:  # 事件列表最多允许25个事件（丢弃超过此数量的事件）
            params = {
                **self.metadata,
                "task": cfg.task,
                "model": cfg.model if cfg.model in GITHUB_ASSETS_NAMES else "custom",
            }
            if cfg.mode == "export":
                params["format"] = cfg.format
            self.events.append({"name": cfg.mode, "params": params})

        # 检查速率限制
        t = time.time()
        if (t - self.t) < self.rate_limit:
            # 时间未达到速率限制，等待发送
            return

        # 时间超过速率限制，立即发送
        data = {"client_id": SETTINGS["uuid"], "events": self.events}  # SHA-256匿名化UUID哈希和事件列表

        # 等效于 requests.post(self.url, json=data)
        smart_request("post", self.url, json=data, retry=0, verbose=False)

        # 重置事件列表和速率限制定时器
        self.events = []
        self.t = t


# 在hub/utils初始化时运行以下代码 -------------------------------------------------------------------------------------
events = Events()
