# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import shutil
import threading
import time
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

from ultralytics.hub.utils import HELP_MSG, HUB_WEB_ROOT, PREFIX, TQDM
from ultralytics.utils import IS_COLAB, LOGGER, SETTINGS, __version__, checks, emojis
from ultralytics.utils.errors import HUBModelError

AGENT_NAME = f"python-{__version__}-colab" if IS_COLAB else f"python-{__version__}-local"


class HUBTrainingSession:
    """
    用于Ultralytics HUB YOLO模型的训练会话。处理模型初始化、心跳和检查点。

    属性:
        model_id (str): 被训练的YOLO模型的标识符。
        model_url (str): Ultralytics HUB中模型的URL。
        rate_limits (dict): 不同API调用的速率限制（以秒为单位）。
        timers (dict): 用于速率限制的计时器。
        metrics_queue (dict): 存储每个周期的模型指标，直到上传。
        model (dict): 从Ultralytics HUB获取的模型数据。
    """

    def __init__(self, identifier):
        """
        使用提供的模型标识符初始化HUBTrainingSession。

        参数:
            identifier (str): 用于初始化HUB训练会话的模型标识符。
                它可以是一个URL字符串，或者具有特定格式的模型密钥。

        异常:
            ValueError: 如果提供的模型标识符无效。
            ConnectionError: 如果无法使用全局API密钥进行连接。
            ModuleNotFoundError: 如果未安装hub-sdk包。
        """
        from hub_sdk import HUBClient

        self.rate_limits = {"metrics": 3, "ckpt": 900, "heartbeat": 300}  # 速率限制（秒）
        self.metrics_queue = {}  # 存储每个周期的指标，直到上传
        self.metrics_upload_failed_queue = {}  # 如果上传失败，则存储每个周期的指标
        self.timers = {}  # 存储计时器，用于 ultralytics/utils/callbacks/hub.py
        self.model = None
        self.model_url = None
        self.model_file = None
        self.train_args = None

        # 解析输入
        api_key, model_id, self.filename = self._parse_identifier(identifier)

        # 获取凭证
        active_key = api_key or SETTINGS.get("api_key")
        credentials = {"api_key": active_key} if active_key else None  # 设置凭证

        # 初始化客户端
        self.client = HUBClient(credentials)

        # 加载模型
        try:
            if model_id:
                self.load_model(model_id)  # 加载现有模型
            else:
                self.model = self.client.model()  # 加载空模型
        except Exception:
            if identifier.startswith(f"{HUB_WEB_ROOT}/models/") and not self.client.authenticated:
                LOGGER.warning(
                    f"{PREFIX}警告 ⚠️ 请使用 'yolo login API_KEY' 登录。"
                    "你可以在以下位置找到你的API密钥: https://hub.ultralytics.com/settings?tab=api+keys."
                )

    @classmethod
    def create_session(cls, identifier, args=None):
        """类方法，用于创建经过认证的HUBTrainingSession，或者返回None。"""
        try:
            session = cls(identifier)
            if args and not identifier.startswith(f"{HUB_WEB_ROOT}/models/"):  # 不是HUB模型URL
                session.create_model(args)
                assert session.model.id, "HUB模型未正确加载"
            return session
        # PermissionError和ModuleNotFoundError表示未安装hub-sdk
        except (PermissionError, ModuleNotFoundError, AssertionError):
            return None

    def load_model(self, model_id):
        """使用提供的模型标识符从Ultralytics HUB加载现有模型。"""
        self.model = self.client.model(model_id)
        if not self.model.data:  # 如果模型不存在
            raise ValueError(emojis("❌ 指定的HUB模型不存在"))  # TODO: 改进错误处理

        self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"
        if self.model.is_trained():
            print(emojis(f"加载已训练的HUB模型 {self.model_url} 🚀"))
            url = self.model.get_weights_url("best")  # 具有认证的下载URL
            self.model_file = checks.check_file(url, download_dir=Path(SETTINGS["weights_dir"]) / "hub" / self.model.id)
            return

        # 设置训练参数并开始心跳，以便HUB监控代理
        self._set_train_args()
        self.model.start_heartbeat(self.rate_limits["heartbeat"])
        LOGGER.info(f"{PREFIX}查看模型 {self.model_url} 🚀")

    def create_model(self, model_args):
        """使用指定的模型标识符初始化HUB训练会话。"""
        payload = {
            "config": {
                "batchSize": model_args.get("batch", -1),
                "epochs": model_args.get("epochs", 300),
                "imageSize": model_args.get("imgsz", 640),
                "patience": model_args.get("patience", 100),
                "device": str(model_args.get("device", "")),  # 将None转换为字符串
                "cache": str(model_args.get("cache", "ram")),  # 将True, False, None转换为字符串
            },
            "dataset": {"name": model_args.get("data")},
            "lineage": {
                "architecture": {"name": self.filename.replace(".pt", "").replace(".yaml", "")},
                "parent": {},
            },
            "meta": {"name": self.filename},
        }

        if self.filename.endswith(".pt"):
            payload["lineage"]["parent"]["name"] = self.filename

        self.model.create_model(payload)

        # 如果模型未能创建
        # TODO: 改进错误处理
        if not self.model.id:
            return None

        self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"

        # 开始心跳，以便HUB监控代理
        self.model.start_heartbeat(self.rate_limits["heartbeat"])

        LOGGER.info(f"{PREFIX}查看模型 {self.model_url} 🚀")

    @staticmethod
    def _parse_identifier(identifier):
        """
        解析给定的标识符以确定标识符类型并提取相关组件。

        该方法支持不同的标识符格式：
            - 一个HUB模型URL https://hub.ultralytics.com/models/MODEL
            - 一个带有API密钥的HUB模型URL https://hub.ultralytics.com/models/MODEL?api_key=APIKEY
            - 一个以'.pt'或'.yaml'结尾的本地文件名

        参数：
            identifier (str): 要解析的标识符字符串。

        返回：
            (tuple): 一个元组，包含API密钥、模型ID和文件名（如适用）。

        异常：
            HUBModelError: 如果标识符格式无法识别。
        """
        api_key, model_id, filename = None, None, None
        if Path(identifier).suffix in {".pt", ".yaml"}:
            filename = identifier
        elif identifier.startswith(f"{HUB_WEB_ROOT}/models/"):
            parsed_url = urlparse(identifier)
            model_id = Path(parsed_url.path).stem  # 处理可能存在的末尾斜杠
            query_params = parse_qs(parsed_url.query)  # 字典形式，例：{"api_key": ["API_KEY_HERE"]}
            api_key = query_params.get("api_key", [None])[0]
        else:
            raise HUBModelError(f"model='{identifier} 无效，正确格式是 {HUB_WEB_ROOT}/models/MODEL_ID")
        return api_key, model_id, filename

    def _set_train_args(self):
        """
        初始化训练参数并在Ultralytics HUB上创建模型条目。

        该方法根据模型的状态设置训练参数，并根据提供的额外参数更新它们。它处理模型的不同状态，例如是否可以恢复训练，是否是预训练模型，或是否需要特定的文件设置。

        异常：
            ValueError: 如果模型已训练完成，缺少所需的数据集信息，或提供的训练参数存在问题。
        """
        if self.model.is_resumable():
            # 模型有已保存的权重
            self.train_args = {"data": self.model.get_dataset_url(), "resume": True}
            self.model_file = self.model.get_weights_url("last")
        else:
            # 模型没有已保存的权重
            self.train_args = self.model.data.get("train_args")  # 新的响应

            # 将模型文件设置为*.pt或*.yaml文件
            self.model_file = (
                self.model.get_weights_url("parent") if self.model.is_pretrained() else self.model.get_architecture()
            )

        if "data" not in self.train_args:
            # RF bug - 数据集有时未导出
            raise ValueError("数据集可能仍在处理。请稍等片刻再试。")

        self.model_file = checks.check_yolov5u_filename(self.model_file, verbose=False)  # YOLOv5->YOLOv5u
        self.model_id = self.model.id

    def request_queue(
        self,
        request_func,
        retry=3,
        timeout=30,
        thread=True,
        verbose=True,
        progress_total=None,
        stream_response=None,
        *args,
        **kwargs,
    ):
        """尝试执行`request_func`，并支持重试、超时处理、可选的多线程和进度显示。"""

        def retry_request():
            """尝试调用`request_func`，支持重试、超时和可选的多线程。"""
            t0 = time.time()  # 记录超时的开始时间
            response = None
            for i in range(retry + 1):
                if (time.time() - t0) > timeout:
                    LOGGER.warning(f"{PREFIX}请求超时。 {HELP_MSG}")
                    break  # 超时，退出循环

                response = request_func(*args, **kwargs)
                if response is None:
                    LOGGER.warning(f"{PREFIX}未收到请求的响应。 {HELP_MSG}")
                    time.sleep(2**i)  # 指数退避，再次重试前等待
                    continue  # 跳过进一步处理，重新尝试

                if progress_total:
                    self._show_upload_progress(progress_total, response)
                elif stream_response:
                    self._iterate_content(response)

                if HTTPStatus.OK <= response.status_code < HTTPStatus.MULTIPLE_CHOICES:
                    # 如果请求与指标上传相关
                    if kwargs.get("metrics"):
                        self.metrics_upload_failed_queue = {}
                    return response  # 成功，不需要重试

                if i == 0:
                    # 初次尝试，检查状态码并提供提示信息
                    message = self._get_failure_message(response, retry, timeout)

                    if verbose:
                        LOGGER.warning(f"{PREFIX}{message} {HELP_MSG} ({response.status_code})")

                if not self._should_retry(response.status_code):
                    LOGGER.warning(f"{PREFIX}请求失败。 {HELP_MSG} ({response.status_code})")
                    break  # 不是需要重试的错误，退出循环

                time.sleep(2**i)  # 重试时指数退避

            # 如果请求与指标上传相关并且重试次数超过
            if response is None and kwargs.get("metrics"):
                self.metrics_upload_failed_queue.update(kwargs.get("metrics"))

            return response

        if thread:
            # 启动一个新线程运行retry_request函数
            threading.Thread(target=retry_request, daemon=True).start()
        else:
            # 如果在主线程中运行，直接调用retry_request
            return retry_request()

    @staticmethod
    def _should_retry(status_code):
        """根据HTTP状态码判断请求是否需要重试。"""
        retry_codes = {
            HTTPStatus.REQUEST_TIMEOUT,
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.GATEWAY_TIMEOUT,
        }
        return status_code in retry_codes

    def _get_failure_message(self, response: requests.Response, retry: int, timeout: int):
        """
        根据响应状态码生成重试消息。

        参数:
            response: HTTP 响应对象。
            retry: 允许的重试次数。
            timeout: 最大超时时间。

        返回:
            (str): 重试消息。
        """
        if self._should_retry(response.status_code):
            return f"正在重试 {retry} 次，超时 {timeout} 秒。" if retry else ""
        elif response.status_code == HTTPStatus.TOO_MANY_REQUESTS:  # 达到速率限制
            headers = response.headers
            return (
                f"达到速率限制（{headers['X-RateLimit-Remaining']}/{headers['X-RateLimit-Limit']}）。"
                f" 请在 {headers['Retry-After']} 秒后重试。"
            )
        else:
            try:
                return response.json().get("message", "没有 JSON 消息。")
            except AttributeError:
                return "无法读取 JSON。"

    def upload_metrics(self):
        """将模型指标上传到 Ultralytics HUB。"""
        return self.request_queue(self.model.upload_metrics, metrics=self.metrics_queue.copy(), thread=True)

    def upload_model(
        self,
        epoch: int,
        weights: str,
        is_best: bool = False,
        map: float = 0.0,
        final: bool = False,
    ) -> None:
        """
        将模型检查点上传到 Ultralytics HUB。

        参数:
            epoch (int): 当前的训练周期。
            weights (str): 模型权重文件的路径。
            is_best (bool): 表示当前模型是否是迄今为止最佳的模型。
            map (float): 模型的平均精度（Mean Average Precision）。
            final (bool): 表示该模型是否是训练结束后的最终模型。
        """
        weights = Path(weights)
        if not weights.is_file():
            last = weights.with_name(f"last{weights.suffix}")
            if final and last.is_file():
                LOGGER.warning(
                    f"{PREFIX} 警告 ⚠️ 未找到模型 'best.pt'，将 'last.pt' 复制到 'best.pt' 并上传。"
                    "当在像 Google Colab 这样的临时环境中恢复训练时，常常会发生这种情况。"
                    "为了更可靠的训练，建议使用 Ultralytics HUB Cloud。"
                    "更多信息请访问 https://docs.ultralytics.com/hub/cloud-training."
                )
                shutil.copy(last, weights)  # 将 last.pt 复制到 best.pt
            else:
                LOGGER.warning(f"{PREFIX} 警告 ⚠️ 模型上传问题。缺少模型 {weights}。")
                return

        self.request_queue(
            self.model.upload_model,
            epoch=epoch,
            weights=str(weights),
            is_best=is_best,
            map=map,
            final=final,
            retry=10,
            timeout=3600,
            thread=not final,
            progress_total=weights.stat().st_size if final else None,  # 仅在最终模型时显示进度
            stream_response=True,
        )

    @staticmethod
    def _show_upload_progress(content_length: int, response: requests.Response) -> None:
        """
        显示进度条以跟踪文件上传的进度。

        参数:
            content_length (int): 要下载的内容的总大小（字节数）。
            response (requests.Response): 来自文件下载请求的响应对象。

        返回:
            None
        """
        with TQDM(total=content_length, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            for data in response.iter_content(chunk_size=1024):
                pbar.update(len(data))

    @staticmethod
    def _iterate_content(response: requests.Response) -> None:
        """
        处理流式 HTTP 响应数据。

        参数:
            response (requests.Response): 来自文件下载请求的响应对象。

        返回:
            None
        """
        for _ in response.iter_content(chunk_size=1024):
            pass  # 不处理数据块
