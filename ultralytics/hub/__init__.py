# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import requests

from ultralytics.data.utils import HUBDatasetStats
from ultralytics.hub.auth import Auth
from ultralytics.hub.session import HUBTrainingSession
from ultralytics.hub.utils import HUB_API_ROOT, HUB_WEB_ROOT, PREFIX, events
from ultralytics.utils import LOGGER, SETTINGS, checks

__all__ = (
    "PREFIX",
    "HUB_WEB_ROOT",
    "HUBTrainingSession",
    "login",
    "logout",
    "reset_model",
    "export_fmts_hub",
    "export_model",
    "get_export",
    "check_dataset",
    "events",
)


def login(api_key: str = None, save=True) -> bool:
    """
    使用提供的API密钥登录到Ultralytics HUB API。

    会话不会被存储；当需要时，会使用保存的SETTINGS或HUB_API_KEY环境变量创建一个新的会话，前提是身份验证成功。

    参数:
        api_key (str, 可选): 用于身份验证的API密钥。
            如果未提供，将从SETTINGS或HUB_API_KEY环境变量中获取。
        save (bool, 可选): 是否在身份验证成功后将API密钥保存到SETTINGS中。

    返回:
        (bool): 如果身份验证成功，则返回True，否则返回False。
    """
    checks.check_requirements("hub-sdk>=0.0.12")
    from hub_sdk import HUBClient

    api_key_url = f"{HUB_WEB_ROOT}/settings?tab=api+keys"  # 设置重定向URL
    saved_key = SETTINGS.get("api_key")
    active_key = api_key or saved_key
    credentials = {"api_key": active_key} if active_key and active_key != "" else None  # 设置凭证

    client = HUBClient(credentials)  # 初始化HUBClient

    if client.authenticated:
        # 成功使用HUB进行身份验证

        if save and client.api_key != saved_key:
            SETTINGS.update({"api_key": client.api_key})  # 使用有效的API密钥更新设置

        # 根据是否提供了密钥或从设置中获取的密钥，设置消息
        log_message = (
            "新认证成功 ✅" if client.api_key == api_key or not credentials else "已认证 ✅"
        )
        LOGGER.info(f"{PREFIX}{log_message}")

        return True
    else:
        # 使用HUB身份验证失败
        LOGGER.info(f"{PREFIX}从 {api_key_url} 获取API密钥，然后运行 'yolo login API_KEY'")
        return False


def logout():
    """
    通过从设置文件中删除API密钥来注销Ultralytics HUB。要重新登录，请使用'yolo login'。

    示例:
        ```python
        from ultralytics import hub

        hub.logout()
        ```
    """
    SETTINGS["api_key"] = ""
    LOGGER.info(f"{PREFIX}已注销 ✅。要重新登录，请使用 'yolo login'.")


def reset_model(model_id=""):
    """将训练后的模型重置为未训练状态。"""
    r = requests.post(f"{HUB_API_ROOT}/model-reset", json={"modelId": model_id}, headers={"x-api-key": Auth().api_key})
    if r.status_code == 200:
        LOGGER.info(f"{PREFIX}模型重置成功")
        return
    LOGGER.warning(f"{PREFIX}模型重置失败 {r.status_code} {r.reason}")


def export_fmts_hub():
    """返回HUB支持的导出格式列表。"""
    from ultralytics.engine.exporter import export_formats

    return list(export_formats()["Argument"][1:]) + ["ultralytics_tflite", "ultralytics_coreml"]


def export_model(model_id="", format="torchscript"):
    """将模型导出为所有格式。"""
    assert format in export_fmts_hub(), f"不支持的导出格式'{format}'，有效格式为 {export_fmts_hub()}"
    r = requests.post(
        f"{HUB_API_ROOT}/v1/models/{model_id}/export", json={"format": format}, headers={"x-api-key": Auth().api_key}
    )
    assert r.status_code == 200, f"{PREFIX}{format} 导出失败 {r.status_code} {r.reason}"
    LOGGER.info(f"{PREFIX}{format} 导出已开始 ✅")


def get_export(model_id="", format="torchscript"):
    """获取已导出的模型字典及下载链接。"""
    assert format in export_fmts_hub(), f"不支持的导出格式'{format}'，有效格式为 {export_fmts_hub()}"
    r = requests.post(
        f"{HUB_API_ROOT}/get-export",
        json={"apiKey": Auth().api_key, "modelId": model_id, "format": format},
        headers={"x-api-key": Auth().api_key},
    )
    assert r.status_code == 200, f"{PREFIX}{format} get_export 失败 {r.status_code} {r.reason}"
    return r.json()


def check_dataset(path: str, task: str) -> None:
    """
    用于在上传之前检查HUB数据集Zip文件的错误。它会在数据集上传到HUB之前进行错误检查。
    使用示例如下所示。

    参数:
        path (str): 数据集Zip文件的路径（数据集zip中包含data.yaml）。
        task (str): 数据集任务。选项有'detect'，'segment'，'pose'，'classify'，'obb'。

    示例:
        从 https://github.com/ultralytics/hub/tree/main/example_datasets 下载 *.zip 文件
            例如：https://github.com/ultralytics/hub/raw/main/example_datasets/coco8.zip  获取coco8.zip。
        ```python
        from ultralytics.hub import check_dataset

        check_dataset("path/to/coco8.zip", task="detect")  # 检查检测数据集
        check_dataset("path/to/coco8-seg.zip", task="segment")  # 检查分割数据集
        check_dataset("path/to/coco8-pose.zip", task="pose")  # 检查姿态数据集
        check_dataset("path/to/dota8.zip", task="obb")  # 检查OBB数据集
        check_dataset("path/to/imagenet10.zip", task="classify")  # 检查分类数据集
        ```
    """
    HUBDatasetStats(path=path, task=task).get_json()
    LOGGER.info(f"检查已成功完成 ✅。将此数据集上传至 {HUB_WEB_ROOT}/datasets/.")
