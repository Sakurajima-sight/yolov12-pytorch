# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

import cv2

from ultralytics.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_PATH,
    DEFAULT_SOL_DICT,
    IS_VSCODE,
    LOGGER,
    RANK,
    ROOT,
    RUNS_DIR,
    SETTINGS,
    SETTINGS_FILE,
    TESTS_RUNNING,
    IterableSimpleNamespace,
    __version__,
    checks,
    colorstr,
    deprecation_warn,
    vscode_msg,
    yaml_load,
    yaml_print,
)

# 定义有效的解决方案
SOLUTION_MAP = {
    "count": ("ObjectCounter", "count"),
    "heatmap": ("Heatmap", "generate_heatmap"),
    "queue": ("QueueManager", "process_queue"),
    "speed": ("SpeedEstimator", "estimate_speed"),
    "workout": ("AIGym", "monitor"),
    "analytics": ("Analytics", "process_data"),
    "trackzone": ("TrackZone", "trackzone"),
    "inference": ("Inference", "inference"),
    "help": None,
}

# 定义有效的任务和模式
MODES = {"train", "val", "predict", "export", "track", "benchmark"}
TASKS = {"detect", "segment", "classify", "pose", "obb"}
TASK2DATA = {
    "detect": "coco8.yaml",
    "segment": "coco8-seg.yaml",
    "classify": "imagenet10",
    "pose": "coco8-pose.yaml",
    "obb": "dota8.yaml",
}
TASK2MODEL = {
    "detect": "yolo11n.pt",
    "segment": "yolo11n-seg.pt",
    "classify": "yolo11n-cls.pt",
    "pose": "yolo11n-pose.pt",
    "obb": "yolo11n-obb.pt",
}
TASK2METRIC = {
    "detect": "metrics/mAP50-95(B)",
    "segment": "metrics/mAP50-95(M)",
    "classify": "metrics/accuracy_top1",
    "pose": "metrics/mAP50-95(P)",
    "obb": "metrics/mAP50-95(B)",
}
MODELS = {TASK2MODEL[task] for task in TASKS}

ARGV = sys.argv or ["", ""]  # 有时 sys.argv 为空 []
SOLUTIONS_HELP_MSG = f"""
    收到的参数: {str(["yolo"] + ARGV[1:])}。Ultralytics 'yolo solutions' 用法概述：

        yolo solutions SOLUTION ARGS

        其中 SOLUTION（可选）是以下之一 {list(SOLUTION_MAP.keys())[:-1]}
              ARGS（可选）是任何数量的自定义 'arg=value' 对，如 'show_in=True'，覆盖默认值 
                  详见 https://docs.ultralytics.com/usage/cfg
                
    1. 调用物体计数解决方案
        yolo solutions count source="path/to/video/file.mp4" region=[(20, 400), (1080, 400), (1080, 360), (20, 360)]

    2. 调用热力图解决方案
        yolo solutions heatmap colormap=cv2.COLORMAP_PARULA model=yolo11n.pt

    3. 调用队列管理解决方案
        yolo solutions queue region=[(20, 400), (1080, 400), (1080, 360), (20, 360)] model=yolo11n.pt

    4. 调用运动监控解决方案（例如俯卧撑）
        yolo solutions workout model=yolo11n-pose.pt kpts=[6, 8, 10]

    5. 生成分析图表
        yolo solutions analytics analytics_type="pie"
    
    6. 在特定区域内跟踪物体
        yolo solutions trackzone source="path/to/video/file.mp4" region=[(150, 150), (1130, 150), (1130, 570), (150, 570)]
        
    7. Streamlit 实时摄像头推理 GUI
        yolo streamlit-predict
    """
CLI_HELP_MSG = f"""
    收到的参数: {str(["yolo"] + ARGV[1:])}。Ultralytics 'yolo' 命令使用以下语法：

        yolo TASK MODE ARGS

        其中   TASK（可选）是以下之一 {TASKS}
                MODE（必需）是以下之一 {MODES}
                ARGS（可选）是任何数量的自定义 'arg=value' 对，如 'imgsz=320'，覆盖默认值。
                    查看所有 ARGS 详见 https://docs.ultralytics.com/usage/cfg 或使用 'yolo cfg'

    1. 训练一个检测模型，训练 10 个周期，初始学习率为 0.01
        yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01

    2. 使用预训练的分割模型预测 YouTube 视频，图像大小为 320：
        yolo predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

    3. 使用批量大小为 1，图像大小为 640 的预训练检测模型进行验证：
        yolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640

    4. 将 YOLO11n 分类模型导出为 ONNX 格式，图像大小为 224x128（不需要 TASK）
        yolo export model=yolo11n-cls.pt format=onnx imgsz=224,128

    5. Ultralytics 解决方案用法
        yolo solutions count 或在 {list(SOLUTION_MAP.keys())[1:-1]} 中 source="path/to/video/file.mp4"

    6. 运行特殊命令：
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        yolo solutions help

    文档: https://docs.ultralytics.com
    解决方案: https://docs.ultralytics.com/solutions/
    社区: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

# 定义参数类型检查的键
CFG_FLOAT_KEYS = {  # 整数或浮动参数，即 x=2 和 x=2.0
    "warmup_epochs",
    "box",
    "cls",
    "dfl",
    "degrees",
    "shear",
    "time",
    "workspace",
    "batch",
}
CFG_FRACTION_KEYS = {  # 分数浮动参数，值范围为 0.0<=value<=1.0
    "dropout",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_momentum",
    "warmup_bias_lr",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "translate",
    "scale",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "copy_paste",
    "conf",
    "iou",
    "fraction",
}
CFG_INT_KEYS = {  # 仅整数的参数
    "epochs",
    "patience",
    "workers",
    "seed",
    "close_mosaic",
    "mask_ratio",
    "max_det",
    "vid_stride",
    "line_width",
    "nbs",
    "save_period",
}
CFG_BOOL_KEYS = {  # 仅布尔类型的参数
    "save",
    "exist_ok",
    "verbose",
    "deterministic",
    "single_cls",
    "rect",
    "cos_lr",
    "overlap_mask",
    "val",
    "save_json",
    "save_hybrid",
    "half",
    "dnn",
    "plots",
    "show",
    "save_txt",
    "save_conf",
    "save_crop",
    "save_frames",
    "show_labels",
    "show_conf",
    "visualize",
    "augment",
    "agnostic_nms",
    "retina_masks",
    "show_boxes",
    "keras",
    "optimize",
    "int8",
    "dynamic",
    "simplify",
    "nms",
    "profile",
    "multi_scale",
}


def cfg2dict(cfg):
    """
    将配置对象转换为字典。

    参数:
        cfg (str | Path | Dict | SimpleNamespace): 要转换的配置对象。可以是文件路径、字符串、字典或 SimpleNamespace 对象。

    返回:
        (Dict): 转换后的配置字典。

    示例:
        将 YAML 文件路径转换为字典：
        >>> config_dict = cfg2dict("config.yaml")

        将 SimpleNamespace 转换为字典：
        >>> from types import SimpleNamespace
        >>> config_sn = SimpleNamespace(param1="value1", param2="value2")
        >>> config_dict = cfg2dict(config_sn)

        传入已存在的字典：
        >>> config_dict = cfg2dict({"param1": "value1", "param2": "value2"})

    说明:
        - 如果 cfg 是路径或字符串，则将其作为 YAML 加载并转换为字典。
        - 如果 cfg 是 SimpleNamespace 对象，则使用 vars() 将其转换为字典。
        - 如果 cfg 已经是字典，则直接返回原字典。
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # 加载字典
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # 转换为字典
    return cfg


def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):
    """
    从文件或字典加载并合并配置数据，并可选地应用覆盖。

    参数:
        cfg (str | Path | Dict | SimpleNamespace): 配置数据源。可以是文件路径、字典或 SimpleNamespace 对象。
        overrides (Dict | None): 包含键值对的字典，用于覆盖基础配置。

    返回:
        (SimpleNamespace): 包含合并配置参数的命名空间。

    示例:
        >>> from ultralytics.cfg import get_cfg
        >>> config = get_cfg()  # 加载默认配置
        >>> config_with_overrides = get_cfg("path/to/config.yaml", overrides={"epochs": 50, "batch_size": 16})

    说明:
        - 如果同时提供了 `cfg` 和 `overrides`，则 `overrides` 中的值会优先。
        - 特别处理确保配置的一致性和正确性，例如将数字类型的 `project` 和 `name` 转换为字符串，并验证配置键值。
        - 函数对配置数据进行了类型和值检查。
    """
    cfg = cfg2dict(cfg)

    # 合并覆盖配置
    if overrides:
        overrides = cfg2dict(overrides)
        if "save_dir" not in cfg:
            overrides.pop("save_dir", None)  # 特殊覆盖键值需忽略
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}  # 合并 cfg 和 overrides 字典（优先使用 overrides）

    # 特殊处理数字类型的 project/name
    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":  # 将 model 赋值给 'name' 参数
        cfg["name"] = str(cfg.get("model", "")).split(".")[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' 自动更新为 'name={cfg['name']}'。")

    # 类型和值检查
    check_cfg(cfg)

    # 返回命名空间实例
    return IterableSimpleNamespace(**cfg)


def check_cfg(cfg, hard=True):
    """
    检查 Ultralytics 库的配置参数类型和值。

    该函数验证配置参数的类型和值，确保其正确性，并在必要时进行转换。
    它检查全局变量中定义的特定键类型，如 CFG_FLOAT_KEYS、CFG_FRACTION_KEYS、CFG_INT_KEYS 和 CFG_BOOL_KEYS。

    参数:
        cfg (Dict): 需要验证的配置字典。
        hard (bool): 如果为 True，则对无效的类型和值抛出异常；如果为 False，则尝试进行转换。

    示例:
        >>> config = {
        ...     "epochs": 50,  # 有效的整数
        ...     "lr0": 0.01,  # 有效的浮动数
        ...     "momentum": 1.2,  # 无效的浮动数（超出 0.0-1.0 范围）
        ...     "save": "true",  # 无效的布尔值
        ... }
        >>> check_cfg(config, hard=False)
        >>> print(config)
        {'epochs': 50, 'lr0': 0.01, 'momentum': 1.2, 'save': False}  # 修正后的 'save' 键

    说明:
        - 该函数会直接修改输入字典。
        - None 值会被忽略，因为它们可能来自可选参数。
        - 分数键会检查值是否在范围 [0.0, 1.0] 之间。
    """
    for k, v in cfg.items():
        if v is not None:  # None 值可能来自可选参数
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' 的类型 {type(v).__name__} 无效。"
                        f"'{k}' 的有效类型是 int（如 '{k}=0'）或 float（如 '{k}=0.5'）"
                    )
                cfg[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' 的类型 {type(v).__name__} 无效。"
                            f"'{k}' 的有效类型是 int（如 '{k}=0'）或 float（如 '{k}=0.5'）"
                        )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' 是无效值。'{k}' 的有效值应在 0.0 到 1.0 之间。")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' 的类型 {type(v).__name__} 无效。'{k}' 必须是整数（如 '{k}=8'）"
                    )
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' 的类型 {type(v).__name__} 无效。"
                        f"'{k}' 必须是布尔值（如 '{k}=True' 或 '{k}=False'）"
                    )
                cfg[k] = bool(v)


def get_save_dir(args, name=None):
    """
    返回用于保存输出的目录路径，来自参数或默认设置。

    参数:
        args (SimpleNamespace): 包含配置的命名空间对象，如 'project'、'name'、'task'、'mode' 和 'save_dir'。
        name (str | None): 可选的输出目录名称。如果未提供，则默认为 'args.name' 或 'args.mode'。

    返回:
        (Path): 输出应该保存的目录路径。

    示例:
        >>> from types import SimpleNamespace
        >>> args = SimpleNamespace(project="my_project", task="detect", mode="train", exist_ok=True)
        >>> save_dir = get_save_dir(args)
        >>> print(save_dir)
        my_project/detect/train
    """
    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path

        project = args.project or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)

    return Path(save_dir)


def _handle_deprecation(custom):
    """
    处理已弃用的配置键，将其映射到当前的等效项，并给出弃用警告。

    参数:
        custom (Dict): 可能包含已弃用键的配置字典。

    示例:
        >>> custom_config = {"boxes": True, "hide_labels": "False", "line_thickness": 2}
        >>> _handle_deprecation(custom_config)
        >>> print(custom_config)
        {'show_boxes': True, 'show_labels': True, 'line_width': 2}

    说明:
        该函数会直接修改输入字典，将已弃用的键替换为其当前的等效项。如果需要，还会处理值的转换，
        例如对 'hide_labels' 和 'hide_conf' 进行布尔值反转。
    """
    for key in custom.copy().keys():
        if key == "boxes":
            deprecation_warn(key, "show_boxes")
            custom["show_boxes"] = custom.pop("boxes")
        if key == "hide_labels":
            deprecation_warn(key, "show_labels")
            custom["show_labels"] = custom.pop("hide_labels") == "False"
        if key == "hide_conf":
            deprecation_warn(key, "show_conf")
            custom["show_conf"] = custom.pop("hide_conf") == "False"
        if key == "line_thickness":
            deprecation_warn(key, "line_width")
            custom["line_width"] = custom.pop("line_thickness")
        if key == "label_smoothing":
            deprecation_warn(key)
            custom.pop("label_smoothing")

    return custom


def check_dict_alignment(base: Dict, custom: Dict, e=None):
    """
    检查自定义配置字典与基础配置字典之间的对齐情况，处理已弃用的键，并为不匹配的键提供错误信息。

    参数:
        base (Dict): 基础配置字典，包含有效的键。
        custom (Dict): 需要检查对齐的自定义配置字典。
        e (Exception | None): 可选的异常实例，由调用函数传递。

    异常:
        SystemExit: 如果自定义字典与基础字典的键不匹配，抛出此异常。

    示例:
        >>> base_cfg = {"epochs": 50, "lr0": 0.01, "batch_size": 16}
        >>> custom_cfg = {"epoch": 100, "lr": 0.02, "batch_size": 32}
        >>> try:
        ...     check_dict_alignment(base_cfg, custom_cfg)
        ... except SystemExit:
        ...     print("发现不匹配的键")

    说明:
        - 根据与有效键的相似性，建议修正不匹配的键。
        - 自动将自定义配置中的已弃用键替换为更新的等效项。
        - 打印每个不匹配的键的详细错误信息，以帮助用户修正配置。
    """
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (set(x.keys()) for x in (base, custom))
    if mismatched := [k for k in custom_keys if k not in base_keys]:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)  # 键列表
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f"类似的参数是 i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' 不是有效的 YOLO 参数。 {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e


def merge_equals_args(args: List[str]) -> List[str]:
    """
    合并带有孤立 '=' 的参数，并将带有括号的片段连接起来。

    该函数处理以下几种情况：
    1. ['arg', '=', 'val'] 变成 ['arg=val']
    2. ['arg=', 'val'] 变成 ['arg=val']
    3. ['arg', '=val'] 变成 ['arg=val']
    4. 将带括号的片段连接，如 ['imgsz=[3,', '640,', '640]'] 变成 ['imgsz=[3,640,640]']

    参数:
        args (List[str]): 一个字符串列表，每个元素代表一个参数或片段。

    返回:
        List[str]: 一个字符串列表，其中孤立的 '=' 周围的参数已合并，带括号的片段已连接。

    示例:
        >>> args = ["arg1", "=", "value", "arg2=", "value2", "arg3", "=value3", "imgsz=[3,", "640,", "640]"]
        >>> merge_and_join_args(args)
        ['arg1=value', 'arg2=value2', 'arg3=value3', 'imgsz=[3,640,640]']
    """
    new_args = []
    current = ""
    depth = 0

    i = 0
    while i < len(args):
        arg = args[i]

        # 处理等号合并
        if arg == "=" and 0 < i < len(args) - 1:  # 合并 ['arg', '=', 'val']
            new_args[-1] += f"={args[i + 1]}"
            i += 2
            continue
        elif arg.endswith("=") and i < len(args) - 1 and "=" not in args[i + 1]:  # 合并 ['arg=', 'val']
            new_args.append(f"{arg}{args[i + 1]}")
            i += 2
            continue
        elif arg.startswith("=") and i > 0:  # 合并 ['arg', '=val']
            new_args[-1] += arg
            i += 1
            continue

        # 处理括号合并
        depth += arg.count("[") - arg.count("]")
        current += arg
        if depth == 0:
            new_args.append(current)
            current = ""

        i += 1

    # 如果还有剩余的当前字符串，添加到新列表中
    if current:
        new_args.append(current)

    return new_args


def handle_yolo_hub(args: List[str]) -> None:
    """
    处理 Ultralytics HUB 命令行接口（CLI）命令，用于身份验证。

    该函数处理 Ultralytics HUB 的 CLI 命令，如登录和注销。它应该在执行与 HUB 身份验证相关的脚本时调用。

    参数:
        args (List[str]): 一组命令行参数。第一个参数应为 'login' 或 'logout'。对于 'login'，第二个可选参数可以是 API 密钥。

    示例:
        ```bash
        yolo login YOUR_API_KEY
        ```

    说明:
        - 该函数从 ultralytics 导入 'hub' 模块来执行登录和注销操作。
        - 对于 'login' 命令，如果没有提供 API 密钥，将传递一个空字符串给登录函数。
        - 'logout' 命令不需要任何额外的参数。
    """
    from ultralytics import hub

    if args[0] == "login":
        key = args[1] if len(args) > 1 else ""
        # 使用提供的 API 密钥登录到 Ultralytics HUB
        hub.login(key)
    elif args[0] == "logout":
        # 从 Ultralytics HUB 注销
        hub.logout()


def handle_yolo_settings(args: List[str]) -> None:
    """
    处理 YOLO 设置命令行接口（CLI）命令。

    该函数处理 YOLO 设置 CLI 命令，如重置和更新单个设置。它应该在执行与 YOLO 设置管理相关的脚本时调用。

    参数:
        args (List[str]): 一组命令行参数，用于 YOLO 设置管理。

    示例:
        >>> handle_yolo_settings(["reset"])  # 重置 YOLO 设置
        >>> handle_yolo_settings(["default_cfg_path=yolo11n.yaml"])  # 更新特定设置

    说明:
        - 如果没有提供任何参数，函数将显示当前的设置。
        - 'reset' 命令将删除现有的设置文件，并创建新的默认设置。
        - 其他参数被视为键值对，用于更新特定的设置。
        - 函数将检查提供的设置与现有设置之间的一致性。
        - 处理后，将显示更新后的设置。
        - 有关处理 YOLO 设置的更多信息，请访问:
          https://docs.ultralytics.com/quickstart/#ultralytics-settings
    """
    url = "https://docs.ultralytics.com/quickstart/#ultralytics-settings"  # 帮助 URL
    try:
        if any(args):
            if args[0] == "reset":
                SETTINGS_FILE.unlink()  # 删除设置文件
                SETTINGS.reset()  # 创建新的设置
                LOGGER.info("设置已成功重置")  # 通知用户设置已重置
            else:  # 保存新的设置
                new = dict(parse_key_value_pair(a) for a in args)
                check_dict_alignment(SETTINGS, new)
                SETTINGS.update(new)

        print(SETTINGS)  # 打印当前设置
        LOGGER.info(f"💡 了解更多关于 Ultralytics 设置的信息，请访问 {url}")
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ 设置错误: '{e}'。请参阅 {url} 获取帮助。")


def handle_yolo_solutions(args: List[str]) -> None:
    """
    处理 YOLO 解决方案参数并运行指定的计算机视觉解决方案管道。

    参数:
        args (List[str]): 配置和运行 Ultralytics YOLO 解决方案的命令行参数: https://docs.ultralytics.com/solutions/
            这些参数可以包括解决方案名称、源以及其他配置参数。

    返回:
        None: 该函数处理视频帧并保存输出，但不返回任何值。

    示例:
        使用默认设置运行人员计数解决方案：
        >>> handle_yolo_solutions(["count"])

        使用自定义配置运行分析：
        >>> handle_yolo_solutions(["analytics", "conf=0.25", "source=path/to/video/file.mp4"])

        使用自定义配置运行推理，需要 Streamlit 版本 1.29.0 或更高：
        >>> handle_yolo_solutions(["inference", "model=yolo11n.pt"])

    说明:
        - 默认配置从 DEFAULT_SOL_DICT 和 DEFAULT_CFG_DICT 合并。
        - 参数可以以 'key=value' 格式提供，也可以作为布尔标志。
        - 可用的解决方案在 SOLUTION_MAP 中定义，包含相应的类和方法。
        - 如果提供了无效的解决方案，将默认使用 'count' 解决方案。
        - 输出视频将保存在 'runs/solution/{solution_name}' 目录中。
        - 对于 'analytics' 解决方案，帧数会跟踪以生成分析图表。
        - 视频处理可以通过按 'q' 中断。
        - 按顺序处理视频帧，并以 .avi 格式保存输出。
        - 如果未指定源，则会下载并使用默认的示例视频。
        - 推理解决方案将使用 'streamlit run' 命令启动。
        - Streamlit 应用文件位于 Ultralytics 包目录中。
    """
    full_args_dict = {**DEFAULT_SOL_DICT, **DEFAULT_CFG_DICT}  # 参数字典
    overrides = {}

    # 检查字典对齐
    for arg in merge_equals_args(args):
        arg = arg.lstrip("-").rstrip(",")
        if "=" in arg:
            try:
                k, v = parse_key_value_pair(arg)
                overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {arg: ""}, e)
        elif arg in full_args_dict and isinstance(full_args_dict.get(arg), bool):
            overrides[arg] = True
    check_dict_alignment(full_args_dict, overrides)  # 字典对齐

    # 获取解决方案名称
    if args and args[0] in SOLUTION_MAP:
        if args[0] != "help":
            s_n = args.pop(0)  # 直接提取解决方案名称
        else:
            LOGGER.info(SOLUTIONS_HELP_MSG)
    else:
        LOGGER.warning(
            f"⚠️ 没有提供有效的解决方案。使用默认的 'count'。可用的解决方案: {', '.join(SOLUTION_MAP.keys())}"
        )
        s_n = "count"  # 如果没有提供，使用默认的解决方案 'count'

    if args and args[0] == "help":  # 如果用户调用 `yolo solutions help`，则返回
        return

    if s_n == "inference":
        checks.check_requirements("streamlit>=1.29.0")
        LOGGER.info("💡 加载 Ultralytics 实时推理应用...")
        subprocess.run(
            [  # 使用 Streamlit 自定义参数运行子进程
                "streamlit",
                "run",
                str(ROOT / "solutions/streamlit_inference.py"),
                "--server.headless",
                "true",
                overrides.pop("model", "yolo11n.pt"),
            ]
        )
    else:
        cls, method = SOLUTION_MAP[s_n]  # 解决方案类名、方法名和默认源

        from ultralytics import solutions  # 导入 ultralytics 解决方案

        solution = getattr(solutions, cls)(IS_CLI=True, **overrides)  # 获取解决方案类，如 ObjectCounter
        process = getattr(
            solution, method
        )  # 获取特定类的方法进行处理，例如，ObjectCounter 的 count 方法

        cap = cv2.VideoCapture(solution.CFG["source"])  # 读取视频文件

        # 提取视频文件的宽度、高度和 fps，创建保存目录并初始化视频写入器
        import os  # 用于目录创建
        from pathlib import Path

        from ultralytics.utils.files import increment_path  # 用于更新输出目录路径

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        if s_n == "analytics":  # 对于分析图形，输出具有固定的形状，如 w=1920, h=1080
            w, h = 1920, 1080
        save_dir = increment_path(Path("runs") / "solutions" / "exp", exist_ok=False)
        save_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录
        vw = cv2.VideoWriter(os.path.join(save_dir, "solution.avi"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        try:  # 处理视频帧
            f_n = 0  # 帧号，分析图表时需要
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                frame = process(frame, f_n := f_n + 1) if s_n == "analytics" else process(frame)
                vw.write(frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()


def parse_key_value_pair(pair: str = "key=value"):
    """
    解析键值对字符串，分离成单独的键和值。

    参数:
        pair (str): 包含键值对的字符串，格式为 "key=value"。

    返回:
        key (str): 解析出的键。
        value (str): 解析出的值。

    异常:
        AssertionError: 如果值为空或缺失。

    示例:
        >>> key, value = parse_key_value_pair("model=yolo11n.pt")
        >>> print(f"Key: {key}, Value: {value}")
        Key: model, Value: yolo11n.pt

        >>> key, value = parse_key_value_pair("epochs=100")
        >>> print(f"Key: {key}, Value: {value}")
        Key: epochs, Value: 100

    备注:
        - 该函数会在第一个 '=' 字符处分割输入字符串。
        - 会去除键和值两端的空格。
        - 如果值为空，会抛出断言错误。
    """
    k, v = pair.split("=", 1)  # 在第一个 '=' 处分割
    k, v = k.strip(), v.strip()  # 去除两端空格
    assert v, f"missing '{k}' value"
    return k, smart_value(v)


def smart_value(v):
    """
    将字符串表示的值转换为相应的 Python 类型。

    该函数会尝试将字符串转换为最适合的 Python 对象类型，支持转换为 None、bool、int、float 以及其他可以安全评估的类型。

    参数:
        v (str): 要转换的值的字符串表示。

    返回:
        (Any): 转换后的值。可以是 None、bool、int、float，或者如果没有合适的转换，返回原始字符串。

    示例:
        >>> smart_value("42")
        42
        >>> smart_value("3.14")
        3.14
        >>> smart_value("True")
        True
        >>> smart_value("None")
        None
        >>> smart_value("some_string")
        'some_string'

    备注:
        - 该函数对布尔值和 None 值的比较是不区分大小写的。
        - 对于其他类型，它会尝试使用 Python 的 eval() 函数，虽然如果用在不信任的输入上可能存在安全风险。
        - 如果无法转换，返回原始字符串。
    """
    v_lower = v.lower()
    if v_lower == "none":
        return None
    elif v_lower == "true":
        return True
    elif v_lower == "false":
        return False
    else:
        try:
            return eval(v)
        except Exception:
            return v


def entrypoint(debug=""):
    """
    Ultralytics 入口函数，用于解析和执行命令行参数。

    该函数是 Ultralytics CLI 的主要入口，负责解析命令行参数并执行相应任务，如训练、验证、预测、导出模型等。

    参数:
        debug (str): 用于调试目的的空格分隔的命令行参数字符串。

    示例:
        训练一个检测模型 10 个 epoch，初始学习率为 0.01：
        >>> entrypoint("train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01")

        使用预训练的分割模型预测 YouTube 视频，图像大小为 320：
        >>> entrypoint("predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320")

        验证一个预训练的检测模型，批量大小为 1，图像大小为 640：
        >>> entrypoint("val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640")

    备注:
        - 如果没有传递任何参数，函数将显示用法帮助信息。
        - 有关所有可用命令及其参数，请参阅提供的帮助信息和 Ultralytics 文档：https://docs.ultralytics.com。
    """
    args = (debug.split(" ") if debug else ARGV)[1:]
    if not args:  # 没有传递参数
        LOGGER.info(CLI_HELP_MSG)
        return

    special = {
        "help": lambda: LOGGER.info(CLI_HELP_MSG),
        "checks": checks.collect_system_info,
        "version": lambda: LOGGER.info(__version__),
        "settings": lambda: handle_yolo_settings(args[1:]),
        "cfg": lambda: yaml_print(DEFAULT_CFG_PATH),
        "hub": lambda: handle_yolo_hub(args[1:]),
        "login": lambda: handle_yolo_hub(args),
        "logout": lambda: handle_yolo_hub(args),
        "copy-cfg": copy_default_cfg,
        "solutions": lambda: handle_yolo_solutions(args[1:]),
    }
    full_args_dict = {**DEFAULT_CFG_DICT, **{k: None for k in TASKS}, **{k: None for k in MODES}, **special}

    # 定义特殊命令的常见误用，如 -h, -help, --help
    special.update({k[0]: v for k, v in special.items()})  # 单数
    special.update({k[:-1]: v for k, v in special.items() if len(k) > 1 and k.endswith("s")})  # 单数
    special = {**special, **{f"-{k}": v for k, v in special.items()}, **{f"--{k}": v for k, v in special.items()}}

    overrides = {}  # 基本的覆盖，i.e. imgsz=320
    for a in merge_equals_args(args):  # 合并 "=" 符号周围的空格
        if a.startswith("--"):
            LOGGER.warning(f"WARNING ⚠️ 参数 '{a}' 不需要前导破折号 '--'，已更新为 '{a[2:]}'。")
            a = a[2:]
        if a.endswith(","):
            LOGGER.warning(f"WARNING ⚠️ 参数 '{a}' 不需要尾部逗号 ','，已更新为 '{a[:-1]}'。")
            a = a[:-1]
        if "=" in a:
            try:
                k, v = parse_key_value_pair(a)
                if k == "cfg" and v is not None:  # 自定义.yaml 文件
                    LOGGER.info(f"用 {v} 替换 {DEFAULT_CFG_PATH}")
                    overrides = {k: val for k, val in yaml_load(checks.check_yaml(v)).items() if k != "cfg"}
                else:
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {a: ""}, e)

        elif a in TASKS:
            overrides["task"] = a
        elif a in MODES:
            overrides["mode"] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True  # 默认布尔参数自动设置为 True，例如 'yolo show' 设置 show=True
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(
                f"'{colorstr('red', 'bold', a)}' 是有效的 YOLO 参数，但缺少 '=' 来设置其值，例如尝试 '{a}={DEFAULT_CFG_DICT[a]}'\n{CLI_HELP_MSG}"
            )
        else:
            check_dict_alignment(full_args_dict, {a: ""})

    # 检查键
    check_dict_alignment(full_args_dict, overrides)

    # 模式
    mode = overrides.get("mode")
    if mode is None:
        mode = DEFAULT_CFG.mode or "predict"
        LOGGER.warning(f"WARNING ⚠️ 缺少 'mode' 参数。有效的模式是 {MODES}。使用默认 'mode={mode}'。")
    elif mode not in MODES:
        raise ValueError(f"无效的 'mode={mode}'。有效的模式是 {MODES}。\n{CLI_HELP_MSG}")

    # 任务
    task = overrides.pop("task", None)
    if task:
        if task == "classify" and mode == "track":
            raise ValueError(
                f"❌ 分类不支持 'mode=track'。分类的有效模式是 {MODES - {'track'}}。\n{CLI_HELP_MSG}"
            )
        elif task not in TASKS:
            if task == "track":
                LOGGER.warning(
                    "WARNING ⚠️ 无效的 'task=track'，设置 'task=detect' 和 'mode=track'。有效的任务是 {TASKS}。\n{CLI_HELP_MSG}."
                )
                task, mode = "detect", "track"
            else:
                raise ValueError(f"无效的 'task={task}'。有效的任务是 {TASKS}。\n{CLI_HELP_MSG}")
        if "model" not in overrides:
            overrides["model"] = TASK2MODEL[task]

    # 模型
    model = overrides.pop("model", DEFAULT_CFG.model)
    if model is None:
        model = "yolo11n.pt"
        LOGGER.warning(f"WARNING ⚠️ 缺少 'model' 参数。使用默认 'model={model}'。")
    overrides["model"] = model
    stem = Path(model).stem.lower()
    if "rtdetr" in stem:  # 猜测架构
        from ultralytics import RTDETR

        model = RTDETR(model)  # 无任务参数
    elif "fastsam" in stem:
        from ultralytics import FastSAM

        model = FastSAM(model)
    elif "sam_" in stem or "sam2_" in stem or "sam2.1_" in stem:
        from ultralytics import SAM

        model = SAM(model)
    else:
        from ultralytics import YOLO

        model = YOLO(model, task=task)
    if isinstance(overrides.get("pretrained"), str):
        model.load(overrides["pretrained"])

    # 任务更新
    if task != model.task:
        if task:
            LOGGER.warning(
                f"WARNING ⚠️ 'task={task}' 与 'task={model.task}' 模型冲突。忽略 'task={task}' 并更新为 'task={model.task}' 来匹配模型。"
            )
        task = model.task

    # 模式
    if mode in {"predict", "track"} and "source" not in overrides:
        overrides["source"] = (
            "https://ultralytics.com/images/boats.jpg" if task == "obb" else DEFAULT_CFG.source or ASSETS
        )
        LOGGER.warning(f"WARNING ⚠️ 缺少 'source' 参数。使用默认 'source={overrides['source']}'。")
    elif mode in {"train", "val"}:
        if "data" not in overrides and "resume" not in overrides:
            overrides["data"] = DEFAULT_CFG.data or TASK2DATA.get(task or DEFAULT_CFG.task, DEFAULT_CFG.data)
            LOGGER.warning(f"WARNING ⚠️ 缺少 'data' 参数。使用默认 'data={overrides['data']}'。")
    elif mode == "export":
        if "format" not in overrides:
            overrides["format"] = DEFAULT_CFG.format or "torchscript"
            LOGGER.warning(f"WARNING ⚠️ 缺少 'format' 参数。使用默认 'format={overrides['format']}'。")

    # 运行命令
    getattr(model, mode)(**overrides)  # 模型的默认参数

    # 显示帮助
    LOGGER.info(f"💡 更多信息请访问 https://docs.ultralytics.com/modes/{mode}")

    # 推荐 VS Code 扩展
    if IS_VSCODE and SETTINGS.get("vscode_msg", True):
        LOGGER.info(vscode_msg())


# 特殊模式 --------------------------------------------------------------------------------------------------------
def copy_default_cfg():
    """
    复制默认配置文件，并在文件名后添加 '_copy' 后缀创建一个新文件。

    该函数会复制现有的默认配置文件（DEFAULT_CFG_PATH），并将其保存为带有 '_copy' 后缀的新文件
    ，该文件会保存在当前工作目录中。它为用户提供了基于默认设置创建自定义配置文件的方便方式。

    示例:
        >>> copy_default_cfg()
        # 输出: default.yaml 复制到 /path/to/current/directory/default_copy.yaml
        # 使用该自定义配置的 YOLO 命令示例：
        #   yolo cfg='/path/to/current/directory/default_copy.yaml' imgsz=320 batch=8

    备注:
        - 新的配置文件会创建在当前工作目录中。
        - 复制后，函数会打印新文件的位置，并提供一个使用该新配置文件的 YOLO 命令示例。
        - 该函数对于那些希望修改默认配置而不改变原始文件的用户非常有用。
    """
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(
        f"{DEFAULT_CFG_PATH} 已复制到 {new_file}\n"
        f"使用该新自定义配置的 YOLO 命令示例：\n    yolo cfg='{new_file}' imgsz=320 batch=8"
    )


if __name__ == "__main__":
    # 示例: entrypoint(debug='yolo predict model=yolo11n.pt')
    entrypoint(debug="")
